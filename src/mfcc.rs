//! Audio feature extraction for `SpeechAligner` training.
//!
//! Supported feature modes:
//! - `mfcc39`: pre-emphasis -> framing -> Hann -> FFT -> 40-bin mel bank -> log
//!   energy -> DCT-II (13 coefficients) -> delta + delta-delta = 39 total.
//! - `logmel80`: pre-emphasis -> framing -> Hann -> FFT -> 80-bin mel bank ->
//!   log energy.

use std::f32::consts::PI;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Sample rate expected by the MFCC pipeline.
pub const SAMPLE_RATE: usize = 16_000;

/// MFCC feature dimension (13 static + 13 delta + 13 delta-delta).
pub const MFCC_DIM: usize = 39;

/// Log-mel feature dimension.
pub const LOG_MEL_DIM: usize = 80;

/// Number of static MFCC coefficients before delta computation.
const NUM_CEPSTRAL: usize = 13;

const FFT_SIZE: usize = 512;

/// Frame length in samples (25ms @ 16kHz).
const FRAME_LEN: usize = 400;

/// Hop length in samples (10ms @ 16kHz).
const HOP_LEN: usize = 160;

const PRE_EMPHASIS: f32 = 0.97;

const MEL_FREQ_MIN: f32 = 0.0;

const MEL_FREQ_MAX: f32 = 8000.0;

/// Added to mel energies before the log to avoid `ln(0)`.
const LOG_FLOOR: f32 = 1e-10;

/// Audio feature representation used by `SpeechAligner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FeatureMode {
    #[default]
    Mfcc39,
    LogMel80,
}

impl FeatureMode {
    /// Human-readable label for logs/reports.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mfcc39 => "mfcc39",
            Self::LogMel80 => "logmel80",
        }
    }

    /// Feature dimension emitted per frame.
    pub const fn feature_dim(self) -> usize {
        match self {
            Self::Mfcc39 => MFCC_DIM,
            Self::LogMel80 => LOG_MEL_DIM,
        }
    }

    const fn mel_filter_count(self) -> usize {
        match self {
            Self::Mfcc39 => 40,
            Self::LogMel80 => LOG_MEL_DIM,
        }
    }
}

/// MFCC feature extractor with precomputed filterbank and DCT matrix.
pub struct MfccExtractor {
    feature_mode: FeatureMode,
    mel_filterbank: Vec<Vec<f32>>,
    dct_matrix: Vec<Vec<f32>>,
    hann_window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl Default for MfccExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MfccExtractor {
    /// Create a new feature extractor with default `mfcc39` mode.
    pub fn new() -> Self {
        Self::with_mode(FeatureMode::default())
    }

    /// Create a feature extractor for the requested mode.
    pub fn with_mode(feature_mode: FeatureMode) -> Self {
        let dct_matrix = if feature_mode == FeatureMode::Mfcc39 {
            build_dct_matrix(feature_mode.mel_filter_count())
        } else {
            Vec::new()
        };

        Self {
            feature_mode,
            mel_filterbank: build_mel_filterbank(feature_mode.mel_filter_count()),
            dct_matrix,
            hann_window: build_hann_window(),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE),
        }
    }

    /// Active feature mode.
    pub const fn feature_mode(&self) -> FeatureMode {
        self.feature_mode
    }

    /// Feature dimension emitted per frame.
    pub const fn feature_dim(&self) -> usize {
        self.feature_mode.feature_dim()
    }

    /// Extract one feature vector per 10 ms hop from 16 kHz mono audio.
    ///
    /// Fails if `samples` is shorter than one 25 ms frame.
    pub fn extract_frames(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.len() < FRAME_LEN {
            return Err(Error::config(format!(
                "Audio too short for {}: {} samples < {FRAME_LEN} frame length",
                self.feature_mode.label(),
                samples.len(),
            )));
        }

        let emphasized = pre_emphasis(samples);
        let num_frames = (emphasized.len() - FRAME_LEN) / HOP_LEN + 1;
        let mut windowed = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let log_mel_frames = (0..num_frames)
            .map(|frame_idx| {
                let start = frame_idx * HOP_LEN;
                self.frame_to_log_mel(
                    &emphasized[start..start + FRAME_LEN],
                    &mut windowed,
                    &mut spectrum,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        match self.feature_mode {
            FeatureMode::Mfcc39 => {
                let static_mfcc: Vec<[f32; NUM_CEPSTRAL]> = log_mel_frames
                    .iter()
                    .map(|frame| self.log_mel_to_cepstral(frame))
                    .collect();
                let deltas = compute_deltas(&static_mfcc);
                let delta_deltas = compute_deltas(&deltas);

                Ok(static_mfcc
                    .iter()
                    .zip(&deltas)
                    .zip(&delta_deltas)
                    .map(|((s, d), dd)| [s.as_slice(), d, dd].concat())
                    .collect())
            }
            FeatureMode::LogMel80 => Ok(log_mel_frames),
        }
    }

    /// Extract features as a `[1, frames, feature_dim]` tensor.
    pub fn extract_tensor(
        &self,
        samples: &[f32],
        device: &burn::tensor::Device,
    ) -> Result<burn::tensor::Tensor<3>> {
        let frames = self.extract_frames(samples)?;
        let shape = [1, frames.len(), self.feature_dim()];
        let data = burn::tensor::TensorData::new(frames.concat(), shape);
        Ok(burn::tensor::Tensor::from_data(data, device))
    }

    /// Hann window -> FFT -> mel filterbank -> log for one frame.
    ///
    /// `windowed` and `spectrum` are FFT scratch buffers; their contents on
    /// entry are ignored.
    fn frame_to_log_mel(
        &self,
        frame: &[f32],
        windowed: &mut [f32],
        spectrum: &mut [Complex<f32>],
    ) -> Result<Vec<f32>> {
        let (head, tail) = windowed.split_at_mut(FRAME_LEN);
        for ((out, &sample), &win) in head.iter_mut().zip(frame).zip(&self.hann_window) {
            *out = sample * win;
        }
        tail.fill(0.0);

        self.fft
            .process(windowed, spectrum)
            .map_err(|e| Error::data(format!("FFT failed: {e}")))?;

        Ok(self
            .mel_filterbank
            .iter()
            .map(|filter| {
                let energy: f32 = filter
                    .iter()
                    .zip(spectrum.iter())
                    .map(|(f, c)| f * c.norm_sqr())
                    .sum();
                (energy + LOG_FLOOR).ln()
            })
            .collect())
    }

    fn log_mel_to_cepstral(&self, mel_energies: &[f32]) -> [f32; NUM_CEPSTRAL] {
        let mut cepstral = [0.0f32; NUM_CEPSTRAL];
        for (coeff, row) in cepstral.iter_mut().zip(&self.dct_matrix) {
            *coeff = row.iter().zip(mel_energies).map(|(d, m)| d * m).sum();
        }
        cepstral
    }
}

/// Pre-emphasis filter: `y[n] = x[n] - alpha * x[n-1]`.
fn pre_emphasis(samples: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    if let Some(&first) = samples.first() {
        out.push(first);
    }
    for pair in samples.windows(2) {
        out.push((-PRE_EMPHASIS).mul_add(pair[0], pair[1]));
    }
    out
}

fn build_hann_window() -> Vec<f32> {
    let len = FRAME_LEN as f32;
    (0..FRAME_LEN)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (len - 1.0)).cos()))
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular mel filters over the `FFT_SIZE / 2 + 1` power-spectrum bins.
fn build_mel_filterbank(num_filters: usize) -> Vec<Vec<f32>> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let mel_min = hz_to_mel(MEL_FREQ_MIN);
    let mel_max = hz_to_mel(MEL_FREQ_MAX);

    let num_points = num_filters + 2;
    let bin_indices: Vec<f32> = (0..num_points)
        .map(|i| {
            let frac = i as f32 / (num_points - 1) as f32;
            let mel = mel_min + frac * (mel_max - mel_min);
            mel_to_hz(mel) * FFT_SIZE as f32 / SAMPLE_RATE as f32
        })
        .collect();

    bin_indices
        .windows(3)
        .map(|edges| {
            let [left, center, right] = [edges[0], edges[1], edges[2]];
            (0..fft_bins)
                .map(|k| {
                    let k = k as f32;
                    if k > left && k <= center && (center - left).abs() > f32::EPSILON {
                        (k - left) / (center - left)
                    } else if k > center && k < right && (right - center).abs() > f32::EPSILON {
                        (right - k) / (right - center)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

/// Unnormalised DCT-II matrix, `NUM_CEPSTRAL x num_filters`.
fn build_dct_matrix(num_filters: usize) -> Vec<Vec<f32>> {
    (0..NUM_CEPSTRAL)
        .map(|i| {
            (0..num_filters)
                .map(|j| (PI * i as f32 * (j as f32 + 0.5) / num_filters as f32).cos())
                .collect()
        })
        .collect()
}

/// Regression deltas over a +/-2 frame window, clamping at the edges.
fn compute_deltas(features: &[[f32; NUM_CEPSTRAL]]) -> Vec<[f32; NUM_CEPSTRAL]> {
    const WINDOW: usize = 2;
    let last = features.len().saturating_sub(1);
    let denom: f32 = 2.0 * (1..=WINDOW).map(|d| (d * d) as f32).sum::<f32>();

    (0..features.len())
        .map(|t| {
            let mut delta = [0.0f32; NUM_CEPSTRAL];
            for (c, value) in delta.iter_mut().enumerate() {
                let mut numerator = 0.0f32;
                for d in 1..=WINDOW {
                    let plus = features[(t + d).min(last)][c];
                    let minus = features[t.saturating_sub(d)][c];
                    numerator += d as f32 * (plus - minus);
                }
                *value = numerator / denom;
            }
            delta
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    fn sine_wave(num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_pre_emphasis_first_sample_unchanged() {
        let result = pre_emphasis(&[1.0, 2.0, 3.0, 4.0]);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 1.03).abs() < 1e-6);
    }

    #[test]
    fn test_hann_window_endpoints_near_zero() {
        let win = build_hann_window();
        assert_eq!(win.len(), FRAME_LEN);
        assert!(win[0].abs() < 1e-6, "first sample should be ~0");
    }

    #[test]
    fn test_mel_hz_roundtrip() {
        let back = mel_to_hz(hz_to_mel(1000.0));
        assert!((1000.0 - back).abs() < 0.01, "mel roundtrip gave {back}");
    }

    #[test]
    fn test_filterbank_shape() {
        let fb = build_mel_filterbank(FeatureMode::Mfcc39.mel_filter_count());
        assert_eq!(fb.len(), FeatureMode::Mfcc39.mel_filter_count());
        for filter in &fb {
            assert_eq!(filter.len(), FFT_SIZE / 2 + 1);
        }
    }

    #[test]
    fn every_mel_filter_has_a_nonzero_weight() {
        for mode in [FeatureMode::Mfcc39, FeatureMode::LogMel80] {
            let filterbank = build_mel_filterbank(mode.mel_filter_count());
            for (i, filter) in filterbank.iter().enumerate() {
                assert!(
                    filter.iter().any(|&w| w > 0.0),
                    "{} mel filter {i} is all zeros",
                    mode.label()
                );
            }
        }
    }

    #[test]
    fn test_dct_matrix_shape() {
        let dct = build_dct_matrix(FeatureMode::Mfcc39.mel_filter_count());
        assert_eq!(dct.len(), NUM_CEPSTRAL);
        for row in &dct {
            assert_eq!(row.len(), FeatureMode::Mfcc39.mel_filter_count());
        }
    }

    #[test]
    fn test_extract_frames_sine_wave() {
        let expected_frames = (SAMPLE_RATE - FRAME_LEN) / HOP_LEN + 1;
        for mode in [FeatureMode::Mfcc39, FeatureMode::LogMel80] {
            let frames = MfccExtractor::with_mode(mode)
                .extract_frames(&sine_wave(SAMPLE_RATE))
                .expect("extract");
            assert_eq!(frames.len(), expected_frames);
            for frame in &frames {
                assert_eq!(frame.len(), mode.feature_dim());
                assert!(frame.iter().all(|v| v.is_finite()), "{}", mode.label());
            }
        }
    }

    #[test]
    fn test_extract_too_short() {
        let extractor = MfccExtractor::new();
        assert!(extractor.extract_frames(&[0.0; FRAME_LEN - 1]).is_err());
        let frames = extractor.extract_frames(&[0.0; FRAME_LEN]).expect("frames");
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_deltas_shape() {
        let features = vec![[1.0; NUM_CEPSTRAL]; 10];
        assert_eq!(compute_deltas(&features).len(), features.len());
    }

    #[test]
    fn test_deltas_constant_input_yields_zero() {
        let deltas = compute_deltas(&[[5.0; NUM_CEPSTRAL]; 10]);
        for &val in deltas.iter().flatten() {
            assert!(val.abs() < 1e-6, "delta should be ~0, got {val}");
        }
    }

    #[test]
    fn test_extract_tensor_shape() {
        let samples = sine_wave(SAMPLE_RATE);
        let device = burn::tensor::Device::default();
        for mode in [FeatureMode::Mfcc39, FeatureMode::LogMel80] {
            let extractor = MfccExtractor::with_mode(mode);
            let tensor = extractor.extract_tensor(&samples, &device).expect("tensor");
            let num_frames = extractor.extract_frames(&samples).expect("frames").len();
            assert_eq!(tensor.dims(), [1, num_frames, mode.feature_dim()]);
        }
    }
}
