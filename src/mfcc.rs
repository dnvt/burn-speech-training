//! Audio feature extraction for `SpeechAligner` training.
//!
//! Supported feature modes:
//! - `mfcc39`: pre-emphasis -> framing -> Hann -> FFT -> 40-bin mel bank -> log
//!   energy -> DCT-II (13 coefficients) -> delta + delta-delta = 39 total.
//! - `logmel80`: pre-emphasis -> framing -> Hann -> FFT -> 80-bin mel bank ->
//!   log energy.

#![allow(clippy::indexing_slicing)] // Bounded by construction (all indices < known array sizes)

use std::f32::consts::PI;

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

/// FFT size.
const FFT_SIZE: usize = 512;

/// Frame length in samples (25ms @ 16kHz).
const FRAME_LEN: usize = 400;

/// Hop length in samples (10ms @ 16kHz).
const HOP_LEN: usize = 160;

/// Pre-emphasis coefficient.
const PRE_EMPHASIS: f32 = 0.97;

/// Minimum frequency for mel filterbank (Hz).
const MEL_FREQ_MIN: f32 = 0.0;

/// Maximum frequency for mel filterbank (Hz).
const MEL_FREQ_MAX: f32 = 8000.0;

/// Floor for log energy to avoid log(0).
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
    fft: std::sync::Arc<dyn RealToComplex<f32>>,
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
        let mel_filterbank = build_mel_filterbank(feature_mode.mel_filter_count());
        let dct_matrix = if feature_mode == FeatureMode::Mfcc39 {
            build_dct_matrix(feature_mode.mel_filter_count())
        } else {
            Vec::new()
        };
        let hann_window = build_hann_window();

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        Self {
            feature_mode,
            mel_filterbank,
            dct_matrix,
            hann_window,
            fft,
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

    /// Extract frame features from raw 16kHz mono audio samples.
    ///
    /// Returns a vector of per-frame feature vectors, one per frame.
    pub fn extract_frames(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.len() < FRAME_LEN {
            return Err(Error::config(format!(
                "Audio too short for {}: {} samples < {FRAME_LEN} frame length",
                self.feature_mode.label(),
                samples.len(),
            )));
        }

        // 1. Pre-emphasis
        let emphasized = pre_emphasis(samples);

        // 2. Framing + windowing + FFT + mel filterbank + log + DCT
        let num_frames = (emphasized.len().saturating_sub(FRAME_LEN)) / HOP_LEN + 1;
        let mut log_mel_frames = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LEN;
            let end = start + FRAME_LEN;
            if end > emphasized.len() {
                break;
            }

            let frame = &emphasized[start..end];
            let log_mel = self.frame_to_log_mel(frame)?;
            log_mel_frames.push(log_mel);
        }

        if log_mel_frames.is_empty() {
            return Err(Error::config(format!(
                "No {} frames extracted",
                self.feature_mode.label()
            )));
        }

        match self.feature_mode {
            FeatureMode::Mfcc39 => {
                let static_mfcc: Vec<[f32; NUM_CEPSTRAL]> = log_mel_frames
                    .iter()
                    .map(|frame| self.log_mel_to_cepstral(frame))
                    .collect();

                let deltas = compute_deltas(&static_mfcc);
                let delta_deltas = compute_deltas(&deltas);

                let mut features = Vec::with_capacity(static_mfcc.len());
                for i in 0..static_mfcc.len() {
                    let mut feat = vec![0.0_f32; MFCC_DIM];
                    feat[..NUM_CEPSTRAL].copy_from_slice(&static_mfcc[i]);
                    feat[NUM_CEPSTRAL..2 * NUM_CEPSTRAL].copy_from_slice(&deltas[i]);
                    feat[2 * NUM_CEPSTRAL..].copy_from_slice(&delta_deltas[i]);
                    features.push(feat);
                }

                Ok(features)
            }
            FeatureMode::LogMel80 => Ok(log_mel_frames),
        }
    }

    /// Extract MFCC features in legacy fixed-size form.
    pub fn extract(&self, samples: &[f32]) -> Result<Vec<[f32; MFCC_DIM]>> {
        if self.feature_mode != FeatureMode::Mfcc39 {
            return Err(Error::config(
                "extract() is only supported in mfcc39 mode; use extract_frames() instead",
            ));
        }

        let frames = self.extract_frames(samples)?;
        let mut mfcc_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let mut mfcc = [0.0_f32; MFCC_DIM];
            mfcc.copy_from_slice(&frame);
            mfcc_frames.push(mfcc);
        }

        Ok(mfcc_frames)
    }

    /// Extract features and return as a Burn tensor `[1, T, D]`.
    #[allow(dead_code)] // Public API — used in tests and by future callers
    pub fn extract_tensor<B: burn::tensor::backend::Backend>(
        &self,
        samples: &[f32],
        device: &B::Device,
    ) -> Result<burn::tensor::Tensor<B, 3>> {
        let frames = self.extract_frames(samples)?;
        let num_frames = frames.len();
        let feature_dim = self.feature_dim();

        let mut flat = Vec::with_capacity(num_frames * feature_dim);
        for frame in &frames {
            flat.extend_from_slice(frame);
        }

        let data = burn::tensor::TensorData::new(
            flat,
            burn::tensor::Shape::new([1, num_frames, feature_dim]),
        );
        Ok(burn::tensor::Tensor::from_data(data, device))
    }

    /// Process a single frame: window -> FFT -> mel -> log.
    fn frame_to_log_mel(&self, frame: &[f32]) -> Result<Vec<f32>> {
        // Apply Hann window
        let mut windowed = vec![0.0f32; FFT_SIZE];
        for (i, (&sample, &win)) in frame.iter().zip(self.hann_window.iter()).enumerate() {
            windowed[i] = sample * win;
        }

        // FFT
        let mut spectrum = self.fft.make_output_vec();
        self.fft
            .process(&mut windowed, &mut spectrum)
            .map_err(|e| Error::config(format!("FFT failed: {e}")))?;

        // Power spectrum
        let power: Vec<f32> = spectrum
            .iter()
            .map(realfft::num_complex::Complex::norm_sqr)
            .collect();

        // Mel filterbank application
        let mut mel_energies = vec![0.0_f32; self.mel_filterbank.len()];
        for (mel_energy, filter) in mel_energies.iter_mut().zip(self.mel_filterbank.iter()) {
            let energy: f32 = filter.iter().zip(power.iter()).map(|(f, p)| f * p).sum();
            *mel_energy = (energy + LOG_FLOOR).ln();
        }

        Ok(mel_energies)
    }

    /// Convert log-mel energies to static cepstral coefficients.
    fn log_mel_to_cepstral(&self, mel_energies: &[f32]) -> [f32; NUM_CEPSTRAL] {
        let mut cepstral = [0.0f32; NUM_CEPSTRAL];
        for (coeff, row) in cepstral.iter_mut().zip(self.dct_matrix.iter()) {
            *coeff = row
                .iter()
                .zip(mel_energies.iter())
                .map(|(d, m)| d * m)
                .sum();
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

/// Build a Hann window of length `FRAME_LEN`.
fn build_hann_window() -> Vec<f32> {
    (0..FRAME_LEN)
        .map(|n| {
            #[allow(clippy::cast_precision_loss)]
            let x = n as f32;
            #[allow(clippy::cast_precision_loss)]
            let len = FRAME_LEN as f32;
            0.5 * (1.0 - (2.0 * PI * x / (len - 1.0)).cos())
        })
        .collect()
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build triangular mel filterbank.
fn build_mel_filterbank(num_filters: usize) -> Vec<Vec<f32>> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let mel_min = hz_to_mel(MEL_FREQ_MIN);
    let mel_max = hz_to_mel(MEL_FREQ_MAX);

    let num_points = num_filters + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let frac = i as f32 / (num_points - 1) as f32;
            mel_min + frac * (mel_max - mel_min)
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    let bin_indices: Vec<f32> = mel_points
        .iter()
        .map(|&mel| mel_to_hz(mel) * FFT_SIZE as f32 / SAMPLE_RATE as f32)
        .collect();

    let mut filterbank = Vec::with_capacity(num_filters);
    for i in 0..num_filters {
        let left = bin_indices[i];
        let center = bin_indices[i + 1];
        let right = bin_indices[i + 2];

        let mut filter = vec![0.0f32; fft_bins];
        for (k, val) in filter.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let k_f = k as f32;
            if k_f > left && k_f <= center {
                let denom = center - left;
                if denom.abs() > f32::EPSILON {
                    *val = (k_f - left) / denom;
                }
            } else if k_f > center && k_f < right {
                let denom = right - center;
                if denom.abs() > f32::EPSILON {
                    *val = (right - k_f) / denom;
                }
            }
        }
        filterbank.push(filter);
    }

    filterbank
}

/// Build DCT-II matrix.
fn build_dct_matrix(num_filters: usize) -> Vec<Vec<f32>> {
    let mut matrix = Vec::with_capacity(NUM_CEPSTRAL);
    for i in 0..NUM_CEPSTRAL {
        let mut row = Vec::with_capacity(num_filters);
        for j in 0..num_filters {
            #[allow(clippy::cast_precision_loss)]
            let val = (PI * i as f32 * (j as f32 + 0.5) / num_filters as f32).cos();
            row.push(val);
        }
        matrix.push(row);
    }
    matrix
}

/// Compute delta features using a window of +/-2 frames.
fn compute_deltas(features: &[[f32; NUM_CEPSTRAL]]) -> Vec<[f32; NUM_CEPSTRAL]> {
    let n = features.len();
    let mut deltas = vec![[0.0f32; NUM_CEPSTRAL]; n];
    let delta_window = 2i32;

    // Denominator: 2 * sum(d^2 for d=1..window) = 2*(1+4) = 10
    #[allow(clippy::cast_precision_loss)]
    let denom: f32 = 2.0 * (1..=delta_window).map(|d| (d * d) as f32).sum::<f32>();

    for (t, delta_frame) in deltas.iter_mut().enumerate() {
        for c in 0..NUM_CEPSTRAL {
            let mut numerator = 0.0f32;
            for d in 1..=delta_window {
                #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                let t_plus = (t as i32 + d).min((n as i32) - 1) as usize;
                #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                let t_minus = (t as i32 - d).max(0) as usize;
                #[allow(clippy::cast_precision_loss)]
                let d_f = d as f32;
                numerator += d_f * (features[t_plus][c] - features[t_minus][c]);
            }
            delta_frame[c] = numerator / denom;
        }
    }

    deltas
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_emphasis_first_sample_unchanged() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = pre_emphasis(&samples);
        let eps = 1e-6;
        assert!((result[0] - 1.0).abs() < eps);
        // y[1] = 2.0 - 0.97 * 1.0 = 1.03
        assert!((result[1] - 1.03).abs() < eps);
    }

    #[test]
    fn test_hann_window_endpoints_near_zero() {
        let win = build_hann_window();
        assert_eq!(win.len(), FRAME_LEN);
        assert!(win[0].abs() < 1e-6, "first sample should be ~0");
    }

    #[test]
    fn test_mel_hz_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!(
            (hz - back).abs() < 0.01,
            "mel roundtrip failed: {hz} -> {mel} -> {back}"
        );
    }

    #[test]
    fn test_filterbank_shape() {
        let fb = build_mel_filterbank(FeatureMode::Mfcc39.mel_filter_count());
        assert_eq!(fb.len(), FeatureMode::Mfcc39.mel_filter_count());
        let expected_bins = FFT_SIZE / 2 + 1;
        for filter in &fb {
            assert_eq!(filter.len(), expected_bins);
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
    fn test_extract_sine_wave() {
        let extractor = MfccExtractor::new();
        let num_samples = SAMPLE_RATE;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let frames = extractor.extract(&samples).expect("extract");
        assert!(!frames.is_empty(), "should produce frames");

        let expected_frames = (num_samples - FRAME_LEN) / HOP_LEN + 1;
        assert_eq!(frames.len(), expected_frames);

        for frame in &frames {
            assert_eq!(frame.len(), MFCC_DIM);
            for &val in frame {
                assert!(val.is_finite(), "MFCC value should be finite");
            }
        }
    }

    #[test]
    fn test_extract_log_mel_sine_wave() {
        let extractor = MfccExtractor::with_mode(FeatureMode::LogMel80);
        let num_samples = SAMPLE_RATE;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let frames = extractor.extract_frames(&samples).expect("extract");
        assert!(!frames.is_empty(), "should produce frames");
        for frame in &frames {
            assert_eq!(frame.len(), LOG_MEL_DIM);
            for &val in frame {
                assert!(val.is_finite(), "log-mel value should be finite");
            }
        }
    }

    #[test]
    fn test_extract_too_short() {
        let extractor = MfccExtractor::new();
        let short = vec![0.0; FRAME_LEN - 1];
        assert!(extractor.extract(&short).is_err());
    }

    #[test]
    fn test_deltas_shape() {
        let features: Vec<[f32; NUM_CEPSTRAL]> = (0..10).map(|_| [1.0; NUM_CEPSTRAL]).collect();
        let deltas = compute_deltas(&features);
        assert_eq!(deltas.len(), features.len());
    }

    #[test]
    fn test_deltas_constant_input_yields_zero() {
        let features: Vec<[f32; NUM_CEPSTRAL]> = (0..10).map(|_| [5.0; NUM_CEPSTRAL]).collect();
        let deltas = compute_deltas(&features);
        for frame in &deltas[2..8] {
            for &val in frame {
                assert!(
                    val.abs() < 1e-6,
                    "delta should be ~0 for constant input, got {val}"
                );
            }
        }
    }

    #[test]
    fn test_extract_tensor_shape() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::NdArray;

        type B = NdArray<f32>;
        let device = NdArrayDevice::default();

        let extractor = MfccExtractor::new();
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let tensor = extractor
            .extract_tensor::<B>(&samples, &device)
            .expect("tensor");
        let dims = tensor.dims();
        assert_eq!(dims[0], 1, "batch dim should be 1");
        assert!(dims[1] > 0, "time dim should be > 0");
        assert_eq!(dims[2], MFCC_DIM, "feature dim should be {MFCC_DIM}");
    }

    #[test]
    fn test_extract_tensor_shape_log_mel() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::NdArray;

        type B = NdArray<f32>;
        let device = NdArrayDevice::default();

        let extractor = MfccExtractor::with_mode(FeatureMode::LogMel80);
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let tensor = extractor
            .extract_tensor::<B>(&samples, &device)
            .expect("tensor");
        let dims = tensor.dims();
        assert_eq!(dims[0], 1, "batch dim should be 1");
        assert!(dims[1] > 0, "time dim should be > 0");
        assert_eq!(dims[2], LOG_MEL_DIM, "feature dim should be {LOG_MEL_DIM}");
    }
}
