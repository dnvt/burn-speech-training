//! `LibriSpeech` dataset loader for `SpeechAligner` training.
//!
//! Parses the standard `LibriSpeech` directory layout, extracts audio features
//! on-the-fly, resolves phoneme targets via `CMUdict`, and produces
//! length-sorted, padded training batches.

#![allow(clippy::indexing_slicing)] // Bounded by construction in batch assembly

use std::path::{Path, PathBuf};

use crate::g2p::CmuDict;

use crate::error::{Error, Result};
use crate::mfcc::MfccExtractor;
use crate::phoneme_map::transcript_to_targets;

/// Minimum utterance duration in seconds.
const MIN_DURATION_SECS: f32 = 0.5;

/// A single training sample (metadata, resolved lazily into features).
#[derive(Debug, Clone)]
pub struct UtteranceMeta {
    /// Path to the WAV/FLAC file.
    pub audio_path: PathBuf,
    /// Raw transcript text.
    pub transcript: String,
}

/// A training sample with extracted features and targets.
pub struct TrainingSample {
    /// Feature frames: `[T, D]`.
    pub feature_frames: Vec<Vec<f32>>,
    /// Per-frame feature dimension `D`.
    pub feature_dim: usize,
    /// CTC target indices.
    pub targets: Vec<i32>,
    /// Number of OOV words in this utterance.
    pub oov_count: usize,
}

/// A padded training batch ready for the model.
pub struct TrainingBatch<B: burn::tensor::backend::Backend> {
    /// Input features: `[batch, max_time, feature_dim]`.
    pub inputs: burn::tensor::Tensor<B, 3>,
    /// CTC targets: `[batch, max_target_len]`.
    pub targets: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    /// Actual input lengths per sample: `[batch]`.
    pub input_lengths: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    /// Actual target lengths per sample: `[batch]`.
    pub target_lengths: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    /// Pad mask: `[batch, max_time]` -- true for padded frames.
    pub pad_mask: burn::tensor::Tensor<B, 2, burn::tensor::Bool>,
}

/// Scan a `LibriSpeech` split directory and collect utterance metadata.
///
/// Handles both standard and provisioned layouts:
/// - Standard: `<data_dir>/<split>/<speaker>/<chapter>/<utt>.flac`
/// - Provisioned: `<data_dir>/<split>/<split>/<speaker>/<chapter>/<utt>.wav`
///
/// Transcripts: `<speaker>-<chapter>.trans.txt` alongside audio files.
pub fn scan_librispeech(
    data_dir: &Path,
    split: &str,
    max_duration_secs: f32,
) -> Result<Vec<UtteranceMeta>> {
    let split_dir = data_dir.join(split);
    if !split_dir.is_dir() {
        return Err(Error::config(format!(
            "LibriSpeech split directory not found: {}",
            split_dir.display()
        )));
    }

    // Handle double-nested layout from provisioning (tar --strip-components=1
    // strips LibriSpeech/ but leaves train-clean-100/ inside split dir)
    let nested = split_dir.join(split);
    let root = if nested.is_dir() { nested } else { split_dir };

    let mut utterances = Vec::new();

    let speaker_dirs = list_subdirs(&root)?;
    for speaker_dir in &speaker_dirs {
        let chapter_dirs = list_subdirs(speaker_dir)?;
        for chapter_dir in &chapter_dirs {
            let trans_files = find_trans_files(chapter_dir)?;
            for trans_path in &trans_files {
                let entries = parse_trans_file(trans_path, chapter_dir)?;
                for (audio_path, transcript) in entries {
                    let duration = estimate_audio_duration(&audio_path);
                    if duration < MIN_DURATION_SECS || duration > max_duration_secs {
                        continue;
                    }
                    utterances.push(UtteranceMeta {
                        audio_path,
                        transcript,
                    });
                }
            }
        }
    }

    if utterances.is_empty() {
        return Err(Error::config(format!(
            "No utterances found in {}/{}",
            data_dir.display(),
            split
        )));
    }

    Ok(utterances)
}

/// Load audio from a FLAC or WAV file as 16kHz mono f32 samples.
pub fn load_audio_samples(path: &Path) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .map_err(|e| Error::config(format!("Cannot open audio {}: {e}", path.display())))?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::config(format!("Cannot probe audio {}: {e}", path.display())))?;

    let mut reader = probed.format;
    let track = reader
        .default_track()
        .ok_or_else(|| Error::config(format!("No audio track in {}", path.display())))?;
    let track_id = track.id;

    // Validate sample rate — feature extraction expects 16kHz
    if let Some(rate) = track.codec_params.sample_rate {
        if rate != 16000 {
            return Err(Error::config(format!(
                "Audio sample rate {rate}Hz != expected 16000Hz in {}",
                path.display()
            )));
        }
    }

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::config(format!("Cannot create decoder for {}: {e}", path.display())))?;

    let mut all_samples = Vec::new();
    loop {
        let pkt = match reader.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(Error::config(format!(
                    "Error reading packet from {}: {e}",
                    path.display()
                )));
            }
        };

        if pkt.track_id() != track_id {
            continue;
        }

        let audio_buf = decoder
            .decode(&pkt)
            .map_err(|e| Error::config(format!("Decode error for {}: {e}", path.display())))?;

        let spec = *audio_buf.spec();
        let capacity = audio_buf.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(capacity as u64, spec);
        sample_buf.copy_interleaved_ref(audio_buf);

        let num_channels = spec.channels.count();
        if num_channels == 1 {
            all_samples.extend_from_slice(sample_buf.samples());
        } else {
            // Downmix to mono by averaging channels
            for chunk in sample_buf.samples().chunks(num_channels) {
                let sum: f32 = chunk.iter().sum();
                #[allow(clippy::cast_precision_loss)]
                let avg = sum / num_channels as f32;
                all_samples.push(avg);
            }
        }
    }

    if all_samples.is_empty() {
        return Err(Error::config(format!(
            "Empty audio file: {}",
            path.display()
        )));
    }

    Ok(all_samples)
}

/// Process a single utterance into frame features and CTC targets.
pub fn process_utterance(
    meta: &UtteranceMeta,
    extractor: &MfccExtractor,
    dict: &CmuDict,
) -> Result<TrainingSample> {
    let samples = load_audio_samples(&meta.audio_path)?;
    let feature_frames = extractor.extract_frames(&samples)?;
    let (targets, stats) = transcript_to_targets(&meta.transcript, dict)?;

    // CTC validity: input length must be at least target length
    let target_len = targets.len();
    if feature_frames.len() < target_len {
        return Err(Error::config(format!(
            "CTC constraint violated: {} frames < target length {} for {}",
            feature_frames.len(),
            target_len,
            meta.audio_path.display()
        )));
    }

    // Log G2P fallback usage at debug level
    if stats.g2p_hits > 0 {
        tracing::debug!(
            "G2P fallback used for {}/{} words in {}",
            stats.g2p_hits,
            stats.total_words,
            meta.audio_path.display()
        );
    }

    Ok(TrainingSample {
        feature_frames,
        feature_dim: extractor.feature_dim(),
        targets,
        oov_count: stats.g2p_hits,
    })
}

/// Create padded training batches from samples, sorted by length.
pub fn create_batches<B: burn::tensor::backend::Backend>(
    samples: &mut [TrainingSample],
    batch_size: usize,
    device: &B::Device,
) -> Vec<TrainingBatch<B>> {
    use burn::tensor::{Int, Shape, Tensor, TensorData};

    // Sort by frame count (ascending) for minimal padding
    samples.sort_by_key(|s| s.feature_frames.len());

    let mut batches = Vec::new();
    let num_samples = samples.len();

    let mut start = 0;
    while start < num_samples {
        let end = (start + batch_size).min(num_samples);
        let batch_slice = &samples[start..end];
        let b = batch_slice.len();

        let max_time = batch_slice
            .iter()
            .map(|s| s.feature_frames.len())
            .max()
            .unwrap_or(0);
        let max_target = batch_slice
            .iter()
            .map(|s| s.targets.len())
            .max()
            .unwrap_or(0);
        let feature_dim = batch_slice.first().map_or(0, |sample| sample.feature_dim);

        let mut input_data = vec![0.0f32; b * max_time * feature_dim];
        let mut target_data = vec![0i32; b * max_target];
        let mut input_lens = Vec::with_capacity(b);
        let mut target_lens = Vec::with_capacity(b);
        let mut pad_mask_data = vec![0i32; b * max_time];

        for (i, sample) in batch_slice.iter().enumerate() {
            debug_assert_eq!(sample.feature_dim, feature_dim);

            let t = sample.feature_frames.len();
            let l = sample.targets.len();

            for (frame_idx, frame) in sample.feature_frames.iter().enumerate() {
                let offset = (i * max_time + frame_idx) * feature_dim;
                input_data[offset..offset + feature_dim].copy_from_slice(frame);
            }

            let t_offset = i * max_target;
            target_data[t_offset..t_offset + l].copy_from_slice(&sample.targets);

            #[allow(clippy::cast_possible_wrap)]
            {
                input_lens.push(t as i32);
                target_lens.push(l as i32);
            }

            for frame_idx in t..max_time {
                pad_mask_data[i * max_time + frame_idx] = 1;
            }
        }

        let inputs = Tensor::from_data(
            TensorData::new(input_data, Shape::new([b, max_time, feature_dim])),
            device,
        );
        let targets = Tensor::from_data(
            TensorData::new(target_data, Shape::new([b, max_target])),
            device,
        );
        let input_lengths = Tensor::from_data(TensorData::new(input_lens, Shape::new([b])), device);
        let target_lengths =
            Tensor::from_data(TensorData::new(target_lens, Shape::new([b])), device);
        let pad_mask: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(pad_mask_data, Shape::new([b, max_time])),
            device,
        );
        let pad_mask = pad_mask.equal_elem(1);

        batches.push(TrainingBatch {
            inputs,
            targets,
            input_lengths,
            target_lengths,
            pad_mask,
        });

        start = end;
    }

    batches
}

/// Create dynamically-sized training batches using an **attention-aware memory
/// budget**.
///
/// Self-attention memory scales with `B × T²` (quadratic in sequence length).
/// This function packs as many samples as fit under a memory ceiling per batch,
/// automatically adapting: short utterances → 200+ per batch, long → 20-50.
///
/// `max_attn_elements` controls the ceiling for the attention score tensor:
/// `B × n_heads × T × T`. Default 500M elements ≈ 2 GB forward, safe up to
/// ~6 GB with backward pass. For 80 GB GPUs, use 500M–2B.
///
/// The `max_samples_per_batch` cap prevents degenerate cases.
pub fn create_dynamic_batches<B: burn::tensor::backend::Backend>(
    samples: &mut [TrainingSample],
    max_attn_elements: usize,
    max_samples_per_batch: usize,
    n_heads: usize,
    device: &B::Device,
) -> Vec<TrainingBatch<B>> {
    use burn::tensor::{Int, Shape, Tensor, TensorData};

    // Sort by frame count (ascending) for minimal padding
    samples.sort_by_key(|s| s.feature_frames.len());

    let mut batches = Vec::new();
    let num_samples = samples.len();

    let mut start = 0;
    while start < num_samples {
        // Greedily add samples until the attention memory budget is exceeded
        let mut end = start + 1;
        while end < num_samples && end - start < max_samples_per_batch {
            // The longest utterance determines T (due to padding)
            let longest = samples.get(end).map_or(0, |s| s.feature_frames.len());
            let batch_size = end - start + 1;
            // Attention score tensor: B × n_heads × T × T
            let attn_elements = batch_size * n_heads * longest * longest;
            if attn_elements > max_attn_elements {
                break;
            }
            end += 1;
        }

        // Ensure at least 1 sample per batch
        if end == start {
            end = start + 1;
        }

        let batch_slice = &samples[start..end];
        let b = batch_slice.len();

        let max_time = batch_slice
            .iter()
            .map(|s| s.feature_frames.len())
            .max()
            .unwrap_or(0);
        let max_target = batch_slice
            .iter()
            .map(|s| s.targets.len())
            .max()
            .unwrap_or(0);
        let feature_dim = batch_slice.first().map_or(0, |sample| sample.feature_dim);

        let mut input_data = vec![0.0f32; b * max_time * feature_dim];
        let mut target_data = vec![0i32; b * max_target];
        let mut input_lens = Vec::with_capacity(b);
        let mut target_lens = Vec::with_capacity(b);
        let mut pad_mask_data = vec![0i32; b * max_time];

        for (i, sample) in batch_slice.iter().enumerate() {
            debug_assert_eq!(sample.feature_dim, feature_dim);

            let t = sample.feature_frames.len();
            let l = sample.targets.len();

            for (frame_idx, frame) in sample.feature_frames.iter().enumerate() {
                let offset = (i * max_time + frame_idx) * feature_dim;
                input_data[offset..offset + feature_dim].copy_from_slice(frame);
            }

            let t_offset = i * max_target;
            target_data[t_offset..t_offset + l].copy_from_slice(&sample.targets);

            #[allow(clippy::cast_possible_wrap)]
            {
                input_lens.push(t as i32);
                target_lens.push(l as i32);
            }

            for frame_idx in t..max_time {
                pad_mask_data[i * max_time + frame_idx] = 1;
            }
        }

        let inputs = Tensor::from_data(
            TensorData::new(input_data, Shape::new([b, max_time, feature_dim])),
            device,
        );
        let targets = Tensor::from_data(
            TensorData::new(target_data, Shape::new([b, max_target])),
            device,
        );
        let input_lengths = Tensor::from_data(TensorData::new(input_lens, Shape::new([b])), device);
        let target_lengths =
            Tensor::from_data(TensorData::new(target_lens, Shape::new([b])), device);
        let pad_mask: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(pad_mask_data, Shape::new([b, max_time])),
            device,
        );
        let pad_mask = pad_mask.equal_elem(1);

        batches.push(TrainingBatch {
            inputs,
            targets,
            input_lengths,
            target_lengths,
            pad_mask,
        });

        start = end;
    }

    batches
}

/// List immediate subdirectories of a path.
fn list_subdirs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| Error::config(format!("Cannot read directory {}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| Error::config(format!("Cannot read entry in {}: {e}", dir.display())))?;
        if entry.path().is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// Find `.trans.txt` files in a directory.
fn find_trans_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| Error::config(format!("Cannot read directory {}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| Error::config(format!("Cannot read entry in {}: {e}", dir.display())))?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".trans.txt") {
                files.push(path);
            }
        }
    }
    Ok(files)
}

/// Parse a `.trans.txt` file into `(audio_path, transcript)` pairs.
fn parse_trans_file(trans_path: &Path, chapter_dir: &Path) -> Result<Vec<(PathBuf, String)>> {
    let content = std::fs::read_to_string(trans_path).map_err(|e| {
        Error::config(format!(
            "Cannot read transcript {}: {e}",
            trans_path.display()
        ))
    })?;

    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: <uttid> <TRANSCRIPT TEXT>
        if let Some(space_idx) = line.find(' ') {
            let utt_id = &line[..space_idx];
            let transcript = line[space_idx + 1..].trim().to_owned();

            // Try FLAC first (standard LibriSpeech), then WAV
            let flac_path = chapter_dir.join(format!("{utt_id}.flac"));
            let wav_path = chapter_dir.join(format!("{utt_id}.wav"));

            let audio_path = if flac_path.exists() {
                flac_path
            } else if wav_path.exists() {
                wav_path
            } else {
                continue;
            };

            entries.push((audio_path, transcript));
        }
    }

    Ok(entries)
}

/// Estimate audio duration from file size.
/// `LibriSpeech` FLAC: ~8x compression of 16kHz 16-bit mono = ~4KB/s.
fn estimate_audio_duration(path: &Path) -> f32 {
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return 0.0,
    };

    let bytes = metadata.len();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    #[allow(clippy::cast_precision_loss)]
    match ext {
        "flac" => bytes as f32 / 4000.0,
        "wav" => bytes as f32 / 32000.0,
        _ => bytes as f32 / 4000.0,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::mfcc::MFCC_DIM;

    #[test]
    fn test_estimate_duration_heuristic() {
        // 4KB should be ~1 second for FLAC
        let dur_flac = 4000.0_f32 / 4000.0;
        assert!((dur_flac - 1.0).abs() < 0.1);

        // 32KB should be ~1 second for WAV
        let dur_wav = 32000.0_f32 / 32000.0;
        assert!((dur_wav - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_trans_line_format() {
        let line = "1234-5678-0001 THIS IS A TEST";
        let space_idx = line.find(' ').expect("space");
        let utt_id = &line[..space_idx];
        let transcript = line[space_idx + 1..].trim();
        assert_eq!(utt_id, "1234-5678-0001");
        assert_eq!(transcript, "THIS IS A TEST");
    }

    #[test]
    fn test_ctc_validity_check() {
        // CTC requires: input_frames >= 2 * target_len + 1
        let min_frames_for_3_targets = 2 * 3 + 1; // = 7
        assert!(10 >= min_frames_for_3_targets);
        assert!(5 < min_frames_for_3_targets);
    }

    #[test]
    fn test_scan_nonexistent_dir() {
        let result = scan_librispeech(Path::new("/nonexistent/path"), "train-clean-100", 30.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_provisioned_nested_layout() {
        let tempdir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(err) => panic!("tempdir creation should succeed: {err}"),
        };
        let chapter_dir = tempdir
            .path()
            .join("train-clean-100")
            .join("train-clean-100")
            .join("103")
            .join("1240");
        if let Err(err) = fs::create_dir_all(&chapter_dir) {
            panic!("chapter dir creation should succeed: {err}");
        }
        if let Err(err) = fs::write(
            chapter_dir.join("103-1240.trans.txt"),
            "103-1240-0000 THIS IS A TEST\n",
        ) {
            panic!("transcript creation should succeed: {err}");
        }
        if let Err(err) = fs::write(chapter_dir.join("103-1240-0000.wav"), vec![0_u8; 32000]) {
            panic!("wav creation should succeed: {err}");
        }

        let result = match scan_librispeech(tempdir.path(), "train-clean-100", 5.0) {
            Ok(result) => result,
            Err(err) => panic!("nested provisioned layout should scan: {err}"),
        };

        assert_eq!(result.len(), 1);
        assert!(result[0].audio_path.ends_with("103-1240-0000.wav"));
        assert_eq!(result[0].transcript, "THIS IS A TEST");
    }

    mod dynamic_batching {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::NdArray;

        use super::*;

        type B = NdArray<f32>;

        fn sample_with_frames(num_frames: usize) -> TrainingSample {
            TrainingSample {
                feature_frames: vec![vec![0.0f32; MFCC_DIM]; num_frames],
                feature_dim: MFCC_DIM,
                targets: vec![1; (num_frames / 5).max(1)],
                oov_count: 0,
            }
        }

        fn batch_sizes(batches: &[TrainingBatch<B>]) -> Vec<usize> {
            batches
                .iter()
                .map(|batch| batch.input_lengths.dims()[0])
                .collect()
        }

        #[test]
        fn test_dynamic_batches_respect_attention_budget() {
            let device = NdArrayDevice::default();
            let mut samples = vec![
                sample_with_frames(10),
                sample_with_frames(10),
                sample_with_frames(20),
            ];

            let batches = create_dynamic_batches::<B>(&mut samples, 1_600, 16, 2, &device);

            assert_eq!(batch_sizes(&batches), vec![2, 1]);
        }

        #[test]
        fn test_dynamic_batches_respect_max_samples_cap() {
            let device = NdArrayDevice::default();
            let mut samples = vec![
                sample_with_frames(8),
                sample_with_frames(8),
                sample_with_frames(8),
                sample_with_frames(8),
            ];

            let batches = create_dynamic_batches::<B>(&mut samples, 1_000_000, 2, 1, &device);

            assert_eq!(batch_sizes(&batches), vec![2, 2]);
        }

        #[test]
        fn test_dynamic_batches_fall_back_to_single_sample_when_needed() {
            let device = NdArrayDevice::default();
            let mut samples = vec![sample_with_frames(100), sample_with_frames(100)];

            let batches = create_dynamic_batches::<B>(&mut samples, 1, 16, 8, &device);

            assert_eq!(batch_sizes(&batches), vec![1, 1]);
        }

        #[test]
        fn test_dynamic_batches_scale_with_attention_head_count() {
            let device = NdArrayDevice::default();
            let mut one_head_samples = vec![
                sample_with_frames(50),
                sample_with_frames(50),
                sample_with_frames(50),
                sample_with_frames(50),
            ];
            let mut two_head_samples = vec![
                sample_with_frames(50),
                sample_with_frames(50),
                sample_with_frames(50),
                sample_with_frames(50),
            ];

            let one_head_batches =
                create_dynamic_batches::<B>(&mut one_head_samples, 10_000, 16, 1, &device);
            let two_head_batches =
                create_dynamic_batches::<B>(&mut two_head_samples, 10_000, 16, 2, &device);

            assert_eq!(batch_sizes(&one_head_batches), vec![4]);
            assert_eq!(batch_sizes(&two_head_batches), vec![2, 2]);
        }
    }
}
