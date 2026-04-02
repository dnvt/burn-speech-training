//! Audio-feature precomputation for `SpeechAligner` training.
//!
//! Extracts features from audio and saves as binary cache files,
//! eliminating the CPU bottleneck during GPU training. Each sample
//! becomes a flat binary file with a compact header.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use crate::g2p::CmuDict;
use serde::{Deserialize, Serialize};

use crate::dataset::{load_audio_samples, scan_librispeech, UtteranceMeta};
use crate::error::{Error, Result};
use crate::mfcc::{FeatureMode, MfccExtractor};
use crate::phoneme_map::transcript_to_targets;
use crate::ui::{detail, section, step, success};

/// Binary file magic bytes for version detection.
const MAGIC: [u8; 4] = *b"MFCC";
/// Format version.
const VERSION: u32 = 1;

/// Manifest entry for one precomputed sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputedEntry {
    /// Index in the manifest.
    pub index: usize,
    /// Number of feature frames.
    pub num_frames: usize,
    /// Number of CTC target indices.
    pub num_targets: usize,
    /// OOV word count from G2P resolution.
    pub oov_count: usize,
    /// Original audio path (for debugging).
    pub audio_path: String,
}

/// Manifest for a precomputed dataset.
#[derive(Debug, Serialize, Deserialize)]
pub struct PrecomputedManifest {
    /// Feature dimension for each frame.
    pub feature_dim: usize,
    /// Number of precomputed samples.
    pub num_samples: usize,
    /// Total feature frames across all samples.
    pub total_frames: usize,
    /// Total OOV words.
    pub total_oov: usize,
    /// Data source.
    pub data_dir: String,
    /// Split name.
    pub split: String,
    /// Feature representation used for this cache.
    #[serde(default)]
    pub feature_mode: FeatureMode,
    /// Maximum utterance duration included in this cache.
    #[serde(default)]
    pub max_duration_secs: Option<f32>,
    /// Individual sample entries.
    pub entries: Vec<PrecomputedEntry>,
}

/// Arguments for the precompute command.
pub struct PrecomputeArgs {
    pub data_dir: PathBuf,
    pub split: String,
    pub output_dir: PathBuf,
    pub max_duration_secs: f32,
    pub feature_mode: FeatureMode,
}

/// Execute feature precomputation: scan → extract → save binary files +
/// manifest.
pub fn execute_precompute(args: &PrecomputeArgs) -> Result<()> {
    section("SpeechAligner Feature Precomputation");

    // 1. Load resources
    step("Loading resources");
    let dict =
        CmuDict::load().map_err(|e| Error::config(format!("Failed to load CMUdict: {e}")))?;
    let extractor = MfccExtractor::with_mode(args.feature_mode);
    detail(&format!(
        "  CMUdict + feature extractor loaded ({})",
        args.feature_mode.label()
    ));

    // 2. Scan dataset
    step(&format!(
        "Scanning {} / {}",
        args.data_dir.display(),
        args.split
    ));
    let utterances = scan_librispeech(&args.data_dir, &args.split, args.max_duration_secs)?;
    detail(&format!("  Found {} utterances", utterances.len()));

    // 3. Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .map_err(|e| Error::process(format!("Cannot create output dir: {e}")))?;

    // 4. Extract and save
    step("Extracting features");
    let mut entries = Vec::new();
    let mut skipped = 0usize;
    let mut total_oov = 0usize;
    let mut total_frames = 0usize;
    let total = utterances.len();

    for (i, meta) in utterances.iter().enumerate() {
        match extract_and_save(i, meta, &extractor, &dict, &args.output_dir) {
            Ok(entry) => {
                total_frames += entry.num_frames;
                total_oov += entry.oov_count;
                entries.push(entry);
            }
            Err(e) => {
                tracing::debug!("Skipping {}: {e}", meta.audio_path.display());
                skipped += 1;
            }
        }
        if (i + 1) % 1000 == 0 || i + 1 == total {
            detail(&format!(
                "  [{}/{}] extracted, {} skipped, {} total frames",
                entries.len(),
                i + 1,
                skipped,
                total_frames
            ));
        }
    }

    if entries.is_empty() {
        return Err(Error::config("No valid samples after extraction"));
    }

    // 5. Write manifest
    step("Writing manifest");
    let manifest = PrecomputedManifest {
        feature_dim: extractor.feature_dim(),
        feature_mode: args.feature_mode,
        num_samples: entries.len(),
        total_frames,
        total_oov,
        data_dir: args.data_dir.display().to_string(),
        split: args.split.clone(),
        max_duration_secs: Some(args.max_duration_secs),
        entries,
    };

    let manifest_path = args.output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| Error::process(format!("Failed to serialize manifest: {e}")))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| Error::process(format!("Failed to write manifest: {e}")))?;

    // 6. Summary
    #[allow(clippy::cast_precision_loss)]
    let size_mb = estimate_cache_size(&manifest) as f64 / (1024.0 * 1024.0);

    success(&format!(
        "Precomputed {} samples ({} frames, {:.1} MB) to {}",
        manifest.num_samples,
        total_frames,
        size_mb,
        args.output_dir.display()
    ));
    detail(&format!("  Skipped: {skipped}, OOV words: {total_oov}"));
    detail(&format!("  Feature mode: {}", args.feature_mode.label()));
    detail(&format!("  Max duration: {:.1}s", args.max_duration_secs));
    detail(&format!("  Manifest: {}", manifest_path.display()));

    Ok(())
}

/// Load precomputed samples from a cache directory.
/// Called by the precomputed training loop.
#[allow(dead_code)]
pub fn load_precomputed(
    cache_dir: &Path,
) -> Result<(PrecomputedManifest, Vec<super::dataset::TrainingSample>)> {
    let manifest_path = cache_dir.join("manifest.json");
    let manifest_data = std::fs::read_to_string(&manifest_path)
        .map_err(|e| Error::config(format!("Cannot read manifest: {e}")))?;
    let manifest: PrecomputedManifest = serde_json::from_str(&manifest_data)
        .map_err(|e| Error::config(format!("Invalid manifest JSON: {e}")))?;

    if manifest.feature_dim == 0 {
        return Err(Error::config(format!(
            "Feature dimension mismatch: manifest has invalid feature_dim={}",
            manifest.feature_dim
        )));
    }

    let mut samples = Vec::with_capacity(manifest.num_samples);
    for entry in &manifest.entries {
        let bin_path = cache_dir.join(format!("sample_{:06}.bin", entry.index));
        let sample = load_binary_sample(
            &bin_path,
            entry.num_frames,
            entry.num_targets,
            manifest.feature_dim,
        )?;
        samples.push(super::dataset::TrainingSample {
            feature_frames: sample.0,
            feature_dim: manifest.feature_dim,
            targets: sample.1,
            oov_count: entry.oov_count,
        });
    }

    Ok((manifest, samples))
}

// ── Internal helpers ────────────────────────────────────────────────

fn extract_and_save(
    index: usize,
    meta: &UtteranceMeta,
    extractor: &MfccExtractor,
    dict: &CmuDict,
    output_dir: &Path,
) -> Result<PrecomputedEntry> {
    let audio = load_audio_samples(&meta.audio_path)?;
    let feature_frames = extractor.extract_frames(&audio)?;
    let (targets, stats) = transcript_to_targets(&meta.transcript, dict)?;

    // CTC validity: input length must be at least target length
    let target_len = targets.len();
    if feature_frames.len() < target_len {
        return Err(Error::config(format!(
            "CTC constraint: {} frames < {} targets",
            feature_frames.len(),
            target_len
        )));
    }

    // Write binary file
    let bin_path = output_dir.join(format!("sample_{index:06}.bin"));
    write_binary_sample(
        &bin_path,
        &feature_frames,
        extractor.feature_dim(),
        &targets,
    )?;

    Ok(PrecomputedEntry {
        index,
        num_frames: feature_frames.len(),
        num_targets: targets.len(),
        oov_count: stats.g2p_hits,
        audio_path: meta.audio_path.display().to_string(),
    })
}

/// Binary format: `MAGIC(4) | VERSION(4) | num_frames(4) | num_targets(4) |`
/// `feature_data(num_frames * feature_dim * 4) | targets(num_targets * 4)`
fn write_binary_sample(
    path: &Path,
    feature_frames: &[Vec<f32>],
    feature_dim: usize,
    targets: &[i32],
) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| Error::process(format!("Cannot create {}: {e}", path.display())))?;

    file.write_all(&MAGIC)
        .map_err(|e| Error::process(format!("Write error: {e}")))?;
    #[allow(clippy::cast_possible_truncation)]
    let num_frames = feature_frames.len() as u32;
    #[allow(clippy::cast_possible_truncation)]
    let num_targets = targets.len() as u32;
    file.write_all(&VERSION.to_le_bytes())
        .map_err(|e| Error::process(format!("Write error: {e}")))?;
    file.write_all(&num_frames.to_le_bytes())
        .map_err(|e| Error::process(format!("Write error: {e}")))?;
    file.write_all(&num_targets.to_le_bytes())
        .map_err(|e| Error::process(format!("Write error: {e}")))?;

    // Write feature data as flat f32 array
    for frame in feature_frames {
        if frame.len() != feature_dim {
            return Err(Error::config(format!(
                "Feature frame width mismatch while writing {}: expected {}, got {}",
                path.display(),
                feature_dim,
                frame.len()
            )));
        }
        for &val in frame {
            file.write_all(&val.to_le_bytes())
                .map_err(|e| Error::process(format!("Write error: {e}")))?;
        }
    }

    // Write targets as i32 array
    for &t in targets {
        file.write_all(&t.to_le_bytes())
            .map_err(|e| Error::process(format!("Write error: {e}")))?;
    }

    Ok(())
}

#[allow(dead_code)]
fn load_binary_sample(
    path: &Path,
    expected_frames: usize,
    expected_targets: usize,
    feature_dim: usize,
) -> Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let data = std::fs::read(path)
        .map_err(|e| Error::config(format!("Cannot read {}: {e}", path.display())))?;

    let mut cursor = &data[..];

    // Read and validate magic
    let mut magic = [0u8; 4];
    cursor
        .read_exact(&mut magic)
        .map_err(|e| Error::config(format!("Corrupt file {}: {e}", path.display())))?;
    if magic != MAGIC {
        return Err(Error::config(format!("Bad magic in {}", path.display())));
    }

    // Read version
    let mut buf4 = [0u8; 4];
    cursor
        .read_exact(&mut buf4)
        .map_err(|e| Error::config(format!("Corrupt file {}: {e}", path.display())))?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(Error::config(format!(
            "Unsupported version {version} in {}",
            path.display()
        )));
    }

    // Read frame/target counts
    cursor
        .read_exact(&mut buf4)
        .map_err(|e| Error::config(format!("Corrupt file {}: {e}", path.display())))?;
    let num_frames = u32::from_le_bytes(buf4) as usize;
    cursor
        .read_exact(&mut buf4)
        .map_err(|e| Error::config(format!("Corrupt file {}: {e}", path.display())))?;
    let num_targets = u32::from_le_bytes(buf4) as usize;

    // Sanity bound: ~16 min of audio at 10ms hop = 100K frames max
    let max_reasonable: usize = 100_000;
    if num_frames > max_reasonable || num_targets > max_reasonable {
        return Err(Error::config(format!(
            "Unreasonable sizes in {}: {num_frames}f/{num_targets}t (max {max_reasonable})",
            path.display()
        )));
    }

    if num_frames != expected_frames || num_targets != expected_targets {
        return Err(Error::config(format!(
            "Size mismatch in {}: expected {expected_frames}f/{expected_targets}t, got \
             {num_frames}f/{num_targets}t",
            path.display()
        )));
    }

    // Read feature frames
    let mut frames = Vec::with_capacity(num_frames);
    for _ in 0..num_frames {
        let mut frame = vec![0.0_f32; feature_dim];
        for val in &mut frame {
            cursor.read_exact(&mut buf4).map_err(|e| {
                Error::config(format!("Corrupt feature data in {}: {e}", path.display()))
            })?;
            *val = f32::from_le_bytes(buf4);
        }
        frames.push(frame);
    }

    // Read targets
    let mut targets = Vec::with_capacity(num_targets);
    for _ in 0..num_targets {
        cursor.read_exact(&mut buf4).map_err(|e| {
            Error::config(format!("Corrupt target data in {}: {e}", path.display()))
        })?;
        targets.push(i32::from_le_bytes(buf4));
    }

    Ok((frames, targets))
}

fn estimate_cache_size(manifest: &PrecomputedManifest) -> usize {
    // Per sample: 16 bytes header + frames * feature_dim * 4 + targets * 4
    manifest
        .entries
        .iter()
        .map(|e| 16 + e.num_frames * manifest.feature_dim * 4 + e.num_targets * 4)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_roundtrip() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.bin");

        let frames = vec![
            vec![1.0f32; FeatureMode::Mfcc39.feature_dim()],
            vec![2.0; FeatureMode::Mfcc39.feature_dim()],
            vec![3.0; FeatureMode::Mfcc39.feature_dim()],
        ];
        let targets = vec![0i32, 5, 10, 41];

        write_binary_sample(&path, &frames, FeatureMode::Mfcc39.feature_dim(), &targets)
            .expect("write");
        let (loaded_frames, loaded_targets) =
            load_binary_sample(&path, 3, 4, FeatureMode::Mfcc39.feature_dim()).expect("load");

        assert_eq!(loaded_frames.len(), 3);
        assert_eq!(loaded_targets, vec![0, 5, 10, 41]);
        for (i, frame) in loaded_frames.iter().enumerate() {
            let expected = (i + 1) as f32;
            assert!((frame[0] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_binary_magic_validation() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, b"BAAD").expect("write");

        let result = load_binary_sample(&path, 1, 1, FeatureMode::Mfcc39.feature_dim());
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_version_validation() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("badver.bin");
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // bad version
        std::fs::write(&path, &data).expect("write");

        let result = load_binary_sample(&path, 0, 0, FeatureMode::Mfcc39.feature_dim());
        assert!(result.is_err());
    }

    #[test]
    fn test_manifest_serialization() {
        let manifest = PrecomputedManifest {
            feature_dim: 39,
            feature_mode: FeatureMode::Mfcc39,
            num_samples: 2,
            total_frames: 100,
            total_oov: 3,
            data_dir: "/data".to_owned(),
            split: "train-clean-100".to_owned(),
            max_duration_secs: Some(15.0),
            entries: vec![
                PrecomputedEntry {
                    index: 0,
                    num_frames: 50,
                    num_targets: 10,
                    oov_count: 1,
                    audio_path: "a.wav".to_owned(),
                },
                PrecomputedEntry {
                    index: 1,
                    num_frames: 50,
                    num_targets: 15,
                    oov_count: 2,
                    audio_path: "b.wav".to_owned(),
                },
            ],
        };

        let json = serde_json::to_string(&manifest).expect("serialize");
        let loaded: PrecomputedManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.num_samples, 2);
        assert_eq!(loaded.total_frames, 100);
        assert_eq!(loaded.feature_mode, FeatureMode::Mfcc39);
        assert_eq!(loaded.max_duration_secs, Some(15.0));
        assert_eq!(loaded.entries.len(), 2);
    }

    #[test]
    fn test_manifest_deserializes_legacy_without_duration_cap() {
        let json = r#"{
          "feature_dim": 39,
          "num_samples": 1,
          "total_frames": 10,
          "total_oov": 0,
          "data_dir": "/data",
          "split": "train-clean-100",
          "entries": [
            {
              "index": 0,
              "num_frames": 10,
              "num_targets": 4,
              "oov_count": 0,
              "audio_path": "a.wav"
            }
          ]
        }"#;

        let loaded: PrecomputedManifest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(loaded.feature_mode, FeatureMode::Mfcc39);
        assert_eq!(loaded.max_duration_secs, None);
    }

    #[test]
    fn test_size_mismatch_rejected() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.bin");

        let frames = vec![vec![1.0f32; FeatureMode::Mfcc39.feature_dim()]];
        let targets = vec![0i32];
        write_binary_sample(&path, &frames, FeatureMode::Mfcc39.feature_dim(), &targets)
            .expect("write");

        // Request wrong counts
        let result = load_binary_sample(&path, 2, 1, FeatureMode::Mfcc39.feature_dim());
        assert!(result.is_err());
    }
}
