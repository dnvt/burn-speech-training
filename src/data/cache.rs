//! Precomputed feature cache.
//!
//! Decoding audio and computing MFCCs is the slowest part of a training run
//! on GPU. `precompute` does it once and writes one binary file per
//! utterance plus a `manifest.json`; `load` reads them back as
//! [`TrainingSample`]s.
//!
//! # File format
//!
//! `sample_NNNNNN.bin`, all integers little-endian:
//!
//! | bytes | content |
//! |---|---|
//! | 4 | magic `MFCC` |
//! | 4 | format version, `u32` (currently 2) |
//! | 4 | `num_frames`, `u32` |
//! | 4 | `num_targets`, `u32` |
//! | 4 | `feature_dim`, `u32` (version 2 only) |
//! | `num_frames × feature_dim × 4` | features, `f32`, row-major |
//! | `num_targets × 4` | CTC targets, `i32` |
//!
//! Version 1 files (no `feature_dim` field) are still readable; their
//! feature dimension comes from the manifest.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::librispeech::{self, LibriSpeechSource, SkipCounts};
use crate::data::TrainingSample;
use crate::error::{self, Error, Result};
use crate::mfcc::{FeatureMode, MfccExtractor};

const MAGIC: [u8; 4] = *b"MFCC";
const VERSION: u32 = 2;
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheEntry {
    pub index: usize,
    pub num_frames: usize,
    pub num_targets: usize,
    pub oov_count: usize,
    pub audio_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheManifest {
    pub feature_dim: usize,
    /// Missing in manifests written before this field existed; `load`
    /// infers it from `feature_dim`.
    #[serde(default)]
    pub feature_mode: Option<FeatureMode>,
    pub num_samples: usize,
    pub total_frames: usize,
    pub total_oov: usize,
    pub data_dir: String,
    pub split: String,
    #[serde(default)]
    pub max_duration_secs: Option<f32>,
    pub entries: Vec<CacheEntry>,
}

fn sample_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("sample_{index:06}.bin"))
}

/// Featurize a LibriSpeech split into `output_dir`.
pub fn precompute(source: &LibriSpeechSource, output_dir: &Path) -> Result<CacheManifest> {
    let utterances = librispeech::scan(&source.data_dir, &source.split)?;
    error::create_dir_all(output_dir)?;
    tracing::info!(
        "precomputing {} features for {} utterances into {}",
        source.feature_mode.label(),
        utterances.len(),
        output_dir.display()
    );

    let extractor = MfccExtractor::with_mode(source.feature_mode);
    let mut entries = Vec::new();
    let mut skips = SkipCounts::default();
    for (index, utterance) in utterances.iter().enumerate() {
        match librispeech::load_sample(utterance, &extractor, source.max_duration_secs) {
            Ok(sample) => {
                write_sample(&sample_path(output_dir, index), &sample)?;
                entries.push(CacheEntry {
                    index,
                    num_frames: sample.num_frames(),
                    num_targets: sample.targets.len(),
                    oov_count: sample.oov_words,
                    audio_path: utterance.audio_path.display().to_string(),
                });
            }
            Err(skip) => skips.record(utterance, skip),
        }
        if (index + 1) % 1000 == 0 {
            tracing::info!("  {}/{} processed", index + 1, utterances.len());
        }
    }
    skips.report(entries.len());
    if entries.is_empty() {
        return Err(Error::data("no usable utterances"));
    }

    let manifest = CacheManifest {
        feature_dim: extractor.feature_dim(),
        feature_mode: Some(source.feature_mode),
        num_samples: entries.len(),
        total_frames: entries.iter().map(|e| e.num_frames).sum(),
        total_oov: entries.iter().map(|e| e.oov_count).sum(),
        data_dir: source.data_dir.display().to_string(),
        split: source.split.clone(),
        max_duration_secs: Some(source.max_duration_secs),
        entries,
    };
    error::write(
        &output_dir.join(MANIFEST_FILE),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

/// Read a cache written by [`precompute`].
pub fn load(dir: &Path) -> Result<(CacheManifest, Vec<TrainingSample>)> {
    let manifest_path = dir.join(MANIFEST_FILE);
    let mut manifest: CacheManifest =
        serde_json::from_str(&error::read_to_string(&manifest_path)?)?;

    let inferred_mode = [FeatureMode::Mfcc39, FeatureMode::LogMel80]
        .into_iter()
        .find(|mode| mode.feature_dim() == manifest.feature_dim);
    match (manifest.feature_mode, inferred_mode) {
        (Some(mode), _) if mode.feature_dim() != manifest.feature_dim => {
            return Err(Error::data(format!(
                "{}: feature_mode {} does not match feature_dim {}",
                manifest_path.display(),
                mode.label(),
                manifest.feature_dim
            )));
        }
        (None, Some(mode)) => manifest.feature_mode = Some(mode),
        (None, None) => {
            return Err(Error::data(format!(
                "{}: unsupported feature_dim {}",
                manifest_path.display(),
                manifest.feature_dim
            )));
        }
        _ => {}
    }
    if manifest.num_samples != manifest.entries.len() {
        return Err(Error::data(format!(
            "{}: num_samples is {} but there are {} entries",
            manifest_path.display(),
            manifest.num_samples,
            manifest.entries.len()
        )));
    }

    let samples = manifest
        .entries
        .iter()
        .map(|entry| {
            let mut sample = read_sample(&sample_path(dir, entry.index), manifest.feature_dim)?;
            if sample.num_frames() != entry.num_frames || sample.targets.len() != entry.num_targets
            {
                return Err(Error::data(format!(
                    "{}: manifest says {} frames / {} targets, file has {} / {}",
                    sample_path(dir, entry.index).display(),
                    entry.num_frames,
                    entry.num_targets,
                    sample.num_frames(),
                    sample.targets.len()
                )));
            }
            sample.oov_words = entry.oov_count;
            Ok(sample)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((manifest, samples))
}

fn write_sample(path: &Path, sample: &TrainingSample) -> Result<()> {
    let to_u32 = |n: usize| {
        u32::try_from(n).map_err(|_| Error::data(format!("{}: sample too large", path.display())))
    };
    let mut buf = Vec::with_capacity(20 + 4 * (sample.features.len() + sample.targets.len()));
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&to_u32(sample.num_frames())?.to_le_bytes());
    buf.extend_from_slice(&to_u32(sample.targets.len())?.to_le_bytes());
    buf.extend_from_slice(&to_u32(sample.feature_dim)?.to_le_bytes());
    for value in &sample.features {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    for target in &sample.targets {
        buf.extend_from_slice(&target.to_le_bytes());
    }
    error::write(path, buf)
}

/// `feature_dim` is the manifest's value; version 2 files must agree with it.
fn read_sample(path: &Path, feature_dim: usize) -> Result<TrainingSample> {
    let data = std::fs::read(path).map_err(|err| Error::io(path, err))?;
    let corrupt =
        |why: &str| Error::data(format!("{}: corrupt cache file ({why})", path.display()));
    let word = |i: usize| -> Result<[u8; 4]> {
        data.get(i * 4..i * 4 + 4)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| corrupt("truncated header"))
    };

    if word(0)? != MAGIC {
        return Err(corrupt("bad magic"));
    }
    let version = u32::from_le_bytes(word(1)?);
    let num_frames = u32::from_le_bytes(word(2)?) as usize;
    let num_targets = u32::from_le_bytes(word(3)?) as usize;
    let header_words = match version {
        1 => 4,
        2 => {
            if u32::from_le_bytes(word(4)?) as usize != feature_dim {
                return Err(corrupt("feature_dim differs from manifest"));
            }
            5
        }
        other => {
            return Err(corrupt(&format!("unsupported version {other}")));
        }
    };

    let num_features = num_frames
        .checked_mul(feature_dim)
        .ok_or_else(|| corrupt("size overflow"))?;
    let expected_len = 4 * (header_words + num_features + num_targets);
    if data.len() != expected_len {
        return Err(corrupt(&format!(
            "expected {expected_len} bytes, found {}",
            data.len()
        )));
    }

    let body = &data[4 * header_words..];
    let (features, targets) = body.split_at(4 * num_features);
    Ok(TrainingSample {
        features: features
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        feature_dim,
        targets: targets
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        oov_words: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::librispeech::tests::fake_librispeech;

    fn sample() -> TrainingSample {
        TrainingSample {
            features: (0..3 * 39).map(|v| v as f32 * 0.5).collect(),
            feature_dim: 39,
            targets: vec![5, 10, 41],
            oov_words: 0,
        }
    }

    #[test]
    fn sample_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("s.bin");
        write_sample(&path, &sample()).expect("write");
        assert_eq!(read_sample(&path, 39).expect("read"), sample());
    }

    #[test]
    fn reads_version_1_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("v1.bin");
        let s = sample();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        s.features
            .iter()
            .for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        s.targets
            .iter()
            .for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        std::fs::write(&path, bytes).expect("write");

        assert_eq!(read_sample(&path, 39).expect("read"), s);
    }

    #[test]
    fn rejects_corrupt_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("s.bin");

        std::fs::write(&path, b"BAAD").expect("write");
        assert!(read_sample(&path, 39).is_err(), "bad magic");

        write_sample(&path, &sample()).expect("write");
        assert!(read_sample(&path, 80).is_err(), "feature_dim mismatch");

        let mut bytes = std::fs::read(&path).expect("read");
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, &bytes).expect("write");
        assert!(read_sample(&path, 39).is_err(), "truncated");

        bytes.extend_from_slice(&[0; 5]);
        std::fs::write(&path, &bytes).expect("write");
        assert!(read_sample(&path, 39).is_err(), "trailing bytes");
    }

    #[test]
    fn precompute_then_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        fake_librispeech(dir.path(), "dev", false);
        let cache_dir = dir.path().join("cache");
        let source = LibriSpeechSource {
            data_dir: dir.path().to_path_buf(),
            split: "dev".into(),
            feature_mode: FeatureMode::Mfcc39,
            max_duration_secs: 10.0,
        };

        let written = precompute(&source, &cache_dir).expect("precompute");
        assert_eq!(written.num_samples, 2);

        let (manifest, samples) = load(&cache_dir).expect("load");
        assert_eq!(manifest, written);
        assert_eq!(samples, librispeech::load(&source).expect("direct load"));
    }

    #[test]
    fn legacy_manifest_infers_feature_mode() {
        let json = r#"{
          "feature_dim": 80, "num_samples": 0, "total_frames": 0, "total_oov": 0,
          "data_dir": "/data", "split": "dev", "entries": []
        }"#;
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(MANIFEST_FILE), json).expect("write");

        let (manifest, samples) = load(dir.path()).expect("load");
        assert_eq!(manifest.feature_mode, Some(FeatureMode::LogMel80));
        assert!(samples.is_empty());
    }
}
