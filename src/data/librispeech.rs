//! LibriSpeech loader.
//!
//! Expects the standard layout:
//!
//! ```text
//! <data_dir>/<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
//! <data_dir>/<split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
//! ```
//!
//! `<data_dir>/<split>/<split>/...` (an extra level, as left by some tar
//! extractions) and `.wav` files are also accepted.

use std::path::{Path, PathBuf};

use crate::data::{audio, ctc_min_frames, transcript_targets, TrainingSample};
use crate::error::{Error, Result};
use crate::mfcc::{FeatureMode, MfccExtractor, SAMPLE_RATE};

/// Utterances shorter than this are skipped.
const MIN_DURATION_SECS: f32 = 0.5;

/// Which LibriSpeech split to load and how to featurize it.
#[derive(Debug, Clone)]
pub struct LibriSpeechSource {
    /// Directory that contains the split directories, e.g. `.../LibriSpeech`.
    pub data_dir: PathBuf,
    /// Split name, e.g. `train-clean-100`.
    pub split: String,
    pub feature_mode: FeatureMode,
    /// Longer utterances are skipped. Attention memory grows with the square
    /// of utterance length, so this bounds peak memory.
    pub max_duration_secs: f32,
}

/// An audio file and its transcript.
#[derive(Debug, Clone)]
pub struct Utterance {
    pub audio_path: PathBuf,
    pub transcript: String,
}

/// Why an utterance was not turned into a training sample.
#[derive(Debug)]
pub enum Skip {
    TooShort,
    TooLong,
    /// Fewer frames than CTC needs for the transcript.
    TooFewFrames,
    Failed(Error),
}

/// List all utterances in a split.
pub fn scan(data_dir: &Path, split: &str) -> Result<Vec<Utterance>> {
    let split_dir = data_dir.join(split);
    if !split_dir.is_dir() {
        return Err(Error::data(format!(
            "LibriSpeech split directory not found: {}",
            split_dir.display()
        )));
    }
    let nested = split_dir.join(split);
    let root = if nested.is_dir() { nested } else { split_dir };

    let mut utterances = Vec::new();
    for speaker in subdirs(&root)? {
        for chapter in subdirs(&speaker)? {
            for transcript in files_with_suffix(&chapter, ".trans.txt")? {
                utterances.extend(parse_transcripts(&transcript, &chapter)?);
            }
        }
    }

    if utterances.is_empty() {
        return Err(Error::data(format!(
            "no utterances found under {}",
            root.display()
        )));
    }
    Ok(utterances)
}

/// Decode, featurize and align one utterance.
pub fn load_sample(
    utterance: &Utterance,
    extractor: &MfccExtractor,
    max_duration_secs: f32,
) -> std::result::Result<TrainingSample, Skip> {
    let samples = audio::load(&utterance.audio_path).map_err(Skip::Failed)?;
    let duration = samples.len() as f32 / SAMPLE_RATE as f32;
    if duration < MIN_DURATION_SECS {
        return Err(Skip::TooShort);
    }
    if duration > max_duration_secs {
        return Err(Skip::TooLong);
    }

    let frames = extractor.extract_frames(&samples).map_err(Skip::Failed)?;
    let (targets, oov_words) = transcript_targets(&utterance.transcript).map_err(Skip::Failed)?;
    if targets.is_empty() || frames.len() < ctc_min_frames(&targets) {
        return Err(Skip::TooFewFrames);
    }

    let mut sample = TrainingSample::from_frames(&frames, extractor.feature_dim(), targets);
    sample.oov_words = oov_words;
    Ok(sample)
}

/// Load every usable utterance in a split.
pub fn load(source: &LibriSpeechSource) -> Result<Vec<TrainingSample>> {
    let utterances = scan(&source.data_dir, &source.split)?;
    tracing::info!(
        "loading {} utterances from {}/{} ({})",
        utterances.len(),
        source.data_dir.display(),
        source.split,
        source.feature_mode.label()
    );

    let extractor = MfccExtractor::with_mode(source.feature_mode);
    let mut samples = Vec::with_capacity(utterances.len());
    let mut skips = SkipCounts::default();
    for (i, utterance) in utterances.iter().enumerate() {
        match load_sample(utterance, &extractor, source.max_duration_secs) {
            Ok(sample) => samples.push(sample),
            Err(skip) => skips.record(utterance, skip),
        }
        if (i + 1) % 1000 == 0 {
            tracing::info!("  {}/{} processed", i + 1, utterances.len());
        }
    }
    skips.report(samples.len());

    if samples.is_empty() {
        return Err(Error::data("no usable utterances"));
    }
    Ok(samples)
}

/// Tallies skipped utterances so they can be reported once at the end.
#[derive(Default)]
pub(crate) struct SkipCounts {
    too_short: usize,
    too_long: usize,
    too_few_frames: usize,
    failed: usize,
    first_failure: Option<String>,
}

impl SkipCounts {
    pub(crate) fn record(&mut self, utterance: &Utterance, skip: Skip) {
        match skip {
            Skip::TooShort => self.too_short += 1,
            Skip::TooLong => self.too_long += 1,
            Skip::TooFewFrames => self.too_few_frames += 1,
            Skip::Failed(err) => {
                tracing::debug!("skipping {}: {err}", utterance.audio_path.display());
                self.failed += 1;
                self.first_failure.get_or_insert_with(|| err.to_string());
            }
        }
    }

    pub(crate) fn report(&self, kept: usize) {
        tracing::info!(
            "kept {kept} utterances; skipped {} too short, {} too long, {} with fewer frames \
             than CTC targets, {} failed",
            self.too_short,
            self.too_long,
            self.too_few_frames,
            self.failed
        );
        if let Some(first) = &self.first_failure {
            tracing::warn!(
                "{} utterances failed to load, first error: {first}",
                self.failed
            );
        }
    }
}

fn read_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = std::fs::read_dir(dir)
        .and_then(|entries| {
            entries
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<Vec<_>>>()
        })
        .map_err(|err| Error::io(dir, err))?;
    paths.sort();
    Ok(paths)
}

fn subdirs(dir: &Path) -> Result<Vec<PathBuf>> {
    Ok(read_dir(dir)?.into_iter().filter(|p| p.is_dir()).collect())
}

fn files_with_suffix(dir: &Path, suffix: &str) -> Result<Vec<PathBuf>> {
    Ok(read_dir(dir)?
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(suffix))
        })
        .collect())
}

/// Each line of a `.trans.txt` file is `<utterance-id> <TRANSCRIPT>`.
fn parse_transcripts(path: &Path, chapter_dir: &Path) -> Result<Vec<Utterance>> {
    let content = crate::error::read_to_string(path)?;
    let mut utterances = Vec::new();
    for line in content.lines() {
        let Some((id, transcript)) = line.trim().split_once(' ') else {
            continue;
        };
        let audio_path = ["flac", "wav"]
            .iter()
            .map(|ext| chapter_dir.join(format!("{id}.{ext}")))
            .find(|p| p.exists());
        match audio_path {
            Some(audio_path) => utterances.push(Utterance {
                audio_path,
                transcript: transcript.trim().to_owned(),
            }),
            None => tracing::warn!("no audio file for {id} listed in {}", path.display()),
        }
    }
    Ok(utterances)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use super::*;
    use crate::data::audio::{test_tone, write_test_wav};

    /// Build a two-utterance LibriSpeech tree with synthetic audio.
    pub(crate) fn fake_librispeech(root: &Path, split: &str, nested: bool) -> PathBuf {
        let mut chapter = root.join(split);
        if nested {
            chapter = chapter.join(split);
        }
        let chapter = chapter.join("103").join("1240");
        fs::create_dir_all(&chapter).expect("mkdir");
        fs::write(
            chapter.join("103-1240.trans.txt"),
            "103-1240-0000 THE CAT SAT\n103-1240-0001 A DOG\n",
        )
        .expect("write transcript");
        write_test_wav(&chapter.join("103-1240-0000.wav"), &test_tone(1.5), 16_000);
        write_test_wav(&chapter.join("103-1240-0001.wav"), &test_tone(1.0), 16_000);
        chapter
    }

    #[test]
    fn scan_reads_transcripts() {
        let dir = tempfile::tempdir().expect("tempdir");
        fake_librispeech(dir.path(), "train-clean-100", false);

        let utterances = scan(dir.path(), "train-clean-100").expect("scan");
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].transcript, "THE CAT SAT");
        assert!(utterances[0].audio_path.ends_with("103-1240-0000.wav"));
    }

    #[test]
    fn scan_accepts_nested_split_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        fake_librispeech(dir.path(), "train-clean-100", true);
        assert_eq!(scan(dir.path(), "train-clean-100").expect("scan").len(), 2);
    }

    #[test]
    fn scan_missing_split_is_an_error() {
        assert!(scan(Path::new("/nonexistent"), "train-clean-100").is_err());
    }

    #[test]
    fn load_featurizes_and_filters_by_duration() {
        let dir = tempfile::tempdir().expect("tempdir");
        fake_librispeech(dir.path(), "dev", false);
        let source = |max_duration_secs| LibriSpeechSource {
            data_dir: dir.path().to_path_buf(),
            split: "dev".into(),
            feature_mode: FeatureMode::Mfcc39,
            max_duration_secs,
        };

        let samples = load(&source(10.0)).expect("load");
        assert_eq!(samples.len(), 2);
        assert!(samples
            .iter()
            .all(|s| s.feature_dim == 39 && !s.targets.is_empty()));
        assert!(samples.iter().all(|s| s.num_frames() >= s.targets.len()));

        // The 1.5 s utterance is over a 1.2 s cap.
        assert_eq!(load(&source(1.2)).expect("load").len(), 1);
    }
}
