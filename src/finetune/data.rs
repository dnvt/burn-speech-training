//! Turning SpeechOcean762 word scores into featurized utterances.

use std::path::Path;

use crate::data::librispeech::{Skip, SkipCounts, Utterance};
use crate::data::so762::{self, WordSample};
use crate::data::{audio, ctc_min_frames, transcript_targets, TrainingSample};
use crate::error::{Error, Result};
use crate::mfcc::{FeatureMode, MfccExtractor};

/// A scored word inside an utterance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordTarget {
    pub position: usize,
    pub num_words: usize,
    /// Human score in `[0, 1]`.
    pub score: f32,
    /// Human score on the original 0–10 scale.
    pub raw_score: i32,
    /// Loss weight; 1.0 unless an experiment reweights words.
    pub weight: f32,
}

/// One audio file with its features, CTC targets and scored words.
#[derive(Debug, Clone)]
pub struct ScoredUtterance {
    pub audio_file: String,
    /// Features and CTC targets for the full sentence.
    pub sample: TrainingSample,
    pub words: Vec<WordTarget>,
}

/// Load a word-score file and featurize every utterance in it.
///
/// Utterances whose audio fails to load, or whose transcript needs more
/// frames than the audio has, are skipped and counted in the log.
pub fn load(
    dataset: &Path,
    data_root: &Path,
    feature_mode: FeatureMode,
    max_samples: Option<usize>,
) -> Result<Vec<ScoredUtterance>> {
    let words = so762::load(dataset, max_samples)?;
    let utterances = featurize(&words, data_root, feature_mode);
    if utterances.is_empty() {
        return Err(Error::data(format!(
            "{}: no usable utterances (check data_root: {})",
            dataset.display(),
            data_root.display()
        )));
    }
    Ok(utterances)
}

fn featurize(
    words: &[WordSample],
    data_root: &Path,
    feature_mode: FeatureMode,
) -> Vec<ScoredUtterance> {
    let extractor = MfccExtractor::with_mode(feature_mode);
    let mut skips = SkipCounts::default();
    let mut utterances = Vec::new();

    for (audio_file, words) in so762::group_by_audio(words) {
        let sentence = &words[0].sentence_text;
        let utterance = Utterance {
            audio_path: data_root.join(audio_file),
            transcript: sentence.clone(),
        };
        let sample = audio::load(&utterance.audio_path)
            .and_then(|samples| extractor.extract_frames(&samples))
            .and_then(|frames| {
                let (targets, oov_words) = transcript_targets(sentence)?;
                let mut sample =
                    TrainingSample::from_frames(&frames, extractor.feature_dim(), targets);
                sample.oov_words = oov_words;
                Ok(sample)
            });
        let sample = match sample {
            Ok(sample) if sample.num_frames() >= ctc_min_frames(&sample.targets) => sample,
            Ok(_) => {
                skips.record(&utterance, Skip::TooFewFrames);
                continue;
            }
            Err(err) => {
                skips.record(&utterance, Skip::Failed(err));
                continue;
            }
        };

        utterances.push(ScoredUtterance {
            audio_file: audio_file.to_owned(),
            sample,
            words: words
                .iter()
                .map(|w| WordTarget {
                    position: w.word_position,
                    num_words: w.num_words(),
                    score: w.human_score as f32,
                    raw_score: w.raw_score(),
                    weight: 1.0,
                })
                .collect(),
        });
    }
    skips.report(utterances.len());
    utterances
}
