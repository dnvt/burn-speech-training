//! Word-level SpeechOcean762 scores.
//!
//! SpeechOcean762 ships per-utterance JSON with nested word scores. This
//! crate reads a flattened file with one entry per word:
//!
//! ```json
//! {
//!   "samples": [
//!     {
//!       "audio_file": "WAVE/SPEAKER0001/000010011.WAV",
//!       "sentence_text": "WE CALL IT BEAR",
//!       "word_position": 3,
//!       "human_score": 0.8,
//!       "human_score_raw": 8,
//!       "phonemes": ["B", "EH", "R"]
//!     }
//!   ]
//! }
//! ```
//!
//! - `audio_file` is relative to the data root passed to fine-tuning or
//!   evaluation. Audio must be 16 kHz.
//! - `word_position` is the 0-based index of the word in `sentence_text`.
//! - `human_score` is the word accuracy score scaled to `[0, 1]`;
//!   `human_score_raw` is the original 0–10 score (optional, defaults to
//!   `round(human_score × 10)`).
//! - `phonemes` (ARPABET, optional) is only needed for the GOP baselines in
//!   evaluation.
//!
//! Other fields are ignored.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{self, Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct WordSample {
    pub audio_file: String,
    pub sentence_text: String,
    pub word_position: usize,
    pub human_score: f64,
    #[serde(default)]
    human_score_raw: Option<i32>,
    #[serde(default)]
    pub phonemes: Vec<String>,
}

impl WordSample {
    /// The 0–10 score.
    pub fn raw_score(&self) -> i32 {
        self.human_score_raw
            .unwrap_or_else(|| (self.human_score * 10.0).round() as i32)
    }

    pub fn num_words(&self) -> usize {
        self.sentence_text.split_whitespace().count()
    }
}

#[derive(Deserialize)]
struct WordFile {
    samples: Vec<WordSample>,
}

/// Read the first `max_samples` words (all when `None`).
pub fn load(path: &Path, max_samples: Option<usize>) -> Result<Vec<WordSample>> {
    let file: WordFile = serde_json::from_str(&error::read_to_string(path)?)
        .map_err(|err| Error::data(format!("{}: {err}", path.display())))?;
    let mut samples = file.samples;
    if let Some(max) = max_samples {
        samples.truncate(max);
    }
    let bad = samples
        .iter()
        .filter(|s| s.word_position >= s.num_words() || !(0.0..=1.0).contains(&s.human_score))
        .count();
    if bad > 0 {
        tracing::warn!(
            "{}: {bad} words have word_position outside the sentence or human_score outside \
             [0, 1]; they will be skipped",
            path.display()
        );
    }
    Ok(samples)
}

/// Group words by audio file, keeping only words whose position is inside
/// the sentence. Groups are sorted by path for reproducible ordering.
pub fn group_by_audio(samples: &[WordSample]) -> BTreeMap<&str, Vec<&WordSample>> {
    let mut groups: BTreeMap<&str, Vec<&WordSample>> = BTreeMap::new();
    for sample in samples {
        if sample.word_position < sample.num_words() && (0.0..=1.0).contains(&sample.human_score) {
            groups.entry(&sample.audio_file).or_default().push(sample);
        }
    }
    for words in groups.values_mut() {
        words.sort_by_key(|w| w.word_position);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    const JSON: &str = r#"{
      "metadata": {"ignored": true},
      "samples": [
        {"audio_file": "b.wav", "sentence_text": "A B", "word_position": 1, "human_score": 0.5,
         "human_score_raw": 5, "word_id": "ignored"},
        {"audio_file": "a.wav", "sentence_text": "X", "word_position": 0, "human_score": 1.0},
        {"audio_file": "b.wav", "sentence_text": "A B", "word_position": 0, "human_score": 0.9},
        {"audio_file": "b.wav", "sentence_text": "A B", "word_position": 7, "human_score": 0.9}
      ]
    }"#;

    fn load_str(json: &str, max: Option<usize>) -> Vec<WordSample> {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("words.json");
        std::fs::write(&path, json).expect("write");
        load(&path, max).expect("load")
    }

    #[test]
    fn loads_minimal_and_full_entries() {
        let samples = load_str(JSON, None);
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].raw_score(), 5);
        assert_eq!(samples[1].raw_score(), 10, "derived from human_score");
        assert!(samples[1].phonemes.is_empty());
        assert_eq!(load_str(JSON, Some(2)).len(), 2);
    }

    #[test]
    fn groups_by_audio_and_drops_out_of_range_words() {
        let samples = load_str(JSON, None);
        let groups = group_by_audio(&samples);
        assert_eq!(
            groups.keys().copied().collect::<Vec<_>>(),
            ["a.wav", "b.wav"]
        );
        let positions: Vec<usize> = groups["b.wav"].iter().map(|w| w.word_position).collect();
        assert_eq!(positions, [0, 1]);
    }
}
