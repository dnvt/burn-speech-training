//! Word-level pronunciation scoring head.
//!
//! The head pools the backbone's frame features over a word's frame span and
//! maps the pooled vector to a score in `[0, 1]` with a two-layer MLP:
//! `Linear(input_dim → hidden_dim) → ReLU → Linear(hidden_dim → 1) → sigmoid`.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};
use serde::{Deserialize, Serialize};

use crate::checkpoint;
use crate::error::{self, Error, Result};
use crate::mfcc::FeatureMode;
use crate::model::proportional_word_frame_span;

const CONFIG_FILE: &str = "scoring-head-config.json";

/// How the head turns logits into a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreOutput {
    /// One logit, squashed with a sigmoid.
    #[default]
    Sigmoid,
    /// Four cumulative-threshold logits over five score bins.
    #[cfg(feature = "experiments")]
    Ordinal,
    /// Five-way softmax over score bins.
    #[cfg(feature = "experiments")]
    OrdinalSoftmax,
}

impl ScoreOutput {
    pub fn num_logits(self) -> usize {
        match self {
            Self::Sigmoid => 1,
            #[cfg(feature = "experiments")]
            Self::Ordinal => 4,
            #[cfg(feature = "experiments")]
            Self::OrdinalSoftmax => 5,
        }
    }
}

/// How frame features are pooled over a word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    #[default]
    Mean,
    /// Softmax-weighted mean with learned per-frame weights.
    #[cfg(feature = "experiments")]
    Attention,
}

/// Where the head's input features come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSource {
    /// `SpeechAligner` frame features.
    #[default]
    Backbone,
    /// Precomputed wav2vec2 features (see `experiments::teacher`).
    #[cfg(feature = "experiments")]
    TeacherCache,
}

/// Saved next to every scoring-head checkpoint as `scoring-head-config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoringHeadConfig {
    /// Features the backbone was trained on.
    pub feature_mode: FeatureMode,
    pub input_dim: usize,
    pub hidden_dim: usize,
    #[serde(default)]
    pub output: ScoreOutput,
    #[serde(default)]
    pub pooling: Pooling,
    #[serde(default)]
    pub input_source: InputSource,
}

impl ScoringHeadConfig {
    pub fn new(feature_mode: FeatureMode, input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            feature_mode,
            input_dim,
            hidden_dim,
            output: ScoreOutput::default(),
            pooling: Pooling::default(),
            input_source: InputSource::default(),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ScoringHead<B> {
        let attention_pool = match self.pooling {
            Pooling::Mean => None,
            #[cfg(feature = "experiments")]
            Pooling::Attention => Some(LinearConfig::new(self.input_dim, 1).init(device)),
        };
        ScoringHead {
            hidden: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            output: LinearConfig::new(self.hidden_dim, self.output.num_logits()).init(device),
            attention_pool,
        }
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        error::write(&dir.join(CONFIG_FILE), serde_json::to_string_pretty(self)?)
    }

    /// Read the config saved in the same directory as `head_checkpoint`.
    pub fn load_for(head_checkpoint: &Path) -> Result<Self> {
        let path = config_path(head_checkpoint);
        serde_json::from_str(&error::read_to_string(&path)?).map_err(|err| {
            Error::config(format!(
                "{}: {err} (heads trained with the `experiments` feature need that feature \
                 to load)",
                path.display()
            ))
        })
    }
}

fn config_path(head_checkpoint: &Path) -> PathBuf {
    head_checkpoint
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(CONFIG_FILE)
}

#[derive(Module, Debug)]
pub struct ScoringHead<B: Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
    attention_pool: Option<Linear<B>>,
}

impl<B: Backend> ScoringHead<B> {
    /// Load a head checkpoint together with its config.
    pub fn load(path: &Path, device: &B::Device) -> Result<(Self, ScoringHeadConfig)> {
        let config = ScoringHeadConfig::load_for(path)?;
        let head = checkpoint::load(config.init(device), path, device)?;
        Ok((head, config))
    }

    /// `[frames, dim]` → `[1, dim]`.
    pub fn pool(&self, span: Tensor<B, 2>) -> Tensor<B, 2> {
        let [frames, dim] = span.dims();
        match &self.attention_pool {
            None => span.mean_dim(0),
            Some(attention) => {
                let weights = burn::tensor::activation::softmax(
                    attention.forward(span.clone()).reshape([1, frames]),
                    1,
                )
                .reshape([frames, 1]);
                (span * weights).sum_dim(0).reshape([1, dim])
            }
        }
    }

    /// `[words, input_dim]` → `[words, num_logits]`.
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.output.forward(relu(self.hidden.forward(pooled)))
    }

    /// Score each word of an utterance. `frame_features` is `[frames, dim]`;
    /// each word is `(word_position, num_words)` and gets the proportional
    /// frame span used during training. Words whose span is empty get `None`.
    pub fn score_words(
        &self,
        frame_features: &Tensor<B, 2>,
        config: &ScoringHeadConfig,
        words: &[(usize, usize)],
    ) -> Vec<Option<f32>> {
        let [frames, dim] = frame_features.dims();
        let spans: Vec<Option<(usize, usize)>> = words
            .iter()
            .map(|&(position, num_words)| proportional_word_frame_span(frames, num_words, position))
            .collect();
        let pooled: Vec<Tensor<B, 2>> = spans
            .iter()
            .flatten()
            .map(|&(start, end)| self.pool(frame_features.clone().slice([start..end, 0..dim])))
            .collect();
        if pooled.is_empty() {
            return vec![None; words.len()];
        }

        let scores = predict(self.forward(Tensor::cat(pooled, 0)), config.output);
        let mut scores = scores
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>()
            .into_iter();
        spans
            .iter()
            .map(|span| span.and_then(|_| scores.next()))
            .collect()
    }
}

/// Logits `[words, num_logits]` → scores `[words]` in `[0, 1]`.
pub fn predict<B: Backend>(logits: Tensor<B, 2>, output: ScoreOutput) -> Tensor<B, 1> {
    let words = logits.dims()[0];
    match output {
        ScoreOutput::Sigmoid => sigmoid(logits).reshape([words]),
        #[cfg(feature = "experiments")]
        other => crate::experiments::losses::ordinal_predictions(logits, other),
    }
}

/// Mean of a one-element (or any) tensor as `f32`.
pub(crate) fn scalar<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> f32 {
    tensor.mean().into_scalar().elem()
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::{Distribution, Shape, TensorData};

    use super::*;

    type B = NdArray;

    #[test]
    fn scores_are_in_unit_interval_and_skip_empty_spans() {
        let device = Default::default();
        let config = ScoringHeadConfig::new(FeatureMode::Mfcc39, 8, 4);
        let head = config.init::<B>(&device);
        let features = Tensor::<B, 2>::random([10, 8], Distribution::Default, &device);

        // Three words over 10 frames, plus one position past the end.
        let scores = head.score_words(&features, &config, &[(0, 3), (1, 3), (2, 3), (5, 3)]);
        assert_eq!(scores.len(), 4);
        assert!(scores[..3]
            .iter()
            .all(|s| s.is_some_and(|v| (0.0..=1.0).contains(&v))));
        assert!(scores[3].is_none());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let device = Default::default();
        let config = ScoringHeadConfig::new(FeatureMode::LogMel80, 6, 3);
        let head = config.init::<B>(&device);
        let path = dir.path().join("head");
        config.save(dir.path()).expect("save config");
        checkpoint::save(&head, &path).expect("save head");

        let (loaded, loaded_config) = ScoringHead::<B>::load(&path, &device).expect("load");
        assert_eq!(loaded_config, config);
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5f32; 12], Shape::new([2, 6])),
            &device,
        );
        let a: Vec<f32> = head.forward(x.clone()).into_data().to_vec().expect("vec");
        let b: Vec<f32> = loaded.forward(x).into_data().to_vec().expect("vec");
        assert_eq!(a, b);
    }

    #[test]
    fn missing_config_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(ScoringHead::<B>::load(&dir.path().join("head"), &Default::default()).is_err());
    }
}
