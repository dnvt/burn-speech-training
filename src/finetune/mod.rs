//! Fine-tune `SpeechAligner` to predict word-level pronunciation scores.
//!
//! A [`ScoringHead`] is trained on SpeechOcean762 human scores (see
//! [`crate::data::so762`] for the input format) together with the backbone.
//! The loss is `ctc_weight × CTC + (1 − ctc_weight) × MSE(score)`.
//!
//! The defaults follow the best recipe in `docs/experiment-log.md`: CTC
//! weight 0 (the CTC gradient hurt ranking), 3 warmup epochs then cosine
//! decay, and keeping the checkpoint with the best holdout ρ rather than the
//! last one. Variants that did not help are in [`crate::experiments`]
//! behind the `experiments` feature.

pub mod data;
pub mod head;

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::nn::loss::CTCLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::json;

use crate::backend::{self, InferBackend, TrainBackend};
use crate::checkpoint;
use crate::data::batch::collate;
use crate::data::TrainingSample;
use crate::error::{self, Error, Result};
use crate::metrics::MetricsWriter;
use crate::mfcc::FeatureMode;
use crate::model::{proportional_word_frame_span, SpeechAligner, SpeechAlignerConfig};
use crate::stats::Correlation;
use crate::train::learning_rate_at;

pub use data::{ScoredUtterance, WordTarget};
pub use head::{ScoringHead, ScoringHeadConfig};

pub const FINAL_MODEL: &str = "speech-aligner-ft-final";
pub const FINAL_HEAD: &str = "scoring-head-final";
pub const BEST_MODEL: &str = "speech-aligner-ft-best";
pub const BEST_HEAD: &str = "scoring-head-best";
const HOLDOUT_BOOTSTRAP: usize = 1000;
/// Separate from `train`'s metrics so fine-tuning into a pre-training
/// directory does not overwrite them.
pub const METRICS_FILE: &str = "finetune-metrics.jsonl";

#[derive(Debug, Clone)]
pub struct FineTuneConfig {
    /// Word-score JSON for training (see [`crate::data::so762`]).
    pub dataset: PathBuf,
    /// Directory that `audio_file` paths in the datasets are relative to.
    pub data_root: PathBuf,
    /// Word-score JSON for model selection. When set, the model is evaluated
    /// every `eval_interval` epochs and the best one is saved separately.
    pub holdout: Option<PathBuf>,
    pub eval_interval: usize,
    /// Pre-trained backbone (e.g. from [`crate::train::train`]). `None`
    /// starts from random weights.
    pub base_checkpoint: Option<PathBuf>,
    /// Backbone shape when starting from random weights. `None` uses the
    /// default for `feature_mode`. Ignored with a `base_checkpoint`, whose
    /// saved config is used instead.
    pub model: Option<SpeechAlignerConfig>,
    /// Must match the base checkpoint's training features.
    pub feature_mode: FeatureMode,
    pub epochs: usize,
    pub learning_rate: f64,
    pub warmup_epochs: usize,
    /// Weight of the CTC loss in `[0, 1]`; the score MSE gets the rest.
    pub ctc_weight: f32,
    /// Utterances per batch. All scored words in them are trained together.
    pub batch_size: usize,
    pub head_hidden_dim: usize,
    /// Use only the first N words of `dataset`.
    pub max_samples: Option<usize>,
    pub checkpoint_dir: PathBuf,
    pub seed: u64,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            dataset: PathBuf::new(),
            data_root: PathBuf::new(),
            holdout: None,
            eval_interval: 1,
            base_checkpoint: None,
            model: None,
            feature_mode: FeatureMode::Mfcc39,
            epochs: 20,
            learning_rate: 3e-4,
            warmup_epochs: 3,
            ctc_weight: 0.0,
            batch_size: 8,
            head_hidden_dim: 256,
            max_samples: None,
            checkpoint_dir: PathBuf::from("checkpoints-ft"),
            seed: 42,
        }
    }
}

impl FineTuneConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.ctc_weight) {
            return Err(Error::config("ctc_weight must be within [0, 1]"));
        }
        if self.epochs == 0 || self.batch_size == 0 || self.head_hidden_dim == 0 {
            return Err(Error::config(
                "epochs, batch_size and head_hidden_dim must be > 0",
            ));
        }
        if self.learning_rate.is_nan() || self.learning_rate <= 0.0 {
            return Err(Error::config("learning_rate must be > 0"));
        }
        if self.holdout.is_some() && self.eval_interval == 0 {
            return Err(Error::config(
                "eval_interval must be > 0 when holdout is set",
            ));
        }
        Ok(())
    }

    pub(crate) fn backbone_config(&self) -> Result<SpeechAlignerConfig> {
        let config = match &self.base_checkpoint {
            Some(path) => checkpoint::model_config_for(path, self.feature_mode),
            None => Ok(self
                .model
                .clone()
                .unwrap_or_else(|| SpeechAlignerConfig::for_feature_mode(self.feature_mode))),
        }?;
        checkpoint::check_feature_mode(&config, self.feature_mode)?;
        Ok(config)
    }

    pub(crate) fn init_backbone(
        &self,
        config: &SpeechAlignerConfig,
        device: &backend::Device,
    ) -> Result<SpeechAligner<TrainBackend>> {
        TrainBackend::seed(device, self.seed);
        let model = config.init(device)?;
        match &self.base_checkpoint {
            Some(path) => checkpoint::load(model, path, device),
            None => Ok(model),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochStats {
    pub epoch: usize,
    pub ctc_loss: f32,
    pub score_loss: f32,
    pub total_loss: f32,
    pub words: usize,
    pub seconds: f64,
    /// Holdout result, when evaluated this epoch.
    pub holdout: Option<Correlation>,
}

#[derive(Debug, Clone)]
pub struct FineTuneSummary {
    pub epochs: Vec<EpochStats>,
    pub final_model: PathBuf,
    pub final_head: PathBuf,
    /// Epoch and checkpoints with the best holdout ρ, when a holdout was set.
    pub best: Option<(usize, Correlation, PathBuf, PathBuf)>,
}

/// Pooled word features and their targets for one batch.
pub(crate) struct WordBatch<B: Backend> {
    /// `[words, dim]`
    pub pooled: Tensor<B, 2>,
    pub targets: Vec<f32>,
    pub weights: Vec<f32>,
    /// Used by the experimental losses.
    #[cfg_attr(not(feature = "experiments"), allow(dead_code))]
    pub raw_scores: Vec<i32>,
}

/// Pool each scored word of each utterance. `frame_features(i)` returns the
/// `[frames, dim]` features of utterance `i`, or `None` to skip it.
pub(crate) fn pool_words<B: Backend>(
    head: &ScoringHead<B>,
    utterances: &[&ScoredUtterance],
    frame_features: impl Fn(usize) -> Option<Tensor<B, 2>>,
) -> Option<WordBatch<B>> {
    let mut pooled = Vec::new();
    let (mut targets, mut weights, mut raw_scores) = (Vec::new(), Vec::new(), Vec::new());
    for (i, utterance) in utterances.iter().enumerate() {
        let Some(features) = frame_features(i) else {
            continue;
        };
        let [frames, dim] = features.dims();
        for word in &utterance.words {
            let Some((start, end)) =
                proportional_word_frame_span(frames, word.num_words, word.position)
            else {
                continue;
            };
            pooled.push(head.pool(features.clone().slice([start..end, 0..dim])));
            targets.push(word.score);
            weights.push(word.weight);
            raw_scores.push(word.raw_score);
        }
    }
    (!pooled.is_empty()).then(|| WordBatch {
        pooled: Tensor::cat(pooled, 0),
        targets,
        weights,
        raw_scores,
    })
}

pub(crate) fn tensor_1d<B: Backend>(values: Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, Shape::new([n])), device)
}

/// Correlate head predictions with human scores. `frame_features(i)` gives
/// the `[frames, dim]` input for utterance `i` (or `None` to skip it).
pub(crate) fn correlate<B: Backend>(
    head: &ScoringHead<B>,
    config: &ScoringHeadConfig,
    utterances: &[ScoredUtterance],
    frame_features: impl Fn(usize) -> Option<Tensor<B, 2>>,
) -> Correlation {
    let (mut predicted, mut human, mut groups) = (Vec::new(), Vec::new(), Vec::new());
    for (i, utterance) in utterances.iter().enumerate() {
        let Some(features) = frame_features(i) else {
            continue;
        };
        let words: Vec<(usize, usize)> = utterance
            .words
            .iter()
            .map(|w| (w.position, w.num_words))
            .collect();
        for (word, score) in utterance
            .words
            .iter()
            .zip(head.score_words(&features, config, &words))
        {
            if let Some(score) = score {
                predicted.push(f64::from(score));
                human.push(f64::from(word.score));
                groups.push(i);
            }
        }
    }
    Correlation::compute(&predicted, &human, &groups, HOLDOUT_BOOTSTRAP)
}

/// Backbone frame features for a single unpadded utterance, `[frames, dim]`.
pub(crate) fn backbone_features<B: Backend>(
    model: &SpeechAligner<B>,
    sample: &TrainingSample,
    device: &B::Device,
) -> Tensor<B, 2> {
    let input = Tensor::<B, 3>::from_data(
        TensorData::new(
            sample.features.clone(),
            Shape::new([1, sample.num_frames(), sample.feature_dim]),
        ),
        device,
    );
    model.forward(input).frame_features.squeeze_dim::<2>(0)
}

pub fn evaluate_holdout(
    model: &SpeechAligner<InferBackend>,
    head: &ScoringHead<InferBackend>,
    config: &ScoringHeadConfig,
    holdout: &[ScoredUtterance],
) -> Correlation {
    let device = backend::default_device();
    correlate(head, config, holdout, |i| {
        Some(backbone_features(model, &holdout[i].sample, &device))
    })
}

/// Save the model, head and their configs under the given checkpoint names.
pub(crate) fn save_pair<B: Backend>(
    dir: &Path,
    model: &SpeechAligner<B>,
    head: &ScoringHead<B>,
    model_name: &str,
    head_name: &str,
) -> Result<(PathBuf, PathBuf)> {
    let (model_path, head_path) = (dir.join(model_name), dir.join(head_name));
    checkpoint::save(model, &model_path)?;
    checkpoint::save(head, &head_path)?;
    Ok((model_path, head_path))
}

/// Fine-tune a backbone and scoring head. See the module docs.
pub fn finetune(config: &FineTuneConfig) -> Result<FineTuneSummary> {
    config.validate()?;
    let device = backend::default_device();

    let mut utterances = data::load(
        &config.dataset,
        &config.data_root,
        config.feature_mode,
        config.max_samples,
    )?;
    let holdout = match &config.holdout {
        Some(path) => data::load(path, &config.data_root, config.feature_mode, None)?,
        None => Vec::new(),
    };

    let backbone_config = config.backbone_config()?;
    let mut model = config.init_backbone(&backbone_config, &device)?;
    let head_config = ScoringHeadConfig::new(
        config.feature_mode,
        backbone_config.channels[3],
        config.head_hidden_dim,
    );
    let mut head = head_config.init::<TrainBackend>(&device);

    error::create_dir_all(&config.checkpoint_dir)?;
    checkpoint::save_model_config(&config.checkpoint_dir, &backbone_config)?;
    head_config.save(&config.checkpoint_dir)?;
    let mut metrics = MetricsWriter::create(&config.checkpoint_dir.join(METRICS_FILE))?;

    tracing::info!(
        "fine-tuning on {} utterances ({} words), {} holdout utterances, backend {}",
        utterances.len(),
        utterances.iter().map(|u| u.words.len()).sum::<usize>(),
        holdout.len(),
        backend::BACKEND_NAME
    );

    utterances.sort_by_key(|u| u.sample.num_frames());
    let mut batches: Vec<&[ScoredUtterance]> = utterances.chunks(config.batch_size).collect();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let ctc = CTCLossConfig::new().with_zero_infinity(true).init();
    let mut model_optimizer = AdamConfig::new().init::<TrainBackend, SpeechAligner<TrainBackend>>();
    let mut head_optimizer = AdamConfig::new().init::<TrainBackend, ScoringHead<TrainBackend>>();

    let mut epochs = Vec::with_capacity(config.epochs);
    let mut best: Option<(usize, Correlation, PathBuf, PathBuf)> = None;

    for epoch_idx in 0..config.epochs {
        let epoch = epoch_idx + 1;
        let lr = learning_rate_at(
            config.learning_rate,
            epoch_idx,
            config.epochs,
            config.warmup_epochs,
        );
        let started = Instant::now();
        batches.shuffle(&mut rng);
        let (mut ctc_sum, mut score_sum, mut total_sum, mut steps, mut words) =
            (0.0, 0.0, 0.0, 0, 0);

        for batch_utts in &batches {
            let samples: Vec<TrainingSample> =
                batch_utts.iter().map(|u| u.sample.clone()).collect();
            let batch = collate::<TrainBackend>(&samples, &device);
            let output = model.forward_with_pad_mask(batch.inputs, batch.pad_mask);

            let ctc_loss = if config.ctc_weight > 0.0 {
                ctc.forward(
                    output.ctc_log_probs,
                    batch.targets,
                    batch.input_lengths,
                    batch.target_lengths,
                )
                .mean()
            } else {
                Tensor::zeros([1], &device)
            };

            let refs: Vec<&ScoredUtterance> = batch_utts.iter().collect();
            let dim = backbone_config.channels[3];
            let Some(word_batch) = pool_words(&head, &refs, |i| {
                let frames = refs[i].sample.num_frames();
                Some(
                    output
                        .frame_features
                        .clone()
                        .slice([i..i + 1, 0..frames, 0..dim])
                        .reshape([frames, dim]),
                )
            }) else {
                continue;
            };

            let predicted = head::predict(head.forward(word_batch.pooled), head_config.output);
            let error = predicted - tensor_1d(word_batch.targets.clone(), &device);
            let score_loss =
                (error.clone() * error * tensor_1d(word_batch.weights, &device)).mean();
            let total = ctc_loss.clone() * config.ctc_weight
                + score_loss.clone() * (1.0 - config.ctc_weight);

            let total_value = head::scalar(total.clone());
            if !total_value.is_finite() {
                tracing::warn!("epoch {epoch}: non-finite loss, skipping batch");
                continue;
            }
            let mut grads = total.backward();
            let model_grads = GradientsParams::from_module(&mut grads, &model);
            model = model_optimizer.step(lr, model, model_grads);
            let head_grads = GradientsParams::from_module(&mut grads, &head);
            head = head_optimizer.step(lr, head, head_grads);

            ctc_sum += f64::from(head::scalar(ctc_loss));
            score_sum += f64::from(head::scalar(score_loss));
            total_sum += f64::from(total_value);
            steps += 1;
            words += word_batch.targets.len();
        }

        if steps == 0 {
            return Err(Error::training(format!(
                "epoch {epoch}: every batch was skipped"
            )));
        }
        let mean = |sum: f64| (sum / f64::from(steps)) as f32;
        let mut stats = EpochStats {
            epoch,
            ctc_loss: mean(ctc_sum),
            score_loss: mean(score_sum),
            total_loss: mean(total_sum),
            words,
            seconds: started.elapsed().as_secs_f64(),
            holdout: None,
        };

        if !holdout.is_empty() && epoch % config.eval_interval == 0 {
            let (infer_model, infer_head) = (model.valid(), head.valid());
            let result = evaluate_holdout(&infer_model, &infer_head, &head_config, &holdout);
            stats.holdout = Some(result);
            if best
                .as_ref()
                .is_none_or(|(_, b, _, _)| result.spearman > b.spearman)
            {
                let (m, h) = save_pair(
                    &config.checkpoint_dir,
                    &infer_model,
                    &infer_head,
                    BEST_MODEL,
                    BEST_HEAD,
                )?;
                best = Some((epoch, result, m, h));
            }
        }

        log_epoch(&mut metrics, &stats, lr);
        epochs.push(stats);
    }

    let (final_model, final_head) = save_pair(
        &config.checkpoint_dir,
        &model.valid(),
        &head.valid(),
        FINAL_MODEL,
        FINAL_HEAD,
    )?;
    if let Some((epoch, result, _, _)) = &best {
        tracing::info!("best holdout: epoch {epoch}, {result}");
    }
    Ok(FineTuneSummary {
        epochs,
        final_model,
        final_head,
        best,
    })
}

pub(crate) fn log_epoch(metrics: &mut MetricsWriter, stats: &EpochStats, lr: f64) {
    let mut fields = json!({
        "epoch": stats.epoch,
        "loss": stats.total_loss,
        "ctc_loss": stats.ctc_loss,
        "score_loss": stats.score_loss,
        "words": stats.words,
        "lr": lr,
        "epoch_secs": stats.seconds,
    });
    if let Some(h) = &stats.holdout {
        fields["holdout_spearman"] = json!(h.spearman);
        fields["holdout_pearson"] = json!(h.pearson);
        fields["holdout_ci95"] = json!([h.spearman_ci95.0, h.spearman_ci95.1]);
    }
    metrics.log("epoch", fields);
    match &stats.holdout {
        Some(h) => tracing::info!(
            "epoch {}: loss {:.4} (ctc {:.4}, score {:.5}), holdout {h}",
            stats.epoch,
            stats.total_loss,
            stats.ctc_loss,
            stats.score_loss
        ),
        None => tracing::info!(
            "epoch {}: loss {:.4} (ctc {:.4}, score {:.5})",
            stats.epoch,
            stats.total_loss,
            stats.ctc_loss,
            stats.score_loss
        ),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use super::*;
    use crate::data::audio::{test_tone, write_test_wav};

    /// Write a tiny SpeechOcean762-style word file with synthetic audio.
    /// Returns the JSON path; audio is relative to `root`.
    pub(crate) fn fake_so762(root: &Path, name: &str, utterances: usize) -> PathBuf {
        let mut samples = Vec::new();
        for u in 0..utterances {
            let audio = format!("{name}-{u}.wav");
            write_test_wav(&root.join(&audio), &test_tone(1.0 + u as f32 * 0.2), 16_000);
            let phonemes = [vec!["DH", "AH"], vec!["K", "AE", "T"], vec!["S", "AE", "T"]];
            for (position, score) in [(0usize, 0.9f64), (1, 0.4), (2, 1.0)] {
                let score = (score - u as f64 * 0.1).clamp(0.0, 1.0);
                samples.push(json!({
                    "audio_file": audio,
                    "sentence_text": "THE CAT SAT",
                    "word_position": position,
                    "human_score": score,
                    "phonemes": phonemes[position],
                }));
            }
        }
        let path = root.join(format!("{name}.json"));
        std::fs::write(&path, json!({ "samples": samples }).to_string()).expect("write");
        path
    }

    /// Narrow backbone for MFCC input so tests run quickly.
    pub(crate) fn tiny_backbone() -> SpeechAlignerConfig {
        SpeechAlignerConfig {
            channels: [8, 8, 16, 16],
            n_heads: 2,
            ..SpeechAlignerConfig::default()
        }
    }

    #[test]
    fn rejects_bad_config() {
        let bad = |config: FineTuneConfig| config.validate().is_err();
        assert!(bad(FineTuneConfig {
            ctc_weight: 1.5,
            ..Default::default()
        }));
        assert!(bad(FineTuneConfig {
            epochs: 0,
            ..Default::default()
        }));
        assert!(bad(FineTuneConfig {
            holdout: Some("h.json".into()),
            eval_interval: 0,
            ..Default::default()
        }));
        assert!(FineTuneConfig::default().validate().is_ok());
    }

    #[test]
    fn finetune_end_to_end_on_synthetic_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let train = fake_so762(dir.path(), "train", 4);
        let holdout = fake_so762(dir.path(), "holdout", 4);
        let config = FineTuneConfig {
            dataset: train,
            data_root: dir.path().to_path_buf(),
            holdout: Some(holdout),
            epochs: 2,
            warmup_epochs: 1,
            batch_size: 2,
            head_hidden_dim: 8,
            model: Some(tiny_backbone()),
            checkpoint_dir: dir.path().join("ckpt"),
            ..Default::default()
        };

        let summary = finetune(&config).expect("finetune");
        assert_eq!(summary.epochs.len(), 2);
        assert!(summary
            .epochs
            .iter()
            .all(|e| e.total_loss.is_finite() && e.words == 12));
        assert!(summary.epochs.iter().all(|e| e.holdout.is_some()));
        let (_, _, best_model, best_head) = summary.best.expect("best");
        for path in [
            &summary.final_model,
            &summary.final_head,
            &best_model,
            &best_head,
        ] {
            assert!(path.exists(), "{} missing", path.display());
        }

        // The saved head reloads with its config.
        let (_, head_config) =
            ScoringHead::<InferBackend>::load(&summary.final_head, &backend::default_device())
                .expect("load");
        assert_eq!(head_config.input_dim, 16);
    }
}
