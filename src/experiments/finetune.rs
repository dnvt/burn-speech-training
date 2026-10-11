//! Fine-tuning with the variants from rounds 3–6 of the experiment log.

use std::collections::BTreeMap;
use std::time::Instant;

use burn::module::Module;
use burn::nn::loss::CTCLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use super::losses::{self, ordinal_bin};
use super::teacher::{self, TeacherCache, TeacherFeatures, TEACHER_DIM};
use crate::checkpoint;
use crate::data::batch::collate;
use crate::data::TrainingSample;
use crate::error::{self, Error, Result};
use crate::finetune::head::{self, InputSource, Pooling, ScoreOutput};
use crate::finetune::{
    self as core, backbone_features, correlate, log_epoch, pool_words, save_pair, tensor_1d,
    EpochStats, FineTuneConfig, FineTuneSummary, ScoredUtterance, ScoringHeadConfig, WordBatch,
};
use crate::metrics::MetricsWriter;
use crate::stats::Correlation;
use crate::train::learning_rate_at;

/// Loss on the predicted word scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreLoss {
    /// Weighted mean squared error (the baseline).
    Mse,
    /// MSE weighted by `|error|^gamma` (runs 4f, 4i–4m).
    Focal { gamma: f32 },
    /// Balanced MSE with Gaussian noise `noise_sigma`.
    BalancedMse { noise_sigma: f32 },
    /// Cumulative-threshold ordinal head over five score bins.
    Ordinal,
    /// Five-way softmax over score bins (runs 6e, 6f).
    OrdinalSoftmax,
}

impl ScoreLoss {
    fn output(self) -> ScoreOutput {
        match self {
            Self::Ordinal => ScoreOutput::Ordinal,
            Self::OrdinalSoftmax => ScoreOutput::OrdinalSoftmax,
            _ => ScoreOutput::Sigmoid,
        }
    }

    fn is_ordinal(self) -> bool {
        matches!(self, Self::Ordinal | Self::OrdinalSoftmax)
    }
}

/// Per-word loss weights from how common each score is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WordWeighting {
    None,
    /// `max_count / count`, capped at 100 (run 4g).
    InverseFrequency,
    /// Effective-number weights with parameter `beta` (e.g. 0.999).
    EffectiveNumber {
        beta: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedTerm {
    pub weight: f32,
    pub margin: f32,
}

#[derive(Debug, Clone)]
pub enum TeacherMode {
    /// Add `weight × MSE(student, projected teacher)` to the loss.
    Distill { weight: f32 },
    /// Feed teacher features to the scoring head instead of the backbone's.
    /// The backbone is not used, so CTC is off.
    Input,
}

#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub base: FineTuneConfig,
    pub score_loss: ScoreLoss,
    pub weighting: WordWeighting,
    /// Repeat words with raw score below `below` this many times in total.
    pub oversample: Option<(usize, i32)>,
    /// Mix pooled embeddings of words with raw score ≤ `below` with random
    /// partners; `alpha` in `(0, 1]` sets how far from 0.5 the mixing
    /// coefficient can be.
    pub mixup: Option<(f32, i32)>,
    pub pooling: Pooling,
    pub freeze_backbone: bool,
    /// `(weight, tau)` for [`losses::rank_regularization`].
    pub rank_regularization: Option<(f32, f32)>,
    pub pairwise_ranking: Option<WeightedTerm>,
    pub triplet_ranking: Option<WeightedTerm>,
    /// `(cache_dir, mode)`, see [`teacher`].
    pub teacher: Option<(std::path::PathBuf, TeacherMode)>,
}

impl ExperimentConfig {
    /// The baseline recipe: no experimental options enabled.
    pub fn new(base: FineTuneConfig) -> Self {
        Self {
            base,
            score_loss: ScoreLoss::Mse,
            weighting: WordWeighting::None,
            oversample: None,
            mixup: None,
            pooling: Pooling::Mean,
            freeze_backbone: false,
            rank_regularization: None,
            pairwise_ranking: None,
            triplet_ranking: None,
            teacher: None,
        }
    }

    fn teacher_input(&self) -> bool {
        matches!(self.teacher, Some((_, TeacherMode::Input)))
    }

    fn validate(&self) -> Result<()> {
        self.base.validate()?;
        let positive = |name: &str, v: f32| {
            if v > 0.0 {
                Ok(())
            } else {
                Err(Error::config(format!("{name} must be > 0")))
            }
        };
        match self.score_loss {
            ScoreLoss::Focal { gamma } => positive("focal gamma", gamma)?,
            ScoreLoss::BalancedMse { noise_sigma } => {
                positive("balanced MSE noise_sigma", noise_sigma)?;
                if self.weighting != WordWeighting::None {
                    return Err(Error::config("balanced MSE does not use word weights"));
                }
            }
            _ => {}
        }
        if let WordWeighting::EffectiveNumber { beta } = self.weighting {
            if !(0.0..1.0).contains(&beta) {
                return Err(Error::config("effective-number beta must be in [0, 1)"));
            }
        }
        if let Some((alpha, _)) = self.mixup {
            positive("mixup alpha", alpha)?;
            if self.score_loss.is_ordinal()
                || matches!(self.score_loss, ScoreLoss::BalancedMse { .. })
            {
                return Err(Error::config(
                    "mixup needs a regression loss (Mse or Focal)",
                ));
            }
        }
        if let Some((weight, tau)) = self.rank_regularization {
            positive("rank regularization weight", weight)?;
            positive("rank regularization tau", tau)?;
        }
        for term in [self.pairwise_ranking, self.triplet_ranking]
            .into_iter()
            .flatten()
        {
            positive("ranking weight", term.weight)?;
            positive("ranking margin", term.margin)?;
        }
        if let Some((_, TeacherMode::Distill { weight })) = &self.teacher {
            positive("distillation weight", *weight)?;
        }
        Ok(())
    }
}

/// Set word weights and oversample low scores, as configured.
fn reweight(utterances: &mut [ScoredUtterance], config: &ExperimentConfig) {
    let bucket = |raw: i32| {
        if config.score_loss.is_ordinal() {
            ordinal_bin(raw) as i32
        } else {
            raw
        }
    };
    let mut counts = BTreeMap::new();
    for word in utterances.iter().flat_map(|u| &u.words) {
        *counts.entry(bucket(word.raw_score)).or_insert(0usize) += 1;
    }
    let weights = match config.weighting {
        WordWeighting::None => None,
        WordWeighting::InverseFrequency => Some(losses::inverse_frequency_weights(&counts)),
        WordWeighting::EffectiveNumber { beta } => {
            Some(losses::effective_number_weights(&counts, beta))
        }
    };

    for utterance in utterances {
        if let Some(weights) = &weights {
            for word in &mut utterance.words {
                word.weight = weights[&bucket(word.raw_score)];
            }
        }
        if let Some((factor, below)) = config.oversample {
            let extra: Vec<_> = utterance
                .words
                .iter()
                .filter(|w| w.raw_score < below)
                .flat_map(|w| std::iter::repeat_n(*w, factor.saturating_sub(1)))
                .collect();
            utterance.words.extend(extra);
            utterance.words.sort_by_key(|w| w.position);
        }
    }
}

/// Append mixed copies of low-scored words to the batch.
fn mixup(batch: &mut WordBatch, alpha: f32, below: i32, rng: &mut StdRng) {
    let n = batch.targets.len();
    if n < 2 {
        return;
    }
    let radius = alpha.clamp(0.0, 1.0) * 0.5;
    let mut mixed = Vec::new();
    let anchors: Vec<usize> = (0..n).filter(|&i| batch.raw_scores[i] <= below).collect();
    for anchor in anchors {
        let partner = (anchor + rng.random_range(1..n)) % n;
        let lam = rng.random_range((0.5 - radius)..=(0.5 + radius));
        let row = |i: usize| batch.pooled.clone().narrow(0, i, 1);
        mixed.push(row(anchor) * lam + row(partner) * (1.0 - lam));
        batch
            .targets
            .push(batch.targets[anchor] * lam + batch.targets[partner] * (1.0 - lam));
        batch
            .weights
            .push((batch.weights[anchor] + batch.weights[partner]) * 0.5);
        let raw =
            batch.raw_scores[anchor] as f32 * lam + batch.raw_scores[partner] as f32 * (1.0 - lam);
        batch.raw_scores.push(raw.round().clamp(0.0, 10.0) as i32);
    }
    if !mixed.is_empty() {
        mixed.insert(0, batch.pooled.clone());
        batch.pooled = Tensor::cat(mixed, 0);
    }
}

/// Fine-tune with experimental options. With [`ExperimentConfig::new`] this
/// matches [`crate::finetune::finetune`].
pub fn finetune(config: &ExperimentConfig) -> Result<FineTuneSummary> {
    config.validate()?;
    let base = &config.base;
    let device = Device::default().autodiff();
    let teacher_input = config.teacher_input();
    let ctc_weight = if teacher_input { 0.0 } else { base.ctc_weight };
    let freeze_backbone = teacher_input || config.freeze_backbone;

    let mut utterances = core::data::load(
        &base.dataset,
        &base.data_root,
        base.feature_mode,
        base.max_samples,
    )?;
    reweight(&mut utterances, config);
    let holdout = match &base.holdout {
        Some(path) => core::data::load(path, &base.data_root, base.feature_mode, None)?,
        None => Vec::new(),
    };

    let cache = config
        .teacher
        .as_ref()
        .map(|(dir, _)| TeacherCache::open(dir))
        .transpose()?;
    let mut items: Vec<(ScoredUtterance, Option<TeacherFeatures>)> =
        utterances.into_iter().map(|u| (u, None)).collect();
    let mut holdout_teacher = vec![None; holdout.len()];
    if let Some(cache) = &cache {
        for (utterance, features) in &mut items {
            *features = cache.get(&utterance.audio_file)?;
        }
        let matched = items.iter().filter(|(_, f)| f.is_some()).count();
        tracing::info!("teacher cache matched {matched}/{} utterances", items.len());
        if teacher_input {
            items.retain(|(_, f)| f.is_some());
            if items.is_empty() {
                return Err(Error::data("no training utterance has teacher features"));
            }
            holdout_teacher = cache.for_utterances(&holdout)?;
        }
    }

    let backbone_config = base.backbone_config()?;
    let mut model = base.init_backbone(&backbone_config, &device)?;
    let dim = backbone_config.channels[3];
    let head_config = ScoringHeadConfig {
        output: config.score_loss.output(),
        pooling: config.pooling,
        input_source: if teacher_input {
            InputSource::TeacherCache
        } else {
            InputSource::Backbone
        },
        ..ScoringHeadConfig::new(
            base.feature_mode,
            if teacher_input { TEACHER_DIM } else { dim },
            base.head_hidden_dim,
        )
    };
    let mut head = head_config.init(&device);
    let mut projection: Option<Linear> = match &config.teacher {
        Some((_, TeacherMode::Distill { .. })) => {
            Some(LinearConfig::new(TEACHER_DIM, dim).init(&device))
        }
        _ => None,
    };

    error::create_dir_all(&base.checkpoint_dir)?;
    checkpoint::save_model_config(&base.checkpoint_dir, &backbone_config)?;
    head_config.save(&base.checkpoint_dir)?;
    let mut metrics = MetricsWriter::create(&base.checkpoint_dir.join(core::METRICS_FILE))?;

    items.sort_by_key(|(u, _)| u.sample.num_frames());
    let mut batches: Vec<&[(ScoredUtterance, Option<TeacherFeatures>)]> =
        items.chunks(base.batch_size).collect();
    let mut rng = StdRng::seed_from_u64(base.seed);
    let ctc = CTCLossConfig::new().with_zero_infinity(true).init();
    let mut model_optimizer = AdamConfig::new().init();
    let mut head_optimizer = AdamConfig::new().init();
    let mut projection_optimizer = AdamConfig::new().init();

    let mut epochs = Vec::with_capacity(base.epochs);
    let mut best: Option<(usize, Correlation, std::path::PathBuf, std::path::PathBuf)> = None;

    for epoch_idx in 0..base.epochs {
        let epoch = epoch_idx + 1;
        let lr = learning_rate_at(
            base.learning_rate,
            epoch_idx,
            base.epochs,
            base.warmup_epochs,
        );
        let started = Instant::now();
        batches.shuffle(&mut rng);
        let (mut ctc_sum, mut score_sum, mut total_sum, mut steps, mut words) =
            (0.0, 0.0, 0.0, 0, 0);

        for batch in &batches {
            let refs: Vec<&ScoredUtterance> = batch.iter().map(|(u, _)| u).collect();
            let output = (!teacher_input).then(|| {
                let samples: Vec<TrainingSample> = refs.iter().map(|u| u.sample.clone()).collect();
                let b = collate(&samples, &device);
                let output = model.forward_with_pad_mask(b.inputs, b.pad_mask);
                (output, b.targets, b.input_lengths, b.target_lengths)
            });
            let student = |i: usize| {
                output.as_ref().map(|(o, ..)| {
                    let frames = refs[i].sample.num_frames();
                    o.frame_features
                        .clone()
                        .slice([i..i + 1, 0..frames, 0..dim])
                        .reshape([frames, dim])
                })
            };

            let ctc_loss = match &output {
                Some((o, targets, input_lengths, target_lengths)) if ctc_weight > 0.0 => ctc
                    .forward(
                        o.ctc_log_probs.clone(),
                        targets.clone(),
                        input_lengths.clone(),
                        target_lengths.clone(),
                    )
                    .mean(),
                _ => Tensor::zeros([1], &device),
            };

            let Some(mut wb) = pool_words(&head, &refs, |i| {
                if teacher_input {
                    batch[i].1.as_ref().map(|f| f.tensor(&device))
                } else {
                    student(i)
                }
            }) else {
                continue;
            };
            if let Some((alpha, below)) = config.mixup {
                mixup(&mut wb, alpha, below, &mut rng);
            }

            let targets = tensor_1d(wb.targets.clone(), &device);
            let weights = tensor_1d(wb.weights.clone(), &device);
            let logits = head.forward(wb.pooled.clone());
            let predicted = head::predict(logits.clone(), head_config.output);
            let score_loss = match config.score_loss {
                ScoreLoss::Mse => {
                    let e = predicted.clone() - targets.clone();
                    (e.clone() * e * weights).mean()
                }
                ScoreLoss::Focal { gamma } => {
                    losses::focal_mse(predicted.clone(), targets.clone(), weights, gamma)
                }
                ScoreLoss::BalancedMse { noise_sigma } => losses::balanced_mse(
                    predicted.clone(),
                    targets.clone(),
                    &wb.raw_scores,
                    noise_sigma,
                ),
                ScoreLoss::Ordinal => {
                    losses::ordinal_threshold_loss(logits, &wb.raw_scores, weights)
                }
                ScoreLoss::OrdinalSoftmax => {
                    losses::ordinal_softmax_loss(logits, &wb.raw_scores, weights)
                }
            };

            let mut total = ctc_loss.clone() * ctc_weight + score_loss.clone() * (1.0 - ctc_weight);
            if let Some((weight, tau)) = config.rank_regularization {
                total = total
                    + losses::rank_regularization(wb.pooled.clone(), targets.clone(), tau) * weight;
            }
            if let Some(t) = config.pairwise_ranking {
                total = total
                    + losses::pairwise_ranking(predicted.clone(), &wb.raw_scores, t.margin)
                        * t.weight;
            }
            if let Some(t) = config.triplet_ranking {
                total = total
                    + losses::triplet_ranking(wb.pooled.clone(), &wb.raw_scores, t.margin)
                        * t.weight;
            }
            if let (Some(proj), Some((_, TeacherMode::Distill { weight }))) =
                (&projection, &config.teacher)
            {
                let parts: Vec<Tensor<1>> = batch
                    .iter()
                    .enumerate()
                    .filter_map(|(i, (_, f))| {
                        teacher::distillation_loss(student(i)?, f.as_ref()?, proj)
                    })
                    .collect();
                if !parts.is_empty() {
                    total = total + Tensor::cat(parts, 0).mean() * *weight;
                }
            }

            let total_value = head::scalar(total.clone());
            if !total_value.is_finite() {
                tracing::warn!("epoch {epoch}: non-finite loss, skipping batch");
                continue;
            }
            let mut grads = total.backward();
            if !freeze_backbone {
                let g = GradientsParams::from_module(&mut grads, &model);
                model = model_optimizer.step(lr, model, g);
            }
            let g = GradientsParams::from_module(&mut grads, &head);
            head = head_optimizer.step(lr, head, g);
            if let Some(proj) = projection.take() {
                let g = GradientsParams::from_module(&mut grads, &proj);
                projection = Some(projection_optimizer.step(lr, proj, g));
            }

            ctc_sum += f64::from(head::scalar(ctc_loss));
            score_sum += f64::from(head::scalar(score_loss));
            total_sum += f64::from(total_value);
            steps += 1;
            words += wb.targets.len();
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

        if !holdout.is_empty() && epoch % base.eval_interval == 0 {
            let (infer_model, infer_head) = (model.valid(), head.valid());
            let result = correlate(&infer_head, &head_config, &holdout, |i| {
                if teacher_input {
                    holdout_teacher[i].as_ref().map(|f| f.tensor(&device))
                } else {
                    Some(backbone_features(&infer_model, &holdout[i].sample, &device))
                }
            });
            stats.holdout = Some(result);
            if best
                .as_ref()
                .is_none_or(|(_, b, _, _)| result.spearman > b.spearman)
            {
                let (m, h) = save_pair(
                    &base.checkpoint_dir,
                    &infer_model,
                    &infer_head,
                    core::BEST_MODEL,
                    core::BEST_HEAD,
                )?;
                best = Some((epoch, result, m, h));
            }
        }
        log_epoch(&mut metrics, &stats, lr);
        epochs.push(stats);
    }

    let (final_model, final_head) = save_pair(
        &base.checkpoint_dir,
        &model.valid(),
        &head.valid(),
        core::FINAL_MODEL,
        core::FINAL_HEAD,
    )?;
    Ok(FineTuneSummary {
        epochs,
        final_model,
        final_head,
        best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finetune::tests::fake_so762;

    fn base(dir: &std::path::Path) -> FineTuneConfig {
        FineTuneConfig {
            dataset: fake_so762(dir, "train", 3),
            data_root: dir.to_path_buf(),
            holdout: Some(fake_so762(dir, "holdout", 3)),
            epochs: 1,
            warmup_epochs: 0,
            batch_size: 3,
            head_hidden_dim: 8,
            model: Some(crate::finetune::tests::tiny_backbone()),
            checkpoint_dir: dir.join("ckpt"),
            ..FineTuneConfig::default()
        }
    }

    #[test]
    fn rejects_incompatible_options() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ExperimentConfig::new(base(dir.path()));
        config.score_loss = ScoreLoss::Ordinal;
        config.mixup = Some((0.5, 7));
        assert!(config.validate().is_err());

        let mut config = ExperimentConfig::new(base(dir.path()));
        config.score_loss = ScoreLoss::BalancedMse { noise_sigma: 0.2 };
        config.weighting = WordWeighting::InverseFrequency;
        assert!(config.validate().is_err());
    }

    #[test]
    fn runs_with_every_option_enabled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("teacher");
        std::fs::create_dir_all(&cache_dir).expect("mkdir");
        teacher::tests::write_cache(
            &cache_dir,
            &[
                ("train-0.wav", 50),
                ("train-1.wav", 60),
                ("train-2.wav", 70),
            ],
        );

        let config = ExperimentConfig {
            score_loss: ScoreLoss::Focal { gamma: 2.0 },
            weighting: WordWeighting::EffectiveNumber { beta: 0.999 },
            oversample: Some((2, 8)),
            mixup: Some((0.5, 7)),
            pooling: Pooling::Attention,
            rank_regularization: Some((0.1, 0.15)),
            pairwise_ranking: Some(WeightedTerm {
                weight: 0.1,
                margin: 0.05,
            }),
            triplet_ranking: Some(WeightedTerm {
                weight: 0.1,
                margin: 0.1,
            }),
            teacher: Some((cache_dir, TeacherMode::Distill { weight: 0.1 })),
            ..ExperimentConfig::new(base(dir.path()))
        };
        let summary = finetune(&config).expect("finetune");
        assert!(summary.epochs[0].total_loss.is_finite());
        assert!(summary.best.is_some());
    }

    #[test]
    fn ordinal_softmax_head_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ExperimentConfig {
            score_loss: ScoreLoss::OrdinalSoftmax,
            ..ExperimentConfig::new(base(dir.path()))
        };
        let summary = finetune(&config).expect("finetune");
        let (_, head_config) =
            crate::finetune::ScoringHead::load(&summary.final_head, &Device::default())
                .expect("load");
        assert_eq!(head_config.output, ScoreOutput::OrdinalSoftmax);
    }
}
