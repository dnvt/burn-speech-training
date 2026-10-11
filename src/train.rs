//! CTC training for `SpeechAligner`.
//!
//! [`train`] takes any list of [`TrainingSample`]s, so the same loop serves
//! LibriSpeech ([`crate::data::librispeech::load`]), the precomputed cache
//! ([`crate::data::cache::load`]) or your own data.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CTCLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::ElementConversion;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{json, Value};

use crate::backend::{self, TrainBackend};
use crate::checkpoint;
use crate::data::batch::{collate, plan_batches};
use crate::data::{Batching, TrainingSample};
use crate::error::{self, Error, Result};
use crate::metrics::MetricsWriter;
use crate::model::{SpeechAligner, SpeechAlignerConfig};

const BATCH_LOG_INTERVAL: usize = 50;
pub const METRICS_FILE: &str = "training-metrics.jsonl";
const EPOCH_CHECKPOINT_PREFIX: &str = "speech-aligner-epoch-";
pub const FINAL_CHECKPOINT: &str = "speech-aligner-final";

#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Must have `input_dim` equal to the samples' feature dimension.
    pub model: SpeechAlignerConfig,
    pub epochs: usize,
    pub learning_rate: f64,
    /// Linear warmup over this many epochs, then cosine decay to zero.
    /// `0` keeps the learning rate constant.
    pub warmup_epochs: usize,
    pub batching: Batching,
    pub checkpoint_dir: PathBuf,
    /// Save `speech-aligner-epoch-N` every this many epochs; `0` saves only
    /// the final checkpoint.
    pub checkpoint_interval: usize,
    /// Continue from a checkpoint. Epoch numbering, the learning-rate
    /// schedule and the metrics file continue from where that run stopped;
    /// `epochs` is the number of *additional* epochs. Optimizer state is not
    /// saved, so Adam's moment estimates restart.
    pub resume_from: Option<PathBuf>,
    /// Seeds weight initialisation and batch order.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model: SpeechAlignerConfig::default(),
            epochs: 10,
            learning_rate: 3e-4,
            warmup_epochs: 0,
            batching: Batching::default(),
            checkpoint_dir: PathBuf::from("checkpoints"),
            checkpoint_interval: 5,
            resume_from: None,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainSummary {
    /// Mean CTC loss per epoch (sample-weighted), for the epochs run in this
    /// call. `NaN` for an epoch where every batch was skipped.
    pub epoch_losses: Vec<f32>,
    pub final_checkpoint: PathBuf,
    pub num_params: usize,
}

/// Learning rate for a 0-based epoch: linear warmup, then cosine decay.
pub(crate) fn learning_rate_at(base: f64, epoch: usize, total: usize, warmup: usize) -> f64 {
    if warmup == 0 {
        return base;
    }
    if epoch < warmup {
        return base * (epoch + 1) as f64 / warmup as f64;
    }
    let progress = (epoch - warmup) as f64 / (total.saturating_sub(warmup)).max(1) as f64;
    base * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Train a `SpeechAligner` with CTC loss and save checkpoints to
/// `config.checkpoint_dir`.
pub fn train(mut samples: Vec<TrainingSample>, config: &TrainConfig) -> Result<TrainSummary> {
    validate(&samples, config)?;
    let device = backend::default_device();
    error::create_dir_all(&config.checkpoint_dir)?;
    checkpoint::save_model_config(&config.checkpoint_dir, &config.model)?;

    TrainBackend::seed(&device, config.seed);
    let mut model: SpeechAligner<TrainBackend> = config.model.init(&device)?;
    if let Some(path) = &config.resume_from {
        model = checkpoint::load(model, path, &device)?;
        tracing::info!("resumed from {}", path.display());
    }
    let num_params = model.num_params();

    let resume = collect_resume_state(&config.checkpoint_dir, config.resume_from.as_deref())?;
    let mut metrics = if resume.completed_epochs > 0 {
        MetricsWriter::append(
            &config.checkpoint_dir.join(METRICS_FILE),
            resume.elapsed_offset_secs,
        )?
    } else {
        MetricsWriter::create(&config.checkpoint_dir.join(METRICS_FILE))?
    };

    let plan = plan_batches(&mut samples, config.batching, config.model.n_heads);
    tracing::info!(
        "training {num_params} params on {} samples in {} batches, {} epochs, backend {}",
        samples.len(),
        plan.len(),
        config.epochs,
        backend::BACKEND_NAME
    );

    let ctc = CTCLossConfig::new().with_zero_infinity(true).init();
    let mut optimizer = AdamConfig::new().init::<TrainBackend, SpeechAligner<TrainBackend>>();
    // Offset by completed epochs so a resumed run doesn't replay the batch
    // order of the first run.
    let mut rng = StdRng::seed_from_u64(config.seed + resume.completed_epochs as u64);
    let total_epochs = resume.completed_epochs + config.epochs;
    let mut order: Vec<usize> = (0..plan.len()).collect();
    let mut epoch_losses = Vec::with_capacity(config.epochs);

    for epoch_idx in 0..config.epochs {
        let epoch = resume.completed_epochs + epoch_idx + 1;
        let lr = learning_rate_at(
            config.learning_rate,
            epoch - 1,
            total_epochs,
            config.warmup_epochs,
        );
        let epoch_start = Instant::now();
        order.shuffle(&mut rng);

        let (mut loss_sum, mut samples_seen, mut skipped) = (0.0f64, 0usize, 0usize);
        for (step, &batch_idx) in order.iter().enumerate() {
            let batch_samples = &samples[plan[batch_idx].clone()];
            let batch = collate::<TrainBackend>(batch_samples, &device);
            let output = model.forward_with_pad_mask(batch.inputs, batch.pad_mask);
            let loss = ctc
                .forward(
                    output.ctc_log_probs,
                    batch.targets,
                    batch.input_lengths,
                    batch.target_lengths,
                )
                .mean();
            let loss_value: f32 = loss.clone().into_scalar().elem();

            if loss_value.is_finite() {
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(lr, model, grads);
                loss_sum += f64::from(loss_value) * batch_samples.len() as f64;
                samples_seen += batch_samples.len();
            } else {
                skipped += 1;
                tracing::warn!(
                    "epoch {epoch} batch {}: loss is {loss_value}, skipping",
                    step + 1
                );
            }

            if (step + 1) % BATCH_LOG_INTERVAL == 0 {
                metrics.log(
                    "batch",
                    json!({ "epoch": epoch, "batch": step + 1, "batches": plan.len(), "loss": loss_value, "lr": lr }),
                );
            }
        }

        let loss = if samples_seen > 0 {
            (loss_sum / samples_seen as f64) as f32
        } else {
            f32::NAN
        };
        epoch_losses.push(loss);
        metrics.log(
            "epoch",
            json!({
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
                "skipped_batches": skipped,
                "epoch_secs": epoch_start.elapsed().as_secs_f64(),
            }),
        );
        tracing::info!(
            "epoch {epoch}: loss {loss:.4}, lr {lr:.2e}, {:.1}s",
            epoch_start.elapsed().as_secs_f64()
        );

        if config.checkpoint_interval > 0 && epoch % config.checkpoint_interval == 0 {
            let path = config
                .checkpoint_dir
                .join(format!("{EPOCH_CHECKPOINT_PREFIX}{epoch}"));
            checkpoint::save(&model, &path)?;
        }
    }

    if epoch_losses.iter().all(|loss| loss.is_nan()) {
        return Err(Error::training("every batch produced a non-finite loss"));
    }

    let final_checkpoint = config.checkpoint_dir.join(FINAL_CHECKPOINT);
    checkpoint::save(&model.valid(), &final_checkpoint)?;
    metrics.log(
        "complete",
        json!({ "epochs": resume.completed_epochs + config.epochs }),
    );
    tracing::info!("saved {}", final_checkpoint.display());

    Ok(TrainSummary {
        epoch_losses,
        final_checkpoint,
        num_params,
    })
}

fn validate(samples: &[TrainingSample], config: &TrainConfig) -> Result<()> {
    if samples.is_empty() {
        return Err(Error::config("no training samples"));
    }
    if config.epochs == 0 {
        return Err(Error::config("epochs must be > 0"));
    }
    if config.learning_rate.is_nan() || config.learning_rate <= 0.0 {
        return Err(Error::config("learning_rate must be > 0"));
    }
    config.batching.validate()?;
    if let Some(sample) = samples
        .iter()
        .find(|s| s.feature_dim != config.model.input_dim)
    {
        return Err(Error::config(format!(
            "samples have feature_dim {} but the model expects input_dim {}",
            sample.feature_dim, config.model.input_dim
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ResumeState {
    completed_epochs: usize,
    elapsed_offset_secs: f64,
}

fn checkpoint_epoch(path: &Path) -> Option<usize> {
    path.file_name()?
        .to_str()?
        .strip_prefix(EPOCH_CHECKPOINT_PREFIX)?
        .parse()
        .ok()
}

/// Work out how many epochs a resumed run has already completed.
///
/// A fresh run (`resume_from == None`) always starts at epoch 0, even if the
/// directory holds files from an earlier run. When resuming from
/// `speech-aligner-epoch-N`, N is authoritative. For any other checkpoint
/// name (e.g. `speech-aligner-final`) the newest epoch recorded in the
/// directory is used. Elapsed time comes from the metrics lines up to that
/// epoch.
fn collect_resume_state(checkpoint_dir: &Path, resume_from: Option<&Path>) -> Result<ResumeState> {
    let mut state = ResumeState::default();
    let Some(resume_path) = resume_from else {
        return Ok(state);
    };
    let resume_epoch = checkpoint_epoch(resume_path);

    if let Some(epoch) = resume_epoch {
        state.completed_epochs = epoch;
    } else if let Ok(entries) = std::fs::read_dir(checkpoint_dir) {
        for entry in entries.flatten() {
            if let Some(epoch) = checkpoint_epoch(&entry.path()) {
                state.completed_epochs = state.completed_epochs.max(epoch);
            }
        }
    }

    let metrics_path = checkpoint_dir.join(METRICS_FILE);
    let Ok(file) = std::fs::File::open(&metrics_path) else {
        return Ok(state);
    };
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|err| Error::io(&metrics_path, err))?;
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let line_epoch = ["epoch", "epochs"]
            .iter()
            .find_map(|key| value.get(*key)?.as_u64())
            .map(|epoch| epoch as usize);
        if resume_epoch.is_none() {
            if let Some(epoch) = line_epoch {
                state.completed_epochs = state.completed_epochs.max(epoch);
            }
        }
        if line_epoch.is_some_and(|epoch| epoch > state.completed_epochs) {
            continue;
        }
        if let Some(elapsed) = value.get("elapsed_secs").and_then(Value::as_f64) {
            state.elapsed_offset_secs = state.elapsed_offset_secs.max(elapsed);
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn tiny_model() -> SpeechAlignerConfig {
        SpeechAlignerConfig {
            input_dim: 4,
            num_classes: 6,
            n_heads: 1,
            se_reduction: 2,
            channels: [4, 4, 4, 8],
            kernel_size: 3,
        }
    }

    fn sample(frames: usize, offset: f32) -> TrainingSample {
        TrainingSample {
            features: (0..frames * 4)
                .map(|i| ((i % 7) as f32 + offset) * 0.1)
                .collect(),
            feature_dim: 4,
            targets: vec![1, 2, 3],
            oov_words: 0,
        }
    }

    #[test]
    fn learning_rate_schedule() {
        assert_eq!(learning_rate_at(1.0, 5, 10, 0), 1.0);
        assert!((learning_rate_at(1.0, 0, 10, 2) - 0.5).abs() < 1e-12);
        assert!((learning_rate_at(1.0, 1, 10, 2) - 1.0).abs() < 1e-12);
        assert!((learning_rate_at(1.0, 2, 10, 2) - 1.0).abs() < 1e-12);
        assert!(learning_rate_at(1.0, 9, 10, 2) < 0.1);
    }

    #[test]
    fn rejects_mismatched_feature_dim() {
        let mut bad = sample(20, 0.0);
        bad.feature_dim = 2;
        let config = TrainConfig {
            model: tiny_model(),
            ..TrainConfig::default()
        };
        let err = train(vec![bad], &config).expect_err("mismatch");
        assert!(err.to_string().contains("feature_dim"), "{err}");
    }

    #[test]
    fn loss_decreases_and_checkpoints_are_written() {
        let dir = tempdir().expect("tempdir");
        let config = TrainConfig {
            model: tiny_model(),
            epochs: 8,
            learning_rate: 1e-2,
            batching: Batching::Fixed { batch_size: 2 },
            checkpoint_dir: dir.path().to_path_buf(),
            checkpoint_interval: 4,
            ..TrainConfig::default()
        };
        let samples = vec![sample(20, 0.0), sample(24, 1.0), sample(16, 2.0)];

        let summary = train(samples, &config).expect("train");
        let first = summary.epoch_losses[0];
        let last = *summary.epoch_losses.last().expect("losses");
        assert!(last < first, "loss should fall: {first} -> {last}");
        for name in [
            "speech-aligner-epoch-4",
            "speech-aligner-epoch-8",
            FINAL_CHECKPOINT,
            METRICS_FILE,
            "model-config.json",
        ] {
            assert!(dir.path().join(name).exists(), "{name} missing");
        }

        // The final checkpoint loads back into the same architecture.
        let device = backend::default_device();
        let model = tiny_model()
            .init::<backend::InferBackend>(&device)
            .expect("init");
        checkpoint::load(model, &summary.final_checkpoint, &device).expect("reload");
    }

    #[test]
    fn resume_state_uses_resume_checkpoint_epoch() {
        let dir = tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(METRICS_FILE),
            concat!(
                "{\"event\":\"epoch\",\"epoch\":2,\"elapsed_secs\":18.0}\n",
                "{\"event\":\"batch\",\"epoch\":2,\"elapsed_secs\":20.0}\n",
                "{\"event\":\"epoch\",\"epoch\":3,\"elapsed_secs\":30.0}\n"
            ),
        )
        .expect("write metrics");
        std::fs::write(dir.path().join("speech-aligner-epoch-3"), b"").expect("write");

        let state =
            collect_resume_state(dir.path(), Some(Path::new("/tmp/speech-aligner-epoch-2")))
                .expect("state");
        assert_eq!(
            state,
            ResumeState {
                completed_epochs: 2,
                elapsed_offset_secs: 20.0
            }
        );
    }

    #[test]
    fn resume_state_falls_back_to_newest_epoch() {
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("speech-aligner-epoch-3"), b"").expect("write");
        let state = collect_resume_state(dir.path(), Some(&dir.path().join(FINAL_CHECKPOINT)))
            .expect("state");
        assert_eq!(state.completed_epochs, 3);
    }

    #[test]
    fn fresh_run_ignores_old_checkpoints() {
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("speech-aligner-epoch-7"), b"").expect("write");
        assert_eq!(
            collect_resume_state(dir.path(), None).expect("state"),
            ResumeState::default()
        );
    }
}
