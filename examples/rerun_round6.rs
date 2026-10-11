//! Preflight or run one controlled Round 6 condition; see docs/round6-rerun.md.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use burn::tensor::Device;
use burn_speech_training::checkpoint;
use burn_speech_training::data::{audio, so762};
use burn_speech_training::evaluate::{evaluate, EvaluateConfig};
use burn_speech_training::experiments::{
    finetune, teacher::TeacherCache, ExperimentConfig, TeacherMode,
};
use burn_speech_training::finetune::FineTuneConfig;
use burn_speech_training::mfcc::FeatureMode;
use burn_speech_training::model::SpeechAlignerConfig;
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Condition {
    Baseline,
    Distill01,
    Distill05,
    Rank,
    Combined,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    condition: Condition,
    train: PathBuf,
    holdout: PathBuf,
    /// Optional independent test set; never used for model selection.
    test: Option<PathBuf>,
    data_root: PathBuf,
    base_checkpoint: PathBuf,
    teacher_cache: Option<PathBuf>,
    out: PathBuf,
    epochs: usize,
    learning_rate: f64,
    warmup_epochs: usize,
    batch_size: usize,
    eval_interval: usize,
    seed: u64,
    /// Limits training words for a pilot; holdout and test remain complete.
    max_samples: Option<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("usage: rerun_round6 CONFIG.json [--run]")?,
    );
    let run = match args.next() {
        None => false,
        Some(arg) if arg == "--run" => true,
        Some(_) => return Err("usage: rerun_round6 CONFIG.json [--run]".into()),
    };
    if args.next().is_some() {
        return Err("usage: rerun_round6 CONFIG.json [--run]".into());
    }
    let config: Config = serde_json::from_slice(&read_bytes(&path)?)?;
    preflight(&config)?;
    println!(
        "condition: {:?}; at most {} epochs; seed {}",
        config.condition, config.epochs, config.seed
    );
    if !run {
        println!("Preflight passed. Add --run to execute this condition.");
        return Ok(());
    }

    // Reserve the result directory atomically, including concurrent launches.
    if let Some(parent) = config.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(&config.out).map_err(|e| {
        format!(
            "cannot reserve result directory {}: {e}",
            config.out.display()
        )
    })?;
    std::fs::write(
        config.out.join("rerun-config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let mut experiment = ExperimentConfig::new(FineTuneConfig {
        dataset: config.train.clone(),
        holdout: Some(config.holdout.clone()),
        data_root: config.data_root.clone(),
        base_checkpoint: Some(config.base_checkpoint.clone()),
        feature_mode: FeatureMode::Mfcc39,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        warmup_epochs: config.warmup_epochs,
        ctc_weight: 0.0,
        batch_size: config.batch_size,
        eval_interval: config.eval_interval,
        head_hidden_dim: 256,
        max_samples: config.max_samples,
        checkpoint_dir: config.out.clone(),
        seed: config.seed,
        ..FineTuneConfig::default()
    });
    if let Some(weight) = distillation_weight(config.condition) {
        experiment.teacher = Some((
            config.teacher_cache.clone().unwrap(),
            TeacherMode::Distill { weight },
        ));
    }
    if matches!(config.condition, Condition::Rank | Condition::Combined) {
        experiment.rank_regularization = Some((0.1, 0.15));
    }
    let summary = finetune(&experiment)?;
    let (epoch, rho, model, head) = summary.best.ok_or("no holdout checkpoint was selected")?;
    println!(
        "selected epoch {epoch}; holdout rho={:.6} (selection metric)",
        rho.spearman
    );
    if let Some(test) = config.test {
        let report = evaluate(&EvaluateConfig {
            checkpoint: model,
            scoring_head: Some(head),
            dataset: test,
            data_root: config.data_root,
            max_samples: None,
            feature_mode: FeatureMode::Mfcc39,
        })?;
        std::fs::write(config.out.join("test-evaluation.md"), report.to_markdown())?;
    }
    Ok(())
}

fn distillation_weight(condition: Condition) -> Option<f32> {
    match condition {
        Condition::Distill01 => Some(0.1),
        Condition::Distill05 => Some(0.5),
        Condition::Combined => Some(0.3),
        Condition::Baseline | Condition::Rank => None,
    }
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()).into())
}

fn preflight(config: &Config) -> Result<()> {
    if !(1..=20).contains(&config.epochs)
        || config.batch_size == 0
        || config.eval_interval == 0
        || config.eval_interval > config.epochs
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.max_samples == Some(0)
    {
        return Err("use 1..20 epochs, finite positive learning rate, and positive batch/eval/sample counts".into());
    }
    if config.out.exists() {
        return Err("output directory already exists; choose a fresh directory".into());
    }
    if config.base_checkpoint.extension().and_then(|e| e.to_str()) != Some("bpk") {
        return Err(
            "the original pre-0.3 checkpoint requires a verified conversion to .bpk".into(),
        );
    }
    let saved_config = config
        .base_checkpoint
        .parent()
        .unwrap_or(Path::new("."))
        .join("model-config.json");
    let model_config: SpeechAlignerConfig = serde_json::from_slice(&read_bytes(&saved_config)?)?;
    if model_config != SpeechAlignerConfig::default() {
        return Err("Round 6 requires the original 39-dimensional, 1.7M-parameter SpeechAligner; synthetic quickstart weights are incompatible".into());
    }
    let device = Device::default();
    let _model = checkpoint::load(model_config.init(&device)?, &config.base_checkpoint)?;
    let cache = if distillation_weight(config.condition).is_some() {
        Some(TeacherCache::open(
            config
                .teacher_cache
                .as_ref()
                .ok_or("distillation requires teacher_cache")?,
        )?)
    } else {
        None
    };
    let mut previous_audio = BTreeSet::new();
    for (label, path, max_samples) in [
        ("train", Some(&config.train), config.max_samples),
        ("holdout", Some(&config.holdout), None),
        ("test", config.test.as_ref(), None),
    ] {
        let Some(path) = path else { continue };
        let words = so762::load(path, max_samples)?;
        let groups = so762::group_by_audio(&words);
        let usable: usize = groups.values().map(Vec::len).sum();
        if usable < 10 {
            return Err(format!("{label} contains fewer than 10 usable words").into());
        }
        for name in groups.keys() {
            if !previous_audio.insert((*name).to_owned()) {
                return Err(
                    "audio overlaps between train/holdout/test; use disjoint splits".into(),
                );
            }
            // Decode before training so non-16k or missing files cannot silently
            // reduce the experiment cohort in the library's skip-on-error path.
            audio::load(&config.data_root.join(name))?;
            if label == "train" {
                if let Some(cache) = &cache {
                    let features = cache
                        .get(name)?
                        .ok_or("teacher cache does not cover every training utterance")?;
                    if features.frames == 0 || features.values.iter().any(|v| !v.is_finite()) {
                        return Err("teacher cache must contain nonempty finite features for every training utterance".into());
                    }
                }
            }
        }
        println!(
            "{label}: {usable} usable words / {} utterances; 16k audio verified",
            groups.len()
        );
    }
    Ok(())
}
