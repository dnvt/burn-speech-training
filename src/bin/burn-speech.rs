//! Command-line front end: precompute features, train, fine-tune, evaluate.
//!
//! Run `burn-speech --help` or `burn-speech <command> --help` for options.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};

use burn_speech_training::checkpoint;
use burn_speech_training::data::librispeech::{self, LibriSpeechSource};
use burn_speech_training::data::{cache, Batching};
use burn_speech_training::evaluate::{evaluate, EvaluateConfig};
use burn_speech_training::finetune::{finetune, FineTuneConfig};
use burn_speech_training::mfcc::FeatureMode;
use burn_speech_training::model::SpeechAlignerConfig;
use burn_speech_training::train::{train, TrainConfig};

#[derive(Parser)]
#[command(version, about = "Speech model training with Burn")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Featurize a LibriSpeech split into a binary cache for faster training.
    Precompute {
        #[command(flatten)]
        source: SourceArgs,
        /// Output directory for the cache.
        #[arg(long)]
        out: PathBuf,
    },
    /// Train SpeechAligner with CTC loss on LibriSpeech or a feature cache.
    Train(TrainArgs),
    /// Fine-tune a scoring head on SpeechOcean762 word scores.
    Finetune(FinetuneArgs),
    /// Report GOP baselines and scoring-head correlation with human scores.
    Evaluate(EvaluateArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Features {
    /// 13 MFCCs + deltas + delta-deltas (39 dims).
    Mfcc39,
    /// 80 log-mel filterbank energies.
    Logmel80,
}

impl From<Features> for FeatureMode {
    fn from(f: Features) -> Self {
        match f {
            Features::Mfcc39 => Self::Mfcc39,
            Features::Logmel80 => Self::LogMel80,
        }
    }
}

#[derive(Args)]
struct SourceArgs {
    /// Directory containing the LibriSpeech split directories.
    #[arg(long)]
    data_dir: PathBuf,
    /// Split name, e.g. train-clean-100.
    #[arg(long, default_value = "train-clean-100")]
    split: String,
    #[arg(long, value_enum, default_value = "mfcc39")]
    features: Features,
    /// Skip utterances longer than this (seconds).
    #[arg(long, default_value_t = 15.0)]
    max_duration: f32,
}

impl SourceArgs {
    fn source(&self) -> LibriSpeechSource {
        LibriSpeechSource {
            data_dir: self.data_dir.clone(),
            split: self.split.clone(),
            feature_mode: self.features.into(),
            max_duration_secs: self.max_duration,
        }
    }
}

#[derive(Args)]
struct TrainArgs {
    /// LibriSpeech root (use with --split); or pass --cache instead.
    #[arg(long, conflicts_with = "cache")]
    data_dir: Option<PathBuf>,
    #[arg(long, default_value = "train-clean-100")]
    split: String,
    #[arg(long, value_enum, default_value = "mfcc39")]
    features: Features,
    #[arg(long, default_value_t = 15.0)]
    max_duration: f32,
    /// Feature cache written by `precompute`.
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Checkpoint directory.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    /// Linear warmup epochs before cosine decay (0 = constant LR).
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// Fixed number of utterances per batch.
    #[arg(long, default_value_t = 16, conflicts_with = "attn_budget")]
    batch_size: usize,
    /// Size batches by attention memory instead: max batch × heads × frames².
    #[arg(long)]
    attn_budget: Option<usize>,
    /// Upper bound on utterances per batch with --attn-budget.
    #[arg(long, default_value_t = 256)]
    max_batch: usize,
    /// Save a checkpoint every N epochs (0 = final only).
    #[arg(long, default_value_t = 5)]
    checkpoint_every: usize,
    /// Continue from this checkpoint.
    #[arg(long)]
    resume: Option<PathBuf>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args)]
struct FinetuneArgs {
    /// SpeechOcean762 word-score JSON for training.
    #[arg(long)]
    dataset: PathBuf,
    /// Directory the audio paths in the JSON are relative to.
    #[arg(long)]
    data_root: PathBuf,
    /// Word-score JSON used to pick the best epoch.
    #[arg(long)]
    holdout: Option<PathBuf>,
    /// Pre-trained checkpoint from `train`; omit to start from scratch.
    #[arg(long)]
    base: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "mfcc39")]
    features: Features,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Weight of the CTC loss (0 = scoring loss only, the best recipe).
    #[arg(long, default_value_t = 0.0)]
    ctc_weight: f32,
    /// Utterances per batch.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    /// Use only the first N words of the dataset.
    #[arg(long)]
    max_samples: Option<usize>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args)]
struct EvaluateArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    /// Scoring head from `finetune`.
    #[arg(long)]
    head: Option<PathBuf>,
    /// SpeechOcean762 word-score JSON (with `phonemes` for GOP).
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    data_root: PathBuf,
    /// Features the checkpoint was trained on (taken from --head if given).
    #[arg(long, value_enum, default_value = "mfcc39")]
    features: Features,
    #[arg(long)]
    max_samples: Option<usize>,
    /// Also write the results table to this Markdown file.
    #[arg(long)]
    report: Option<PathBuf>,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: Command) -> burn_speech_training::error::Result<()> {
    match command {
        Command::Precompute { source, out } => {
            let manifest = cache::precompute(&source.source(), &out)?;
            println!(
                "wrote {} samples ({} frames) to {}",
                manifest.num_samples,
                manifest.total_frames,
                out.display()
            );
        }
        Command::Train(args) => {
            let (samples, feature_mode) = match (&args.cache, &args.data_dir) {
                (Some(dir), _) => {
                    let (manifest, samples) = cache::load(dir)?;
                    (samples, manifest.feature_mode.unwrap_or_default())
                }
                (None, Some(data_dir)) => {
                    let source = LibriSpeechSource {
                        data_dir: data_dir.clone(),
                        split: args.split.clone(),
                        feature_mode: args.features.into(),
                        max_duration_secs: args.max_duration,
                    };
                    (librispeech::load(&source)?, source.feature_mode)
                }
                (None, None) => {
                    return Err(burn_speech_training::error::Error::config(
                        "pass --data-dir or --cache",
                    ))
                }
            };
            let batching = match args.attn_budget {
                Some(max_attn_elements) => Batching::AttentionBudget {
                    max_attn_elements,
                    max_samples: args.max_batch,
                },
                None => Batching::Fixed {
                    batch_size: args.batch_size,
                },
            };
            let summary = train(
                samples,
                &TrainConfig {
                    // A resumed checkpoint keeps its own architecture.
                    model: match &args.resume {
                        Some(path) => checkpoint::model_config_for(path, feature_mode)?,
                        None => SpeechAlignerConfig::for_feature_mode(feature_mode),
                    },
                    epochs: args.epochs,
                    learning_rate: args.lr,
                    warmup_epochs: args.warmup,
                    batching,
                    checkpoint_dir: args.out,
                    checkpoint_interval: args.checkpoint_every,
                    resume_from: args.resume,
                    seed: args.seed,
                },
            )?;
            println!("final checkpoint: {}", summary.final_checkpoint.display());
        }
        Command::Finetune(args) => {
            let summary = finetune(&FineTuneConfig {
                dataset: args.dataset,
                data_root: args.data_root,
                holdout: args.holdout,
                base_checkpoint: args.base,
                feature_mode: args.features.into(),
                epochs: args.epochs,
                learning_rate: args.lr,
                warmup_epochs: args.warmup,
                ctc_weight: args.ctc_weight,
                batch_size: args.batch_size,
                max_samples: args.max_samples,
                checkpoint_dir: args.out,
                seed: args.seed,
                ..FineTuneConfig::default()
            })?;
            println!("final model: {}", summary.final_model.display());
            println!("final head:  {}", summary.final_head.display());
            if let Some((epoch, result, model, head)) = summary.best {
                println!("best holdout (epoch {epoch}): {result}");
                println!("best model: {}", model.display());
                println!("best head:  {}", head.display());
            }
        }
        Command::Evaluate(args) => {
            let report = evaluate(&EvaluateConfig {
                checkpoint: args.checkpoint,
                scoring_head: args.head,
                dataset: args.dataset,
                data_root: args.data_root,
                max_samples: args.max_samples,
                feature_mode: args.features.into(),
            })?;
            let markdown = report.to_markdown();
            println!("{markdown}");
            if let Some(path) = args.report {
                std::fs::write(&path, markdown)
                    .map_err(|err| burn_speech_training::error::Error::io(&path, err))?;
            }
        }
    }
    Ok(())
}
