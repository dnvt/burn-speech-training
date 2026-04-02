# burn-speech-training

[![CI](https://github.com/dnvt/burn-speech-training/actions/workflows/ci.yml/badge.svg)](https://github.com/dnvt/burn-speech-training/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

End-to-end speech model training in Rust, built on [Burn](https://burn.dev).

A complete speech training pipeline for the Burn ML framework. Trains a
pronunciation scoring model from raw audio using CTC loss, evaluates against
human-labeled data, and runs on CPU or GPU.

I built this while working on pronunciation scoring infrastructure and couldn't
find speech training examples for Burn — so I'm open-sourcing it as a reference
for anyone working in this space.

## Quick start

```bash
git clone https://github.com/dnvt/burn-speech-training
cd burn-speech-training
cargo run --example train_small --features ndarray --release
```

Output:

```
Training SpeechAligner on synthetic data (CPU)...

Model: SpeechAligner (122.0K parameters)

  Epoch 1/5: loss = 94.41  (0.0s)
  Epoch 2/5: loss = 83.76  (0.0s)
  Epoch 3/5: loss = 68.96  (0.0s)
  Epoch 4/5: loss = 37.35  (0.0s)
  Epoch 5/5: loss = 26.14  (0.0s)
```

The example uses a tiny model on synthetic data. For real training, see below.

## What's inside

```
src/
├── model.rs          SpeechAligner: CNN+SE+Attention, ~1.7M params
├── train.rs          LibriSpeech training loop (CPU + GPU)
├── finetune.rs       SpeechOcean762 fine-tuning with scoring head
├── evaluate.rs       Spearman ρ evaluation against human labels
├── dataset.rs        LibriSpeech loader + dynamic batching
├── mfcc.rs           MFCC (39-dim) and log-mel (80-dim) feature extraction
├── phoneme_map.rs    ARPABET → CTC index mapping, 8-stage OOV fallback
├── precompute.rs     Binary feature cache for fast ablation
├── loss.rs           CTC loss wrapper
├── attention.rs      Self-attention with residual
├── conv_block.rs     Conv1d + LayerNorm + SE block
├── se_block.rs       Squeeze-and-Excitation
├── ui.rs             Training output helpers
├── error.rs          Error types
└── g2p/              CMU Dict G2P (135K words, embedded at compile time)
    ├── cmudict.rs
    ├── arpabet.rs
    └── types.rs
```

## Pipeline

```
.flac/.wav audio ─→ MFCC extraction ─→ SpeechAligner model ─→ CTC loss ─→ checkpoint
                     (mfcc.rs)          (model.rs)             (loss.rs)
                                             │
        transcript ─→ G2P phoneme lookup ────┘ targets
                       (g2p/ + phoneme_map.rs)
```

**Training**: `train.rs` orchestrates the loop — loads LibriSpeech, extracts
features, batches dynamically by memory budget, trains with Adam + CTC loss,
checkpoints at intervals.

**Fine-tuning**: `finetune.rs` adds a scoring head (MLP) on top of a
pre-trained checkpoint and trains against human pronunciation labels from
SpeechOcean762.

**Evaluation**: `evaluate.rs` computes Spearman ρ between predicted and human
scores with bootstrap confidence intervals.

## Model

```
Input [B, T, 39] → 4× ConvSE blocks → Self-attention → 3 heads
                    39→64→128→256→512    + residual

  Phoneme head [B, T, 42]    frame-level phoneme logits
  Boundary head [B, T, 1]    word boundary probability
  CTC head [T, B, 42]        log-probabilities for CTC loss
```

~1.7M parameters with default config. Adjustable via `SpeechAlignerConfig`.

## Training on real data

### Prerequisites

1. [LibriSpeech](https://www.openslr.org/12/) — download `train-clean-100` or
   `train-clean-360` and extract
2. Rust stable (1.87+)

### Using as a library

This is a library crate. To train on real data, call the training functions
from your own binary:

```rust
use burn_speech_training::train::{TrainRealArgs, execute_train_real};
use burn_speech_training::mfcc::FeatureMode;

let args = TrainRealArgs {
    data_dir: "/path/to/LibriSpeech".into(),
    split: "train-clean-100".into(),
    epochs: 10,
    batch_size: 16,
    learning_rate: 0.0003,
    checkpoint_dir: "./checkpoints".into(),
    checkpoint_interval: 5,
    max_duration_secs: 15.0,
    feature_mode: FeatureMode::Mfcc39,
};

execute_train_real(&args)?;
```

Enable GPU training by compiling with `--features cuda` (NVIDIA) or
`--features wgpu` (Vulkan/Metal).

### Precomputed features

For fast ablation, precompute MFCC features to a binary cache. This eliminates
the CPU bottleneck and turns 2-hour runs into 30-minute runs. See
`src/precompute.rs` for the cache format and `src/train.rs` for the
precomputed training path.

## Experiment results

I ran 35 experiments across 6 rounds on a GPU, totaling ~$135 in compute. The
goal was to maximize Spearman ρ (rank correlation with human pronunciation
scores on [SpeechOcean762](https://www.openslr.org/83/)).

### Summary

| Round | Runs | Best ρ | Key finding |
|-------|------|--------|-------------|
| 1. CTC pre-training | 1 | 0.106 | CTC alignment training works, but log-prob GOP alone can't rank pronunciation quality |
| 2. Hyperparameter tuning | 2 | 0.106 | Learning rate must scale down with batch size — diverges otherwise |
| 3. Scoring head | 1 | 0.221 | Adding a pronunciation scoring MLP trained on human labels reaches 0.22, then plateaus |
| 4. Loss ablation | 13 | 0.292 | Disabling CTC loss during scoring is the single biggest gain (+0.07 ρ). CTC gradient hurts scoring. |
| 5. Schedule search | 5 | 0.292 | Warmup, freeze schedules, LR decay — marginal gains. ≈0.29 ceiling is reproducible. |
| 6. Architecture search | 13 | 0.288 | Rank regularization, ordinal loss, attention pooling, distillation — none broke through |

**Best result**: ρ = 0.292 (Spearman correlation with human pronunciation
scores).

### What worked

- **Disable CTC loss during scoring fine-tuning.** The top experiments all set
  CTC weight to zero when training the scoring head. CTC gradient actively
  interferes with pronunciation ranking. This was the single biggest gain.
- **Warmup + cosine decay.** Prevents late-epoch regression. Small but
  consistent improvement.
- **Dynamic batching by attention memory budget.** Prevents OOM on variable-
  length audio. Essential for GPU training.
- **Promote best-eval checkpoint, not last.** Models peak at epochs 6-14, not
  at the final epoch.

### What didn't work

- **Focal loss** — hurts ranking ability
- **Inverse-frequency class weighting** — no improvement
- **Larger scoring head** (512 → 256 vs 256 → 128) — no effect
- **Rank regularization** — matched baseline, didn't exceed
- **Ordinal softmax CE** — worse than MSE
- **Attention pooling** — regressed
- **Knowledge distillation** — reproduced baseline, no gain

### Why it plateaued

SpeechOcean762 has a severe class imbalance — ~91% of samples score 10/10. MSE
optimization learns to predict ~1.0 for everything, which minimizes loss but
destroys ranking signal. The ≈0.29 ceiling is a data limitation, not a model
limitation. Richer input features (e.g., self-supervised speech representations)
are the likely path forward.

See [`docs/experiment-log.md`](docs/experiment-log.md) for the full experiment
log with per-run configs and results.

## Lessons learned

1. **CTC gradients hurt pronunciation scoring.** Train alignment and scoring
   separately. Don't multi-task them.
2. **LR must scale with batch size.** When dynamic batching changes effective
   batch size, scale LR proportionally or training diverges.
3. **Feature representation matters more than loss engineering.** 35 experiments
   on loss geometry gained +0.07 ρ total. Richer features are the higher-
   leverage path.
4. **Precompute features for ablation.** MFCC extraction is a CPU bottleneck.
   Binary cache turns 2-hour runs into 30-minute runs.
5. **Evaluate at every checkpoint.** The best model is rarely the last one.

## Adapting for your own task

### Different audio features

```rust
use burn_speech_training::mfcc::FeatureMode;

// 39-dim MFCC (default)
let mode = FeatureMode::Mfcc39;

// 80-dim log-mel spectrogram
let mode = FeatureMode::LogMel80;
```

### Different model size

```rust
use burn_speech_training::model::SpeechAlignerConfig;

// Tiny (for testing)
let config = SpeechAlignerConfig {
    channels: [16, 32, 64, 128],
    n_heads: 2,
    ..SpeechAlignerConfig::default()
};

// Large
let config = SpeechAlignerConfig {
    channels: [128, 256, 512, 1024],
    n_heads: 16,
    ..SpeechAlignerConfig::default()
};
```

### Different dataset

The dataset loader expects LibriSpeech directory structure:

```
<data_dir>/<split>/<speaker_id>/<chapter_id>/
  ├── <speaker>-<chapter>-<utterance>.flac
  └── <speaker>-<chapter>.trans.txt
```

To use a different dataset, implement `load_audio_samples()` and
`scan_librispeech()` equivalents in `src/dataset.rs`.

## Feature flags

| Flag | Backend | Use case |
|------|---------|----------|
| `ndarray` (default) | NdArray + Autodiff | CPU training, testing |
| `cuda` | CUDA + Autodiff | NVIDIA GPU training |
| `wgpu` | WGPU + Autodiff | Vulkan/Metal GPU training |

## License

MIT OR Apache-2.0
