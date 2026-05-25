# burn-speech-training

[![CI](https://github.com/dnvt/burn-speech-training/actions/workflows/ci.yml/badge.svg)](https://github.com/dnvt/burn-speech-training/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

Reference speech model training in Rust, built on [Burn](https://burn.dev).

This repo is a practical reference for the full training loop: audio features,
phoneme targets, a Burn model, CTC loss, checkpoints, and pronunciation-scoring
evaluation. The CPU path is covered in CI. CUDA and WGPU feature flags are
included, but GPU behavior is hardware-dependent and should be validated on your
machine.

I built this while working on pronunciation scoring infrastructure and couldn't
find speech training examples for Burn — so I'm open-sourcing it as a reference
for anyone working in this space.

## Quick start

```bash
git clone https://github.com/dnvt/burn-speech-training
cd burn-speech-training
cargo run --example train_small --features ndarray --release
```

Abridged output. Exact loss values vary with model initialization:

```
burn-speech-training: quick-start example

Training SpeechAligner on synthetic data (CPU)...

Model: SpeechAligner (122.0K parameters)
Config: input_dim=13, num_classes=42, heads=2

Synthetic smoke test: fixed batch=2, time=30, targets=6/sample
Training for 5 epochs...

  Epoch 1/5: loss = ...
  Epoch 2/5: loss = ...
  Epoch 3/5: loss = ...
  Epoch 4/5: loss = ...
  Epoch 5/5: loss = ...
```

The example is a synthetic smoke test. It verifies model initialization, forward
pass, CTC loss, backward pass, optimizer step, and inference shapes. It is not
evidence of real speech accuracy. For real training, see below.

## What's inside

```
src/
├── model.rs          SpeechAligner: CNN+SE+Attention, about 1.7M params
├── train.rs          LibriSpeech training loop; ndarray path is CI-verified
├── finetune.rs       SpeechOcean762 fine-tuning with scoring head
├── evaluate.rs       Spearman ρ evaluation against human labels
├── dataset.rs        LibriSpeech loader + dynamic batching
├── mfcc.rs           MFCC and log-mel feature extraction
├── phoneme_map.rs    ARPABET -> CTC index mapping, OOV fallback
├── precompute.rs     Binary feature cache for faster ablation
├── loss.rs           CTC loss wrapper
├── attention.rs      Self-attention with residual
├── conv_block.rs     Conv1d + LayerNorm + SE block
├── se_block.rs       Squeeze-and-Excitation
├── ui.rs             Training output helpers
├── error.rs          Error types
└── g2p/              CMU Dict G2P embedded at compile time
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
`--features wgpu` (Vulkan/Metal). The `ndarray` CPU path is the CI-verified
default; treat GPU backends as local hardware targets to validate.

### Precomputed features

For fast ablation, precompute MFCC features to a binary cache. In the original
experiment setup this moved runs from roughly 2 hours to roughly 30 minutes,
but the exact speedup depends on hardware, dataset size, and feature mode. See
`src/precompute.rs` for the cache format and `src/train.rs` for the precomputed
training path.

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
| 4. Loss ablation | 13 | 0.292 | Disabling CTC loss during scoring is the single biggest gain (+0.07 ρ). In this setup, the CTC gradient hurt scoring. |
| 5. Schedule search | 5 | 0.292 | Warmup, freeze schedules, LR decay — marginal gains. ≈0.29 ceiling is reproducible. |
| 6. Architecture search | 13 | 0.288 | Rank regularization, ordinal loss, attention pooling, distillation — none broke through |

**Best result**: ρ = 0.292 (Spearman correlation with human pronunciation
scores).

### What worked

- **Disable CTC loss during scoring fine-tuning.** The top experiments all set
  CTC weight to zero when training the scoring head. In this setup, the CTC
  gradient appeared to interfere with pronunciation ranking. This was the single
  biggest gain.
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
destroys ranking signal. The evidence points to a representation/data
bottleneck more than a loss-function problem. Richer input features (for
example, self-supervised speech representations) are the likely path forward.

See [`docs/experiment-log.md`](docs/experiment-log.md) for the full experiment
log with per-run configs and results.

## Trust And Scope Docs

- [`docs/datasets.md`](docs/datasets.md): what data is used, what is not
  included, and how to keep provenance clear.
- [`docs/model-card.md`](docs/model-card.md): intended use, non-goals, reported
  result, and known failure modes.
- [`SECURITY.md`](SECURITY.md): how to report sensitive issues without exposing
  private audio or credentials.

## Lessons learned

1. **CTC gradients can hurt pronunciation scoring.** In this experiment set,
   training alignment and scoring separately worked better than multi-tasking
   them.
2. **LR must scale with batch size.** When dynamic batching changes effective
   batch size, scale LR proportionally or training diverges.
3. **Feature representation matters more than loss engineering.** 35 experiments
   on loss geometry gained +0.07 ρ total. Richer features are the higher-
   leverage path.
4. **Precompute features for ablation.** MFCC extraction is a CPU bottleneck.
   A binary cache made iteration much faster in the original experiment setup.
5. **Evaluate at every checkpoint.** The best model is rarely the last one.

## Current limits

- The quick-start example uses synthetic data and should be read as a smoke
  test, not a quality benchmark.
- The strongest reported pronunciation result is modest: ρ = 0.292 on
  SpeechOcean762 word scores.
- The CPU `ndarray` path is the default CI target. CUDA and WGPU support need
  local validation on matching hardware.
- The code favors being inspectable over being a polished training framework.

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

| Flag | Backend | Use case | Verification |
|------|---------|----------|--------------|
| `ndarray` (default) | NdArray + Autodiff | CPU training, testing | CI check, test, clippy, docs, package |
| `cuda` | CUDA + Autodiff | NVIDIA GPU training | Local hardware validation required |
| `wgpu` | WGPU + Autodiff | Vulkan/Metal GPU training | Local hardware validation required |

## License

MIT OR Apache-2.0
