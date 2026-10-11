# burn-speech-training

[![CI](https://github.com/dnvt/burn-speech-training/actions/workflows/ci.yml/badge.svg)](https://github.com/dnvt/burn-speech-training/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/burn-speech-training)](https://crates.io/crates/burn-speech-training)
[![docs.rs](https://img.shields.io/docsrs/burn-speech-training)](https://docs.rs/burn-speech-training)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/dnvt/burn-speech-training/blob/main/LICENSE)

A complete speech training pipeline in Rust on [Burn](https://burn.dev):
MFCC features, CTC training on LibriSpeech, and fine-tuning for
pronunciation scoring on SpeechOcean762.

I wrote it while working on pronunciation scoring and couldn't find a
speech training example for Burn. It is meant to be read and adapted. It is
not a polished framework, and the model is small: the best pronunciation
result is ρ = 0.29 (see [Results](#results)).

## Quick start

Needs Rust 1.95 or newer.

```bash
git clone https://github.com/dnvt/burn-speech-training
cd burn-speech-training
cargo run --release --example train_small
```

This trains a 122K-parameter model on synthetic data for a few seconds, then
reloads the checkpoint. It checks that the pipeline runs end to end and says
nothing about accuracy.

## Pipeline

```text
  LibriSpeech                                       SpeechOcean762
  .flac + transcripts                               word scores (JSON)
        │                                                  │
        ▼                                                  ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  ┌──────────┐
  │precompute │─▶│   train   │─▶│checkpoint │─▶│    finetune     │─▶│ evaluate │
  │MFCC cache │  │ CTC loss  │  │           │  │ + scoring head  │  │ Spearman │
  └───────────┘  └───────────┘  └───────────┘  └─────────────────┘  └──────────┘
   optional        transcripts → phonemes:        MSE on human        GOP baselines
                   CMU dict + OOV rules           scores              + scoring head
```

Each box is a CLI command and a library function.

## Command line

```bash
cargo install burn-speech-training   # or, from a clone: cargo run --release -- <command> ...

# 1. (optional) cache features; training then skips audio decoding
burn-speech precompute --data-dir ~/data/LibriSpeech --split train-clean-100 --out cache/

# 2. CTC pre-training
burn-speech train --cache cache/ --out ckpt/ --epochs 10 --attn-budget 500000000

# 3. pronunciation-scoring fine-tune (best recipe from the experiment log by default)
burn-speech finetune --base ckpt/speech-aligner-final \
    --dataset so762-train.json --holdout so762-dev.json --data-root ~/data/speechocean762 \
    --out ckpt-ft/

# 4. correlation with human scores
burn-speech evaluate --checkpoint ckpt-ft/speech-aligner-ft-best \
    --head ckpt-ft/scoring-head-best \
    --dataset so762-test.json --data-root ~/data/speechocean762
```

`--help` on any command lists every option. Progress is logged to stderr;
set `RUST_LOG=debug` for more. Per-epoch metrics are written as JSON lines
to `training-metrics.jsonl` (train) or `finetune-metrics.jsonl` (finetune) in
the output directory.

LibriSpeech audio is used as is (16 kHz FLAC). SpeechOcean762 needs to be
flattened to one JSON entry per word; the format is documented in
[`src/data/so762.rs`](https://github.com/dnvt/burn-speech-training/blob/main/src/data/so762.rs).

## As a library

```rust,no_run
use burn_speech_training::data::{librispeech, Batching};
use burn_speech_training::mfcc::FeatureMode;
use burn_speech_training::train::{train, TrainConfig};

let samples = librispeech::load(&librispeech::LibriSpeechSource {
    data_dir: "/data/LibriSpeech".into(),
    split: "train-clean-100".into(),
    feature_mode: FeatureMode::Mfcc39,
    max_duration_secs: 15.0,
})?;

let summary = train(samples, &TrainConfig {
    batching: Batching::Fixed { batch_size: 16 },
    checkpoint_dir: "ckpt".into(),
    ..TrainConfig::default()
})?;
# Ok::<(), burn_speech_training::error::Error>(())
```

`train` takes any `Vec<TrainingSample>` (features + CTC targets), so a
different dataset only needs its own loader. Output goes through `tracing`;
install a subscriber (e.g. `tracing_subscriber::fmt::init()`) to see it.

## Model

`SpeechAligner`, 1.7M parameters by default:

```text
features [B, T, 39]
  │
  ├─ 4 × conv block       Conv1d(k=3) → LayerNorm → ReLU → squeeze-excitation
  │                       channels 39 → 64 → 128 → 256 → 512, no downsampling
  ├─ self-attention       LayerNorm → 8-head attention → residual
  │                       (one layer, no feed-forward, no positional encoding)
  │
  ├─▶ CTC head            [T, B, 42]  log-probs: blank + 40 phonemes + silence
  ├─▶ phoneme head        [B, T, 42]  not trained by this crate
  ├─▶ boundary head       [B, T, 1]   not trained by this crate
  └─▶ frame features      [B, T, 512] input to the scoring head
```

Padded frames are masked in the squeeze-excitation pooling and in attention,
so a sample gets the same output whatever it is batched with. Change the size
with `SpeechAlignerConfig` (`channels`, `n_heads`); the config is saved as
`model-config.json` next to each checkpoint.

## Results

35 fine-tuning experiments on SpeechOcean762, measured by Spearman ρ between
predicted and human word scores. Full configs and numbers are in
[`docs/experiment-log.md`](https://github.com/dnvt/burn-speech-training/blob/main/docs/experiment-log.md).

| Round | Runs | Best ρ | Finding |
|---|---|---|---|
| 1. CTC log-prob GOP | 1 | 0.106 | Alignment training works; GOP alone ranks poorly |
| 2. Learning rate | 2 | 0.106 | With variable batch sizes, LR 1e-3 diverged, 3e-4 converged |
| 3. Scoring head | 1 | 0.221 | Supervised head helps, plateaus by epoch 4 |
| 4. Loss ablation | 13 | **0.292** | Setting CTC weight to 0 is the largest gain (+0.07) |
| 5. Schedules | 5 | 0.292 | Warmup + cosine is stable; freezing the backbone hurts |
| 6. Architecture | 13 | 0.292 | No gain over Round 5 (distillation and rank-regularization runs were buggy, see the log) |

What the runs suggest:

- **Train alignment and scoring separately.** The CTC gradient hurt
  ranking when the two shared a loss.
- **Keep the best holdout epoch.** Scores peaked between epochs 6 and 14.
  `finetune` saves `*-best` checkpoints when given a holdout set.
- **The ceiling is probably the features.** About 91% of SpeechOcean762
  words score 10/10, and loss reweighting did not move ρ. Self-supervised
  features such as wav2vec2 are the obvious next step.

The losing variants (focal, ordinal and balanced losses, ranking losses,
mixup, attention pooling, distillation) are kept behind
`--features experiments` in [`src/experiments/`](https://github.com/dnvt/burn-speech-training/blob/main/src/experiments/) for
reproduction.

## Features

| Feature | Default | |
|---|---|---|
| `flex` | yes | pure-Rust CPU backend; the one tested in CI |
| `cli` | yes | the `burn-speech` binary |
| `wgpu` | | GPU via Vulkan/Metal/DX12 (compiled in CI, not run) |
| `cuda` | | NVIDIA GPU (not tested in CI) |
| `experiments` | | variants from the experiment log |

Several backends can be compiled in. The default device is the first
available of `cuda`, `wgpu` and `flex`; set `BURN_DEVICE` (e.g.
`BURN_DEVICE=flex`) to pick one. To use the library without the CLI:
`default-features = false, features = ["flex"]`.

## Layout

| Path | |
|---|---|
| `src/model/` | `SpeechAligner` and its layers |
| `src/mfcc.rs` | MFCC and log-mel features |
| `src/g2p.rs`, `src/phoneme_map.rs` | transcript → ARPABET → CTC class indices |
| `src/data/` | audio decoding, LibriSpeech, SpeechOcean762, batching, feature cache |
| `src/train.rs` | CTC training loop |
| `src/finetune/` | scoring head and fine-tuning |
| `src/evaluate.rs` | GOP baselines and correlation report |
| `src/experiments/` | variants that did not help (`experiments` feature) |
| `src/bin/burn-speech.rs` | command-line tool |

## Limitations

- Word boundaries are approximated by splitting an utterance's frames evenly
  between its words, in both training and evaluation.
- Audio must already be 16 kHz; nothing is resampled.
- Out-of-vocabulary words go through simple letter-to-sound rules.
- GPU backends compile but are only exercised on the author's hardware.

See also [`docs/model-card.md`](https://github.com/dnvt/burn-speech-training/blob/main/docs/model-card.md) (intended use) and
[`docs/datasets.md`](https://github.com/dnvt/burn-speech-training/blob/main/docs/datasets.md) (data sources and licences).

## License

Code: MIT OR Apache-2.0, at your option. The bundled CMU Pronouncing
Dictionary (`src/g2p/data/cmudict.dict`) is © Carnegie Mellon University
under a BSD-style licence, in [`src/g2p/data/LICENSE`](https://github.com/dnvt/burn-speech-training/blob/main/src/g2p/data/LICENSE).
