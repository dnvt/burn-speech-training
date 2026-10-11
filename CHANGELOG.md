# Changelog

## 0.3.0 - 2026-10-11

Burn 0.22. Checkpoints use a new file format, so models trained with 0.2.0
must be retrained.

### Changed

- Burn 0.22. Model, training and checkpoint code no longer take a backend
  type parameter (`SpeechAligner`, not `SpeechAligner<B>`), and the
  `backend` module is gone: pass a `burn::tensor::Device`
  (`Device::default()` picks the first available backend).
- Checkpoints are burnpack files (`speech-aligner-final.bpk`). Paths given
  without an extension still work. `checkpoint::save` returns the file it
  wrote, and `checkpoint::load` no longer takes a device. Files written by
  0.2.0 cannot be read.
- The default CPU backend is `flex` (pure Rust), replacing `ndarray`, which
  Burn deprecated. The `ndarray` feature is gone.
- `rust-version` is 1.95 (required by Burn 0.22).
- symphonia 0.6 (audio decoding) and rand 0.10. Decoded audio is unchanged
  (checked against FLAC fixtures under both versions). Bootstrap confidence
  intervals use a fixed seed, but rand's generator changed, so their exact
  bounds can differ slightly from 0.2.0.

## 0.2.0 - 2026-10-11

The crate was reorganised around a smaller API. Breaking changes throughout.

### Added

- `burn-speech` command-line tool (`cli` feature, on by default) with
  `precompute`, `train`, `finetune` and `evaluate`.
- `train::train(samples, &TrainConfig)` accepts any `Vec<TrainingSample>`.
- Fine-tuning saves the checkpoint with the best holdout ρ (`*-best`).
- `model-config.json` is written next to checkpoints, so non-default model
  sizes can be reloaded.
- `experiments` feature with the variants from the experiment log.
- Documented SpeechOcean762 input format (`src/data/so762.rs`).

### Fixed

- Padded frames leaked into squeeze-excitation pooling, so outputs depended
  on batch composition.
- Distillation detached the student features (no gradient to the backbone);
  rank regularization had almost no gradient. Both affected Round 6 of the
  experiment log.
- Duration filtering used a file-size guess; it now uses decoded length.
- The CTC feasibility check ignored repeated labels; `zero_infinity` is on.
- Batches are shuffled every epoch and collated lazily instead of all being
  moved to the device up front.
- Bootstrap confidence intervals resample by utterance, not by word.
- A fresh run in an existing checkpoint directory no longer inherits epoch
  numbers from old files.
- Precompute wrote one syscall per float.
- Loading a checkpoint into a different architecture succeeded and then
  panicked on first use; it is now an error. A never-used module saved
  different weights than it used.
- Resuming into a new output directory, or with a mismatched model, lost
  or overwrote the earlier run's state.
- Feature-mode and scoring-head width mismatches return errors instead of
  panicking.
- `Phoneme::AH` was mapped to /ɑ/ (ARPABET AA); the IPA layer is gone.

### Changed

- CMU dictionary is embedded as text and parsed on first use; `build.rs` and
  `bitcode` are gone.
- Typed `Error` enum; I/O errors carry the path.
- Cache format version 2 stores `feature_dim`; version 1 files still load.
- One training loop for LibriSpeech and cached features.
- The CMU dictionary licence is included and declared in `Cargo.toml`.
- Scoring heads are saved as `scoring-head-{best,final}` with a required
  `scoring-head-config.json`; heads written by 0.1.0 do not load. Backbone
  checkpoints and version 1 feature caches still load.
- Fine-tuning metrics go to `finetune-metrics.jsonl`.
- Resuming training continues the learning-rate schedule.
- `rust-version` is 1.92 (required by Burn 0.21; 1.87 never built).

## 0.1.0 - 2026-04-02

Initial release.

- SpeechAligner model: CNN + SE + attention, about 1.7M parameters, 3 output heads
- CTC training loop with LibriSpeech data loading
- SpeechOcean762 fine-tuning with pronunciation scoring head
- MFCC and log-mel feature extraction
- CMU Dict G2P embedded at compile time
- OOV fallback path for training data cleanup
- Precomputed feature cache for faster ablation runs
- Spearman correlation evaluation with bootstrap confidence intervals
