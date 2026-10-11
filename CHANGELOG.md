# Changelog

## Unreleased

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
- `Phoneme::AH` was mapped to /ɑ/ (ARPABET AA); the IPA layer is gone.

### Changed

- CMU dictionary is embedded as text and parsed on first use; `build.rs` and
  `bitcode` are gone.
- Typed `Error` enum; I/O errors carry the path.
- Cache format version 2 stores `feature_dim`; version 1 files still load.
- One training loop for LibriSpeech and cached features.
- The CMU dictionary licence is included and declared in `Cargo.toml`.

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
