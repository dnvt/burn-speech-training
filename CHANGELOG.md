# Changelog

## 0.1.0 — 2026-04-02

Initial release.

- SpeechAligner model: CNN+SE+Attention, ~1.7M params, 3 output heads
- CTC training loop with LibriSpeech data loading
- SpeechOcean762 fine-tuning with pronunciation scoring head
- MFCC (39-dim) and log-mel (80-dim) feature extraction
- CMU Dict G2P with 135K words (embedded at compile time)
- 8-stage OOV fallback for training data quality
- Precomputed feature cache for fast ablation
- Spearman correlation evaluation with bootstrap CI
- 137 tests
