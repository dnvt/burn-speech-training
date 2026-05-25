# Changelog

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
