# Model Card: SpeechAligner Reference Model

This is a lightweight model card for the reference model in
`burn-speech-training`. It is meant to make the current limits clear, not to
present the model as production-ready.

## Model

- Name: `SpeechAligner`
- Framework: Burn 0.21
- Default shape: CNN + squeeze-and-excitation blocks + self-attention + CTC head
- Default size: about 1.7M parameters
- Quickstart size: about 122K parameters

## Intended Use

- Learning how to wire a speech training loop in Rust with Burn.
- Inspecting a small CTC-based alignment path.
- Reusing pieces such as feature extraction, batching, model structure, or
  experiment logging.
- Running controlled experiments on pronunciation scoring ideas.

## Not Intended For

- Production ASR.
- Certified language assessment.
- Medical, hiring, immigration, or high-stakes scoring.
- Claims about learner ability without further validation.
- Real-time inference guarantees.

## Training And Evaluation Data

- Synthetic quickstart data: smoke test only.
- LibriSpeech: used by the real-data training path.
- SpeechOcean762: used for pronunciation scoring experiments.
- CMUdict: used for transcript to phoneme lookup.

See `docs/datasets.md` for provenance and inclusion notes.

## Reported Result

The best reported pronunciation-scoring result in the experiment log is about
`0.292` Spearman correlation against SpeechOcean762 word-level human scores.

That number is a limitation marker, not a victory claim. The evidence points to
representation and data limits more than another small loss-function tweak.

## Known Failure Modes

- MFCC/log-mel features may be too weak for robust pronunciation ranking.
- CTC alignment loss can interfere with scoring fine-tuning in this setup.
- SpeechOcean762 word scores are heavily imbalanced toward perfect scores.
- Out-of-vocabulary words depend on fallback phoneme handling.
- GPU features are hardware-dependent and not fully covered by CI.

## Reproducibility Notes

For a quick smoke test:

```bash
cargo run --example train_small --features ndarray --release
```

For experiment claims, prefer linking to `docs/experiment-log.md` and include
the dataset split, command, commit, hardware, and metric definition.
