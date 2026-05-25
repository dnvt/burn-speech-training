# Datasets And Provenance

This repo tries to keep data claims explicit. It does not include full speech
datasets, model weights, private audio, or private transcripts.

## Synthetic Quickstart Data

The `train_small` example generates a small synthetic batch inside the example.
It is useful for checking that the Burn model, CTC loss, autodiff, optimizer
step, and inference shapes are wired together.

It is not evidence of real speech quality.

## LibriSpeech

- Source: https://www.openslr.org/12/
- Used for: real speech alignment training examples.
- Included in this repo: no.
- Notes: download the dataset yourself and follow its license terms. The README
  examples assume an extracted LibriSpeech directory on your machine.

## SpeechOcean762

- Source: https://www.openslr.org/83/
- Used for: pronunciation scoring experiments and Spearman correlation against
  human word-level scores.
- Included in this repo: no.
- Notes: the reported score plateau should be read in the context of this
  dataset and split. The experiment log records the runs and limitations.

## CMU Pronouncing Dictionary

- Source: Carnegie Mellon University Pronouncing Dictionary.
- Used for: transcript to ARPABET phoneme lookup in the G2P path.
- Included in this repo: a compact generated lookup is built into the crate.
- Notes: unknown words use the crate's fallback path, so G2P quality is a known
  limitation for out-of-vocabulary text.

## Private Or User Audio

Do not commit private learner audio, proprietary transcripts, or customer data to
this repository. If you need to discuss a bug involving private audio, reduce it
to a synthetic fixture or open a non-sensitive issue asking for a private contact
path.

## Claim Checklist

When adding a new result, include:

- dataset and split;
- command or script;
- commit hash;
- hardware;
- random seed if relevant;
- metric definition;
- whether the input was synthetic, public data, or private data.
