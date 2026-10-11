# Controlled rerun of the corrected Round 6 experiments

The corrected distillation gradient and rank kernel need a new comparison
against the same baseline. A production resampler change does not affect this
training path: this crate requires audio already at 16 kHz.

`examples/rerun_round6.rs` preflights one condition by default. It verifies the
checkpoint architecture and loading, all audio files at 16 kHz, disjoint splits,
and complete, nonempty, finite training teacher features when needed. Experiment
outputs are created only when invoked with `--run`. Existing result directories are rejected. Each
invocation is bounded to 20 epochs and can use `max_samples` for a pilot.

## Historical recipe and inputs

The source recipe is Cadence's `deploy/train-ops/run-path-a-wave-d.sh` and
`run-path-a-wave-b.sh`: 20 epochs, LR 0.0001, warmup 3, batch size 16,
CTC weight 0, MSE, mean pooling, 256-dimensional head, eval every epoch.
Use the same seed and inputs for every condition:

| Condition | Distillation weight | Rank weight / tau |
|---|---|---|
| `baseline` | off | off |
| `distill01` (6A) | 0.1 | off |
| `distill05` (6B) | 0.5 | off |
| `rank` (4D / standalone 6d) | off | 0.1 / 0.15 |
| `combined` (6C) | 0.3 | 0.1 / 0.15 |

The historical training labels are `word_scores_dev.json`, not the later
official-split training manifest. They contain 25,480 rows / 4,180 utterances;
the standard loader filters three invalid word positions, leaving 25,477 usable
words. `word_scores_holdout.json` contains 6,336 rows / 820 utterances, leaving
6,335 usable words after one invalid position. There is no audio overlap.

Expected SHA256 digests of those original files:

```text
2ffbe08923eb02041fc3bf82dfaee6f82b85fda293413f4caf08c0854b4311b7  word_scores_dev.json
2cec181bf466e19015f9917714a62cc91c3120f4dd1736814fe511013a63a857  word_scores_holdout.json
```

Their `audio_file` paths are relative to Cadence's `data` directory, under
`fixtures/audio/speechocean762/WAVE/`. Record a hash of the recovered base
checkpoint and teacher manifest, the converter commit, backend/hardware,
dependency lockfile, and source commit before starting a full run. Preserve the
original files; archive recovery and checkpoint conversion are separate steps.
This compares conditions in the corrected v0.3 implementation. Report backend,
conversion, and random initialization differences when comparing historical
numbers; matching the recipe alone does not promise bitwise reproduction.

The original remote base is
`/srv/prosody/data/checkpoints/phase-1c/1c-resume-lr0003/prosody-align-final`.
Version 0.3 cannot read checkpoints from before 0.3. A verified conversion must
retain the original weights and produce a compatible `.bpk` plus
`model-config.json`; a newly trained or synthetic checkpoint is not a substitute.

The original cache is
`/srv/prosody/data/distillation/wav2vec2-features`, produced with
`facebook/wav2vec2-large-960h`. It contains version-1 `W2V2` binaries with
1024-dimensional float features at 50 frames/s and a `manifest.json` mapping
audio paths to binaries. This is a different teacher from the public
`mrrubino` phoneme scorer. Do not silently generate replacement features with a
different model revision or preprocessing.

## Running

Copy `examples/round6-rerun.json` outside the repository, replace the paths,
and verify the input digests with `shasum -a 256`. For each condition use a fresh
output directory. The first command only preflights:

```bash
BURN_DEVICE=flex cargo run --release --no-default-features --features flex,experiments \
  --example rerun_round6 -- /path/to/run.json
```

For a resource estimate, start with one baseline epoch and a declared training
word limit (for example `epochs: 1`, `max_samples: 64`). Compare runtime and peak
memory before setting a full-run budget. For a full historical cohort use
`max_samples: null`. To execute the configured condition add `--run`:

```bash
BURN_DEVICE=flex cargo run --release --no-default-features --features flex,experiments \
  --example rerun_round6 -- /path/to/run.json --run
```

Run the five conditions with identical recipe fields and inputs. Save
`Cargo.lock` beside the results. The chosen checkpoint's holdout rho is a model
selection metric. The historical recipe has no independent test split. If an
independent test split is supplied through `test`, it must be disjoint; it is
evaluated only after selecting the checkpoint. A pilot or a replacement split
does not reproduce the published 0.292 result. Confirm any apparent improvement
with additional matched seeds and a fixed independent test set.

Focused regression checks:

```bash
BURN_DEVICE=flex cargo test --no-default-features --features flex,experiments \
  --lib distillation_gradient_reaches_the_student
BURN_DEVICE=flex cargo test --no-default-features --features flex,experiments \
  --lib rank_regularization_has_usable_range
```

## Local recovery status (2026-10-11)

The exact label manifests are present in the local Cadence checkout and match
the digests above. The original full audio directory, trained base checkpoint,
and teacher cache were not found in the local developer/task/runner/temp/cache
locations inspected. A later 920-clip adult intake is a separate split and does
not cover the original cohort. The only local standalone `.bpk` files found are
synthetic 13-dimensional quickstart checkpoints. No real training or paid job
has been started; full reruns await recovery of the original missing assets and
a verified checkpoint conversion.

Read-only GCP checks could not inspect `prosody-train-ops` in
`prosody-studio/us-east4-b`: the existing local principals lack
`compute.instances.get` (the active principal also lacks `compute.instances.list`).
The VM name has no resolvable direct SSH endpoint in this local configuration.
Recovery therefore needs an authorized read/SSH connection to the existing VM
or an accessible archive of the original assets. No VM was started or created.
