# Contributing

Issues and pull requests are welcome. This is a reference implementation, so
changes that make it easier to run, read or adapt are the most useful.

## Before opening a PR

```bash
cargo fmt --check
cargo clippy --all-targets --features experiments -- -D warnings
cargo test --features experiments
```

- Keep examples runnable without private files.
- Back any new result in the docs with the dataset split, command and commit.
- Don't commit model weights, private audio or transcripts.
- If you used AI tools substantially, say so in the PR description.

## Ideas

- Dataset loaders for Common Voice or TIMIT.
- Notes from running the CUDA or WGPU backends on real hardware.
- Resampling so non-16 kHz audio can be used directly.

## Releasing

Bump `version` in `Cargo.toml`, rename the CHANGELOG's "Unreleased" section to
the new version and date, merge, then push a `vX.Y.Z` tag. The release workflow
checks the tag against `Cargo.toml`, creates the GitHub release from the
CHANGELOG and publishes to crates.io.

Contributions are accepted under the project's MIT OR Apache-2.0 license.
