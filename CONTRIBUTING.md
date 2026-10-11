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

Contributions are accepted under the project's MIT OR Apache-2.0 license.
