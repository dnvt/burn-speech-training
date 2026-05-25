# Contributing

Issues and pull requests are welcome.

This repo is meant to be useful as a reference, not a polished training
framework. The best contributions make it easier to run, inspect, and adapt.

## Good first areas

- Make the quick-start example work with a small WAV file.
- Document the precomputed binary feature format.
- Add dataset loaders for TIMIT or Common Voice.
- Improve progress reporting during training.
- Add backend notes for CUDA and WGPU.

## Development

```bash
cargo test --features ndarray
cargo clippy --features ndarray -- -D warnings
cargo fmt --check
```

## AI-assisted work

I use AI tools for drafting, review, refactoring ideas, and test ideation. All
code and documentation in this repository is reviewed, edited, and tested by me
before release.

Contributions that use AI tools are welcome if you can explain the change, keep
the diff focused, run the relevant tests, and disclose substantial AI assistance
in the pull request description.

## Ground rules

- Keep claims tied to commands, data, or experiments.
- Keep examples runnable without private files.
- Do not add model weights or private training artifacts.

## License

Contributions are accepted under the same MIT OR Apache-2.0 license as the
project.
