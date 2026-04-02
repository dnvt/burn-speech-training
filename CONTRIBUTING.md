# Contributing

Issues and pull requests are welcome.

## Good areas to contribute

- New dataset loaders (TIMIT, Common Voice, etc.)
- Additional feature extraction modes
- Training visualization / progress reporting
- Documentation improvements
- Bug fixes

## Development

```bash
# Run tests
cargo test --features ndarray

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --features ndarray -- -D warnings
```

## License

Contributions are accepted under the same MIT OR Apache-2.0 license as the
project.
