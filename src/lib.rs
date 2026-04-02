//! # burn-speech-training
//!
//! End-to-end speech model training pipeline built on [Burn](https://burn.dev).
//!
//! MFCC feature extraction, CTC loss training, LibriSpeech data loading,
//! and evaluation against human-labeled pronunciation data.
//!
//! ## Architecture
//!
//! `SpeechAligner` is a CNN+SE+Attention model (~1.7M parameters) with three
//! output heads: phoneme classification, boundary detection, and CTC alignment.
//!
//! ## Quick Start
//!
//! ```bash
//! cargo run --example train_small --features ndarray --release
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use burn_speech_training::model::SpeechAlignerConfig;
//! use burn_speech_training::mfcc::FeatureMode;
//!
//! // Default config: 39-dim MFCC input, 42 phoneme classes, ~1.7M params
//! let config = SpeechAlignerConfig::default();
//! assert_eq!(config.input_dim, 39);
//! assert_eq!(config.num_classes, 42);
//!
//! // Or configure for log-mel features
//! let config = SpeechAlignerConfig::for_feature_mode(FeatureMode::LogMel80);
//! assert_eq!(config.input_dim, 80);
//! ```

pub mod attention;
pub mod conv_block;
pub mod dataset;
pub mod error;
pub mod evaluate;
pub mod finetune;
pub mod g2p;
pub mod loss;
pub mod mfcc;
pub mod model;
pub mod phoneme_map;
pub mod precompute;
pub mod se_block;
pub mod train;
pub mod ui;
