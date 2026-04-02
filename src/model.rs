//! `SpeechAligner` model: CNN + SE + Attention + triple head.
//!
//! 4 `ConvSE` blocks (39->64->128->256->512) + self-attention + 3 output
//! heads (phoneme classification, boundary detection, CTC alignment).
//! ~1.7M parameters. Config is adjustable for scaling experiments.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use crate::attention::{SelfAttentionConfig, SelfAttentionLayer};
use crate::conv_block::{ConvSEBlock, ConvSEBlockConfig};
use crate::error::{Error, Result};
use crate::mfcc::FeatureMode;

/// Configuration for the `SpeechAligner` model.
#[derive(Debug, Clone)]
pub struct SpeechAlignerConfig {
    /// Input feature dimension (e.g., 39 for MFCC).
    pub input_dim: usize,
    /// Number of output classes (phonemes + blank for CTC).
    pub num_classes: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// SE reduction ratio (passed through to `ConvSEBlockConfig`).
    pub se_reduction: usize,
    /// Channel widths for the 4 conv blocks.
    pub channels: [usize; 4],
    /// Conv kernel size.
    pub kernel_size: usize,
}

impl Default for SpeechAlignerConfig {
    fn default() -> Self {
        Self {
            input_dim: 39,
            num_classes: 42,
            n_heads: 8,
            se_reduction: 8,
            channels: [64, 128, 256, 512],
            kernel_size: 3,
        }
    }
}

impl SpeechAlignerConfig {
    /// Build a config for the requested input feature representation.
    pub fn for_feature_mode(feature_mode: FeatureMode) -> Self {
        Self::with_input_dim(feature_mode.feature_dim())
    }

    /// Build a config with an explicit input feature dimension.
    pub fn with_input_dim(input_dim: usize) -> Self {
        Self {
            input_dim,
            ..Self::default()
        }
    }

    /// Initialize the model on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<SpeechAligner<B>> {
        let d_model = self.channels[3]; // 512
        if self.n_heads == 0 {
            return Err(Error::config("SpeechAlignerConfig.n_heads must be > 0"));
        }
        if !d_model.is_multiple_of(self.n_heads) {
            return Err(Error::config(format!(
                "SpeechAlignerConfig.channels[3] ({d_model}) must be divisible by n_heads ({})",
                self.n_heads
            )));
        }

        let mk_conv = |in_c, out_c| {
            ConvSEBlockConfig::new(in_c, out_c)
                .with_kernel_size(self.kernel_size)
                .with_se_reduction(self.se_reduction)
                .init(device)
        };

        let conv1 = mk_conv(self.input_dim, self.channels[0]);
        let conv2 = mk_conv(self.channels[0], self.channels[1]);
        let conv3 = mk_conv(self.channels[1], self.channels[2]);
        let conv4 = mk_conv(self.channels[2], self.channels[3]);

        let attention = SelfAttentionConfig::new(d_model, self.n_heads).init(device)?;

        let phoneme_head = LinearConfig::new(d_model, self.num_classes).init(device);
        let boundary_head = LinearConfig::new(d_model, 1).init(device);
        let ctc_head = LinearConfig::new(d_model, self.num_classes).init(device);

        Ok(SpeechAligner {
            conv1,
            conv2,
            conv3,
            conv4,
            attention,
            phoneme_head,
            boundary_head,
            ctc_head,
        })
    }
}

/// `SpeechAligner` model output containing all three head outputs.
#[derive(Debug)]
pub struct SpeechAlignerOutput<B: Backend> {
    /// Frame-level encoder features after self-attention: `[batch, time,
    /// d_model]`. Used by the SO762 pronunciation scoring head during
    /// fine-tuning.
    pub frame_features: Tensor<B, 3>,
    /// Phoneme classification logits: `[batch, time, num_classes]`.
    pub phoneme_logits: Tensor<B, 3>,
    /// Boundary detection logits (raw, pre-sigmoid): `[batch, time, 1]`.
    /// Apply sigmoid at the consumer site (loss or inference).
    pub boundary_logits: Tensor<B, 3>,
    /// CTC log-probabilities: `[time, batch, num_classes]` (transposed for
    /// CTC). Post `log_softmax`.
    pub ctc_log_probs: Tensor<B, 3>,
    /// Raw CTC logits: `[time, batch, num_classes]` (pre-softmax).
    /// Used for logit-based GOP scoring (`GOPMaxLogit`, `GOPMargin`).
    pub ctc_logits: Tensor<B, 3>,
}

/// `SpeechAligner`: CNN + SE + Attention + triple head model (~1.7M params).
///
/// Forward pipeline:
/// 1. Transpose input `[B, T, D]` -> `[B, D, T]` (channels-first for Conv1d)
/// 2. 4x `ConvSE` blocks: 39 -> 64 -> 128 -> 256 -> 512
/// 3. Transpose back `[B, 512, T]` -> `[B, T, 512]` (sequence-first for
///    attention)
/// 4. Self-attention with residual
/// 5. Three output heads: phoneme, boundary, CTC
#[derive(Module, Debug)]
pub struct SpeechAligner<B: Backend> {
    conv1: ConvSEBlock<B>,
    conv2: ConvSEBlock<B>,
    conv3: ConvSEBlock<B>,
    conv4: ConvSEBlock<B>,
    attention: SelfAttentionLayer<B>,
    phoneme_head: Linear<B>,
    boundary_head: Linear<B>,
    ctc_head: Linear<B>,
}

impl<B: Backend> SpeechAligner<B> {
    /// Forward pass through the full model.
    ///
    /// Input: `[batch, time, features]` (e.g., `[B, T, 39]` MFCC features)
    /// Returns: [`SpeechAlignerOutput`] with three head outputs.
    pub fn forward(&self, x: Tensor<B, 3>) -> SpeechAlignerOutput<B> {
        self.forward_inner(x, None)
    }

    /// Forward pass with a pad mask for variable-length batches.
    ///
    /// `mask_pad` uses shape `[batch, time]` with `true` for padded frames.
    pub fn forward_with_pad_mask(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> SpeechAlignerOutput<B> {
        self.forward_inner(x, Some(mask_pad))
    }

    fn forward_inner(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
    ) -> SpeechAlignerOutput<B> {
        // 1. Transpose to channels-first: [B, T, D] -> [B, D, T]
        let x = x.swap_dims(1, 2);

        // 2. ConvSE blocks: [B, 39, T] -> [B, 64, T] -> ... -> [B, 512, T]
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);
        let x = self.conv4.forward(x);

        // 3. Transpose back to sequence-first: [B, 512, T] -> [B, T, 512]
        let x = x.swap_dims(1, 2);

        // 4. Self-attention with residual: [B, T, 512] -> [B, T, 512]
        let x = match mask_pad {
            Some(mask_pad) => self.attention.forward_with_pad_mask(x, mask_pad),
            None => self.attention.forward(x),
        };

        // 5. Output heads (all raw logits — activations applied downstream)
        let frame_features = x.clone();
        let phoneme_logits = self.phoneme_head.forward(x.clone());
        let boundary_logits = self.boundary_head.forward(x.clone());
        let ctc_logits = self.ctc_head.forward(x);

        // CTC needs [time, batch, classes] with log-softmax
        let ctc_transposed = ctc_logits.swap_dims(0, 1);
        let ctc_log_probs = log_softmax(ctc_transposed.clone(), 2);

        SpeechAlignerOutput {
            frame_features,
            phoneme_logits,
            boundary_logits,
            ctc_log_probs,
            ctc_logits: ctc_transposed,
        }
    }
}

/// Estimate a word's frame span by partitioning the utterance proportionally.
///
/// This keeps the entire frame axis covered without floating-point rounding
/// drift and matches the heuristic used by the current GOP evaluator.
pub fn proportional_word_frame_span(
    num_frames: usize,
    num_words: usize,
    word_position: usize,
) -> Option<(usize, usize)> {
    if num_frames == 0 || num_words == 0 || word_position >= num_words {
        return None;
    }

    let start = word_position.saturating_mul(num_frames) / num_words;
    let end = (word_position + 1).saturating_mul(num_frames) / num_words;

    if start >= end || end > num_frames {
        return None;
    }

    Some((start, end))
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::activation::sigmoid;
    use burn::tensor::{Distribution, Tensor};

    use super::*;
    use crate::error::Error;

    type B = NdArray<f32>;

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn init_model(config: &SpeechAlignerConfig, device: &NdArrayDevice) -> SpeechAligner<B> {
        match config.init(device) {
            Ok(model) => model,
            Err(err) => panic!("SpeechAligner init should succeed: {err}"),
        }
    }

    #[test]
    fn test_speech_aligner_forward_shape() {
        let device = NdArrayDevice::default();
        let config = SpeechAlignerConfig::default();
        let model = init_model(&config, &device);

        // Input: [batch=2, time=100, features=39]
        let input: Tensor<B, 3> =
            Tensor::random([2, 100, 39], Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        // Phoneme head: [batch=2, time=100, classes=42]
        assert_eq!(output.phoneme_logits.dims(), [2, 100, 42]);
        assert_eq!(output.frame_features.dims(), [2, 100, 512]);

        // Boundary head: [batch=2, time=100, 1]
        assert_eq!(output.boundary_logits.dims(), [2, 100, 1]);

        // CTC head: [time=100, batch=2, classes=42]
        assert_eq!(output.ctc_log_probs.dims(), [100, 2, 42]);
    }

    #[test]
    fn test_ctc_output_is_log_probs() {
        let device = NdArrayDevice::default();
        let config = SpeechAlignerConfig::default();
        let model = init_model(&config, &device);

        let input: Tensor<B, 3> =
            Tensor::random([1, 50, 39], Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        // All CTC log-probs should be <= 0
        let max_val: f32 = output
            .ctc_log_probs
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert!(
            max_val <= 0.001,
            "CTC log-probs should be <= 0, got {max_val}"
        );
    }

    #[test]
    fn test_boundary_output_logits_to_probs() {
        let device = NdArrayDevice::default();
        let config = SpeechAlignerConfig::default();
        let model = init_model(&config, &device);

        let input: Tensor<B, 3> =
            Tensor::random([1, 50, 39], Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        // Boundary logits are raw — apply sigmoid to get probs
        let probs = sigmoid(output.boundary_logits);

        let min_val: f32 = probs
            .clone()
            .min()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let max_val: f32 = probs
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        assert!(
            min_val >= 0.0,
            "boundary probs should be >= 0, got {min_val}"
        );
        assert!(
            max_val <= 1.0,
            "boundary probs should be <= 1, got {max_val}"
        );
    }

    #[test]
    fn test_parameter_count() {
        let device = NdArrayDevice::default();
        let config = SpeechAlignerConfig::default();
        let model = init_model(&config, &device);

        let count = model.num_params();
        // Expected ~1.7M params. Allow a range for implementation details.
        assert!(
            count > 1_000_000,
            "model should have >1M params, got {count}"
        );
        assert!(
            count < 3_000_000,
            "model should have <3M params (not the 6.4M Conv2d variant), got {count}"
        );
    }

    #[test]
    fn test_forward_with_pad_mask_shape() {
        let device = NdArrayDevice::default();
        let config = SpeechAlignerConfig::default();
        let model = init_model(&config, &device);
        let input: Tensor<B, 3> =
            Tensor::random([2, 32, 39], Distribution::Normal(0.0, 1.0), &device);
        let mask_pad = Tensor::<B, 2, burn::tensor::Int>::zeros([2, 32], &device)
            .slice_assign([0..2, 28..32], Tensor::ones([2, 4], &device))
            .equal_elem(1);

        let output = model.forward_with_pad_mask(input, mask_pad);

        assert_eq!(output.phoneme_logits.dims(), [2, 32, 42]);
        assert_eq!(output.frame_features.dims(), [2, 32, 512]);
        assert_eq!(output.boundary_logits.dims(), [2, 32, 1]);
        assert_eq!(output.ctc_log_probs.dims(), [32, 2, 42]);
    }

    #[test]
    fn test_proportional_word_frame_span_partitions_all_frames() {
        let spans = (0..3)
            .map(|word_position| proportional_word_frame_span(10, 3, word_position))
            .collect::<Vec<_>>();

        assert_eq!(spans, vec![Some((0, 3)), Some((3, 6)), Some((6, 10))]);
    }

    #[test]
    fn test_invalid_head_count_returns_config_error() {
        let device = NdArrayDevice::default();
        let config = SpeechAlignerConfig {
            n_heads: 0,
            ..SpeechAlignerConfig::default()
        };

        let err = match config.init::<B>(&device) {
            Ok(_) => panic!("invalid config should be rejected"),
            Err(err) => err,
        };

        assert!(matches!(err, Error::Config { .. }));
    }
}
