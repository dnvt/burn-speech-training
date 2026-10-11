//! `SpeechAligner`: a four-block 1-D convolutional encoder with
//! squeeze-and-excitation, one pre-norm self-attention layer and three linear
//! output heads (phoneme, boundary, CTC).

mod layers;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Bool, Device, Tensor};
use serde::{Deserialize, Serialize};

use self::layers::{ConvSEBlock, SelfAttentionLayer};
use crate::error::{Error, Result};
use crate::mfcc::FeatureMode;

/// Hyperparameters of a [`SpeechAligner`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechAlignerConfig {
    /// Feature dimension of each input frame.
    pub input_dim: usize,
    /// Output classes of the phoneme and CTC heads, including the CTC blank.
    pub num_classes: usize,
    /// Attention heads; must divide `channels[3]`.
    pub n_heads: usize,
    /// Squeeze-and-excitation bottleneck ratio (`channels / se_reduction`).
    pub se_reduction: usize,
    /// Output channels of the four conv blocks; the last is the model width.
    pub channels: [usize; 4],
    /// Temporal kernel size of every conv block.
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
    /// Default config with the input dimension of `feature_mode`.
    pub fn for_feature_mode(feature_mode: FeatureMode) -> Self {
        Self::with_input_dim(feature_mode.feature_dim())
    }

    /// Default config with an explicit input dimension.
    pub fn with_input_dim(input_dim: usize) -> Self {
        Self {
            input_dim,
            ..Self::default()
        }
    }

    /// Initialise a model with random weights on `device`.
    pub fn init(&self, device: &Device) -> Result<SpeechAligner> {
        let d_model = self.channels[3];
        if self.n_heads == 0 || !d_model.is_multiple_of(self.n_heads) {
            return Err(Error::config(format!(
                "SpeechAlignerConfig.n_heads ({}) must be > 0 and divide channels[3] ({d_model})",
                self.n_heads
            )));
        }

        let [c1, c2, c3, c4] = self.channels;
        let block = |in_channels, out_channels| {
            ConvSEBlock::new(
                in_channels,
                out_channels,
                self.kernel_size,
                self.se_reduction,
                device,
            )
        };

        Ok(SpeechAligner {
            conv1: block(self.input_dim, c1),
            conv2: block(c1, c2),
            conv3: block(c2, c3),
            conv4: block(c3, c4),
            attention: SelfAttentionLayer::new(d_model, self.n_heads, device),
            phoneme_head: LinearConfig::new(d_model, self.num_classes).init(device),
            boundary_head: LinearConfig::new(d_model, 1).init(device),
            ctc_head: LinearConfig::new(d_model, self.num_classes).init(device),
        })
    }
}

/// Outputs of [`SpeechAligner::forward`].
///
/// Values at padded frames are unspecified and must be masked by the consumer.
#[derive(Debug)]
pub struct SpeechAlignerOutput {
    /// Encoder output after self-attention: `[batch, time, channels[3]]`.
    pub frame_features: Tensor<3>,
    /// Phoneme logits: `[batch, time, num_classes]`.
    pub phoneme_logits: Tensor<3>,
    /// Boundary logits (pre-sigmoid): `[batch, time, 1]`.
    pub boundary_logits: Tensor<3>,
    /// CTC log-probabilities: `[time, batch, num_classes]`.
    pub ctc_log_probs: Tensor<3>,
    /// CTC logits (pre-softmax): `[time, batch, num_classes]`.
    pub ctc_logits: Tensor<3>,
}

/// Speech encoder with phoneme, boundary and CTC heads.
///
/// Input frames `[batch, time, input_dim]` pass through four conv blocks
/// (Conv1d, `LayerNorm`, `ReLU`, squeeze-and-excitation), one pre-norm
/// self-attention layer with a residual connection, and three linear heads
/// applied per frame. There is no positional encoding; frame order is only
/// seen by the convolutions.
#[derive(Module, Debug)]
pub struct SpeechAligner {
    conv1: ConvSEBlock,
    conv2: ConvSEBlock,
    conv3: ConvSEBlock,
    conv4: ConvSEBlock,
    attention: SelfAttentionLayer,
    phoneme_head: Linear,
    boundary_head: Linear,
    ctc_head: Linear,
}

impl SpeechAligner {
    /// Run the model on a batch in which every frame is real.
    pub fn forward(&self, x: Tensor<3>) -> SpeechAlignerOutput {
        self.forward_masked(x, None)
    }

    /// Run the model on a right-padded batch.
    ///
    /// `mask_pad` is `[batch, time]` and `true` marks padded frames. Padded
    /// frames are excluded from SE pooling and attention, so outputs on real
    /// frames do not depend on how much padding a batch contains.
    pub fn forward_with_pad_mask(
        &self,
        x: Tensor<3>,
        mask_pad: Tensor<2, Bool>,
    ) -> SpeechAlignerOutput {
        self.forward_masked(x, Some(mask_pad))
    }

    fn forward_masked(
        &self,
        x: Tensor<3>,
        mask_pad: Option<Tensor<2, Bool>>,
    ) -> SpeechAlignerOutput {
        let keep = mask_pad
            .clone()
            .map(|mask| mask.bool_not().float().unsqueeze_dim::<3>(1));

        let mut x = x.swap_dims(1, 2);
        if let Some(keep) = &keep {
            x = x * keep.clone();
        }
        let x = self.conv1.forward(x, keep.as_ref());
        let x = self.conv2.forward(x, keep.as_ref());
        let x = self.conv3.forward(x, keep.as_ref());
        let x = self.conv4.forward(x, keep.as_ref());

        let frame_features = self.attention.forward(x.swap_dims(1, 2), mask_pad);

        let phoneme_logits = self.phoneme_head.forward(frame_features.clone());
        let boundary_logits = self.boundary_head.forward(frame_features.clone());
        let ctc_logits = self
            .ctc_head
            .forward(frame_features.clone())
            .swap_dims(0, 1);
        let ctc_log_probs = log_softmax(ctc_logits.clone(), 2);

        SpeechAlignerOutput {
            frame_features,
            phoneme_logits,
            boundary_logits,
            ctc_log_probs,
            ctc_logits,
        }
    }
}

/// Estimate a word's frame span by partitioning the utterance proportionally.
///
/// Spans of consecutive words tile `0..num_frames` exactly, with no
/// floating-point rounding drift.
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
    use burn::tensor::{Distribution, Int, Tensor, Tolerance};

    use super::*;

    /// README "Tiny" widths: debug-build convolutions at the default widths
    /// take tens of seconds per forward pass.
    fn tiny_config() -> SpeechAlignerConfig {
        SpeechAlignerConfig {
            channels: [16, 32, 64, 128],
            n_heads: 2,
            ..SpeechAlignerConfig::default()
        }
    }

    fn init_model(config: &SpeechAlignerConfig) -> SpeechAligner {
        config
            .init(&Device::default())
            .expect("SpeechAligner init should succeed")
    }

    fn random_input(batch: usize, time: usize, dim: usize) -> Tensor<3> {
        let device = Device::default();
        Tensor::random([batch, time, dim], Distribution::Normal(0.0, 1.0), &device)
    }

    /// Pad mask `[lengths.len(), max_time]` that is `true` past each length.
    fn pad_mask(lengths: &[usize], max_time: usize) -> Tensor<2, Bool> {
        let device = Device::default();
        let mut mask = Tensor::<2, Int>::zeros([lengths.len(), max_time], &device);
        for (row, &len) in lengths.iter().enumerate() {
            if len < max_time {
                mask = mask.slice_assign(
                    [row..row + 1, len..max_time],
                    Tensor::ones([1, max_time - len], &device),
                );
            }
        }
        mask.equal_elem(1)
    }

    fn assert_close(actual: Tensor<3>, expected: Tensor<3>, tolerance: f32) {
        actual
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(tolerance));
    }

    #[test]
    fn forward_output_shapes() {
        let model = init_model(&tiny_config());

        for output in [
            model.forward(random_input(2, 32, 39)),
            model.forward_with_pad_mask(random_input(2, 32, 39), pad_mask(&[32, 28], 32)),
        ] {
            assert_eq!(output.frame_features.dims(), [2, 32, 128]);
            assert_eq!(output.phoneme_logits.dims(), [2, 32, 42]);
            assert_eq!(output.boundary_logits.dims(), [2, 32, 1]);
            assert_eq!(output.ctc_log_probs.dims(), [32, 2, 42]);
            assert_eq!(output.ctc_logits.dims(), [32, 2, 42]);
        }
    }

    #[test]
    fn ctc_log_probs_are_normalised() {
        let model = init_model(&tiny_config());
        let output = model.forward(random_input(1, 20, 39));

        let total_prob = output.ctc_log_probs.exp().sum_dim(2);
        let expected = Tensor::ones([20, 1, 1], &Device::default());
        assert_close(total_prob, expected, 1e-4);
    }

    #[test]
    fn padding_does_not_change_outputs_on_real_frames() {
        let device = Device::default();
        let model = init_model(&tiny_config());
        let [short_len, long_len, dim] = [20, 32, 39];

        let short = random_input(1, short_len, dim);
        let short_padded = Tensor::cat(
            vec![
                short.clone(),
                Tensor::zeros([1, long_len - short_len, dim], &device),
            ],
            1,
        );
        let batch = Tensor::cat(vec![short_padded, random_input(1, long_len, dim)], 0);

        let alone = model.forward(short);
        let batched =
            model.forward_with_pad_mask(batch, pad_mask(&[short_len, long_len], long_len));

        let real = |t: Tensor<3>| t.slice([0..1, 0..short_len]);
        assert_close(real(batched.frame_features), alone.frame_features, 1e-4);
        assert_close(real(batched.phoneme_logits), alone.phoneme_logits, 1e-4);
        assert_close(real(batched.boundary_logits), alone.boundary_logits, 1e-4);
        assert_close(
            batched.ctc_log_probs.slice([0..short_len, 0..1]),
            alone.ctc_log_probs,
            1e-4,
        );
    }

    #[test]
    fn empty_pad_mask_matches_unmasked_forward() {
        let model = init_model(&tiny_config());
        let input = random_input(2, 24, 39);

        let unmasked = model.forward(input.clone());
        let masked = model.forward_with_pad_mask(input, pad_mask(&[24, 24], 24));

        assert_close(masked.frame_features, unmasked.frame_features, 1e-5);
        assert_close(masked.ctc_log_probs, unmasked.ctc_log_probs, 1e-5);
    }

    #[test]
    fn default_config_parameter_count() {
        let model = init_model(&SpeechAlignerConfig::default());
        assert_eq!(model.num_params(), 1_709_837);
    }

    #[test]
    fn readme_tiny_and_large_configs_initialise() {
        let large = SpeechAlignerConfig {
            channels: [128, 256, 512, 1024],
            n_heads: 16,
            ..SpeechAlignerConfig::default()
        };
        for config in [tiny_config(), large] {
            let model = init_model(&config);
            let output = model.forward(random_input(1, 4, config.input_dim));
            assert_eq!(output.frame_features.dims(), [1, 4, config.channels[3]]);
        }
    }

    #[test]
    fn invalid_head_count_is_a_config_error() {
        for n_heads in [0, 3] {
            let config = SpeechAlignerConfig {
                n_heads,
                ..SpeechAlignerConfig::default()
            };
            let result = config.init(&Device::default());
            assert!(
                matches!(result, Err(Error::Config(_))),
                "n_heads = {n_heads}"
            );
        }
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = SpeechAlignerConfig::for_feature_mode(FeatureMode::LogMel80);
        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: SpeechAlignerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, config);
    }

    #[test]
    fn proportional_word_frame_span_partitions_all_frames() {
        let spans = (0..3)
            .map(|word_position| proportional_word_frame_span(10, 3, word_position))
            .collect::<Vec<_>>();

        assert_eq!(spans, vec![Some((0, 3)), Some((3, 6)), Some((6, 10))]);
    }
}
