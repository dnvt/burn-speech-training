//! Building blocks of [`SpeechAligner`](super::SpeechAligner).
//!
//! A `keep` mask is `[batch, 1, time]`, 1.0 for real frames and 0.0 for
//! padding.

use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Bool, Device, Tensor};

/// Conv1d (same padding) -> `LayerNorm` over channels -> `ReLU` ->
/// squeeze-and-excitation, on `[batch, channels, time]`.
#[derive(Module, Debug)]
pub struct ConvSEBlock {
    conv: Conv1d,
    ln: LayerNorm,
    se: SEBlock,
}

impl ConvSEBlock {
    pub(super) fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        se_reduction: usize,
        device: &Device,
    ) -> Self {
        Self {
            conv: Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            ln: LayerNormConfig::new(out_channels).init(device),
            se: SEBlock::new(out_channels, se_reduction, device),
        }
    }

    /// Padded frames leave the block zeroed so the next convolution sees the
    /// same zero context as at the end of an unpadded utterance.
    pub(super) fn forward(&self, x: Tensor<3>, keep: Option<&Tensor<3>>) -> Tensor<3> {
        let x = self.conv.forward(x);
        let x = self.ln.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
        let mut x = relu(x);
        if let Some(keep) = keep {
            x = x * keep.clone();
        }
        self.se.forward(x, keep)
    }
}

/// Squeeze-and-excitation (Hu et al., 2018): mean over time, a
/// `channels -> channels / reduction -> channels` bottleneck, and a sigmoid
/// gate that rescales each channel.
#[derive(Module, Debug)]
pub struct SEBlock {
    fc_squeeze: Linear,
    fc_excite: Linear,
}

impl SEBlock {
    pub(super) fn new(channels: usize, reduction: usize, device: &Device) -> Self {
        let bottleneck = (channels / reduction.max(1)).max(1);
        Self {
            fc_squeeze: LinearConfig::new(channels, bottleneck).init(device),
            fc_excite: LinearConfig::new(bottleneck, channels).init(device),
        }
    }

    /// With `keep`, `x` must already be zero at padded frames; the mean then
    /// covers real frames only.
    pub(super) fn forward(&self, x: Tensor<3>, keep: Option<&Tensor<3>>) -> Tensor<3> {
        let [batch, channels, _] = x.dims();
        let pooled = match keep {
            Some(keep) => x.clone().sum_dim(2) / keep.clone().sum_dim(2).clamp_min(1.0),
            None => x.clone().mean_dim(2),
        };
        let scale = relu(self.fc_squeeze.forward(pooled.reshape([batch, channels])));
        let scale = sigmoid(self.fc_excite.forward(scale));
        x * scale.reshape([batch, channels, 1])
    }
}

/// Pre-norm multi-head self-attention with a residual connection and no
/// dropout, on `[batch, time, d_model]`.
#[derive(Module, Debug)]
pub struct SelfAttentionLayer {
    layer_norm: LayerNorm,
    mha: MultiHeadAttention,
}

impl SelfAttentionLayer {
    pub(super) fn new(d_model: usize, n_heads: usize, device: &Device) -> Self {
        Self {
            layer_norm: LayerNormConfig::new(d_model).init(device),
            mha: MultiHeadAttentionConfig::new(d_model, n_heads)
                .with_dropout(0.0)
                .init(device),
        }
    }

    pub(super) fn forward(&self, x: Tensor<3>, mask_pad: Option<Tensor<2, Bool>>) -> Tensor<3> {
        let mut input = MhaInput::self_attn(self.layer_norm.forward(x.clone()));
        if let Some(mask_pad) = mask_pad {
            input = input.mask_pad(mask_pad);
        }
        x + self.mha.forward(input).context
    }
}
