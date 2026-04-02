//! Self-attention layer with `LayerNorm` and residual connection.
//!
//! Wraps Burn's `MultiHeadAttention` with pre-norm (`LayerNorm` before
//! attention) and a residual skip connection.

use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use crate::error::{Error, Result};

/// Configuration for the self-attention layer.
#[derive(Debug, Clone)]
pub struct SelfAttentionConfig {
    /// Model dimension (must match input feature dim).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dropout rate (default: 0.0 for deterministic training).
    pub dropout: f64,
}

impl SelfAttentionConfig {
    /// Create a new self-attention config.
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            dropout: 0.0,
        }
    }

    /// Initialize the self-attention layer on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<SelfAttentionLayer<B>> {
        if self.n_heads == 0 {
            return Err(Error::config("SelfAttentionConfig.n_heads must be > 0"));
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(Error::config(format!(
                "SelfAttentionConfig.d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            )));
        }

        let layer_norm = LayerNormConfig::new(self.d_model).init(device);
        let mha = MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
            .with_dropout(self.dropout)
            .init(device);

        Ok(SelfAttentionLayer { layer_norm, mha })
    }
}

/// Self-attention with pre-norm and residual connection.
///
/// Input: `[batch, seq_len, d_model]` -> Output: `[batch, seq_len, d_model]`
#[derive(Module, Debug)]
pub struct SelfAttentionLayer<B: Backend> {
    layer_norm: LayerNorm<B>,
    mha: MultiHeadAttention<B>,
}

impl<B: Backend> SelfAttentionLayer<B> {
    /// Forward pass: `LayerNorm` -> `MultiHeadAttention` -> residual add.
    ///
    /// Input shape: `[batch, seq_len, d_model]`
    /// Output shape: `[batch, seq_len, d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.layer_norm.forward(x);
        let attn_input = MhaInput::self_attn(normed);
        let attn_output = self.mha.forward(attn_input);
        residual + attn_output.context
    }

    /// Forward pass with an explicit pad mask for variable-length batches.
    ///
    /// `mask_pad` uses shape `[batch, seq_len]` with `true` for padded
    /// positions that should be excluded from attention.
    pub fn forward_with_pad_mask(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.layer_norm.forward(x);
        let attn_input = MhaInput::self_attn(normed).mask_pad(mask_pad);
        let attn_output = self.mha.forward(attn_input);
        residual + attn_output.context
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{Distribution, Tensor, Tolerance};

    use super::*;

    type B = NdArray<f32>;

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn init_attention(
        device: &NdArrayDevice,
        d_model: usize,
        n_heads: usize,
    ) -> SelfAttentionLayer<B> {
        match SelfAttentionConfig::new(d_model, n_heads).init::<B>(device) {
            Ok(layer) => layer,
            Err(err) => panic!("attention init should succeed: {err}"),
        }
    }

    #[test]
    fn test_self_attention_preserves_shape() {
        let device = NdArrayDevice::default();
        let attn = init_attention(&device, 512, 8);

        let input: Tensor<B, 3> =
            Tensor::random([2, 50, 512], Distribution::Normal(0.0, 1.0), &device);
        let output = attn.forward(input);

        assert_eq!(output.dims(), [2, 50, 512]);
    }

    #[test]
    fn test_self_attention_pad_mask_blocks_padded_context() {
        let [batch_size, seq_len, d_model, n_heads, num_padded] = [2, 6, 32, 4, 2];
        let device = NdArrayDevice::default();
        let attn = init_attention(&device, d_model, n_heads);

        let mask_pad: Tensor<B, 2, burn::tensor::Int> =
            Tensor::zeros([batch_size, seq_len], &device);
        let mask_pad = mask_pad.slice_assign(
            [0..batch_size, seq_len - num_padded..seq_len],
            Tensor::ones([batch_size, num_padded], &device),
        );
        let mask_pad = mask_pad.equal_elem(1);

        let input_1: Tensor<B, 3> = Tensor::random(
            [batch_size, seq_len, d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let input_2 = input_1.clone().slice_assign(
            [0..batch_size, seq_len - num_padded..seq_len, 0..d_model],
            Tensor::random(
                [batch_size, num_padded, d_model],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
        );

        let output_1 = attn.forward_with_pad_mask(input_1, mask_pad.clone());
        let output_2 = attn.forward_with_pad_mask(input_2, mask_pad);

        output_1
            .slice([0..batch_size, 0..seq_len - num_padded, 0..d_model])
            .to_data()
            .assert_approx_eq::<f32>(
                &output_2
                    .slice([0..batch_size, 0..seq_len - num_padded, 0..d_model])
                    .to_data(),
                Tolerance::default(),
            );
    }

    #[test]
    fn test_self_attention_rejects_invalid_head_count() {
        let device = NdArrayDevice::default();
        let err = match SelfAttentionConfig::new(32, 0).init::<B>(&device) {
            Ok(_) => panic!("zero attention heads should be rejected"),
            Err(err) => err,
        };

        assert!(matches!(err, crate::error::Error::Config { .. }));
    }
}
