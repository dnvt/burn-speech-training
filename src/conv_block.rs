//! Conv1d block with `LayerNorm`, `ReLU`, and Squeeze-and-Excitation.
//!
//! Each block: Conv1d -> `LayerNorm` -> `ReLU` -> SE. Padding is computed to
//! preserve the time dimension (same padding).
//!
//! `LayerNorm` (over channel dim) is used instead of `BatchNorm` for two
//! reasons:
//! 1. No running-state tracking issues with Burn's `NdArray` backend
//! 2. Modern architectures (`ConvNeXt`, etc.) show `LayerNorm` works well in
//!    convnets

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::se_block::{SEBlock, SEBlockConfig};

/// Configuration for a Conv1d + `LayerNorm` + `ReLU` + SE block.
#[derive(Debug, Clone)]
pub struct ConvSEBlockConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size (default: 3).
    pub kernel_size: usize,
    /// SE reduction ratio (default: 8).
    pub se_reduction: usize,
}

impl ConvSEBlockConfig {
    /// Create a new `ConvSE` block config.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size: 3,
            se_reduction: 8,
        }
    }

    /// Set the kernel size.
    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    /// Set the SE reduction ratio.
    pub fn with_se_reduction(mut self, se_reduction: usize) -> Self {
        self.se_reduction = se_reduction;
        self
    }

    /// Initialize the block on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvSEBlock<B> {
        let padding = burn::nn::PaddingConfig1d::Same;

        let conv = Conv1dConfig::new(self.in_channels, self.out_channels, self.kernel_size)
            .with_padding(padding)
            .init(device);

        // LayerNorm over channel dimension (last dim after transpose).
        let ln = LayerNormConfig::new(self.out_channels).init(device);

        let se = SEBlockConfig::new(self.out_channels)
            .with_reduction(self.se_reduction)
            .init(device);

        ConvSEBlock { conv, ln, se }
    }
}

/// Conv1d + `LayerNorm` + `ReLU` + Squeeze-and-Excitation block.
///
/// Input: `[batch, in_channels, time]` -> Output: `[batch, out_channels, time]`
#[derive(Module, Debug)]
pub struct ConvSEBlock<B: Backend> {
    conv: Conv1d<B>,
    ln: LayerNorm<B>,
    se: SEBlock<B>,
}

impl<B: Backend> ConvSEBlock<B> {
    /// Forward pass: Conv1d -> `LayerNorm` -> `ReLU` -> SE.
    ///
    /// Input shape: `[batch, in_channels, time]`
    /// Output shape: `[batch, out_channels, time]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv.forward(x);
        // LayerNorm expects last dim = channels, so transpose: [B, C, T] -> [B, T, C]
        let x = x.swap_dims(1, 2);
        let x = self.ln.forward(x);
        // Transpose back: [B, T, C] -> [B, C, T]
        let x = x.swap_dims(1, 2);
        let x = relu(x);
        self.se.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{Distribution, Tensor};

    use super::*;

    type B = NdArray<f32>;

    #[test]
    fn test_conv_se_block_shape() {
        let device = NdArrayDevice::default();
        let block = ConvSEBlockConfig::new(39, 64).init::<B>(&device);

        // Input: [batch=2, channels=39, time=100]
        let input: Tensor<B, 3> =
            Tensor::random([2, 39, 100], Distribution::Normal(0.0, 1.0), &device);
        let output = block.forward(input);

        // Output: [batch=2, channels=64, time=100] (time preserved via same padding)
        assert_eq!(output.dims(), [2, 64, 100]);
    }

    #[test]
    fn test_conv_se_block_channel_progression() {
        let device = NdArrayDevice::default();

        let block1 = ConvSEBlockConfig::new(39, 64).init::<B>(&device);
        let block2 = ConvSEBlockConfig::new(64, 128).init::<B>(&device);
        let block3 = ConvSEBlockConfig::new(128, 256).init::<B>(&device);
        let block4 = ConvSEBlockConfig::new(256, 512).init::<B>(&device);

        let x: Tensor<B, 3> = Tensor::random([1, 39, 50], Distribution::Normal(0.0, 1.0), &device);
        let x = block1.forward(x);
        assert_eq!(x.dims(), [1, 64, 50]);
        let x = block2.forward(x);
        assert_eq!(x.dims(), [1, 128, 50]);
        let x = block3.forward(x);
        assert_eq!(x.dims(), [1, 256, 50]);
        let x = block4.forward(x);
        assert_eq!(x.dims(), [1, 512, 50]);
    }
}
