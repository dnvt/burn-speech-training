//! Squeeze-and-Excitation block for channel recalibration.
//!
//! Implements the SE mechanism from Hu et al. (2018): global average pool over
//! the time dimension, bottleneck FC layers, sigmoid gating. This lets each
//! conv block learn which channels matter most for the current input.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for a Squeeze-and-Excitation block.
#[derive(Debug, Clone)]
pub struct SEBlockConfig {
    /// Number of input/output channels.
    pub channels: usize,
    /// Reduction ratio for the bottleneck (default: 8).
    pub reduction: usize,
}

impl SEBlockConfig {
    /// Create a new SE block config.
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            reduction: 8,
        }
    }

    /// Set the reduction ratio.
    pub fn with_reduction(mut self, reduction: usize) -> Self {
        self.reduction = reduction;
        self
    }

    /// Initialize the SE block on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEBlock<B> {
        // Clamp reduction to at least 1 to avoid division by zero.
        let reduction = self.reduction.max(1);
        let bottleneck = (self.channels / reduction).max(1);

        SEBlock {
            fc_squeeze: LinearConfig::new(self.channels, bottleneck).init(device),
            fc_excite: LinearConfig::new(bottleneck, self.channels).init(device),
        }
    }
}

/// Squeeze-and-Excitation block.
///
/// Input: `[batch, channels, time]` -> Output: `[batch, channels, time]`
/// (same shape, channel-wise rescaled).
#[derive(Module, Debug)]
pub struct SEBlock<B: Backend> {
    fc_squeeze: Linear<B>,
    fc_excite: Linear<B>,
}

impl<B: Backend> SEBlock<B> {
    /// Forward pass: squeeze (global avg pool) -> excite (FC bottleneck) ->
    /// scale.
    ///
    /// Input shape: `[batch, channels, time]`
    /// Output shape: `[batch, channels, time]` (same)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];

        // Global average pooling over time dimension: [B, C, T] -> [B, C]
        let squeezed = x.clone().mean_dim(2).reshape([batch, channels]);

        // Bottleneck: FC -> ReLU -> FC -> Sigmoid
        let scale = self.fc_squeeze.forward(squeezed);
        let scale = relu(scale);
        let scale = self.fc_excite.forward(scale);
        let scale = sigmoid(scale);

        // Reshape to [B, C, 1] for broadcast multiplication
        let scale = scale.reshape([batch, channels, 1]);

        x * scale
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
    fn test_se_block_preserves_shape() {
        let device = NdArrayDevice::default();
        let se = SEBlockConfig::new(64).init::<B>(&device);

        let input: Tensor<B, 3> =
            Tensor::random([2, 64, 20], Distribution::Normal(0.0, 1.0), &device);
        let output = se.forward(input);

        assert_eq!(output.dims(), [2, 64, 20]);
    }

    #[test]
    fn test_se_block_small_channels() {
        let device = NdArrayDevice::default();
        // With 4 channels and reduction=8, bottleneck would be 0 -> clamped to 1
        let se = SEBlockConfig::new(4).init::<B>(&device);

        let input: Tensor<B, 3> =
            Tensor::random([1, 4, 10], Distribution::Normal(0.0, 1.0), &device);
        let output = se.forward(input);

        assert_eq!(output.dims(), [1, 4, 10]);
    }
}
