//! CTC loss wrapper for `SpeechAligner`.
//!
//! The model trains against synthetic CTC targets only. The
//! phoneme and boundary heads remain part of the model surface, but their
//! supervised losses land in a later phase when real labels are available.

use burn::nn::loss::{CTCLoss, CTCLossConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Configuration for the active loss surface.
#[derive(Debug, Clone, Default)]
pub struct SpeechAlignerLossConfig;

impl SpeechAlignerLossConfig {
    /// Initialize the Loss module.
    pub fn init() -> SpeechAlignerLoss {
        SpeechAlignerLoss {
            ctc_loss: CTCLossConfig::new().init(),
        }
    }
}

/// Loss module for synthetic CTC training.
pub struct SpeechAlignerLoss {
    ctc_loss: CTCLoss,
}

impl SpeechAlignerLoss {
    /// Compute CTC loss from log-probabilities and sequence lengths.
    ///
    /// - `ctc_log_probs`: `[time, batch, classes]`
    /// - `targets`: `[batch, max_target_len]`
    /// - `input_lengths`: `[batch]`
    /// - `target_lengths`: `[batch]`
    pub fn forward<B: Backend>(
        &self,
        ctc_log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        self.ctc_loss
            .forward(ctc_log_probs, targets, input_lengths, target_lengths)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{Distribution, Shape, Tensor, TensorData};

    use super::*;

    type B = NdArray<f32>;

    #[test]
    fn test_ctc_loss_produces_finite_output() {
        let device = NdArrayDevice::default();
        let loss = SpeechAlignerLossConfig::init();

        // Fake log-probs: [time=10, batch=2, classes=42]
        let log_probs: Tensor<B, 3> =
            Tensor::random([10, 2, 42], Distribution::Normal(-3.0, 0.5), &device);

        let targets = Tensor::from_data(
            TensorData::new(vec![1_i32, 2, 3, 1, 2, 3], Shape::new([2, 3])),
            &device,
        );
        let input_lengths =
            Tensor::from_data(TensorData::new(vec![10_i32, 10], Shape::new([2])), &device);
        let target_lengths =
            Tensor::from_data(TensorData::new(vec![3_i32, 3], Shape::new([2])), &device);

        let result = loss.forward(log_probs, targets, input_lengths, target_lengths);
        let vals: Vec<f32> = match result.to_data().to_vec() {
            Ok(vals) => vals,
            Err(err) => panic!("loss extraction should succeed: {err}"),
        };
        for v in &vals {
            assert!(v.is_finite(), "CTC loss should be finite, got {v}");
        }
    }
}
