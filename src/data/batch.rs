//! Training samples and padded batches.

use std::ops::Range;

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Shape, Tensor, TensorData};

/// One utterance: feature frames and CTC targets.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingSample {
    /// Row-major `[num_frames, feature_dim]` features.
    pub features: Vec<f32>,
    pub feature_dim: usize,
    /// CTC class indices (no blanks).
    pub targets: Vec<i32>,
    /// Words that were not in the pronunciation dictionary.
    pub oov_words: usize,
}

impl TrainingSample {
    pub fn from_frames(frames: &[Vec<f32>], feature_dim: usize, targets: Vec<i32>) -> Self {
        Self {
            features: frames.concat(),
            feature_dim,
            targets,
            oov_words: 0,
        }
    }

    pub fn num_frames(&self) -> usize {
        self.features.len() / self.feature_dim.max(1)
    }
}

/// A padded batch on a device.
pub struct Batch<B: Backend> {
    /// `[batch, max_frames, feature_dim]`, zero-padded.
    pub inputs: Tensor<B, 3>,
    /// `[batch, max_targets]`, zero-padded.
    pub targets: Tensor<B, 2, Int>,
    /// `[batch]`
    pub input_lengths: Tensor<B, 1, Int>,
    /// `[batch]`
    pub target_lengths: Tensor<B, 1, Int>,
    /// `[batch, max_frames]`, `true` on padded frames.
    pub pad_mask: Tensor<B, 2, Bool>,
}

/// How samples are grouped into batches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Batching {
    /// A fixed number of samples per batch.
    Fixed { batch_size: usize },
    /// As many samples as fit a memory budget for the attention scores,
    /// `batch × heads × frames²` elements, capped at `max_samples`.
    ///
    /// Self-attention memory grows with the square of the longest utterance,
    /// so this packs many short utterances or a few long ones per batch.
    AttentionBudget {
        max_attn_elements: usize,
        max_samples: usize,
    },
}

impl Default for Batching {
    fn default() -> Self {
        Self::Fixed { batch_size: 16 }
    }
}

impl Batching {
    pub(crate) fn validate(&self) -> crate::error::Result<()> {
        let ok = match *self {
            Self::Fixed { batch_size } => batch_size > 0,
            Self::AttentionBudget {
                max_attn_elements,
                max_samples,
            } => max_attn_elements > 0 && max_samples > 0,
        };
        if ok {
            Ok(())
        } else {
            Err(crate::error::Error::config(format!(
                "batch sizes must be > 0, got {self:?}"
            )))
        }
    }
}

/// Sort `samples` by length (shortest first) and split them into batches.
/// Returns index ranges into the sorted slice; every range is non-empty.
pub fn plan_batches(
    samples: &mut [TrainingSample],
    batching: Batching,
    n_heads: usize,
) -> Vec<Range<usize>> {
    samples.sort_by_key(TrainingSample::num_frames);

    let mut ranges = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let mut end = start + 1;
        match batching {
            Batching::Fixed { batch_size } => end = (start + batch_size.max(1)).min(samples.len()),
            Batching::AttentionBudget {
                max_attn_elements,
                max_samples,
            } => {
                // Samples are sorted, so the newest one is the longest.
                while end < samples.len() && end - start < max_samples {
                    let frames = samples[end].num_frames();
                    if (end - start + 1) * n_heads * frames * frames > max_attn_elements {
                        break;
                    }
                    end += 1;
                }
            }
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// Pad and stack samples into a batch. All samples must share `feature_dim`.
pub fn collate<B: Backend>(samples: &[TrainingSample], device: &B::Device) -> Batch<B> {
    let batch = samples.len();
    let feature_dim = samples.first().map_or(0, |s| s.feature_dim);
    let max_frames = samples
        .iter()
        .map(TrainingSample::num_frames)
        .max()
        .unwrap_or(0);
    let max_targets = samples.iter().map(|s| s.targets.len()).max().unwrap_or(0);

    let mut inputs = vec![0.0f32; batch * max_frames * feature_dim];
    let mut targets = vec![0i32; batch * max_targets];
    let mut pad_mask = vec![true; batch * max_frames];
    let mut input_lengths = Vec::with_capacity(batch);
    let mut target_lengths = Vec::with_capacity(batch);

    for (i, sample) in samples.iter().enumerate() {
        assert_eq!(
            sample.feature_dim, feature_dim,
            "mixed feature_dim in batch"
        );
        let frames = sample.num_frames();
        let row = i * max_frames * feature_dim;
        inputs[row..row + sample.features.len()].copy_from_slice(&sample.features);
        targets[i * max_targets..i * max_targets + sample.targets.len()]
            .copy_from_slice(&sample.targets);
        pad_mask[i * max_frames..i * max_frames + frames].fill(false);
        input_lengths.push(frames as i32);
        target_lengths.push(sample.targets.len() as i32);
    }

    Batch {
        inputs: Tensor::from_data(
            TensorData::new(inputs, Shape::new([batch, max_frames, feature_dim])),
            device,
        ),
        targets: Tensor::from_data(
            TensorData::new(targets, Shape::new([batch, max_targets])),
            device,
        ),
        input_lengths: Tensor::from_data(
            TensorData::new(input_lengths, Shape::new([batch])),
            device,
        ),
        target_lengths: Tensor::from_data(
            TensorData::new(target_lengths, Shape::new([batch])),
            device,
        ),
        pad_mask: Tensor::from_data(
            TensorData::new(pad_mask, Shape::new([batch, max_frames])),
            device,
        ),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    fn sample(frames: usize) -> TrainingSample {
        TrainingSample {
            features: (0..frames * 2).map(|v| v as f32 + 1.0).collect(),
            feature_dim: 2,
            targets: vec![1; (frames / 5).max(1)],
            oov_words: 0,
        }
    }

    fn sizes(ranges: &[Range<usize>]) -> Vec<usize> {
        ranges.iter().map(ExactSizeIterator::len).collect()
    }

    #[test]
    fn fixed_batches_are_sorted_by_length() {
        let mut samples = vec![sample(30), sample(10), sample(20)];
        let plan = plan_batches(&mut samples, Batching::Fixed { batch_size: 2 }, 1);
        assert_eq!(sizes(&plan), [2, 1]);
        assert_eq!(samples[0].num_frames(), 10);
        assert_eq!(samples[2].num_frames(), 30);
    }

    #[test]
    fn attention_budget_limits_batch_size() {
        let budget = |max_attn_elements, max_samples| Batching::AttentionBudget {
            max_attn_elements,
            max_samples,
        };

        let mut samples = vec![sample(10), sample(10), sample(20)];
        assert_eq!(
            sizes(&plan_batches(&mut samples, budget(1_600, 16), 2)),
            [2, 1]
        );

        let mut samples = vec![sample(8); 4];
        assert_eq!(
            sizes(&plan_batches(&mut samples, budget(1_000_000, 2), 1)),
            [2, 2]
        );

        // A single sample over budget still gets its own batch.
        let mut samples = vec![sample(100); 2];
        assert_eq!(sizes(&plan_batches(&mut samples, budget(1, 16), 8)), [1, 1]);

        // More heads, smaller batches.
        let mut samples = vec![sample(50); 4];
        assert_eq!(
            sizes(&plan_batches(&mut samples, budget(10_000, 16), 1)),
            [4]
        );
        assert_eq!(
            sizes(&plan_batches(&mut samples, budget(10_000, 16), 2)),
            [2, 2]
        );
    }

    #[test]
    fn collate_pads_and_masks() {
        let samples = [sample(3), sample(5)];
        let batch = collate::<NdArray>(&samples, &Default::default());

        assert_eq!(batch.inputs.dims(), [2, 5, 2]);
        assert_eq!(batch.targets.dims(), [2, 1]);

        let lengths: Vec<i64> = batch
            .input_lengths
            .into_data()
            .convert::<i64>()
            .to_vec()
            .unwrap();
        assert_eq!(lengths, [3, 5]);

        let mask: Vec<bool> = batch.pad_mask.into_data().to_vec().unwrap();
        assert_eq!(
            mask,
            [false, false, false, true, true, false, false, false, false, false]
        );

        let inputs: Vec<f32> = batch.inputs.into_data().to_vec().unwrap();
        // First sample: 3 real frames then zero padding.
        assert_eq!(&inputs[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&inputs[6..10], &[0.0; 4]);
    }
}
