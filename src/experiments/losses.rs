//! Scoring losses and word weightings tried in the experiment log.

use std::collections::BTreeMap;

use burn::tensor::activation::{log_softmax, relu, sigmoid, softmax};
use burn::tensor::{Device, Shape, Tensor, TensorData};

use crate::finetune::head::ScoreOutput;

/// Upper edges of the ordinal score bins on the `[0, 1]` scale; raw 0–10
/// scores map to bins {0–2, 3–5, 6–7, 8–9, 10}.
pub const ORDINAL_THRESHOLDS: [f32; 4] = [0.2, 0.5, 0.7, 0.9];
/// Score assigned to each bin when converting bin probabilities to a score.
pub const ORDINAL_BIN_CENTERS: [f32; 5] = [0.1, 0.4, 0.65, 0.85, 1.0];

pub fn ordinal_bin(raw_score: i32) -> usize {
    match raw_score {
        i32::MIN..=2 => 0,
        3..=5 => 1,
        6..=7 => 2,
        8..=9 => 3,
        _ => 4,
    }
}

fn row(values: Vec<f32>, device: &Device) -> Tensor<2> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, Shape::new([1, n])), device)
}

fn matrix(values: Vec<f32>, rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, Shape::new([rows, cols])), device)
}

fn threshold_logits() -> Vec<f32> {
    ORDINAL_THRESHOLDS
        .iter()
        .map(|p| (p / (1.0 - p)).ln())
        .collect()
}

/// Bin probabilities `[words, 5]` for the cumulative-threshold model, where
/// logit `k` is the log-odds that the score exceeds threshold `k`.
fn ordinal_bin_probs(logits: Tensor<2>) -> Tensor<2> {
    let [words, _] = logits.dims();
    let device = logits.device();
    let exceed = sigmoid(logits - row(threshold_logits(), &device));
    let ones = Tensor::<2>::ones([words, 1], &device);
    let zeros = Tensor::<2>::zeros([words, 1], &device);
    // P(bin k) = P(score > t_{k-1}) − P(score > t_k), with P(> t_{-1}) = 1, P(> t_4) = 0.
    let upper = Tensor::cat(vec![ones, exceed.clone()], 1);
    let lower = Tensor::cat(vec![exceed, zeros], 1);
    upper - lower
}

/// Expected score in `[0, 1]` for the ordinal head outputs.
pub fn ordinal_predictions(logits: Tensor<2>, output: ScoreOutput) -> Tensor<1> {
    let [words, _] = logits.dims();
    let device = logits.device();
    let probs = match output {
        ScoreOutput::Ordinal => ordinal_bin_probs(logits),
        ScoreOutput::OrdinalSoftmax => softmax(logits, 1),
        ScoreOutput::Sigmoid => return sigmoid(logits).reshape([words]),
    };
    (probs * row(ORDINAL_BIN_CENTERS.to_vec(), &device))
        .sum_dim(1)
        .reshape([words])
}

/// Weighted binary cross-entropy on the four "score exceeds threshold"
/// targets.
pub fn ordinal_threshold_loss(
    logits: Tensor<2>,
    raw_scores: &[i32],
    weights: Tensor<1>,
) -> Tensor<1> {
    let words = raw_scores.len();
    let device = logits.device();
    let targets: Vec<f32> = raw_scores
        .iter()
        .flat_map(|&raw| [2, 5, 7, 9].map(|edge| if raw > edge { 1.0 } else { 0.0 }))
        .collect();
    let targets = matrix(targets, words, 4, &device);
    let p = sigmoid(logits - row(threshold_logits(), &device));
    let bce = -(targets.clone() * p.clone().clamp_min(1e-6).log()
        + (targets.ones_like() - targets) * (p.ones_like() - p).clamp_min(1e-6).log());
    (bce * weights.reshape([words, 1])).mean()
}

/// Weighted 5-way cross-entropy over score bins.
pub fn ordinal_softmax_loss(
    logits: Tensor<2>,
    raw_scores: &[i32],
    weights: Tensor<1>,
) -> Tensor<1> {
    let words = raw_scores.len();
    let device = logits.device();
    let mut one_hot = vec![0.0f32; words * 5];
    for (i, &raw) in raw_scores.iter().enumerate() {
        one_hot[i * 5 + ordinal_bin(raw)] = 1.0;
    }
    let nll = -(matrix(one_hot, words, 5, &device) * log_softmax(logits, 1))
        .sum_dim(1)
        .reshape([words]);
    (nll * weights).mean()
}

/// Squared error weighted by `(1 − p_t)^gamma`, where `p_t = 1 − |error|`, so
/// words with large errors count more.
pub fn focal_mse(
    predicted: Tensor<1>,
    targets: Tensor<1>,
    weights: Tensor<1>,
    gamma: f32,
) -> Tensor<1> {
    let error = predicted - targets;
    let focal = error.clone().abs().clamp_max(1.0).powf_scalar(gamma);
    (error.clone() * error * focal * weights).mean()
}

/// Balanced MSE (Ren et al. 2022, batch-based Monte Carlo): treats each
/// prediction as a Gaussian with `noise_sigma` and classifies it against
/// every target in the batch. Targets with the same raw score count as
/// positives.
pub fn balanced_mse(
    predicted: Tensor<1>,
    targets: Tensor<1>,
    raw_scores: &[i32],
    noise_sigma: f32,
) -> Tensor<1> {
    let words = raw_scores.len();
    if words < 2 {
        let error = predicted - targets;
        return (error.clone() * error).mean();
    }
    let device = predicted.device();
    let diff = predicted.reshape([words, 1]) - targets.reshape([1, words]);
    let logits = (diff.clone() * diff) * (-0.5 / (noise_sigma * noise_sigma));

    let positives: Vec<f32> = raw_scores
        .iter()
        .flat_map(|a| {
            raw_scores
                .iter()
                .map(move |b| if a == b { 1.0 } else { 0.0 })
        })
        .collect();
    let matched = (softmax(logits, 1) * matrix(positives, words, words, &device))
        .sum_dim(1)
        .reshape([words]);
    -matched.clamp_min(1e-6).log().mean() * (2.0 * noise_sigma * noise_sigma)
}

/// Hinge loss over every pair whose human scores differ: the higher-scored
/// word should be predicted at least `margin` higher.
pub fn pairwise_ranking(predicted: Tensor<1>, raw_scores: &[i32], margin: f32) -> Tensor<1> {
    let words = raw_scores.len();
    let device = predicted.device();
    let mask: Vec<f32> = raw_scores
        .iter()
        .flat_map(|a| {
            raw_scores
                .iter()
                .map(move |b| if a > b { 1.0 } else { 0.0 })
        })
        .collect();
    let pairs = mask.iter().filter(|&&m| m > 0.0).count();
    if pairs == 0 {
        return Tensor::zeros([1], &device);
    }
    let gap = predicted.clone().reshape([words, 1]) - predicted.reshape([1, words]);
    let hinge = relu(gap.ones_like() * margin - gap);
    (hinge * matrix(mask, words, words, &device))
        .sum()
        .reshape([1])
        / pairs as f32
}

/// Triplet loss on pooled word embeddings. For each anchor, words within 1
/// raw point are positives and words 3+ points away are negatives; the mean
/// positive distance should be `margin` below the mean negative distance.
pub fn triplet_ranking(pooled: Tensor<2>, raw_scores: &[i32], margin: f32) -> Tensor<1> {
    let [words, dim] = pooled.dims();
    let device = pooled.device();
    let diff = pooled.clone().reshape([words, 1, dim]) - pooled.reshape([1, words, dim]);
    let distances = (diff.clone() * diff).sum_dim(2).reshape([words, words]);

    let mut positive = vec![0.0f32; words * words];
    let mut negative = vec![0.0f32; words * words];
    let mut valid = vec![0.0f32; words];
    for (a, &sa) in raw_scores.iter().enumerate() {
        for (b, &sb) in raw_scores.iter().enumerate() {
            let gap = (sa - sb).abs();
            if a != b && gap <= 1 {
                positive[a * words + b] = 1.0;
            } else if gap >= 3 {
                negative[a * words + b] = 1.0;
            }
        }
        let has = |m: &[f32]| m[a * words..(a + 1) * words].iter().any(|&v| v > 0.0);
        valid[a] = if has(&positive) && has(&negative) {
            1.0
        } else {
            0.0
        };
    }
    let anchors = valid.iter().filter(|&&v| v > 0.0).count();
    if anchors == 0 {
        return Tensor::zeros([1], &device);
    }

    let mean_distance = |mask: Vec<f32>| {
        let mask = matrix(mask, words, words, &device);
        let count = mask.clone().sum_dim(1).reshape([words]).clamp_min(1.0);
        (distances.clone() * mask).sum_dim(1).reshape([words]) / count
    };
    let valid = Tensor::from_data(TensorData::new(valid, Shape::new([words])), &device);
    let loss = relu(mean_distance(positive) - mean_distance(negative) + margin) * valid;
    loss.sum().reshape([1]) / anchors as f32
}

/// Make pairwise embedding similarity track pairwise score similarity.
///
/// Embedding similarity is `exp(−‖a − b‖²)` on unit-normalised vectors (so
/// it ranges over `[e⁻⁴, 1]`); score similarity is `exp(−|s_a − s_b| / tau)`.
/// The loss is their mean squared difference over off-diagonal pairs.
///
/// The runs in the experiment log used the *mean* squared distance per
/// dimension, which kept embedding similarity within `[0.992, 1]` and gave
/// almost no gradient. This version uses the sum.
pub fn rank_regularization(pooled: Tensor<2>, targets: Tensor<1>, tau: f32) -> Tensor<1> {
    let [words, dim] = pooled.dims();
    let device = pooled.device();
    if words < 2 {
        return Tensor::zeros([1], &device);
    }
    let norm = (pooled.clone() * pooled.clone())
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-6);
    let unit = pooled / norm;
    let diff = unit.clone().reshape([words, 1, dim]) - unit.reshape([1, words, dim]);
    let embedding_similarity = (-(diff.clone() * diff).sum_dim(2).reshape([words, words])).exp();

    let score_gap = (targets.clone().reshape([words, 1]) - targets.reshape([1, words])).abs();
    let score_similarity = (score_gap * (-1.0 / tau.max(1e-3))).exp();

    let off_diagonal: Vec<f32> = (0..words * words)
        .map(|i| if i / words == i % words { 0.0 } else { 1.0 })
        .collect();
    let err = embedding_similarity - score_similarity;
    (err.clone() * err * matrix(off_diagonal, words, words, &device))
        .sum()
        .reshape([1])
        / (words * (words - 1)) as f32
}

/// Per-bucket loss weights `max_count / count`, capped at 100.
pub fn inverse_frequency_weights(counts: &BTreeMap<i32, usize>) -> BTreeMap<i32, f32> {
    let max = counts.values().copied().max().unwrap_or(1) as f32;
    counts
        .iter()
        .map(|(&bucket, &count)| (bucket, (max / count.max(1) as f32).min(100.0)))
        .collect()
}

/// Effective-number weights (Cui et al. 2019), `(1 − β) / (1 − βⁿ)`,
/// normalised to mean 1 and capped at 100 before normalisation.
pub fn effective_number_weights(counts: &BTreeMap<i32, usize>, beta: f32) -> BTreeMap<i32, f32> {
    let mut weights: BTreeMap<i32, f32> = counts
        .iter()
        .map(|(&bucket, &count)| {
            let n = count.max(1) as f32;
            let effective = if beta <= f32::EPSILON {
                n
            } else {
                (1.0 - beta.powf(n)).max(1e-6) / (1.0 - beta)
            };
            (bucket, (1.0 / effective).min(100.0))
        })
        .collect();
    let mean = weights.values().sum::<f32>() / weights.len().max(1) as f32;
    if mean > f32::EPSILON {
        weights.values_mut().for_each(|w| *w /= mean);
    }
    weights
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::finetune::head::scalar;

    fn t1(values: &[f32]) -> Tensor<1> {
        Tensor::from_data(TensorData::from(values), &Device::default())
    }

    #[test]
    fn ordinal_bins() {
        let bins: Vec<usize> = (0..=10).map(ordinal_bin).collect();
        assert_eq!(bins, [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn ordinal_predictions_are_expected_scores() {
        let device = Device::default();
        let neutral = |output: ScoreOutput, n| {
            let p = ordinal_predictions(Tensor::<2>::zeros([1, n], &device), output);
            scalar(p)
        };
        // Threshold model at zero logits: bins weighted by threshold distances.
        assert!((0.45..=0.55).contains(&neutral(ScoreOutput::Ordinal, 4)));
        // Uniform softmax: mean of bin centers.
        assert!((neutral(ScoreOutput::OrdinalSoftmax, 5) - 0.6).abs() < 1e-5);
    }

    #[test]
    fn balanced_mse_is_zero_for_exact_duplicates() {
        let loss = balanced_mse(t1(&[1.0, 1.0]), t1(&[1.0, 1.0]), &[10, 10], 0.2);
        assert!(scalar(loss) < 1e-4);
    }

    #[test]
    fn pairwise_ranking_is_zero_when_order_clears_margin() {
        assert!(scalar(pairwise_ranking(t1(&[0.9, 0.6, 0.2]), &[10, 7, 2], 0.05)) < 1e-6);
        assert!(scalar(pairwise_ranking(t1(&[0.2, 0.6, 0.9]), &[10, 7, 2], 0.05)) > 0.1);
    }

    #[test]
    fn triplet_ranking_is_zero_when_positives_are_closer() {
        let pooled = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 1.1, 0.0, 4.0, 0.0], Shape::new([3, 2])),
            &Device::default(),
        );
        assert!(scalar(triplet_ranking(pooled, &[10, 9, 2], 0.1)) < 1e-6);
    }

    #[test]
    fn rank_regularization_has_usable_range() {
        let device = Device::default();
        // Orthogonal embeddings for identical scores: large loss.
        let orthogonal = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], Shape::new([2, 2])),
            &device,
        );
        let mismatched = scalar(rank_regularization(orthogonal, t1(&[1.0, 1.0]), 0.15));
        // Parallel embeddings for identical scores: no loss.
        let parallel = Tensor::<2>::ones([2, 2], &device);
        let matched = scalar(rank_regularization(parallel, t1(&[1.0, 1.0]), 0.15));
        assert!(matched < 1e-6);
        assert!(mismatched > 0.5, "got {mismatched}");
    }

    #[test]
    fn weights_upweight_rare_buckets() {
        let counts = BTreeMap::from([(10, 90usize), (9, 8), (2, 2)]);
        for weights in [
            inverse_frequency_weights(&counts),
            effective_number_weights(&counts, 0.999),
        ] {
            assert!(weights[&2] > weights[&9] && weights[&9] > weights[&10]);
            assert!(weights.values().all(|w| *w <= 100.0));
        }
    }
}
