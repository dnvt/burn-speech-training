//! Correlation statistics used to compare predicted and human scores.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Agreement between predicted and human scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correlation {
    pub spearman: f64,
    pub pearson: f64,
    /// 95% bootstrap interval for `spearman`, resampled by group.
    pub spearman_ci95: (f64, f64),
    pub count: usize,
}

impl Correlation {
    /// `groups[i]` is the cluster of point `i` (e.g. its utterance index),
    /// see [`bootstrap_ci_95`].
    pub fn compute(predicted: &[f64], human: &[f64], groups: &[usize], n_boot: usize) -> Self {
        Self {
            spearman: spearman(predicted, human),
            pearson: pearson(predicted, human),
            spearman_ci95: bootstrap_ci_95(predicted, human, groups, n_boot),
            count: predicted.len(),
        }
    }
}

impl std::fmt::Display for Correlation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ρ={:.4} [{:.4}, {:.4}] PCC={:.4} n={}",
            self.spearman, self.spearman_ci95.0, self.spearman_ci95.1, self.pearson, self.count
        )
    }
}

/// Spearman rank correlation, with average ranks for ties.
///
/// Returns 0.0 for fewer than two points, mismatched lengths, or zero
/// variance.
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    pearson(&ranks(x), &ranks(y))
}

/// Pearson correlation coefficient. Same edge cases as [`spearman`].
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || n != y.len() {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        let (dx, dy) = (xi - mean_x, yi - mean_y);
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// 1-based ranks; tied values share the average of their ranks.
pub(crate) fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

    let mut result = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[j]] == values[order[i]] {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for &idx in &order[i..j] {
            result[idx] = avg_rank;
        }
        i = j;
    }
    result
}

/// 95% percentile-bootstrap confidence interval for Spearman's ρ.
///
/// Resampling is done over *groups* rather than individual points:
/// `groups[i]` identifies the cluster point `i` belongs to (for example the
/// utterance a word came from). Words from the same utterance are
/// correlated, so resampling them independently gives intervals that are
/// too narrow. Pass a distinct id per point to get the ordinary bootstrap.
///
/// Uses a fixed seed so reports are reproducible. Returns `(0.0, 0.0)` for
/// fewer than 10 points, fewer than 2 groups, or `n_boot == 0`.
pub fn bootstrap_ci_95(x: &[f64], y: &[f64], groups: &[usize], n_boot: usize) -> (f64, f64) {
    if x.len() < 10 || x.len() != y.len() || x.len() != groups.len() || n_boot == 0 {
        return (0.0, 0.0);
    }

    let mut members: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (idx, &group) in groups.iter().enumerate() {
        members.entry(group).or_default().push(idx);
    }
    let members: Vec<Vec<usize>> = members.into_values().collect();
    if members.len() < 2 {
        return (0.0, 0.0);
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut rhos = Vec::with_capacity(n_boot);
    let (mut bx, mut by) = (Vec::with_capacity(x.len()), Vec::with_capacity(x.len()));
    for _ in 0..n_boot {
        bx.clear();
        by.clear();
        for _ in 0..members.len() {
            for &idx in &members[rng.random_range(0..members.len())] {
                bx.push(x[idx]);
                by.push(y[idx]);
            }
        }
        rhos.push(spearman(&bx, &by));
    }

    rhos.sort_by(f64::total_cmp);
    let last = (rhos.len() - 1) as f64;
    (
        rhos[(0.025 * last).round() as usize],
        rhos[(0.975 * last).round() as usize],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn spearman_monotonic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx(spearman(&x, &[10.0, 20.0, 30.0, 40.0, 50.0]), 1.0));
        assert!(approx(spearman(&x, &[50.0, 40.0, 30.0, 20.0, 10.0]), -1.0));
        // Monotonic but non-linear: Spearman is 1, Pearson is not.
        let y = [1.0, 8.0, 27.0, 64.0, 125.0];
        assert!(approx(spearman(&x, &y), 1.0));
        assert!(pearson(&x, &y) < 1.0);
    }

    #[test]
    fn spearman_degenerate_inputs() {
        assert!(approx(spearman(&[], &[]), 0.0));
        assert!(approx(spearman(&[1.0], &[2.0]), 0.0));
        assert!(approx(spearman(&[1.0, 2.0], &[1.0]), 0.0));
        assert!(approx(spearman(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0]), 0.0));
    }

    #[test]
    fn ranks_average_ties() {
        assert_eq!(ranks(&[3.0, 1.0, 4.0, 1.0, 5.0]), [3.0, 1.5, 4.0, 1.5, 5.0]);
        assert_eq!(ranks(&[7.0, 7.0, 7.0]), [2.0, 2.0, 2.0]);
    }

    #[test]
    fn ranks_handle_nan_without_panicking() {
        let r = ranks(&[1.0, f64::NAN, 0.5]);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn bootstrap_is_tight_for_perfect_correlation() {
        let x: Vec<f64> = (0..100).map(f64::from).collect();
        let groups: Vec<usize> = (0..100).collect();
        let (lo, hi) = bootstrap_ci_95(&x, &x, &groups, 200);
        assert!(lo > 0.9 && hi > 0.9 && lo <= hi, "({lo}, {hi})");
    }

    #[test]
    fn bootstrap_clusters_widen_the_interval() {
        // 10 clusters of 10 points; within a cluster x and y move together,
        // across clusters the relationship is noisy.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        for g in 0..10 {
            let offset = [3.0, -2.0, 5.0, 0.0, -4.0, 1.0, 2.0, -1.0, 4.0, -3.0][g];
            for i in 0..10 {
                x.push(g as f64 * 10.0 + i as f64);
                y.push(offset * 10.0 + i as f64);
                groups.push(g);
            }
        }
        let singletons: Vec<usize> = (0..x.len()).collect();
        let (lo_c, hi_c) = bootstrap_ci_95(&x, &y, &groups, 300);
        let (lo_s, hi_s) = bootstrap_ci_95(&x, &y, &singletons, 300);
        assert!(hi_c - lo_c > hi_s - lo_s);
    }

    #[test]
    fn bootstrap_degenerate_inputs() {
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let groups = vec![0; 20];
        assert_eq!(
            bootstrap_ci_95(&x[..2], &x[..2], &groups[..2], 100),
            (0.0, 0.0)
        );
        assert_eq!(bootstrap_ci_95(&x, &x, &groups, 100), (0.0, 0.0));
        let singletons: Vec<usize> = (0..20).collect();
        assert_eq!(bootstrap_ci_95(&x, &x, &singletons, 0), (0.0, 0.0));
    }
}
