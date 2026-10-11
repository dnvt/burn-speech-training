//! Evaluate a checkpoint against SpeechOcean762 word scores.
//!
//! Four goodness-of-pronunciation (GOP) baselines are computed from the CTC
//! output, plus the fine-tuned scoring head when one is given. Each method
//! is reported as Spearman ρ against the human scores, with a 95% bootstrap
//! interval resampled by utterance.
//!
//! Word boundaries are not known, so every method uses the same
//! approximation as training: an utterance's frames are split evenly among
//! its words ([`proportional_word_frame_span`]).

use std::fmt::Write as _;
use std::path::PathBuf;

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

use crate::backend::{self, InferBackend};
use crate::checkpoint;
use crate::data::{audio, so762};
use crate::error::{Error, Result};
use crate::finetune::head::{InputSource, ScoringHead};
use crate::mfcc::{FeatureMode, MfccExtractor};
use crate::model::proportional_word_frame_span;
use crate::phoneme_map::{arpabet_to_idx, BLANK_IDX};
use crate::stats::Correlation;

const BOOTSTRAP_SAMPLES: usize = 2000;
/// Score given to a target phoneme that never appears in the best path.
const GOP_SA_DELETION_SCORE: f64 = -20.0;

#[derive(Debug, Clone)]
pub struct EvaluateConfig {
    pub checkpoint: PathBuf,
    /// Scoring head from [`crate::finetune::finetune`]; its saved config
    /// overrides `feature_mode`.
    pub scoring_head: Option<PathBuf>,
    /// Word-score JSON (see [`crate::data::so762`]); needs `phonemes` for the
    /// GOP methods.
    pub dataset: PathBuf,
    pub data_root: PathBuf,
    pub max_samples: Option<usize>,
    pub feature_mode: FeatureMode,
}

#[derive(Debug, Clone)]
pub struct MethodResult {
    pub name: &'static str,
    pub description: &'static str,
    pub correlation: Correlation,
}

#[derive(Debug, Clone)]
pub struct EvaluationReport {
    pub checkpoint: PathBuf,
    pub scoring_head: Option<PathBuf>,
    pub feature_mode: FeatureMode,
    pub words_scored: usize,
    pub words_skipped: usize,
    pub methods: Vec<MethodResult>,
}

impl EvaluationReport {
    /// Method with the highest Spearman ρ.
    pub fn best(&self) -> Option<&MethodResult> {
        self.methods
            .iter()
            .max_by(|a, b| a.correlation.spearman.total_cmp(&b.correlation.spearman))
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::from("# Evaluation\n\n");
        let _ = writeln!(out, "- Checkpoint: `{}`", self.checkpoint.display());
        if let Some(head) = &self.scoring_head {
            let _ = writeln!(out, "- Scoring head: `{}`", head.display());
        }
        let _ = writeln!(out, "- Features: {}", self.feature_mode.label());
        let _ = writeln!(
            out,
            "- Words: {} scored, {} skipped\n",
            self.words_scored, self.words_skipped
        );
        out.push_str("| Method | Words | Spearman ρ | 95% CI | Pearson | Description |\n");
        out.push_str("|---|---|---|---|---|---|\n");
        for m in &self.methods {
            let c = &m.correlation;
            let _ = writeln!(
                out,
                "| {} | {} | {:.4} | [{:.4}, {:.4}] | {:.4} | {} |",
                m.name,
                c.count,
                c.spearman,
                c.spearman_ci95.0,
                c.spearman_ci95.1,
                c.pearson,
                m.description
            );
        }
        out
    }
}

#[derive(Default)]
struct Scores {
    predicted: Vec<f64>,
    human: Vec<f64>,
    groups: Vec<usize>,
}

impl Scores {
    fn push(&mut self, predicted: f64, human: f64, group: usize) {
        self.predicted.push(predicted);
        self.human.push(human);
        self.groups.push(group);
    }

    fn result(&self, name: &'static str, description: &'static str) -> Option<MethodResult> {
        (!self.predicted.is_empty()).then(|| MethodResult {
            name,
            description,
            correlation: Correlation::compute(
                &self.predicted,
                &self.human,
                &self.groups,
                BOOTSTRAP_SAMPLES,
            ),
        })
    }
}

/// CTC outputs for one utterance, copied to the host.
struct CtcFrames {
    log_probs: Vec<f32>,
    logits: Vec<f32>,
    frames: usize,
    classes: usize,
}

impl CtcFrames {
    fn row<'a>(&self, data: &'a [f32], frame: usize) -> &'a [f32] {
        &data[frame * self.classes..(frame + 1) * self.classes]
    }
}

pub fn evaluate(config: &EvaluateConfig) -> Result<EvaluationReport> {
    let device = backend::default_device();

    let head = match &config.scoring_head {
        Some(path) => {
            let (head, head_config) = ScoringHead::<InferBackend>::load(path, &device)?;
            if head_config.input_source != InputSource::Backbone {
                return Err(Error::config(
                    "this scoring head reads teacher features; evaluate it with \
                     `experiments::teacher::evaluate_head`",
                ));
            }
            Some((head, head_config))
        }
        None => None,
    };
    let feature_mode = head
        .as_ref()
        .map_or(config.feature_mode, |(_, c)| c.feature_mode);

    let model_config = checkpoint::model_config_for(&config.checkpoint, feature_mode)?;
    checkpoint::check_feature_mode(&model_config, feature_mode)?;
    if let Some((_, head_config)) = &head {
        if head_config.input_dim != model_config.channels[3] {
            return Err(Error::config(format!(
                "the scoring head expects {}-dim features but the checkpoint produces {}; \
                 use the backbone the head was fine-tuned with",
                head_config.input_dim, model_config.channels[3]
            )));
        }
    }
    let model = checkpoint::load(
        model_config.init::<InferBackend>(&device)?,
        &config.checkpoint,
        &device,
    )?;
    let extractor = MfccExtractor::with_mode(feature_mode);

    let words = so762::load(&config.dataset, config.max_samples)?;
    let groups = so762::group_by_audio(&words);
    tracing::info!(
        "evaluating {} words in {} utterances",
        groups.values().map(Vec::len).sum::<usize>(),
        groups.len()
    );

    let mut log_prob = Scores::default();
    let mut max_logit = Scores::default();
    let mut margin = Scores::default();
    let mut self_aligned = Scores::default();
    let mut head_scores = Scores::default();
    let (mut scored, mut skipped) = (0usize, 0usize);

    for (utt_idx, (audio_file, utt_words)) in groups.iter().enumerate() {
        let frames = audio::load(&config.data_root.join(audio_file))
            .and_then(|samples| extractor.extract_frames(&samples));
        let frames = match frames {
            Ok(frames) if !frames.is_empty() => frames,
            Ok(_) | Err(_) => {
                if let Err(err) = frames {
                    tracing::debug!("skipping {audio_file}: {err}");
                }
                skipped += utt_words.len();
                continue;
            }
        };
        let input = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(
                frames.concat(),
                Shape::new([1, frames.len(), extractor.feature_dim()]),
            ),
            &device,
        );
        let output = model.forward(input);
        let ctc = ctc_frames(output.ctc_log_probs, output.ctc_logits);
        let best_path = ctc_best_path(&ctc);

        let head_word_scores = head.as_ref().map(|(head, head_config)| {
            let positions: Vec<(usize, usize)> = utt_words
                .iter()
                .map(|w| (w.word_position, w.num_words()))
                .collect();
            let features = output.frame_features.clone().squeeze_dim::<2>(0);
            head.score_words(&features, head_config, &positions)
        });

        for (word_idx, word) in utt_words.iter().enumerate() {
            let Some(span) =
                proportional_word_frame_span(ctc.frames, word.num_words(), word.word_position)
            else {
                skipped += 1;
                continue;
            };
            let targets: Vec<usize> = word
                .phonemes
                .iter()
                .filter_map(|p| arpabet_to_idx(p))
                .map(|idx| idx as usize)
                .collect();

            let mut any = false;
            if let Some((lp, ml, mg)) = gop_scores(&ctc, span, &targets) {
                log_prob.push(lp, word.human_score, utt_idx);
                max_logit.push(ml, word.human_score, utt_idx);
                margin.push(mg, word.human_score, utt_idx);
                any = true;
            }
            if let Some(sa) = gop_self_aligned(&ctc, &best_path, span, &targets) {
                self_aligned.push(sa, word.human_score, utt_idx);
                any = true;
            }
            if let Some(Some(score)) = head_word_scores.as_ref().map(|s| s[word_idx]) {
                head_scores.push(f64::from(score), word.human_score, utt_idx);
                any = true;
            }
            if any {
                scored += 1;
            } else {
                skipped += 1;
            }
        }
    }

    if scored < 10 {
        return Err(Error::data(format!(
            "only {scored} words could be scored; need at least 10"
        )));
    }

    let methods = [
        log_prob.result(
            "GOP-LogProb",
            "mean over phonemes of the max log-probability in the word",
        ),
        max_logit.result("GOP-MaxLogit", "same with raw logits"),
        margin.result(
            "GOP-Margin",
            "target logit minus the best other logit (blank included)",
        ),
        self_aligned.result("GOP-SA", "log-probability over the CTC best-path segment"),
        head_scores.result("ScoringHead", "fine-tuned scoring head"),
    ]
    .into_iter()
    .flatten()
    .collect();

    Ok(EvaluationReport {
        checkpoint: config.checkpoint.clone(),
        scoring_head: config.scoring_head.clone(),
        feature_mode,
        words_scored: scored,
        words_skipped: skipped,
        methods,
    })
}

/// `[frames, 1, classes]` CTC outputs → host vectors.
fn ctc_frames<B: Backend>(log_probs: Tensor<B, 3>, logits: Tensor<B, 3>) -> CtcFrames {
    let [frames, _, classes] = log_probs.dims();
    let to_vec = |t: Tensor<B, 3>| t.into_data().iter::<f32>().collect::<Vec<_>>();
    CtcFrames {
        log_probs: to_vec(log_probs),
        logits: to_vec(logits),
        frames,
        classes,
    }
}

/// `(log-prob, max-logit, margin)` GOP for a word: for each target phoneme,
/// the best value over the word's frames; then the mean over phonemes.
fn gop_scores(
    ctc: &CtcFrames,
    (start, end): (usize, usize),
    targets: &[usize],
) -> Option<(f64, f64, f64)> {
    let targets: Vec<usize> = targets
        .iter()
        .copied()
        .filter(|&t| t < ctc.classes)
        .collect();
    if targets.is_empty() {
        return None;
    }
    let mut sums = (0.0, 0.0, 0.0);
    for &target in &targets {
        let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for frame in start..end {
            let log_probs = ctc.row(&ctc.log_probs, frame);
            let logits = ctc.row(&ctc.logits, frame);
            let best_other = logits
                .iter()
                .enumerate()
                .filter(|&(c, _)| c != target)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
            best.0 = best.0.max(f64::from(log_probs[target]));
            best.1 = best.1.max(f64::from(logits[target]));
            best.2 = best.2.max(f64::from(logits[target] - best_other));
        }
        sums.0 += best.0;
        sums.1 += best.1;
        sums.2 += best.2;
    }
    let n = targets.len() as f64;
    Some((sums.0 / n, sums.1 / n, sums.2 / n))
}

/// Greedy CTC decoding: per-frame argmax, then merge runs and drop blanks.
/// Returns `(class, start_frame, end_frame)` segments.
fn ctc_best_path(ctc: &CtcFrames) -> Vec<(usize, usize, usize)> {
    let argmax: Vec<usize> = (0..ctc.frames)
        .map(|frame| {
            ctc.row(&ctc.log_probs, frame)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(c, _)| c)
        })
        .collect();

    let mut segments = Vec::new();
    let mut start = 0;
    while start < argmax.len() {
        let class = argmax[start];
        let end = start + argmax[start..].iter().take_while(|&&c| c == class).count();
        if class != BLANK_IDX as usize {
            segments.push((class, start, end));
        }
        start = end;
    }
    segments
}

/// Self-aligned GOP: each target phoneme is scored by its mean log-prob over
/// the first matching best-path segment inside the word, or
/// [`GOP_SA_DELETION_SCORE`] if the decoder never emitted it there.
fn gop_self_aligned(
    ctc: &CtcFrames,
    best_path: &[(usize, usize, usize)],
    (start, end): (usize, usize),
    targets: &[usize],
) -> Option<f64> {
    let targets: Vec<usize> = targets
        .iter()
        .copied()
        .filter(|&t| t < ctc.classes)
        .collect();
    if targets.is_empty() {
        return None;
    }
    let total: f64 = targets
        .iter()
        .map(|&target| {
            best_path
                .iter()
                .find(|&&(class, s, e)| class == target && s < end && e > start)
                .map_or(GOP_SA_DELETION_SCORE, |&(_, s, e)| {
                    let (s, e) = (s.max(start), e.min(end));
                    (s..e)
                        .map(|frame| f64::from(ctc.row(&ctc.log_probs, frame)[target]))
                        .sum::<f64>()
                        / (e - s) as f64
                })
        })
        .sum();
    Some(total / targets.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finetune::tests::fake_so762;

    fn ctc(frames: usize, classes: usize, log_probs: Vec<f32>) -> CtcFrames {
        CtcFrames {
            logits: log_probs.clone(),
            log_probs,
            frames,
            classes,
        }
    }

    #[test]
    fn best_path_merges_runs_and_drops_blanks() {
        // Frame argmax: 0 1 1 0 2 2 1
        let rows = [0, 1, 1, 0, 2, 2, 1];
        let mut lp = vec![-10.0f32; rows.len() * 3];
        for (t, &c) in rows.iter().enumerate() {
            lp[t * 3 + c] = 0.0;
        }
        let path = ctc_best_path(&ctc(rows.len(), 3, lp));
        assert_eq!(path, [(1, 1, 3), (2, 4, 6), (1, 6, 7)]);
    }

    #[test]
    fn gop_rewards_present_phonemes() {
        // Two frames, three classes; class 1 is confident in frame 0.
        let lp = vec![-5.0, -0.1, -3.0, -0.2, -4.0, -2.0];
        let c = ctc(2, 3, lp);
        let (present, _, _) = gop_scores(&c, (0, 2), &[1]).expect("score");
        let (absent, _, _) = gop_scores(&c, (0, 2), &[2]).expect("score");
        assert!(present > absent);
        assert!(gop_scores(&c, (0, 2), &[9]).is_none(), "out-of-range class");

        let path = ctc_best_path(&c);
        assert!(gop_self_aligned(&c, &path, (0, 2), &[1]).expect("sa") > GOP_SA_DELETION_SCORE);
        assert_eq!(
            gop_self_aligned(&c, &path, (0, 2), &[2]),
            Some(GOP_SA_DELETION_SCORE)
        );
    }

    #[test]
    fn evaluates_a_trained_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dataset = fake_so762(dir.path(), "eval", 4);
        let ft = crate::finetune::finetune(&crate::finetune::FineTuneConfig {
            dataset: dataset.clone(),
            data_root: dir.path().to_path_buf(),
            epochs: 1,
            warmup_epochs: 0,
            head_hidden_dim: 8,
            model: Some(crate::finetune::tests::tiny_backbone()),
            checkpoint_dir: dir.path().join("ckpt"),
            ..Default::default()
        })
        .expect("finetune");

        let report = evaluate(&EvaluateConfig {
            checkpoint: ft.final_model,
            scoring_head: Some(ft.final_head),
            dataset,
            data_root: dir.path().to_path_buf(),
            max_samples: None,
            feature_mode: FeatureMode::Mfcc39,
        })
        .expect("evaluate");

        assert_eq!(report.words_scored, 12);

        // The same head with a backbone of a different width is rejected.
        let other = dir.path().join("other");
        std::fs::create_dir_all(&other).expect("mkdir");
        std::fs::copy(&report.checkpoint, other.join("model")).expect("copy");
        let wider = crate::model::SpeechAlignerConfig {
            channels: [8, 8, 16, 32],
            ..crate::finetune::tests::tiny_backbone()
        };
        checkpoint::save_model_config(&other, &wider).expect("config");
        let err = evaluate(&EvaluateConfig {
            checkpoint: other.join("model"),
            scoring_head: report.scoring_head.clone(),
            dataset: dir.path().join("eval.json"),
            data_root: dir.path().to_path_buf(),
            max_samples: None,
            feature_mode: FeatureMode::Mfcc39,
        })
        .expect_err("width mismatch");
        assert!(err.to_string().contains("scoring head expects"), "{err}");
        let names: Vec<&str> = report.methods.iter().map(|m| m.name).collect();
        assert_eq!(
            names,
            [
                "GOP-LogProb",
                "GOP-MaxLogit",
                "GOP-Margin",
                "GOP-SA",
                "ScoringHead"
            ]
        );
        assert!(report.to_markdown().contains("| ScoringHead | 12 |"));
    }
}
