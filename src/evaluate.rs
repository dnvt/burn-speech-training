//! Evaluate a `SpeechAligner` checkpoint against `SpeechOcean762` word-level
//! human scores. Computes CTC-based GOP and reports Spearman correlation.

mod inner {
    use std::cmp::Ordering;
    use std::path::{Path, PathBuf};

    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;
    use serde::Deserialize;

    use crate::dataset::load_audio_samples;
    use crate::error::{Error, Result};
    use crate::finetune::{
        load_scoring_head_checkpoint, load_scoring_head_config, load_teacher_feature_cache,
        scoring_input_dim_from_config, scoring_predictions_from_logits, ScoringHead,
        ScoringHeadConfig, ScoringInputSource,
    };
    use crate::mfcc::{FeatureMode, MfccExtractor};
    use crate::model::{proportional_word_frame_span, SpeechAligner, SpeechAlignerConfig};
    use crate::phoneme_map::arpabet_to_idx;
    use crate::ui::{detail, section, step, success};

    const START_GATE: f64 = 0.30;
    const SUCCESS_GATE: f64 = 0.40;

    // ── SpeechOcean762 JSON schema ──────────────────────────────────────

    #[derive(Deserialize)]
    struct WordDataset {
        #[allow(dead_code)]
        metadata: serde_json::Value,
        samples: Vec<WordSample>,
    }

    #[derive(Deserialize)]
    struct WordSample {
        #[allow(dead_code)]
        word_id: String,
        #[allow(dead_code)]
        text: String,
        human_score: f64,
        audio_file: String,
        phonemes: Vec<String>,
        #[allow(dead_code)]
        utterance_id: String,
        word_position: usize,
        sentence_text: String,
    }

    struct EvaluationMetric {
        name: &'static str,
        rho: f64,
        pcc: f64,
        ci: (f64, f64),
        words_scored: usize,
        description: &'static str,
    }

    struct MetricAccumulator {
        name: &'static str,
        description: &'static str,
        predicted: Vec<f64>,
        human: Vec<f64>,
    }

    impl MetricAccumulator {
        fn new(name: &'static str, description: &'static str, capacity: usize) -> Self {
            Self {
                name,
                description,
                predicted: Vec::with_capacity(capacity),
                human: Vec::with_capacity(capacity),
            }
        }

        fn push(&mut self, predicted: f64, human: f64) {
            self.predicted.push(predicted);
            self.human.push(human);
        }

        fn words_scored(&self) -> usize {
            self.predicted.len().min(self.human.len())
        }

        fn finalize(&self) -> Option<EvaluationMetric> {
            let words_scored = self.words_scored();
            if words_scored == 0 {
                return None;
            }
            Some(EvaluationMetric {
                name: self.name,
                rho: spearman_rho(&self.predicted, &self.human),
                pcc: pearson_r(&self.predicted, &self.human),
                ci: bootstrap_ci_95(&self.predicted, &self.human, 2000),
                words_scored,
                description: self.description,
            })
        }
    }

    /// Arguments for checkpoint evaluation.
    pub struct EvaluateArgs {
        pub checkpoint: PathBuf,
        pub scoring_head: Option<PathBuf>,
        pub dataset: PathBuf,
        pub data_root: PathBuf,
        pub max_samples: Option<usize>,
        pub feature_mode: FeatureMode,
        pub distillation_features: Option<PathBuf>,
    }

    /// Run the evaluation pipeline (always on CPU via `NdArray`).
    pub fn execute_evaluate(args: &EvaluateArgs) -> Result<()> {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::NdArray;

        type B = NdArray<f32>;

        let device = NdArrayDevice::default();
        section("SpeechAligner Checkpoint Evaluation (SpeechOcean762)");
        let mut resolved_feature_mode = args.feature_mode;
        let teacher_feature_cache = args
            .distillation_features
            .as_ref()
            .map(|cache_dir| load_teacher_feature_cache(cache_dir))
            .transpose()?;

        // ── 1. Load checkpoint ──────────────────────────────────────
        step("Loading checkpoint");
        let mut scoring_config = ScoringHeadConfig::default();
        if let Some(path) = args.scoring_head.as_ref() {
            scoring_config = load_scoring_head_config(path);
            resolved_feature_mode = scoring_config.feature_mode;
        }
        let config = SpeechAlignerConfig::for_feature_mode(resolved_feature_mode);
        let model = load_checkpoint::<B>(&args.checkpoint, &config, &device)?;
        let param_count: usize = model.num_params();
        detail(&format!("  Parameters: {param_count}"));
        detail(&format!(
            "  Feature mode: {}",
            resolved_feature_mode.label()
        ));
        let scoring_input_dim = scoring_input_dim_from_config(&scoring_config, config.channels[3]);
        if args.scoring_head.is_some()
            && matches!(
                scoring_config.scoring_input_source,
                ScoringInputSource::TeacherCache
            )
            && teacher_feature_cache.is_none()
        {
            return Err(Error::config(
                "Scoring head requires --distillation-features to replay cached teacher inputs",
            ));
        }

        let (scoring_head, scoring_config) = match args.scoring_head.as_ref() {
            Some(path) => {
                step("Loading scoring head");
                let scoring_head =
                    load_scoring_head_checkpoint::<B>(path, scoring_input_dim, &device)?;
                detail(&format!("  Parameters: {}", scoring_head.num_params()));
                detail(&format!(
                    "  Scoring input: {} ({scoring_input_dim} dims)",
                    scoring_config.scoring_input_source.label()
                ));
                detail(&format!(
                    "  Scoring mode: {}",
                    if scoring_config.ordinal_loss {
                        "ordinal"
                    } else if scoring_config.balanced_mse {
                        "balanced_mse"
                    } else {
                        "regression"
                    }
                ));
                (Some(scoring_head), scoring_config)
            }
            None => (None, ScoringHeadConfig::default()),
        };

        // ── 2. Load dataset ─────────────────────────────────────────
        step("Loading SpeechOcean762 holdout");
        let raw = std::fs::read_to_string(&args.dataset)
            .map_err(|e| Error::config(format!("Cannot read dataset: {e}")))?;
        let dataset: WordDataset = serde_json::from_str(&raw)
            .map_err(|e| Error::config(format!("Invalid dataset JSON: {e}")))?;

        let max = args.max_samples.unwrap_or(dataset.samples.len());
        let limit = max.min(dataset.samples.len());
        let samples = dataset.samples.get(..limit).unwrap_or(&dataset.samples);
        detail(&format!(
            "  Samples: {} (of {} total)",
            samples.len(),
            dataset.samples.len()
        ));

        // ── 3. Group by utterance (avoid re-loading same audio) ─────
        step("Grouping samples by utterance");
        let mut utterance_groups: std::collections::BTreeMap<String, Vec<&WordSample>> =
            std::collections::BTreeMap::new();
        for s in samples {
            utterance_groups
                .entry(s.audio_file.clone())
                .or_default()
                .push(s);
        }
        detail(&format!("  Unique utterances: {}", utterance_groups.len()));

        // ── 4. Run inference + compute per-word scores ───────────────
        let inference_label = if scoring_head.is_some() {
            "Running inference (GOP variants + SO762 scoring head)"
        } else {
            "Running inference (log-prob + logit + self-aligned GOP)"
        };
        step(inference_label);
        let mfcc = MfccExtractor::with_mode(resolved_feature_mode);

        let mut logprob_metric = MetricAccumulator::new(
            "GOPLogProb",
            "Max log-softmax prob per phoneme",
            samples.len(),
        );
        let mut max_logit_metric =
            MetricAccumulator::new("GOPMaxLogit", "Max raw logit per phoneme", samples.len());
        let mut margin_metric = MetricAccumulator::new(
            "GOPMargin",
            "Target logit minus next-best logit",
            samples.len(),
        );
        let mut sa_metric = MetricAccumulator::new(
            "GOP-SA",
            "Self-aligned CTC pronunciation score",
            samples.len(),
        );
        let mut scoring_head_metric = scoring_head.as_ref().map(|_| {
            MetricAccumulator::new(
                "ScoringHead",
                if scoring_config.ordinal_softmax_loss {
                    "SO762 fine-tuned 5-class ordinal scorer"
                } else if scoring_config.ordinal_loss {
                    "SO762 fine-tuned threshold-ordinal scorer"
                } else {
                    "SO762 fine-tuned regression scorer"
                },
                samples.len(),
            )
        });
        let mut scored_any_humans: Vec<f64> = Vec::with_capacity(samples.len());
        let mut skipped = 0usize;
        let mut processed = 0usize;
        let total_utterances = utterance_groups.len();

        for (idx, (audio_path, words)) in utterance_groups.iter().enumerate() {
            let teacher_frame_features = if scoring_head.is_some()
                && matches!(
                    scoring_config.scoring_input_source,
                    ScoringInputSource::TeacherCache
                ) {
                match teacher_feature_cache
                    .as_ref()
                    .map(|cache| cache.load(audio_path))
                    .transpose()?
                {
                    Some(Some((teacher_features, teacher_frames))) if teacher_frames > 0 => {
                        Some(Tensor::<B, 2>::from_data(
                            burn::tensor::TensorData::new(
                                teacher_features,
                                burn::tensor::Shape::new([teacher_frames, scoring_input_dim]),
                            ),
                            &device,
                        ))
                    }
                    _ => None,
                }
            } else {
                None
            };

            let gop_bundle = match load_audio_samples(&args.data_root.join(audio_path)) {
                Ok(audio) => match mfcc.extract_tensor::<B>(&audio, &device) {
                    Ok(mfcc_tensor) => {
                        let output = model.forward(mfcc_tensor);
                        let dims = output.ctc_log_probs.dims();
                        let num_frames = dims[0];
                        let num_classes = config.num_classes;
                        let log_probs: Tensor<B, 2> = output.ctc_log_probs.squeeze();
                        let raw_logits: Tensor<B, 2> = output.ctc_logits.squeeze();
                        let frame_features = output.frame_features.clone().squeeze_dim::<2>(0);
                        match log_probs.clone().into_data().to_vec() {
                            Ok(lp_data_full) => {
                                let best_path =
                                    ctc_best_path(&lp_data_full, num_frames, num_classes);
                                Some((
                                    log_probs,
                                    raw_logits,
                                    frame_features,
                                    lp_data_full,
                                    best_path,
                                    num_frames,
                                    num_classes,
                                ))
                            }
                            Err(_) => {
                                tracing::warn!(
                                    audio = %audio_path,
                                    "Tensor extraction failed for GOP metrics"
                                );
                                None
                            }
                        }
                    }
                    Err(_) => None,
                },
                Err(_) => None,
            };

            for word in words {
                let scores = gop_bundle.as_ref().and_then(
                    |(log_probs, raw_logits, _, _, _, num_frames, num_classes)| {
                        compute_word_gop_all::<B>(
                            log_probs,
                            raw_logits,
                            *num_frames,
                            *num_classes,
                            &word.phonemes,
                            &word.sentence_text,
                            word.word_position,
                        )
                    },
                );

                let sa_score = gop_bundle.as_ref().and_then(
                    |(_, _, _, lp_data_full, best_path, num_frames, num_classes)| {
                        compute_word_gop_sa(
                            lp_data_full,
                            best_path,
                            *num_frames,
                            *num_classes,
                            &word.phonemes,
                            &word.sentence_text,
                            word.word_position,
                        )
                    },
                );

                let backbone_frame_features = gop_bundle
                    .as_ref()
                    .map(|(_, _, frame_features, _, _, _, _)| frame_features);
                let head_frame_features = if matches!(
                    scoring_config.scoring_input_source,
                    ScoringInputSource::TeacherCache
                ) {
                    teacher_frame_features.as_ref()
                } else {
                    backbone_frame_features
                };

                let head_score = scoring_head.as_ref().zip(head_frame_features).and_then(
                    |(scoring_head, frame_features)| {
                        compute_word_scoring_head(
                            frame_features,
                            scoring_head,
                            &scoring_config,
                            &device,
                            &word.sentence_text,
                            word.word_position,
                        )
                    },
                );

                let mut word_scored = false;
                if let Some((lp, ml, mg)) = scores {
                    logprob_metric.push(lp, word.human_score);
                    max_logit_metric.push(ml, word.human_score);
                    margin_metric.push(mg, word.human_score);
                    word_scored = true;
                }
                if let Some(sa) = sa_score {
                    sa_metric.push(sa, word.human_score);
                    word_scored = true;
                }
                if let (Some(metric), Some(head_score)) = (scoring_head_metric.as_mut(), head_score)
                {
                    metric.push(head_score, word.human_score);
                    word_scored = true;
                }

                if word_scored {
                    scored_any_humans.push(word.human_score);
                    processed += 1;
                } else {
                    skipped += 1;
                }
            }

            if (idx + 1) % 200 == 0 || idx + 1 == total_utterances {
                detail(&format!(
                    "  [{}/{}] utterances, {processed} words scored, {skipped} skipped",
                    idx + 1,
                    total_utterances,
                ));
            }
        }

        detail(&format!("  Final: {processed} scored, {skipped} skipped"));

        if matches!(
            scoring_config.scoring_input_source,
            ScoringInputSource::TeacherCache
        ) && scoring_head_metric
            .as_ref()
            .map_or(0, MetricAccumulator::words_scored)
            == 0
        {
            return Err(Error::config(
                "Teacher-cache scoring head produced 0 words. Ensure --distillation-features \
                 contains the evaluation split, not only the training set.",
            ));
        }

        if processed < 10 {
            return Err(Error::process(format!(
                "Only {processed} words scored — too few for meaningful correlation"
            )));
        }

        // ── 5. Compute Spearman correlation across all active methods ──
        let correlation_label = if scoring_head_metric.is_some() {
            "Computing Spearman correlation (4 GOP methods + scoring head)"
        } else {
            "Computing Spearman correlation (4 GOP methods)"
        };
        step(correlation_label);
        let mut metrics = Vec::new();
        metrics.extend(logprob_metric.finalize());
        metrics.extend(max_logit_metric.finalize());
        metrics.extend(margin_metric.finalize());
        metrics.extend(sa_metric.finalize());
        metrics.extend(
            scoring_head_metric
                .as_ref()
                .and_then(MetricAccumulator::finalize),
        );

        let best_metric = metrics
            .iter()
            .filter(|metric| metric.words_scored >= 10)
            .max_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap_or(Ordering::Equal))
            .ok_or_else(|| Error::process("No evaluation metrics were produced"))?;

        #[allow(clippy::cast_precision_loss)]
        let human_mean = scored_any_humans.iter().sum::<f64>() / processed as f64;

        let result_label = if scoring_head.is_some() {
            "Evaluation Results — GOP + SO762 Scoring Head"
        } else {
            "Evaluation Results — 4 GOP Methods"
        };
        section(result_label);
        detail(&format!("  Words evaluated: {processed}"));
        detail(&format!("  Words skipped:   {skipped}"));
        detail(&format!("  Human scores:    mean={human_mean:.3}"));
        detail("");
        for metric in &metrics {
            detail(&format!(
                "  {:<12} ρ={:.4}  PCC={:.4}  CI [{:.4}, {:.4}]  words={}  ({})",
                metric.name,
                metric.rho,
                metric.pcc,
                metric.ci.0,
                metric.ci.1,
                metric.words_scored,
                metric.description
            ));
        }
        detail("");
        detail(&format!(
            "  Best method: {} (ρ={:.4}, PCC={:.4})",
            best_metric.name, best_metric.rho, best_metric.pcc
        ));

        if best_metric.rho >= SUCCESS_GATE {
            success(&format!(
                "FULL SUCCESS: ρ = {:.4} ≥ {:.2} — Path A full success gate met",
                best_metric.rho, SUCCESS_GATE
            ));
        } else if best_metric.rho >= START_GATE {
            success(&format!(
                "START GATE PASS: ρ = {:.4} ≥ {:.2} — start gate cleared, but {:.2} \
                 full-success gate remains unmet",
                best_metric.rho, START_GATE, SUCCESS_GATE
            ));
        } else {
            detail(&format!(
                "  GATE FAIL: ρ = {:.4} < {:.2} — start gate not met",
                best_metric.rho, START_GATE
            ));
        }

        let (gop_mean, gop_min, gop_max) = if logprob_metric.predicted.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            #[allow(clippy::cast_precision_loss)]
            let gop_mean = logprob_metric.predicted.iter().sum::<f64>()
                / logprob_metric.predicted.len() as f64;
            let gop_min = logprob_metric
                .predicted
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let gop_max = logprob_metric
                .predicted
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            (gop_mean, gop_min, gop_max)
        };
        let report = format_report_multi(
            &args.checkpoint,
            args.scoring_head.as_deref(),
            processed,
            skipped,
            &metrics,
            gop_mean,
            gop_min,
            gop_max,
            human_mean,
            resolved_feature_mode,
        );
        let report_path = args
            .checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("evaluation-report.md");
        std::fs::write(&report_path, &report)
            .map_err(|e| Error::process(format!("Failed to write report: {e}")))?;
        detail(&format!("  Report: {}", report_path.display()));

        Ok(())
    }

    // ── Checkpoint loading ──────────────────────────────────────────

    fn load_checkpoint<B: Backend>(
        path: &Path,
        config: &SpeechAlignerConfig,
        device: &B::Device,
    ) -> Result<SpeechAligner<B>> {
        let bytes = std::fs::read(path).map_err(|e| {
            Error::config(format!("Cannot read checkpoint {}: {e}", path.display()))
        })?;

        let model = config.init(device)?;

        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load(bytes, device)
            .map_err(|e| Error::process(format!("Failed to load checkpoint: {e}")))?;

        let model = model.load_record(record);
        Ok(model)
    }

    fn compute_word_scoring_head<B: Backend>(
        frame_features: &Tensor<B, 2>,
        scoring_head: &ScoringHead<B>,
        scoring_config: &ScoringHeadConfig,
        device: &B::Device,
        sentence_text: &str,
        word_position: usize,
    ) -> Option<f64> {
        let dims = frame_features.dims();
        let num_frames = dims[0];
        let feature_dim = dims[1];
        let num_words = sentence_text.split_whitespace().count();
        let (start, end) = proportional_word_frame_span(num_frames, num_words, word_position)?;

        let span_features = frame_features.clone().slice([start..end, 0..feature_dim]);
        let score = scoring_predictions_from_logits(
            scoring_head.forward_word_span(span_features, scoring_config.pooling_mode),
            scoring_config,
            device,
        )
        .reshape([1]);
        score
            .to_data()
            .to_vec::<f32>()
            .ok()
            .and_then(|values| values.into_iter().next())
            .map(f64::from)
    }

    // ── GOP computation from CTC log-probs ──────────────────────────

    /// Compute a word-level GOP score from CTC log-probabilities.
    ///
    /// Strategy: divide frames proportionally among words in the sentence,
    /// then for each target phoneme, find the max log-prob for that phoneme
    /// class across the word's frame span. Average the per-phoneme scores.
    #[allow(dead_code)] // Replaced by compute_word_gop_all but kept as reference
    fn compute_word_gop<B: Backend>(
        log_probs: &Tensor<B, 2>, // [time, num_classes]
        num_frames: usize,
        phonemes: &[String],
        sentence_text: &str,
        word_position: usize,
    ) -> Option<f64> {
        if phonemes.is_empty() {
            return None;
        }

        // Map phonemes to class indices
        let indices: Vec<i32> = phonemes.iter().filter_map(|p| arpabet_to_idx(p)).collect();

        if indices.is_empty() {
            return None;
        }

        let num_words = sentence_text.split_whitespace().count();
        let (start_frame, end_frame) =
            proportional_word_frame_span(num_frames, num_words, word_position)?;

        // Extract log-probs for this word's frame span (42 = phonemes + CTC blank)
        let num_classes = 42;
        let word_log_probs = log_probs
            .clone()
            .slice([start_frame..end_frame, 0..num_classes]);
        let word_data: Vec<f32> = word_log_probs.into_data().to_vec().ok()?;
        let word_frames = end_frame - start_frame;

        // For each target phoneme, find the max log-prob across frames
        let mut phoneme_gops: Vec<f64> = Vec::with_capacity(indices.len());
        for &class_idx in &indices {
            if class_idx < 0 || class_idx as usize >= num_classes {
                continue;
            }
            let col = class_idx as usize;
            let mut max_log_prob = f64::NEG_INFINITY;
            for frame in 0..word_frames {
                let val = word_data
                    .get(frame * num_classes + col)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                let val_f64 = f64::from(val);
                if val_f64 > max_log_prob {
                    max_log_prob = val_f64;
                }
            }
            phoneme_gops.push(max_log_prob);
        }

        if phoneme_gops.is_empty() {
            return None;
        }

        // Average per-phoneme GOP (in log-prob space, higher = better)
        let avg = phoneme_gops.iter().sum::<f64>() / phoneme_gops.len() as f64;
        Some(avg)
    }

    /// Compute all 3 GOP variants for a word: `(log-prob, max-logit, margin)`.
    ///
    /// - `GOPLogProb`: max log-softmax probability per target phoneme
    /// - `GOPMaxLogit`: max raw logit per target phoneme (Parikh et al.)
    /// - `GOPMargin`: gap between target phoneme logit and next-highest logit
    #[allow(clippy::too_many_arguments)]
    fn compute_word_gop_all<B: Backend>(
        log_probs: &Tensor<B, 2>,  // [time, num_classes] — post log-softmax
        raw_logits: &Tensor<B, 2>, // [time, num_classes] — pre-softmax
        num_frames: usize,
        num_classes: usize,
        phonemes: &[String],
        sentence_text: &str,
        word_position: usize,
    ) -> Option<(f64, f64, f64)> {
        if phonemes.is_empty() {
            return None;
        }

        let indices: Vec<i32> = phonemes.iter().filter_map(|p| arpabet_to_idx(p)).collect();
        if indices.is_empty() {
            return None;
        }

        let num_words = sentence_text.split_whitespace().count();
        let (start_frame, end_frame) =
            proportional_word_frame_span(num_frames, num_words, word_position)?;

        let word_lp = log_probs
            .clone()
            .slice([start_frame..end_frame, 0..num_classes]);
        let word_rl = raw_logits
            .clone()
            .slice([start_frame..end_frame, 0..num_classes]);
        let lp_data: Vec<f32> = word_lp.into_data().to_vec().ok()?;
        let rl_data: Vec<f32> = word_rl.into_data().to_vec().ok()?;
        let word_frames = end_frame - start_frame;

        let mut sum_logprob = 0.0f64;
        let mut sum_max_logit = 0.0f64;
        let mut sum_margin = 0.0f64;
        let mut count = 0usize;

        for &class_idx in &indices {
            if class_idx < 0 || class_idx as usize >= num_classes {
                continue;
            }
            let col = class_idx as usize;

            let mut best_lp = f64::NEG_INFINITY;
            let mut best_logit = f64::NEG_INFINITY;
            let mut best_margin = f64::NEG_INFINITY;

            for frame in 0..word_frames {
                let base = frame * num_classes;

                // GOPLogProb: max log-prob for target phoneme
                let lp_val = f64::from(
                    lp_data
                        .get(base + col)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY),
                );
                if lp_val > best_lp {
                    best_lp = lp_val;
                }

                // GOPMaxLogit: max raw logit for target phoneme
                let logit_val = f64::from(
                    rl_data
                        .get(base + col)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY),
                );
                if logit_val > best_logit {
                    best_logit = logit_val;
                }

                // GOPMargin: target logit minus max non-target logit
                let mut max_other = f64::NEG_INFINITY;
                for c in 0..num_classes {
                    if c != col {
                        let other =
                            f64::from(rl_data.get(base + c).copied().unwrap_or(f32::NEG_INFINITY));
                        if other > max_other {
                            max_other = other;
                        }
                    }
                }
                let margin = logit_val - max_other;
                if margin > best_margin {
                    best_margin = margin;
                }
            }

            sum_logprob += best_lp;
            sum_max_logit += best_logit;
            sum_margin += best_margin;
            count += 1;
        }

        if count == 0 {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let n = count as f64;
        Some((sum_logprob / n, sum_max_logit / n, sum_margin / n))
    }

    // ── GOP-SA (Self-Aligned) via CTC best-path ──────────────────────

    /// CTC best-path decoding: argmax at each frame, collapse repeats + blanks.
    /// Returns a `Vec` of `(class_index, start_frame, end_frame)` segments.
    #[allow(clippy::indexing_slicing)]
    fn ctc_best_path(
        log_probs: &[f32], // flat [time × num_classes]
        num_frames: usize,
        num_classes: usize,
    ) -> Vec<(usize, usize, usize)> {
        if num_frames == 0 || num_classes == 0 || log_probs.len() < num_frames * num_classes {
            return Vec::new();
        }

        // Argmax at each frame
        let mut best_class = Vec::with_capacity(num_frames);
        for t in 0..num_frames {
            let base = t * num_classes;
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let val = log_probs[base + c];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }
            best_class.push(max_idx);
        }

        // Collapse repeats and blanks (blank = index 0)
        let mut segments = Vec::new();
        let mut i = 0;
        while i < num_frames {
            let cls = best_class[i];
            if cls == 0 {
                // blank — skip
                i += 1;
                continue;
            }
            let start = i;
            while i < num_frames && best_class[i] == cls {
                i += 1;
            }
            segments.push((cls, start, i));
        }
        segments
    }

    /// GOP-SA: score each target phoneme using the CTC self-alignment.
    ///
    /// For each target phoneme in the word, find the best-path segment that
    /// matches it, then score as the average log-prob at those frames.
    /// If a target phoneme has no matching segment, it gets a deletion penalty.
    #[allow(clippy::indexing_slicing, clippy::too_many_arguments)]
    fn compute_word_gop_sa(
        log_probs: &[f32],                   // flat [time × num_classes]
        best_path: &[(usize, usize, usize)], // (class, start, end)
        num_frames: usize,
        num_classes: usize,
        phonemes: &[String],
        sentence_text: &str,
        word_position: usize,
    ) -> Option<f64> {
        if phonemes.is_empty() || log_probs.len() < num_frames * num_classes {
            return None;
        }

        let indices: Vec<i32> = phonemes.iter().filter_map(|p| arpabet_to_idx(p)).collect();
        if indices.is_empty() {
            return None;
        }

        let num_words = sentence_text.split_whitespace().count();
        let (w_start, w_end) = proportional_word_frame_span(num_frames, num_words, word_position)?;

        // Find best-path segments within this word's frame range
        let word_segments: Vec<&(usize, usize, usize)> = best_path
            .iter()
            .filter(|(_, s, e)| *s < w_end && *e > w_start)
            .collect();

        // For each target phoneme, find matching segment and score
        let mut total_score = 0.0f64;
        let mut count = 0usize;

        for &class_idx in &indices {
            if class_idx < 0 || class_idx as usize >= num_classes {
                continue;
            }
            let target_cls = class_idx as usize;

            // Find the segment in the word's range that matches this phoneme
            if let Some(&&(_, seg_start, seg_end)) = word_segments
                .iter()
                .find(|&&(cls, _, _)| *cls == target_cls)
            {
                // Average log-prob for this phoneme across its frames
                let frame_start = seg_start.max(w_start);
                let frame_end = seg_end.min(w_end);
                let mut sum = 0.0f64;
                let mut n = 0usize;
                for t in frame_start..frame_end {
                    sum += f64::from(log_probs[t * num_classes + target_cls]);
                    n += 1;
                }
                if n > 0 {
                    total_score += sum / n as f64;
                    count += 1;
                }
            } else {
                // Phoneme not found in best-path → low score (deletion)
                total_score += -20.0; // strong penalty
                count += 1;
            }
        }

        if count == 0 {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        Some(total_score / count as f64)
    }

    // ── Spearman correlation ────────────────────────────────────────

    #[allow(clippy::similar_names, clippy::indexing_slicing)]
    pub fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 2 || x.len() != y.len() {
            return 0.0;
        }

        let rx = ranks(x);
        let ry = ranks(y);

        // Pearson on ranks
        let avg_rx = rx.iter().sum::<f64>() / n as f64;
        let avg_ry = ry.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (rx_val, ry_val) in rx.iter().zip(ry.iter()) {
            let dx = rx_val - avg_rx;
            let dy = ry_val - avg_ry;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-15 {
            return 0.0;
        }
        cov / denom
    }

    /// Pearson correlation coefficient (PCC) on raw values.
    ///
    /// The standard metric for `SpeechOcean762` comparisons in the literature.
    #[allow(clippy::similar_names)]
    pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 2 || x.len() != y.len() {
            return 0.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_x = x.iter().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let avg_y = y.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - avg_x;
            let dy = yi - avg_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-15 {
            return 0.0;
        }
        cov / denom
    }

    #[allow(clippy::indexing_slicing)]
    pub fn ranks(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
                j += 1;
            }
            // Average rank for ties
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for item in indexed.iter().take(j).skip(i) {
                result[item.0] = avg_rank;
            }
            i = j;
        }
        result
    }

    #[allow(clippy::indexing_slicing)]
    pub fn bootstrap_ci_95(x: &[f64], y: &[f64], n_boot: usize) -> (f64, f64) {
        let n = x.len();
        if n < 10 {
            return (0.0, 0.0);
        }

        // Simple LCG for reproducible bootstrap
        let mut rng_state: u64 = 42;
        let mut rhos: Vec<f64> = Vec::with_capacity(n_boot);

        for _ in 0..n_boot {
            let mut bx = Vec::with_capacity(n);
            let mut by = Vec::with_capacity(n);
            for _ in 0..n {
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let idx = ((rng_state >> 33) as usize) % n;
                bx.push(x[idx]);
                by.push(y[idx]);
            }
            rhos.push(spearman_rho(&bx, &by));
        }

        rhos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = (0.025 * n_boot as f64) as usize;
        let hi_idx = (0.975 * n_boot as f64) as usize;
        (
            rhos.get(lo_idx).copied().unwrap_or(0.0),
            rhos.get(hi_idx.min(rhos.len() - 1)).copied().unwrap_or(0.0),
        )
    }

    // ── Report formatting ───────────────────────────────────────────

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn format_report_multi(
        checkpoint: &Path,
        scoring_head: Option<&Path>,
        scored: usize,
        skipped: usize,
        metrics: &[EvaluationMetric],
        gop_mean: f64,
        gop_min: f64,
        gop_max: f64,
        human_mean: f64,
        feature_mode: FeatureMode,
    ) -> String {
        use std::fmt::Write as _;

        let best_metric = metrics
            .iter()
            .filter(|metric| metric.words_scored >= 10)
            .max_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap_or(Ordering::Equal));
        let best_metric = match best_metric {
            Some(metric) => metric,
            None => {
                return String::from("# SpeechAligner Evaluation Report\n\nNo metrics available.\n")
            }
        };
        let mut r = String::new();
        r.push_str("# SpeechAligner Evaluation Report\n\n");
        let _ = writeln!(
            r,
            "**Date**: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        );
        let _ = writeln!(r, "**Checkpoint**: `{}`", checkpoint.display());
        if let Some(scoring_head) = scoring_head {
            let _ = writeln!(r, "**Scoring head**: `{}`", scoring_head.display());
        }
        let _ = writeln!(r, "**Feature mode**: {}", feature_mode.label());
        let _ = writeln!(r, "**Dataset**: SpeechOcean762 holdout (word-level)");
        let _ = writeln!(r, "**Words Scored**: {scored}");
        let _ = writeln!(r, "**Words Skipped**: {skipped}");
        let _ = writeln!(r, "**Human score mean**: {human_mean:.3}");
        let _ = writeln!(
            r,
            "**GOP LogProb mean**: {gop_mean:.3} [{gop_min:.3}, {gop_max:.3}]"
        );
        r.push_str("\n---\n\n");
        r.push_str("## Method Comparison\n\n");
        r.push_str("| Method | Words | Spearman ρ | Pearson PCC | 95% CI (ρ) | Description |\n");
        r.push_str("|--------|-------|-----------|------------|------------|-------------|\n");
        for metric in metrics {
            let emphasis = if metric.name == best_metric.name {
                "**"
            } else {
                ""
            };
            let _ = writeln!(
                r,
                "| {emphasis}{}{emphasis} | {} | {emphasis}{:.4}{emphasis} | {:.4} | [{:.4}, \
                 {:.4}] | {} |",
                metric.name,
                metric.words_scored,
                metric.rho,
                metric.pcc,
                metric.ci.0,
                metric.ci.1,
                metric.description
            );
        }
        let _ = writeln!(
            r,
            "\n**Best method**: {} (ρ={:.4}, PCC={:.4})",
            best_metric.name, best_metric.rho, best_metric.pcc
        );
        r.push_str("\n## Gate Decision\n\n");
        if best_metric.rho >= SUCCESS_GATE {
            let _ = writeln!(
                r,
                "**FULL SUCCESS** — ρ = {:.4} ≥ {:.2}. Path A full-success gate met.",
                best_metric.rho, SUCCESS_GATE
            );
        } else if best_metric.rho >= START_GATE {
            let _ = writeln!(
                r,
                "**START GATE PASS** — ρ = {:.4} ≥ {:.2}, so start gate cleared. Full-success \
                 gate {:.2} remains unmet.",
                best_metric.rho, START_GATE, SUCCESS_GATE
            );
        } else {
            let _ = writeln!(
                r,
                "**FAIL** — best ρ = {:.4} < {:.2}, so start gate not met.",
                best_metric.rho, START_GATE
            );
            r.push_str("\nNext steps:\n");
            r.push_str(
                "1. Validate richer feature paths (for example logmel80 or teacher features)\n",
            );
            r.push_str(
                "2. Fine-tune scoring head on SpeechOcean762 with inline holdout tracking\n",
            );
            r.push_str("3. Re-evaluate before proceeding to large-scale training\n");
        }
        r
    }
}

pub use inner::{bootstrap_ci_95, execute_evaluate, pearson_r, spearman_rho, EvaluateArgs};

#[cfg(test)]
mod tests {
    use super::inner::{bootstrap_ci_95, ranks, spearman_rho};

    #[test]
    fn test_spearman_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_rho(&x, &y);
        assert!((rho - 1.0).abs() < 1e-10, "Expected ρ=1.0, got {rho}");
    }

    #[test]
    fn test_spearman_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let rho = spearman_rho(&x, &y);
        assert!((rho - (-1.0)).abs() < 1e-10, "Expected ρ=-1.0, got {rho}");
    }

    #[test]
    fn test_spearman_no_correlation() {
        // Shuffled — low correlation expected
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 1.0, 8.0, 3.0, 7.0, 2.0, 6.0, 4.0];
        let rho = spearman_rho(&x, &y);
        assert!(rho.abs() < 0.5, "Expected low ρ, got {rho}");
    }

    #[test]
    fn test_spearman_empty_and_single() {
        assert!((spearman_rho(&[], &[])).abs() < 1e-10);
        assert!((spearman_rho(&[1.0], &[2.0])).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_mismatched_lengths() {
        let rho = spearman_rho(&[1.0, 2.0], &[1.0]);
        assert!((rho).abs() < 1e-10, "Mismatched lengths should return 0");
    }

    #[test]
    fn test_ranks_basic() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let r = ranks(&values);
        let expected = [3.0, 1.5, 4.0, 1.5, 5.0];
        // 1.0 appears twice → tied rank = (1+2)/2 = 1.5
        assert_eq!(r.len(), expected.len());
        for (actual, target) in r.iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ranks_all_equal() {
        let values = vec![7.0, 7.0, 7.0];
        let r = ranks(&values);
        // All tied → average rank = (1+2+3)/3 = 2.0
        for val in &r {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bootstrap_ci_returns_range() {
        let x: Vec<f64> = (0..100).map(f64::from).collect();
        let y: Vec<f64> = (0..100).map(f64::from).collect();
        let (lo, hi) = bootstrap_ci_95(&x, &y, 200);
        // Perfect correlation → CI should be tight around 1.0
        assert!(lo > 0.9, "CI lower bound {lo} should be > 0.9");
        assert!(hi > 0.9, "CI upper bound {hi} should be > 0.9");
        assert!(lo <= hi, "CI should be ordered: {lo} <= {hi}");
    }

    #[test]
    fn test_bootstrap_ci_too_few() {
        let (lo, hi) = bootstrap_ci_95(&[1.0, 2.0], &[1.0, 2.0], 100);
        assert!((lo).abs() < 1e-10);
        assert!((hi).abs() < 1e-10);
    }
}
