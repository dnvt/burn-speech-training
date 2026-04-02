//! Fine-tune `SpeechAligner` on `SpeechOcean762` word-level pronunciation
//! scores.
//!
//! Adds a pronunciation scoring MLP head and trains with a multi-task loss:
//! `alpha * CTC + (1 - alpha) * MSE(predicted_score, human_score)`.
//!
//! The scoring head pools frame-level encoder features for each word's
//! estimated span, then regresses to a `[0, 1]` pronunciation quality score.

mod inner {
    use std::fmt::Write as _;
    use std::io::Write as _;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::time::Instant;

    use crate::g2p::CmuDict;
    use burn::module::{AutodiffModule, Module};
    use burn::nn::loss::CTCLossConfig;
    use burn::nn::{Linear, LinearConfig};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::activation::{log_softmax, relu, sigmoid};
    use burn::tensor::backend::{AutodiffBackend, Backend};
    use burn::tensor::{Bool, Int, Shape, Tensor, TensorData};
    use parking_lot::Mutex;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    use crate::dataset::load_audio_samples;
    use crate::error::{Error, Result};
    use crate::evaluate::{bootstrap_ci_95, pearson_r, spearman_rho};
    use crate::mfcc::{FeatureMode, MfccExtractor};
    use crate::model::{proportional_word_frame_span, SpeechAligner, SpeechAlignerConfig};
    use crate::phoneme_map::transcript_to_targets;
    use crate::ui::{detail, section, step, success};

    const SCORING_HIDDEN_DIM: usize = 256;
    const FINE_TUNE_BATCH_METRIC_INTERVAL: usize = 25;
    const DEFAULT_BALANCED_MSE_NOISE_SIGMA: f32 = 0.2;
    const DEFAULT_RANK_SIMILARITY_TAU: f32 = 0.15;
    const ORDINAL_BOUNDARIES: [f32; 4] = [0.2, 0.5, 0.7, 0.9];
    const ORDINAL_BIN_CENTERS: [f32; 5] = [0.1, 0.4, 0.65, 0.85, 1.0];
    const TRIPLET_POSITIVE_GAP: i32 = 1;
    const TRIPLET_NEGATIVE_GAP: i32 = 3;
    const START_GATE: f64 = 0.30;
    const SUCCESS_GATE: f64 = 0.40;

    fn default_ordinal_boundaries() -> Vec<f32> {
        ORDINAL_BOUNDARIES.to_vec()
    }

    fn default_ordinal_bin_centers() -> Vec<f32> {
        ORDINAL_BIN_CENTERS.to_vec()
    }

    fn default_scoring_hidden_dim() -> usize {
        SCORING_HIDDEN_DIM
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub enum ScoringInputSource {
        #[default]
        Backbone,
        TeacherCache,
    }

    impl ScoringInputSource {
        pub fn label(self) -> &'static str {
            match self {
                Self::Backbone => "backbone",
                Self::TeacherCache => "teacher_cache",
            }
        }
    }

    fn default_rank_similarity_tau() -> f32 {
        DEFAULT_RANK_SIMILARITY_TAU
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub enum ScoringPoolingMode {
        #[default]
        Mean,
        Attention,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ScoringHeadConfig {
        #[serde(default)]
        pub feature_mode: FeatureMode,
        #[serde(default)]
        pub scoring_input_source: ScoringInputSource,
        #[serde(default)]
        pub scoring_input_dim: usize,
        #[serde(default)]
        pub ordinal_loss: bool,
        #[serde(default)]
        pub ordinal_softmax_loss: bool,
        #[serde(default)]
        pub balanced_mse: bool,
        #[serde(default)]
        pub pooling_mode: ScoringPoolingMode,
        #[serde(default = "default_scoring_hidden_dim")]
        pub hidden_dim: usize,
        #[serde(default = "default_scoring_output_dim")]
        pub output_dim: usize,
        #[serde(default = "default_ordinal_boundaries")]
        pub ordinal_boundaries: Vec<f32>,
        #[serde(default = "default_ordinal_bin_centers")]
        pub ordinal_bin_centers: Vec<f32>,
        #[serde(default = "default_balanced_mse_noise_sigma")]
        pub balanced_mse_noise_sigma: f32,
        #[serde(default)]
        pub rank_regularization_weight: f32,
        #[serde(default = "default_rank_similarity_tau")]
        pub rank_similarity_tau: f32,
    }

    impl Default for ScoringHeadConfig {
        fn default() -> Self {
            Self {
                feature_mode: FeatureMode::default(),
                scoring_input_source: ScoringInputSource::default(),
                scoring_input_dim: 0,
                ordinal_loss: false,
                ordinal_softmax_loss: false,
                balanced_mse: false,
                pooling_mode: ScoringPoolingMode::Mean,
                hidden_dim: default_scoring_hidden_dim(),
                output_dim: default_scoring_output_dim(),
                ordinal_boundaries: default_ordinal_boundaries(),
                ordinal_bin_centers: default_ordinal_bin_centers(),
                balanced_mse_noise_sigma: default_balanced_mse_noise_sigma(),
                rank_regularization_weight: 0.0,
                rank_similarity_tau: default_rank_similarity_tau(),
            }
        }
    }

    fn default_balanced_mse_noise_sigma() -> f32 {
        DEFAULT_BALANCED_MSE_NOISE_SIGMA
    }

    fn default_scoring_output_dim() -> usize {
        1
    }

    #[derive(Deserialize)]
    struct WordDataset {
        #[allow(dead_code)]
        metadata: serde_json::Value,
        samples: Vec<WordSample>,
    }

    #[derive(Deserialize, Clone)]
    struct WordSample {
        #[allow(dead_code)]
        word_id: String,
        #[allow(dead_code)]
        text: String,
        human_score: f64,
        human_score_raw: i32,
        audio_file: String,
        #[allow(dead_code)]
        phonemes: Vec<String>,
        #[allow(dead_code)]
        utterance_id: String,
        word_position: usize,
        sentence_text: String,
    }

    #[derive(Module, Debug)]
    pub struct ScoringHead<B: Backend> {
        linear1: Linear<B>,
        linear2: Linear<B>,
        pooling_attention: Linear<B>,
    }

    #[derive(Module, Debug)]
    pub(super) struct LegacyScoringHead<B: Backend> {
        linear1: Linear<B>,
        linear2: Linear<B>,
    }

    impl<B: Backend> ScoringHead<B> {
        /// Initialize a pronunciation scoring MLP with default hidden dim.
        #[cfg(test)]
        pub fn new(input_dim: usize, device: &B::Device) -> Self {
            Self::with_dims(input_dim, SCORING_HIDDEN_DIM, 1, device)
        }

        pub fn with_dims(
            input_dim: usize,
            hidden_dim: usize,
            output_dim: usize,
            device: &B::Device,
        ) -> Self {
            Self {
                linear1: LinearConfig::new(input_dim, hidden_dim).init(device),
                linear2: LinearConfig::new(hidden_dim, output_dim).init(device),
                pooling_attention: LinearConfig::new(input_dim, 1).init(device),
            }
        }

        /// Forward pass: `[batch, input_dim] -> [batch, 1]`.
        pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            let x = self.linear1.forward(x);
            let x = relu(x);
            self.linear2.forward(x)
        }

        /// Pool a word span into a single feature vector.
        pub fn pool_word_span(
            &self,
            span_features: Tensor<B, 2>,
            pooling_mode: ScoringPoolingMode,
        ) -> Tensor<B, 2> {
            let [span_len, feature_dim] = span_features.dims();
            match pooling_mode {
                ScoringPoolingMode::Mean => span_features.mean_dim(0).reshape([1, feature_dim]),
                ScoringPoolingMode::Attention => {
                    let logits = self
                        .pooling_attention
                        .forward(span_features.clone())
                        .reshape([1, span_len]);
                    let weights = log_softmax(logits, 1).exp().reshape([span_len, 1]);
                    (span_features * weights)
                        .sum_dim(0)
                        .reshape([1, feature_dim])
                }
            }
        }

        pub fn forward_word_span(
            &self,
            span_features: Tensor<B, 2>,
            pooling_mode: ScoringPoolingMode,
        ) -> Tensor<B, 2> {
            let pooled = self.pool_word_span(span_features, pooling_mode);
            self.forward(pooled)
        }
    }

    impl<B: Backend> LegacyScoringHead<B> {
        pub(crate) fn with_hidden(input_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
            Self {
                linear1: LinearConfig::new(input_dim, hidden_dim).init(device),
                linear2: LinearConfig::new(hidden_dim, 1).init(device),
            }
        }
    }

    #[allow(clippy::struct_excessive_bools)]
    pub struct FineTuneArgs {
        /// Path to the base checkpoint (for example a pre-training checkpoint).
        /// Omit when `fresh_init` is enabled.
        pub checkpoint: Option<PathBuf>,
        /// Path to SO762 dev set JSON.
        pub dataset: PathBuf,
        /// Root directory for resolving audio paths.
        pub data_root: PathBuf,
        /// Input feature representation expected by the checkpoint.
        pub feature_mode: FeatureMode,
        /// Start from a freshly initialized backbone instead of loading a base
        /// checkpoint.
        pub fresh_init: bool,
        /// Number of fine-tuning epochs.
        pub epochs: usize,
        /// Learning rate.
        pub learning_rate: f64,
        /// CTC loss weight in `[0.0, 1.0]`.
        pub ctc_weight: f32,
        /// Output checkpoint directory.
        pub checkpoint_dir: PathBuf,
        /// Maximum number of word samples to use.
        pub max_samples: Option<usize>,
        /// Oversample words with `human_score_raw < threshold` by this factor.
        /// 0 or 1 = no oversampling. Default: 1 (off).
        pub oversample_factor: usize,
        /// Score threshold for oversampling (raw 0-10 scale). Words below this
        /// are repeated `oversample_factor` times. Default: 8.
        pub oversample_below: i32,
        /// Use inverse-frequency weighting for the MSE loss. Each word's
        /// squared error is weighted by `1 / count(score_bucket)` so every
        /// score level contributes equally. Default: false.
        pub weighted_loss: bool,
        /// Run holdout evaluation every N epochs and log ρ/PCC to metrics.
        /// 0 = disabled. Default: 0.
        pub eval_interval: usize,
        /// Path to the SO762 holdout JSON for inline evaluation.
        pub holdout: Option<PathBuf>,
        /// Mini-batch size for training. Groups utterances with padding.
        /// Higher values improve GPU utilization. Default: 1 (no batching).
        pub batch_size: usize,
        /// Focal loss gamma. When > 0, applies focal weighting `(1 -
        /// p_t)^gamma` to down-weight easy samples exponentially.
        /// Default: 0.0 (standard MSE).
        pub focal_gamma: f32,
        /// Use ordinal cross-entropy loss instead of MSE regression. Maps
        /// scores to 5 ordered classes {0-2, 3-5, 6-7, 8-9, 10}.
        /// Default: false.
        pub ordinal_loss: bool,
        /// Use true 5-class softmax ordinal cross-entropy instead of the
        /// threshold-based ordinal surrogate.
        pub ordinal_softmax_loss: bool,
        /// Use Balanced MSE (BMC-style) instead of plain MSE regression.
        /// Default: false.
        pub balanced_mse: bool,
        /// Observation noise sigma for Balanced MSE. Default: 0.2.
        pub balanced_mse_noise_sigma: f32,
        /// Use effective-number score balancing on the regression loss instead
        /// of naive inverse-frequency weights. Default: false.
        pub score_balanced_loss: bool,
        /// Beta parameter for effective-number score balancing. Default:
        /// `0.999`.
        pub score_balance_beta: f32,
        /// Feature-space mixup alpha for low-score pooled word embeddings.
        /// Default: `0.0` (disabled).
        pub feature_mixup_alpha: f32,
        /// Only apply feature mixup to words with raw score at or below this
        /// threshold. Default: `7`.
        pub feature_mixup_below: i32,
        /// LR warmup epochs. Linearly ramps from 0 to `learning_rate` over this
        /// many epochs, then applies cosine decay. 0 = flat LR. Default: 0.
        pub warmup_epochs: usize,
        /// Freeze the `SpeechAligner` backbone and train only the scoring head.
        /// Default: false.
        pub freeze_backbone: bool,
        /// Scoring head hidden dimension. Default: 256 (512→256→1).
        /// Larger values (e.g., 512) test the capacity hypothesis.
        pub scoring_hidden: usize,
        /// Weight for pooled-word embedding rank regularization. 0 disables
        /// it.
        pub rank_regularization_weight: f32,
        /// Target-similarity temperature for rank regularization.
        pub rank_similarity_tau: f32,
        /// Weight for pairwise ranking hinge loss on predicted scores.
        pub pairwise_ranking_weight: f32,
        /// Margin for pairwise ranking hinge loss.
        pub pairwise_ranking_margin: f32,
        /// Weight for triplet-style ranking loss on pooled word embeddings.
        pub triplet_ranking_weight: f32,
        /// Margin for triplet-style ranking loss on pooled word embeddings.
        pub triplet_ranking_margin: f32,
        /// Word-span pooling mode for the scoring head.
        pub pooling_mode: ScoringPoolingMode,
        /// Path to wav2vec2 distillation feature cache directory. When set,
        /// adds a feature-matching loss between SpeechAligner frame features
        /// and wav2vec2 encoder hidden states. Default: None (disabled).
        pub distillation_features: Option<PathBuf>,
        /// Weight for the distillation feature-matching loss term.
        /// total = scoring_loss + distillation_weight * MSE(student, teacher).
        /// Default: 0.1.
        pub distillation_weight: f32,
        /// Use wav2vec2 teacher features as the PRIMARY input to the scoring
        /// head instead of SpeechAligner frame features. Requires
        /// `--distillation-features`. Skips SpeechAligner backbone entirely.
        /// Tests whether richer features (1024-dim) improve scoring.
        pub teacher_as_input: bool,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct WordTarget {
        pub word_position: usize,
        pub human_score: f32,
        pub human_score_raw: i32,
        pub num_words: usize,
        /// Loss weight for this word (1.0 = default, higher = more important).
        pub weight: f32,
    }

    struct PreparedUtterance {
        feature_frames: Vec<Vec<f32>>,
        feature_dim: usize,
        ctc_targets: Vec<i32>,
        words: Vec<WordTarget>,
        /// Audio file path (for distillation cache matching).
        audio_file: String,
        /// wav2vec2 teacher features for distillation: flat f32 [frames, 1024].
        teacher_features: Option<Vec<f32>>,
        /// Number of teacher frames (may differ from MFCC frames).
        teacher_frames: usize,
    }

    #[derive(Debug, Clone, Copy)]
    struct EvalSnapshot {
        epoch: usize,
        rho: f64,
        pcc: f64,
        ci: (f64, f64),
        words_scored: usize,
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct EpochStats {
        pub(super) ctc_loss: f32,
        pub(super) mse_loss: f32,
        pub(super) total_loss: f32,
        pub(super) words: usize,
        pub(super) seconds: f64,
    }

    pub(super) struct MetricsWriter {
        file: Option<std::fs::File>,
        run_start: Instant,
    }

    impl MetricsWriter {
        pub(super) fn new(checkpoint_dir: &Path) -> Self {
            let path = checkpoint_dir.join("training-metrics.jsonl");
            let file = std::fs::File::create(&path).ok();
            if file.is_some() {
                detail(&format!("  Metrics: {}", path.display()));
            }
            Self {
                file,
                run_start: Instant::now(),
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn emit_batch(
            &mut self,
            epoch: usize,
            batch: usize,
            total_batches: usize,
            batch_utterances: usize,
            total_loss: f32,
            ctc_loss: f32,
            mse_loss: f32,
            batch_words: usize,
            batch_elapsed_secs: f64,
        ) {
            let elapsed = self.run_start.elapsed().as_secs_f64();
            let samples_per_sec = if batch_elapsed_secs > 0.0 {
                batch_utterances as f64 / batch_elapsed_secs
            } else {
                0.0
            };

            tracing::info!(
                target: "training_metrics",
                event = "batch",
                epoch = epoch,
                batch = batch,
                total_batches = total_batches,
                batch_utterances = batch_utterances,
                batch_words = batch_words,
                loss = total_loss,
                ctc_loss = ctc_loss,
                mse_loss = mse_loss,
                samples_per_sec = samples_per_sec,
                elapsed_secs = elapsed,
                "training_batch"
            );

            if let Some(ref mut file) = self.file {
                let _ = writeln!(
                    file,
                    r#"{{"event":"batch","epoch":{epoch},"batch":{batch},"total_batches":{total_batches},"batch_utterances":{batch_utterances},"batch_words":{batch_words},"loss":{total_loss:.6},"ctc_loss":{ctc_loss:.6},"mse_loss":{mse_loss:.6},"samples_per_sec":{samples_per_sec:.6},"elapsed_secs":{elapsed:.1},"batch_elapsed_secs":{batch_elapsed_secs:.6}}}"#,
                );
                let _ = file.flush();
            }
        }

        pub(super) fn emit_epoch(
            &mut self,
            epoch: usize,
            total_epochs: usize,
            utterances: usize,
            batch_count: usize,
            stats: EpochStats,
        ) {
            let elapsed = self.run_start.elapsed().as_secs_f64();
            let remaining_epochs = total_epochs.saturating_sub(epoch);
            #[allow(clippy::cast_precision_loss)]
            let eta_secs = if epoch > 0 {
                (elapsed / epoch as f64) * remaining_epochs as f64
            } else {
                0.0
            };
            #[allow(clippy::cast_precision_loss)]
            let throughput = utterances as f64 / stats.seconds.max(0.001);
            #[allow(clippy::cast_precision_loss)]
            let words_per_sec = stats.words as f64 / stats.seconds.max(0.001);

            tracing::info!(
                target: "training_metrics",
                event = "epoch",
                epoch = epoch,
                total_epochs = total_epochs,
                avg_loss = stats.total_loss,
                ctc_loss = stats.ctc_loss,
                mse_loss = stats.mse_loss,
                words = stats.words,
                batches = batch_count,
                utterances = utterances,
                epoch_secs = stats.seconds,
                throughput_samples_per_sec = throughput,
                words_per_sec = words_per_sec,
                elapsed_secs = elapsed,
                eta_secs = eta_secs,
                "training_epoch"
            );

            if let Some(ref mut file) = self.file {
                let _ = writeln!(
                    file,
                    r#"{{"event":"epoch","epoch":{epoch},"total_epochs":{total_epochs},"avg_loss":{total_loss:.6},"ctc_loss":{ctc_loss:.6},"mse_loss":{mse_loss:.6},"words":{words},"batches":{batch_count},"utterances":{utterances},"epoch_secs":{seconds:.1},"throughput":{throughput:.6},"words_per_sec":{words_per_sec:.6},"elapsed_secs":{elapsed:.1},"eta_secs":{eta_secs:.1}}}"#,
                    total_loss = stats.total_loss,
                    ctc_loss = stats.ctc_loss,
                    mse_loss = stats.mse_loss,
                    words = stats.words,
                    seconds = stats.seconds,
                );
                let _ = file.flush();
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn emit_eval(
            &mut self,
            epoch: usize,
            rho: f64,
            pcc: f64,
            ci_lo: f64,
            ci_hi: f64,
            words_scored: usize,
        ) {
            let elapsed = self.run_start.elapsed().as_secs_f64();

            tracing::info!(
                target: "training_metrics",
                event = "eval",
                epoch = epoch,
                rho = rho,
                pcc = pcc,
                ci_lo = ci_lo,
                ci_hi = ci_hi,
                words_scored = words_scored,
                elapsed_secs = elapsed,
                "holdout_eval"
            );

            if let Some(ref mut file) = self.file {
                let _ = writeln!(
                    file,
                    r#"{{"event":"eval","epoch":{epoch},"rho":{rho:.4},"pcc":{pcc:.4},"ci_lo":{ci_lo:.4},"ci_hi":{ci_hi:.4},"words_scored":{words_scored},"elapsed_secs":{elapsed:.1}}}"#,
                );
                let _ = file.flush();
            }
        }

        fn emit_complete(&mut self, epochs: usize, first_loss: f32, last_loss: f32) {
            let elapsed = self.run_start.elapsed().as_secs_f64();
            let reduction_pct = if first_loss.abs() > f32::EPSILON {
                ((first_loss - last_loss) / first_loss) * 100.0
            } else {
                0.0
            };

            tracing::info!(
                target: "training_metrics",
                event = "complete",
                epochs = epochs,
                first_loss = first_loss,
                last_loss = last_loss,
                reduction_pct = reduction_pct,
                total_secs = elapsed,
                "training_complete"
            );

            if let Some(ref mut file) = self.file {
                let _ = writeln!(
                    file,
                    r#"{{"event":"complete","epochs":{epochs},"first_loss":{first_loss:.6},"last_loss":{last_loss:.6},"reduction_pct":{reduction_pct:.1},"total_secs":{elapsed:.1}}}"#,
                );
                let _ = file.flush();
            }
        }
    }

    fn scoring_loss_mode(args: &FineTuneArgs) -> &'static str {
        if args.ordinal_softmax_loss {
            "ordinal_softmax"
        } else if args.ordinal_loss {
            "ordinal"
        } else if args.balanced_mse {
            "balanced_mse"
        } else if args.score_balanced_loss {
            "score_balanced_regression"
        } else {
            "regression"
        }
    }

    fn scoring_head_output_dim(args: &FineTuneArgs) -> usize {
        if args.ordinal_softmax_loss {
            ORDINAL_BIN_CENTERS.len()
        } else {
            1
        }
    }

    fn scoring_head_output_dim_from_config(config: &ScoringHeadConfig) -> usize {
        if config.output_dim > 0 {
            config.output_dim
        } else if config.ordinal_softmax_loss {
            ORDINAL_BIN_CENTERS.len()
        } else {
            1
        }
    }

    fn scoring_input_source_from_args(args: &FineTuneArgs) -> ScoringInputSource {
        if args.teacher_as_input {
            ScoringInputSource::TeacherCache
        } else {
            ScoringInputSource::Backbone
        }
    }

    fn effective_ctc_weight(args: &FineTuneArgs) -> f32 {
        if args.teacher_as_input {
            0.0
        } else {
            args.ctc_weight
        }
    }

    fn effective_freeze_backbone(args: &FineTuneArgs) -> bool {
        args.teacher_as_input || args.freeze_backbone
    }

    fn effective_distillation_weight(args: &FineTuneArgs) -> f32 {
        if args.teacher_as_input {
            0.0
        } else {
            args.distillation_weight
        }
    }

    fn configured_scoring_input_dim(args: &FineTuneArgs, backbone_dim: usize) -> usize {
        if args.teacher_as_input {
            DISTILL_DIM
        } else {
            backbone_dim
        }
    }

    pub fn scoring_input_dim_from_config(config: &ScoringHeadConfig, backbone_dim: usize) -> usize {
        if config.scoring_input_dim > 0 {
            config.scoring_input_dim
        } else if matches!(
            config.scoring_input_source,
            ScoringInputSource::TeacherCache
        ) {
            DISTILL_DIM
        } else {
            backbone_dim
        }
    }

    fn build_scoring_head_config(args: &FineTuneArgs) -> ScoringHeadConfig {
        let backbone_dim = SpeechAlignerConfig::for_feature_mode(args.feature_mode).channels[3];
        ScoringHeadConfig {
            feature_mode: args.feature_mode,
            scoring_input_source: scoring_input_source_from_args(args),
            scoring_input_dim: configured_scoring_input_dim(args, backbone_dim),
            ordinal_loss: args.ordinal_loss,
            ordinal_softmax_loss: args.ordinal_softmax_loss,
            balanced_mse: args.balanced_mse,
            pooling_mode: args.pooling_mode,
            hidden_dim: args.scoring_hidden,
            output_dim: scoring_head_output_dim(args),
            ordinal_boundaries: default_ordinal_boundaries(),
            ordinal_bin_centers: default_ordinal_bin_centers(),
            balanced_mse_noise_sigma: args.balanced_mse_noise_sigma,
            rank_regularization_weight: args.rank_regularization_weight,
            rank_similarity_tau: args.rank_similarity_tau,
        }
    }

    fn base_checkpoint_label(args: &FineTuneArgs) -> String {
        if args.fresh_init {
            format!("fresh_init ({})", args.feature_mode.label())
        } else {
            args.checkpoint
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| String::from("<missing checkpoint>"))
        }
    }

    fn scoring_head_config_path_for_dir(checkpoint_dir: &Path) -> PathBuf {
        checkpoint_dir.join("scoring-head-config.json")
    }

    fn scoring_head_config_path_for_head(scoring_head_path: &Path) -> PathBuf {
        let parent = scoring_head_path.parent().unwrap_or_else(|| Path::new("."));
        scoring_head_config_path_for_dir(parent)
    }

    fn save_scoring_head_config(args: &FineTuneArgs) -> Result<()> {
        let path = scoring_head_config_path_for_dir(&args.checkpoint_dir);
        let config = build_scoring_head_config(args);
        let json = serde_json::to_string_pretty(&config)
            .map_err(|err| Error::process(format!("Failed to serialize scoring config: {err}")))?;
        std::fs::write(&path, json).map_err(|err| {
            Error::process(format!(
                "Failed to write scoring config {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    pub fn load_scoring_head_config(scoring_head_path: &Path) -> ScoringHeadConfig {
        let path = scoring_head_config_path_for_head(scoring_head_path);
        let Ok(raw) = std::fs::read_to_string(path) else {
            return ScoringHeadConfig::default();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    fn scoring_head_hidden_candidates(config: &ScoringHeadConfig) -> Vec<usize> {
        let mut candidates = vec![config.hidden_dim, SCORING_HIDDEN_DIM, 512];
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|value| *value > 0);
        candidates
    }

    fn catch_unwind_quiet<F, T>(f: F) -> std::result::Result<T, ()>
    where
        F: FnOnce() -> T,
    {
        static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

        let lock = PANIC_HOOK_LOCK.get_or_init(|| Mutex::new(()));
        let guard = lock.lock();

        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = catch_unwind(AssertUnwindSafe(f));
        std::panic::set_hook(previous_hook);
        drop(guard);

        result.map_err(|_| ())
    }

    pub(super) fn ordinal_bin_index(raw_score: i32) -> usize {
        match raw_score {
            i32::MIN..=2 => 0,
            3..=5 => 1,
            6..=7 => 2,
            8..=9 => 3,
            _ => 4,
        }
    }

    fn ordinal_bucket_label(raw_score: i32) -> i32 {
        match ordinal_bin_index(raw_score) {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            _ => 4,
        }
    }

    fn ordinal_threshold_logits() -> [f32; 4] {
        ORDINAL_BOUNDARIES.map(logit_probability)
    }

    fn logit_probability(p: f32) -> f32 {
        let clipped = p.clamp(1e-4, 1.0 - 1e-4);
        (clipped / (1.0 - clipped)).ln()
    }

    fn ordinal_targets_from_raw_scores<B: Backend>(
        raw_scores: &[i32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let mut flat = Vec::with_capacity(raw_scores.len() * ORDINAL_BOUNDARIES.len());
        for &raw_score in raw_scores {
            flat.push(if raw_score > 2 { 1.0 } else { 0.0 });
            flat.push(if raw_score > 5 { 1.0 } else { 0.0 });
            flat.push(if raw_score > 7 { 1.0 } else { 0.0 });
            flat.push(if raw_score > 9 { 1.0 } else { 0.0 });
        }
        Tensor::from_data(
            TensorData::new(
                flat,
                Shape::new([raw_scores.len(), ORDINAL_BOUNDARIES.len()]),
            ),
            device,
        )
    }

    fn ordinal_expected_scores_from_logits<B: Backend>(
        score_logits: Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = score_logits.dims()[0];
        let threshold_logits = Tensor::from_data(
            TensorData::new(
                ordinal_threshold_logits().to_vec(),
                Shape::new([1, ORDINAL_BOUNDARIES.len()]),
            ),
            device,
        );
        let cumulative = sigmoid(score_logits - threshold_logits);
        let p0 = Tensor::<B, 2>::ones([batch_size, 1], device)
            - cumulative.clone().slice([0..batch_size, 0..1]);
        let p1 = cumulative.clone().slice([0..batch_size, 0..1])
            - cumulative.clone().slice([0..batch_size, 1..2]);
        let p2 = cumulative.clone().slice([0..batch_size, 1..2])
            - cumulative.clone().slice([0..batch_size, 2..3]);
        let p3 = cumulative.clone().slice([0..batch_size, 2..3])
            - cumulative.clone().slice([0..batch_size, 3..4]);
        let p4 = cumulative.slice([0..batch_size, 3..4]);
        let class_probs = Tensor::cat(vec![p0, p1, p2, p3, p4], 1);
        let expected = class_probs.clone().slice([0..batch_size, 0..1]) * ORDINAL_BIN_CENTERS[0]
            + class_probs.clone().slice([0..batch_size, 1..2]) * ORDINAL_BIN_CENTERS[1]
            + class_probs.clone().slice([0..batch_size, 2..3]) * ORDINAL_BIN_CENTERS[2]
            + class_probs.clone().slice([0..batch_size, 3..4]) * ORDINAL_BIN_CENTERS[3]
            + class_probs.slice([0..batch_size, 4..5]) * ORDINAL_BIN_CENTERS[4];
        expected.reshape([batch_size])
    }

    fn ordinal_softmax_targets_from_raw_scores<B: Backend>(
        raw_scores: &[i32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let class_count = ORDINAL_BIN_CENTERS.len();
        let mut flat = vec![0.0_f32; raw_scores.len() * class_count];
        for (row, &raw_score) in flat.chunks_exact_mut(class_count).zip(raw_scores.iter()) {
            let bucket = ordinal_bin_index(raw_score);
            if let Some(cell) = row.get_mut(bucket) {
                *cell = 1.0;
            }
        }
        Tensor::from_data(
            TensorData::new(flat, Shape::new([raw_scores.len(), class_count])),
            device,
        )
    }

    fn ordinal_softmax_expected_scores_from_logits<B: Backend>(
        score_logits: Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = score_logits.dims()[0];
        let probs = log_softmax(score_logits, 1).exp();
        let centers = Tensor::from_data(
            TensorData::new(
                ORDINAL_BIN_CENTERS.to_vec(),
                Shape::new([1, ORDINAL_BIN_CENTERS.len()]),
            ),
            device,
        );
        (probs * centers).sum_dim(1).reshape([batch_size])
    }

    pub fn scoring_predictions_from_logits<B: Backend>(
        score_logits: Tensor<B, 2>,
        scoring_config: &ScoringHeadConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        if scoring_config.ordinal_softmax_loss {
            ordinal_softmax_expected_scores_from_logits(score_logits, device)
        } else if scoring_config.ordinal_loss {
            ordinal_expected_scores_from_logits(score_logits, device)
        } else {
            let n = score_logits.dims()[0];
            sigmoid(score_logits).reshape([n])
        }
    }

    fn ordinal_loss_from_logits<B: Backend>(
        score_logits: Tensor<B, 2>,
        raw_scores: &[i32],
        loss_weights: Tensor<B, 1>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = raw_scores.len();
        let threshold_logits = Tensor::from_data(
            TensorData::new(
                ordinal_threshold_logits().to_vec(),
                Shape::new([1, ORDINAL_BOUNDARIES.len()]),
            ),
            device,
        );
        let ordinal_targets = ordinal_targets_from_raw_scores::<B>(raw_scores, device);
        let cumulative_logits = score_logits - threshold_logits;
        let probs = sigmoid(cumulative_logits);
        let ones = Tensor::<B, 2>::ones([batch_size, ORDINAL_BOUNDARIES.len()], device);
        let log_p = probs.clone().clamp_min(1e-6).log();
        let log_not_p = (ones.clone() - probs).clamp_min(1e-6).log();
        let bce = -(ordinal_targets.clone() * log_p + (ones - ordinal_targets) * log_not_p);
        let sample_weights = loss_weights.reshape([batch_size, 1]);
        (bce * sample_weights).mean()
    }

    fn ordinal_softmax_loss_from_logits<B: Backend>(
        score_logits: Tensor<B, 2>,
        raw_scores: &[i32],
        loss_weights: Tensor<B, 1>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = raw_scores.len();
        let targets = ordinal_softmax_targets_from_raw_scores::<B>(raw_scores, device);
        let log_probs = log_softmax(score_logits, 1);
        let per_sample = -(targets * log_probs).sum_dim(1).reshape([batch_size]);
        (per_sample * loss_weights).mean()
    }

    pub(super) fn score_balanced_weights_from_counts(
        counts: &std::collections::BTreeMap<i32, usize>,
        beta: f32,
    ) -> std::collections::BTreeMap<i32, f32> {
        let mut weights = std::collections::BTreeMap::new();
        let mut total = 0.0f32;

        for (&bucket, &count) in counts {
            let effective = if beta <= f32::EPSILON {
                count.max(1) as f32
            } else {
                let count_f = count.max(1) as f32;
                (1.0 - beta.powf(count_f)).max(1e-6) / (1.0 - beta)
            };
            let weight = (1.0 / effective).min(100.0);
            total += weight;
            weights.insert(bucket, weight);
        }

        let mean = if weights.is_empty() {
            1.0
        } else {
            total / weights.len() as f32
        };
        if mean > f32::EPSILON {
            for weight in weights.values_mut() {
                *weight /= mean;
            }
        }
        weights
    }

    fn sample_feature_mixup_lambda(alpha: f32) -> f32 {
        if alpha <= f32::EPSILON {
            return 1.0;
        }
        let mut rng = rand::rng();
        let radius = alpha.clamp(0.0, 1.0) * 0.5;
        rng.random_range((0.5 - radius)..=(0.5 + radius))
    }

    pub(super) fn balanced_mse_loss<B: Backend>(
        predicted_scores: Tensor<B, 1>,
        score_targets: Tensor<B, 1>,
        raw_scores: &[i32],
        noise_sigma: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = predicted_scores.dims()[0];
        if batch_size < 2 {
            let diff = predicted_scores - score_targets;
            let se = diff.clone() * diff;
            return se.mean();
        }

        let pred = predicted_scores.reshape([batch_size, 1]);
        let target = score_targets.reshape([1, batch_size]);
        let diff = pred - target;
        let logits = (diff.clone() * diff) * (-0.5 / (noise_sigma * noise_sigma));
        let log_probs = log_softmax(logits, 1);

        let mut positive_mask = vec![0.0_f32; batch_size * batch_size];
        for (row_score, row_mask) in raw_scores
            .iter()
            .zip(positive_mask.chunks_exact_mut(batch_size))
        {
            for (col_score, cell) in raw_scores.iter().zip(row_mask.iter_mut()) {
                if row_score == col_score {
                    *cell = 1.0;
                }
            }
        }
        let positive_mask = Tensor::from_data(
            TensorData::new(positive_mask, Shape::new([batch_size, batch_size])),
            device,
        );
        let matched_mass = (log_probs.exp() * positive_mask)
            .sum_dim(1)
            .reshape([batch_size]);
        let selected = matched_mass.clamp_min(1e-6).log();
        (-selected.mean()) * (2.0 * noise_sigma * noise_sigma)
    }

    #[allow(clippy::indexing_slicing)] // row/col loops are bounded by batch_size
    pub(super) fn pairwise_ranking_loss<B: Backend>(
        predicted_scores: Tensor<B, 1>,
        raw_scores: &[i32],
        margin: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = predicted_scores.dims()[0];
        if batch_size < 2 {
            return Tensor::<B, 1>::zeros([1], device);
        }

        let score_i = predicted_scores.clone().reshape([batch_size, 1]);
        let score_j = predicted_scores.reshape([1, batch_size]);
        let pairwise_margin = Tensor::<B, 2>::ones([batch_size, batch_size], device) * margin;
        let losses = relu(pairwise_margin - (score_i - score_j));

        let mut mask_data = vec![0.0f32; batch_size * batch_size];
        let mut valid_pairs = 0usize;
        for (row_idx, &lhs) in raw_scores.iter().enumerate() {
            for (col_idx, &rhs) in raw_scores.iter().enumerate() {
                if lhs > rhs {
                    mask_data[row_idx * batch_size + col_idx] = 1.0;
                    valid_pairs += 1;
                }
            }
        }
        if valid_pairs == 0 {
            return Tensor::<B, 1>::zeros([1], device);
        }

        let mask = Tensor::from_data(
            TensorData::new(mask_data, Shape::new([batch_size, batch_size])),
            device,
        );
        (losses * mask).sum().reshape([1]) / valid_pairs as f32
    }

    #[allow(clippy::indexing_slicing)] // anchor/other loops are bounded by batch_size
    pub(super) fn triplet_ranking_loss<B: Backend>(
        pooled_features: Tensor<B, 2>,
        raw_scores: &[i32],
        margin: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let [batch_size, feature_dim] = pooled_features.dims();
        if batch_size < 3 {
            return Tensor::<B, 1>::zeros([1], device);
        }

        let feat_i = pooled_features
            .clone()
            .reshape([batch_size, 1, feature_dim]);
        let feat_j = pooled_features.reshape([1, batch_size, feature_dim]);
        let diff = feat_i - feat_j;
        let distances = (diff.clone() * diff)
            .sum_dim(2)
            .reshape([batch_size, batch_size]);

        let mut positive_mask_data = vec![0.0f32; batch_size * batch_size];
        let mut negative_mask_data = vec![0.0f32; batch_size * batch_size];
        let mut valid_anchor_mask = vec![0.0f32; batch_size];
        let mut positive_counts = vec![0.0f32; batch_size];
        let mut negative_counts = vec![0.0f32; batch_size];

        for (anchor_idx, &anchor_score) in raw_scores.iter().enumerate() {
            for (other_idx, &other_score) in raw_scores.iter().enumerate() {
                if anchor_idx == other_idx {
                    continue;
                }
                let gap = (anchor_score - other_score).abs();
                let offset = anchor_idx * batch_size + other_idx;
                if gap <= TRIPLET_POSITIVE_GAP {
                    positive_mask_data[offset] = 1.0;
                    positive_counts[anchor_idx] += 1.0;
                } else if gap >= TRIPLET_NEGATIVE_GAP {
                    negative_mask_data[offset] = 1.0;
                    negative_counts[anchor_idx] += 1.0;
                }
            }
            if positive_counts[anchor_idx] > 0.0 && negative_counts[anchor_idx] > 0.0 {
                valid_anchor_mask[anchor_idx] = 1.0;
            }
        }

        let valid_anchor_count = valid_anchor_mask.iter().filter(|&&v| v > 0.0).count();
        if valid_anchor_count == 0 {
            return Tensor::<B, 1>::zeros([1], device);
        }

        let positive_mask = Tensor::from_data(
            TensorData::new(positive_mask_data, Shape::new([batch_size, batch_size])),
            device,
        );
        let negative_mask = Tensor::from_data(
            TensorData::new(negative_mask_data, Shape::new([batch_size, batch_size])),
            device,
        );
        let positive_count_tensor = Tensor::from_data(
            TensorData::new(positive_counts, Shape::new([batch_size])),
            device,
        );
        let negative_count_tensor = Tensor::from_data(
            TensorData::new(negative_counts, Shape::new([batch_size])),
            device,
        );
        let valid_anchor_tensor = Tensor::from_data(
            TensorData::new(valid_anchor_mask, Shape::new([batch_size])),
            device,
        );

        let positive_mean = (distances.clone() * positive_mask)
            .sum_dim(1)
            .reshape([batch_size])
            / positive_count_tensor.clamp_min(1.0);
        let negative_mean = (distances * negative_mask).sum_dim(1).reshape([batch_size])
            / negative_count_tensor.clamp_min(1.0);
        let margin_tensor = Tensor::<B, 1>::ones([batch_size], device) * margin;
        let losses = relu(positive_mean - negative_mean + margin_tensor) * valid_anchor_tensor;
        losses.sum().reshape([1]) / valid_anchor_count as f32
    }

    fn rank_regularization_loss<B: Backend>(
        pooled_features: Tensor<B, 2>,
        score_targets: Tensor<B, 1>,
        similarity_tau: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let [batch_size, feature_dim] = pooled_features.dims();
        if batch_size < 2 {
            return Tensor::<B, 1>::zeros([1], device);
        }

        let feature_energy = (pooled_features.clone() * pooled_features.clone())
            .sum_dim(1)
            .reshape([batch_size, 1]);
        let normalized = pooled_features / feature_energy.sqrt().clamp_min(1e-6);
        let feat_i = normalized.clone().reshape([batch_size, 1, feature_dim]);
        let feat_j = normalized.reshape([1, batch_size, feature_dim]);
        let feat_diff = feat_i - feat_j;
        let feature_distance = (feat_diff.clone() * feat_diff)
            .mean_dim(2)
            .reshape([batch_size, batch_size]);
        let feature_similarity = (feature_distance * -1.0).exp();

        let target_i = score_targets.clone().reshape([batch_size, 1]);
        let target_j = score_targets.reshape([1, batch_size]);
        let target_distance = (target_i - target_j).abs();
        let target_similarity = (target_distance * (-1.0 / similarity_tau.max(1e-3))).exp();

        let mut offdiag = vec![1.0_f32; batch_size * batch_size];
        for (idx, row) in offdiag.chunks_exact_mut(batch_size).enumerate() {
            if let Some(cell) = row.get_mut(idx) {
                *cell = 0.0;
            }
        }
        let mask = Tensor::from_data(
            TensorData::new(offdiag, Shape::new([batch_size, batch_size])),
            device,
        );
        let diff = feature_similarity - target_similarity;
        let sq = diff.clone() * diff * mask;
        let denom = (batch_size * batch_size - batch_size).max(1) as f32;
        sq.sum().reshape([1]) / denom
    }

    pub fn execute_finetune(args: &FineTuneArgs) -> Result<()> {
        if !(0.0..=1.0).contains(&args.ctc_weight) {
            return Err(Error::config("ctc_weight must be within 0.0..=1.0"));
        }
        if args.ordinal_loss && args.ordinal_softmax_loss {
            return Err(Error::config(
                "ordinal_loss and ordinal_softmax_loss are mutually exclusive",
            ));
        }
        if args.ordinal_loss && args.balanced_mse {
            return Err(Error::config(
                "ordinal_loss and balanced_mse are mutually exclusive",
            ));
        }
        if args.ordinal_softmax_loss && args.balanced_mse {
            return Err(Error::config(
                "ordinal_softmax_loss and balanced_mse are mutually exclusive",
            ));
        }
        if args.score_balanced_loss && args.balanced_mse {
            return Err(Error::config(
                "score_balanced_loss and balanced_mse are mutually exclusive",
            ));
        }
        if args.score_balanced_loss && args.ordinal_loss {
            return Err(Error::config(
                "score_balanced_loss and ordinal_loss are mutually exclusive",
            ));
        }
        if args.score_balanced_loss && args.ordinal_softmax_loss {
            return Err(Error::config(
                "score_balanced_loss and ordinal_softmax_loss are mutually exclusive",
            ));
        }
        if args.ordinal_loss && args.focal_gamma > f32::EPSILON {
            return Err(Error::config(
                "focal_gamma is not supported with ordinal_loss",
            ));
        }
        if args.ordinal_softmax_loss && args.focal_gamma > f32::EPSILON {
            return Err(Error::config(
                "focal_gamma is not supported with ordinal_softmax_loss",
            ));
        }
        if args.balanced_mse && args.focal_gamma > f32::EPSILON {
            return Err(Error::config(
                "focal_gamma is not supported with balanced_mse",
            ));
        }
        if args.balanced_mse && args.weighted_loss {
            return Err(Error::config(
                "weighted_loss is not supported with balanced_mse",
            ));
        }
        if args.score_balanced_loss && args.weighted_loss {
            return Err(Error::config(
                "weighted_loss is not supported with score_balanced_loss",
            ));
        }
        if args.balanced_mse_noise_sigma <= 0.0 {
            return Err(Error::config("balanced_mse_noise_sigma must be > 0"));
        }
        if !(0.0..1.0).contains(&args.score_balance_beta) {
            return Err(Error::config("score_balance_beta must be within 0.0..1.0"));
        }
        if !(0.0..=1.0).contains(&args.feature_mixup_alpha) {
            return Err(Error::config(
                "feature_mixup_alpha must be within 0.0..=1.0",
            ));
        }
        if args.feature_mixup_alpha > f32::EPSILON
            && (args.ordinal_loss || args.ordinal_softmax_loss || args.balanced_mse)
        {
            return Err(Error::config(
                "feature_mixup_alpha is only supported with regression-style scoring losses",
            ));
        }
        if args.rank_regularization_weight < 0.0 {
            return Err(Error::config("rank_regularization_weight must be >= 0"));
        }
        if args.rank_similarity_tau <= 0.0 {
            return Err(Error::config("rank_similarity_tau must be > 0"));
        }
        if args.pairwise_ranking_weight < 0.0 {
            return Err(Error::config("pairwise_ranking_weight must be >= 0"));
        }
        if args.triplet_ranking_weight < 0.0 {
            return Err(Error::config("triplet_ranking_weight must be >= 0"));
        }
        if args.pairwise_ranking_margin <= 0.0 {
            return Err(Error::config("pairwise_ranking_margin must be > 0"));
        }
        if args.triplet_ranking_margin <= 0.0 {
            return Err(Error::config("triplet_ranking_margin must be > 0"));
        }

        #[cfg(feature = "cuda")]
        {
            use burn::backend::cuda::CudaDevice;
            use burn::backend::{Autodiff, Cuda};

            type TrainB = Autodiff<Cuda>;
            type InferB = Cuda;

            let device = CudaDevice::default();
            finetune_loop::<TrainB, InferB>(args, &device, "Autodiff<Cuda> (CUDA GPU)")
        }

        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            use burn::backend::Autodiff;

            type TrainB = Autodiff<Wgpu>;
            type InferB = Wgpu;

            let device = WgpuDevice::default();
            finetune_loop::<TrainB, InferB>(args, &device, "Autodiff<Wgpu> (Vulkan/Metal GPU)")
        }

        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            use burn::backend::{Autodiff, NdArray};

            type TrainB = Autodiff<NdArray>;
            type InferB = NdArray;

            let device = NdArrayDevice::default();
            finetune_loop::<TrainB, InferB>(args, &device, "Autodiff<NdArray> (CPU)")
        }
    }

    fn finetune_loop<TrainB, InferB>(
        args: &FineTuneArgs,
        device: &TrainB::Device,
        backend_name: &str,
    ) -> Result<()>
    where
        TrainB: AutodiffBackend<InnerBackend = InferB>,
        InferB: Backend<FloatElem = f32>,
        SpeechAligner<TrainB>: AutodiffModule<TrainB, InnerModule = SpeechAligner<InferB>>,
    {
        section("SpeechAligner SO762 Fine-Tuning");

        let config = SpeechAlignerConfig::for_feature_mode(args.feature_mode);
        if !args.fresh_init && args.checkpoint.is_none() {
            return Err(Error::config(
                "--checkpoint is required unless --fresh-init is set",
            ));
        }
        step(if args.fresh_init {
            "Initializing base model"
        } else {
            "Loading base checkpoint"
        });
        let teacher_as_input = args.teacher_as_input;
        let scoring_input_dim = if teacher_as_input {
            if args.distillation_features.is_none() {
                return Err(Error::config(
                    "--teacher-as-input requires --distillation-features",
                ));
            }
            detail(&format!(
                "  MODE: wav2vec2 teacher features as primary input ({}→scoring head)",
                DISTILL_DIM
            ));
            DISTILL_DIM
        } else {
            config.channels[3]
        };
        TrainB::seed(device, 42);
        let mut model = if args.fresh_init {
            config.init(device)?
        } else {
            load_model_checkpoint::<TrainB>(
                args.checkpoint.as_deref().ok_or_else(|| {
                    Error::config("--checkpoint is required unless --fresh-init is set")
                })?,
                &config,
                device,
            )
            .map_err(|err| {
                Error::process(format!(
                    "{err}. If this checkpoint was trained with a different feature shape, rerun \
                     with --fresh-init."
                ))
            })?
        };
        let ctc_weight = effective_ctc_weight(args);
        let mse_weight = 1.0 - ctc_weight;
        let freeze_backbone = effective_freeze_backbone(args);
        detail(&format!("  Feature mode: {}", args.feature_mode.label()));
        detail(&format!("  Base source: {}", base_checkpoint_label(args)));
        if teacher_as_input {
            detail(if args.fresh_init {
                "  Base model: freshly initialized but BYPASSED (teacher features used instead)"
            } else {
                "  Base model: loaded but BYPASSED (teacher features used instead)"
            });
            if args.ctc_weight > f32::EPSILON {
                detail("  Teacher-input mode forces ctc_weight=0.0");
            }
            if !args.freeze_backbone {
                detail("  Teacher-input mode forces freeze_backbone=true");
            }
            if args.distillation_weight > f32::EPSILON {
                detail("  Teacher-input mode disables feature-matching distillation loss");
            }
        } else {
            detail(&format!("  Base model: {} params", model.num_params()));
        }

        let hidden_dim = if args.scoring_hidden > 0 {
            args.scoring_hidden
        } else {
            SCORING_HIDDEN_DIM
        };
        let output_dim = scoring_head_output_dim(args);
        let mut scoring_head =
            ScoringHead::<TrainB>::with_dims(scoring_input_dim, hidden_dim, output_dim, device);
        detail(&format!(
            "  Scoring head: {} params ({} -> {} -> {})",
            scoring_head.num_params(),
            scoring_input_dim,
            hidden_dim,
            output_dim
        ));
        detail(&format!("  Scoring loss mode: {}", scoring_loss_mode(args)));

        step("Loading SpeechOcean762 dev set");
        let raw = std::fs::read_to_string(&args.dataset)
            .map_err(|err| Error::config(format!("Cannot read dataset: {err}")))?;
        let dataset: WordDataset = serde_json::from_str(&raw)
            .map_err(|err| Error::config(format!("Invalid dataset JSON: {err}")))?;
        let max_samples = args.max_samples.unwrap_or(dataset.samples.len());
        let samples = dataset
            .samples
            .get(..max_samples.min(dataset.samples.len()))
            .unwrap_or(&dataset.samples);
        detail(&format!("  Word samples: {}", samples.len()));

        step("Preparing utterance-level fine-tuning batches");
        let mut utterances = prepare_utterances(args, samples)?;
        let total_words = utterances
            .iter()
            .map(|utterance| utterance.words.len())
            .sum::<usize>();
        detail(&format!("  Valid utterances: {}", utterances.len()));
        detail(&format!("  Valid words: {total_words}"));

        if utterances.is_empty() {
            return Err(Error::config(
                "No valid utterances were prepared for SO762 fine-tuning",
            ));
        }

        let teacher_feature_cache = args
            .distillation_features
            .as_ref()
            .map(|cache_dir| load_teacher_feature_cache(cache_dir))
            .transpose()?;

        // Load distillation features (if configured)
        let distillation_weight = if let Some(ref cache) = teacher_feature_cache {
            step("Loading wav2vec2 distillation features");
            let loaded = load_distillation_features(cache, &mut utterances)?;
            detail(&format!(
                "  Distillation: {loaded}/{} utterances matched, weight={}",
                utterances.len(),
                effective_distillation_weight(args)
            ));
            if loaded == 0 && teacher_as_input {
                return Err(Error::config(
                    "--teacher-as-input matched 0 utterances in --distillation-features",
                ));
            }
            if loaded == 0 {
                tracing::warn!("No distillation features matched — distillation disabled");
                0.0
            } else {
                effective_distillation_weight(args)
            }
        } else {
            0.0
        };

        // Load holdout for inline evaluation (if configured)
        if args.holdout.is_some() && args.eval_interval == 0 {
            tracing::warn!(
                "--holdout provided but --eval-interval is 0; holdout will not be evaluated"
            );
        }
        let holdout = if args.eval_interval > 0 {
            if let Some(ref holdout_path) = args.holdout {
                step("Loading holdout set for inline evaluation");
                let holdout_data = load_holdout(
                    holdout_path,
                    &args.data_root,
                    args.feature_mode,
                    teacher_feature_cache.as_ref().filter(|_| teacher_as_input),
                )?;
                let total_words: usize = holdout_data
                    .utterances
                    .iter()
                    .map(|utterance| utterance.words.len())
                    .sum();
                detail(&format!(
                    "  Holdout: {} utterances, {} words, eval every {} epochs",
                    holdout_data.utterances.len(),
                    total_words,
                    args.eval_interval
                ));
                Some(holdout_data)
            } else {
                None
            }
        } else {
            None
        };

        step(&format!(
            "Fine-tuning: {} epochs, lr={}, ctc_weight={}, backend={}",
            args.epochs, args.learning_rate, args.ctc_weight, backend_name
        ));

        std::fs::create_dir_all(&args.checkpoint_dir).map_err(|err| {
            Error::process(format!(
                "Failed to create checkpoint dir {}: {err}",
                args.checkpoint_dir.display()
            ))
        })?;
        save_scoring_head_config(args)?;

        let mut model_optimizer = AdamConfig::new().init::<TrainB, SpeechAligner<TrainB>>();
        let mut head_optimizer = AdamConfig::new().init::<TrainB, ScoringHead<TrainB>>();
        let ctc_loss_fn = CTCLossConfig::new().init();
        let mut metrics = MetricsWriter::new(&args.checkpoint_dir);
        let scoring_config = build_scoring_head_config(args);

        // Distillation: linear projection 1024→512 (teacher→student dim)
        let (mut distill_projection, mut distill_optimizer) = if distillation_weight > f32::EPSILON
        {
            let proj: Linear<TrainB> =
                LinearConfig::new(DISTILL_DIM, config.channels[3]).init(device);
            let opt = AdamConfig::new().init::<TrainB, Linear<TrainB>>();
            detail(&format!(
                "  Distillation projection: {}→{} ({} params, with optimizer)",
                DISTILL_DIM,
                config.channels[3],
                proj.num_params()
            ));
            (Some(proj), Some(opt))
        } else {
            (None, None)
        };

        let focal_gamma = args.focal_gamma;
        let warmup_epochs = args.warmup_epochs;
        let balanced_mse = args.balanced_mse;
        let balanced_mse_noise_sigma = args.balanced_mse_noise_sigma;
        let score_balanced_loss = args.score_balanced_loss;
        let feature_mixup_alpha = args.feature_mixup_alpha;
        let feature_mixup_below = args.feature_mixup_below;
        let ordinal_loss = args.ordinal_loss;
        let ordinal_softmax_loss = args.ordinal_softmax_loss;
        let base_lr = args.learning_rate;
        let rank_regularization_weight = args.rank_regularization_weight;
        let rank_similarity_tau = args.rank_similarity_tau;
        let pairwise_ranking_weight = args.pairwise_ranking_weight;
        let pairwise_ranking_margin = args.pairwise_ranking_margin;
        let triplet_ranking_weight = args.triplet_ranking_weight;
        let triplet_ranking_margin = args.triplet_ranking_margin;
        let total_epochs = args.epochs;
        let mut epoch_stats = Vec::with_capacity(args.epochs);

        // Sort utterances by frame count to minimize padding waste within batches
        let batch_size = args.batch_size.max(1);
        if batch_size > 1 {
            utterances.sort_by_key(|u| u.feature_frames.len());
            detail(&format!(
                "  Sorted utterances by length for batching (batch_size={batch_size})"
            ));
        }

        let mut eval_snapshots = Vec::new();

        for epoch in 0..args.epochs {
            let epoch_start = Instant::now();
            let mut ctc_sum = 0.0f64;
            let mut mse_sum = 0.0f64;
            let mut total_sum = 0.0f64;
            let mut total_words_epoch = 0usize;
            let mut batches = 0usize;

            let total_steps = utterances.len().div_ceil(batch_size);
            for step_idx in 0..total_steps {
                let batch_start_time = Instant::now();
                let start = step_idx * batch_size;
                let end = (start + batch_size).min(utterances.len());
                let Some(batch_utts) = utterances.get(start..end) else {
                    continue;
                };
                let cur_batch = batch_utts.len();

                // Find max lengths in this mini-batch
                let max_frames = batch_utts
                    .iter()
                    .map(|u| u.feature_frames.len())
                    .max()
                    .unwrap_or(0);
                let max_targets = batch_utts
                    .iter()
                    .map(|u| u.ctc_targets.len())
                    .max()
                    .unwrap_or(0);
                let input_feature_dim = batch_utts.first().map_or(0, |utt| utt.feature_dim);

                if max_frames == 0 || max_targets == 0 {
                    continue;
                }

                let output = if teacher_as_input {
                    None
                } else {
                    // Pad and stack input frames: [batch, max_frames, feature_dim]
                    #[allow(clippy::indexing_slicing)]
                    // bounded: offset + feature_dim <= flat.len()
                    let input = {
                        let mut flat = vec![0.0f32; cur_batch * max_frames * input_feature_dim];
                        for (i, utt) in batch_utts.iter().enumerate() {
                            debug_assert_eq!(utt.feature_dim, input_feature_dim);
                            for (f, frame) in utt.feature_frames.iter().enumerate() {
                                let offset = (i * max_frames + f) * input_feature_dim;
                                flat[offset..offset + input_feature_dim].copy_from_slice(frame);
                            }
                        }
                        Tensor::from_data(
                            TensorData::new(
                                flat,
                                Shape::new([cur_batch, max_frames, input_feature_dim]),
                            ),
                            device,
                        )
                    };

                    let mask_data: Vec<bool> = batch_utts
                        .iter()
                        .flat_map(|utt| (0..max_frames).map(move |f| f >= utt.feature_frames.len()))
                        .collect();
                    let mask_pad = Tensor::<TrainB, 2, Bool>::from_data(
                        TensorData::new(mask_data, Shape::new([cur_batch, max_frames])),
                        device,
                    );
                    Some(model.forward_with_pad_mask(input, mask_pad))
                };

                #[allow(clippy::cast_possible_wrap, clippy::indexing_slicing)]
                let ctc_loss = if let Some(output) = output.as_ref() {
                    let mut flat_targets = vec![0i32; cur_batch * max_targets];
                    let mut input_lens = Vec::with_capacity(cur_batch);
                    let mut target_lens = Vec::with_capacity(cur_batch);
                    for (i, utt) in batch_utts.iter().enumerate() {
                        input_lens.push(utt.feature_frames.len() as i32);
                        target_lens.push(utt.ctc_targets.len() as i32);
                        let t_off = i * max_targets;
                        flat_targets[t_off..t_off + utt.ctc_targets.len()]
                            .copy_from_slice(&utt.ctc_targets);
                    }
                    let ctc_tgt = Tensor::<TrainB, 2, Int>::from_data(
                        TensorData::new(flat_targets, Shape::new([cur_batch, max_targets])),
                        device,
                    );
                    let in_lens = Tensor::<TrainB, 1, Int>::from_data(
                        TensorData::new(input_lens, Shape::new([cur_batch])),
                        device,
                    );
                    let tgt_lens = Tensor::<TrainB, 1, Int>::from_data(
                        TensorData::new(target_lens, Shape::new([cur_batch])),
                        device,
                    );
                    ctc_loss_fn
                        .forward(output.ctc_log_probs.clone(), ctc_tgt, in_lens, tgt_lens)
                        .mean()
                } else {
                    Tensor::<TrainB, 1>::zeros([1], device)
                };

                // Scoring head: extract per-utterance frame features, pool per-word
                let feature_dim = if teacher_as_input {
                    DISTILL_DIM // 1024
                } else {
                    config.channels[3] // 512
                };
                let mut all_pooled: Vec<Tensor<TrainB, 2>> = Vec::new();
                let mut all_targets: Vec<f32> = Vec::new();
                let mut all_weights: Vec<f32> = Vec::new();
                let mut all_raw_scores: Vec<i32> = Vec::new();

                for (i, utt) in batch_utts.iter().enumerate() {
                    let utt_frames = utt.feature_frames.len();
                    if utt_frames == 0 || utt.words.is_empty() {
                        continue;
                    }

                    // Get frame features: from teacher cache or model output
                    let (utt_features, num_scoring_frames) = if teacher_as_input {
                        // Use cached wav2vec2 features directly (1024-dim, ~50fps)
                        if let Some(ref teacher_flat) = utt.teacher_features {
                            let t_frames = utt.teacher_frames;
                            if t_frames == 0 {
                                continue;
                            }
                            let feat: Tensor<TrainB, 2> = Tensor::from_data(
                                TensorData::new(
                                    teacher_flat.clone(),
                                    Shape::new([t_frames, DISTILL_DIM]),
                                ),
                                device,
                            );
                            (feat, t_frames)
                        } else {
                            continue; // No teacher features for this utterance
                        }
                    } else {
                        // Use SpeechAligner model output (512-dim, ~100fps)
                        let Some(output) = output.as_ref() else {
                            continue;
                        };
                        let feat = output
                            .frame_features
                            .clone()
                            .slice([i..i + 1, 0..utt_frames, 0..feature_dim])
                            .reshape([utt_frames, feature_dim]);
                        (feat, utt_frames)
                    };

                    for word in &utt.words {
                        let Some((wstart, wend)) = proportional_word_frame_span(
                            num_scoring_frames,
                            word.num_words,
                            word.word_position,
                        ) else {
                            continue;
                        };
                        let span_features =
                            utt_features.clone().slice([wstart..wend, 0..feature_dim]);
                        let pooled = scoring_head.pool_word_span(span_features, args.pooling_mode);
                        all_pooled.push(pooled);
                        all_targets.push(word.human_score);
                        all_weights.push(word.weight);
                        all_raw_scores.push(word.human_score_raw);
                    }
                }

                if all_pooled.is_empty() {
                    continue;
                }

                if feature_mixup_alpha > f32::EPSILON && all_pooled.len() > 1 {
                    let original_len = all_pooled.len();
                    let mut rng = rand::rng();
                    for anchor_idx in 0..original_len {
                        let Some(&anchor_raw) = all_raw_scores.get(anchor_idx) else {
                            continue;
                        };
                        if anchor_raw > feature_mixup_below {
                            continue;
                        }
                        let Some(anchor_pooled) = all_pooled.get(anchor_idx).cloned() else {
                            continue;
                        };
                        let Some(&anchor_target) = all_targets.get(anchor_idx) else {
                            continue;
                        };
                        let Some(&anchor_weight) = all_weights.get(anchor_idx) else {
                            continue;
                        };

                        let mut partner_idx = anchor_idx;
                        if original_len > 1 {
                            while partner_idx == anchor_idx {
                                partner_idx = rng.random_range(0..original_len);
                            }
                        }
                        let Some(partner_pooled) = all_pooled.get(partner_idx).cloned() else {
                            continue;
                        };
                        let Some(&partner_target) = all_targets.get(partner_idx) else {
                            continue;
                        };
                        let Some(&partner_weight) = all_weights.get(partner_idx) else {
                            continue;
                        };
                        let Some(&partner_raw) = all_raw_scores.get(partner_idx) else {
                            continue;
                        };

                        let lam = sample_feature_mixup_lambda(feature_mixup_alpha);
                        let mixed_pooled = anchor_pooled * lam + partner_pooled * (1.0 - lam);
                        let mixed_target = anchor_target.mul_add(lam, partner_target * (1.0 - lam));
                        let mixed_weight = (anchor_weight + partner_weight) * 0.5;
                        let mixed_raw = (anchor_raw as f32)
                            .mul_add(lam, partner_raw as f32 * (1.0 - lam))
                            .round()
                            .clamp(0.0, 10.0) as i32;

                        all_pooled.push(mixed_pooled);
                        all_targets.push(mixed_target);
                        all_weights.push(mixed_weight);
                        all_raw_scores.push(mixed_raw);
                    }
                }

                let batch_words = all_targets.len();
                let pooled_features = Tensor::cat(all_pooled, 0);
                let score_targets = Tensor::from_data(
                    TensorData::new(all_targets, Shape::new([batch_words])),
                    device,
                );
                let loss_weights = Tensor::from_data(
                    TensorData::new(all_weights, Shape::new([batch_words])),
                    device,
                );

                let score_logits = scoring_head.forward(pooled_features.clone());
                let predicted_scores =
                    scoring_predictions_from_logits(score_logits.clone(), &scoring_config, device);
                let diff = predicted_scores.clone() - score_targets.clone();
                let se = diff.clone() * diff;

                let scoring_loss = if ordinal_loss {
                    ordinal_loss_from_logits(score_logits, &all_raw_scores, loss_weights, device)
                } else if ordinal_softmax_loss {
                    ordinal_softmax_loss_from_logits(
                        score_logits,
                        &all_raw_scores,
                        loss_weights,
                        device,
                    )
                } else if balanced_mse {
                    balanced_mse_loss(
                        predicted_scores.clone(),
                        score_targets.clone(),
                        &all_raw_scores,
                        balanced_mse_noise_sigma,
                        device,
                    )
                } else if score_balanced_loss {
                    let weighted_se = se * loss_weights;
                    weighted_se.mean()
                } else if focal_gamma > f32::EPSILON {
                    // Focal loss: weight by (1 - p_t)^gamma where p_t measures how
                    // close the prediction is to the target. Harder samples (large
                    // error) get higher weight. gamma=0 → standard MSE.
                    let abs_diff = (predicted_scores.clone() - score_targets.clone()).abs();
                    let ones = Tensor::ones([batch_words], device);
                    let p_t = (ones - abs_diff).clamp_min(0.0);
                    let focal_w = (Tensor::<TrainB, 1>::ones([batch_words], device) - p_t)
                        .powf_scalar(focal_gamma);
                    let focal_se = se * focal_w * loss_weights;
                    focal_se.mean()
                } else {
                    let weighted_se = se * loss_weights;
                    weighted_se.mean()
                };

                let rank_loss = if rank_regularization_weight > f32::EPSILON {
                    rank_regularization_loss(
                        pooled_features.clone(),
                        score_targets.clone(),
                        rank_similarity_tau,
                        device,
                    )
                } else {
                    Tensor::<TrainB, 1>::zeros([1], device)
                };

                let pairwise_loss = if pairwise_ranking_weight > f32::EPSILON {
                    pairwise_ranking_loss(
                        predicted_scores.clone(),
                        &all_raw_scores,
                        pairwise_ranking_margin,
                        device,
                    )
                } else {
                    Tensor::<TrainB, 1>::zeros([1], device)
                };

                let triplet_loss = if triplet_ranking_weight > f32::EPSILON {
                    triplet_ranking_loss(
                        pooled_features.clone(),
                        &all_raw_scores,
                        triplet_ranking_margin,
                        device,
                    )
                } else {
                    Tensor::<TrainB, 1>::zeros([1], device)
                };

                // Distillation loss: MSE between student and projected teacher features
                let distill_loss = if let (Some(proj), Some(output)) =
                    (distill_projection.as_ref(), output.as_ref())
                {
                    let feature_dim = config.channels[3];
                    let mut distill_parts: Vec<Tensor<TrainB, 1>> = Vec::new();
                    for (i, utt) in batch_utts.iter().enumerate() {
                        if let Some(ref teacher_flat) = utt.teacher_features {
                            let utt_frames = utt.feature_frames.len();
                            let t_frames = utt.teacher_frames;
                            // MFCC=100fps (10ms hop), wav2vec2=50fps (20ms hop).
                            // Stride student frames by 2 to align temporally.
                            let aligned_student_frames = utt_frames / 2;
                            let common = aligned_student_frames.min(t_frames);
                            if common == 0 {
                                continue;
                            }
                            // Student features: take every 2nd frame to match
                            // wav2vec2 temporal positions (frame 0→0ms, 2→20ms,
                            // 4→40ms matching teacher 0→0ms, 1→20ms, 2→40ms)
                            let student_indices: Vec<usize> = (0..common).map(|f| f * 2).collect();
                            let mut student_data = Vec::with_capacity(common * feature_dim);
                            let full_student = output
                                .frame_features
                                .clone()
                                .slice([i..i + 1, 0..utt_frames.min(common * 2), 0..feature_dim])
                                .reshape([utt_frames.min(common * 2), feature_dim]);
                            let student_raw: Vec<f32> =
                                full_student.to_data().to_vec().unwrap_or_default();
                            for &idx in &student_indices {
                                let start = idx * feature_dim;
                                let end = start + feature_dim;
                                if end <= student_raw.len() {
                                    student_data.extend_from_slice(&student_raw[start..end]);
                                }
                            }
                            let actual_common = student_data.len() / feature_dim;
                            if actual_common == 0 {
                                continue;
                            }
                            let student: Tensor<TrainB, 2> = Tensor::from_data(
                                TensorData::new(
                                    student_data,
                                    Shape::new([actual_common, feature_dim]),
                                ),
                                device,
                            );
                            // Teacher features from cache
                            let teacher_data: Vec<f32> = teacher_flat
                                .chunks(DISTILL_DIM)
                                .take(actual_common)
                                .flatten()
                                .copied()
                                .collect();
                            let teacher = Tensor::from_data(
                                TensorData::new(
                                    teacher_data,
                                    Shape::new([actual_common, DISTILL_DIM]),
                                ),
                                device,
                            );
                            // Project teacher 1024→512
                            let projected_teacher = proj.forward(teacher);
                            // Frame-level MSE
                            let diff = student - projected_teacher;
                            let se = diff.clone() * diff;
                            let frame_mse = se.mean_dim(1); // [common, 1]
                            distill_parts.push(frame_mse.reshape([actual_common]));
                        }
                    }
                    if distill_parts.is_empty() {
                        Tensor::<TrainB, 1>::zeros([1], device)
                    } else {
                        Tensor::cat(distill_parts, 0).mean()
                    }
                } else {
                    Tensor::<TrainB, 1>::zeros([1], device)
                };

                let distill_value = if distillation_weight > f32::EPSILON {
                    scalar_from_tensor(distill_loss.clone()).unwrap_or(0.0)
                } else {
                    0.0
                };
                let total_loss = ctc_loss.clone() * ctc_weight
                    + scoring_loss.clone() * mse_weight
                    + rank_loss * rank_regularization_weight
                    + pairwise_loss * pairwise_ranking_weight
                    + triplet_loss * triplet_ranking_weight
                    + distill_loss * distillation_weight;
                let total_loss_value = scalar_from_tensor(total_loss.clone())?;

                if !total_loss_value.is_finite() {
                    tracing::warn!(
                        "Skipping non-finite fine-tune loss at epoch {}: {total_loss_value}",
                        epoch + 1
                    );
                    continue;
                }

                let ctc_value = scalar_from_tensor(ctc_loss.clone())?;
                let mse_value = scalar_from_tensor(scoring_loss.clone())?;

                if distill_value > f32::EPSILON {
                    tracing::debug!(
                        target: "training_metrics",
                        distill_loss = distill_value,
                        distill_weight = distillation_weight,
                        "distillation_loss"
                    );
                }

                // LR schedule: linear warmup then cosine decay
                let current_lr = if warmup_epochs > 0 {
                    let epoch_f = epoch as f64;
                    let warmup_f = warmup_epochs as f64;
                    let total_f = total_epochs as f64;
                    if epoch_f < warmup_f {
                        // Linear warmup: 0 → base_lr
                        base_lr * (epoch_f + 1.0) / warmup_f
                    } else {
                        // Cosine decay: base_lr → 0
                        let progress = (epoch_f - warmup_f) / (total_f - warmup_f).max(1.0);
                        base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
                    }
                } else {
                    base_lr
                };

                let mut grads = total_loss.backward();
                if !freeze_backbone {
                    let model_grads = GradientsParams::from_module(&mut grads, &model);
                    model = model_optimizer.step(current_lr, model, model_grads);
                }
                let head_grads = GradientsParams::from_module(&mut grads, &scoring_head);
                scoring_head = head_optimizer.step(current_lr, scoring_head, head_grads);
                if let (Some(ref mut proj), Some(ref mut opt)) =
                    (&mut distill_projection, &mut distill_optimizer)
                {
                    let proj_grads = GradientsParams::from_module(&mut grads, proj);
                    *proj = opt.step(current_lr, proj.clone(), proj_grads);
                }

                ctc_sum += f64::from(ctc_value);
                mse_sum += f64::from(mse_value);
                total_sum += f64::from(total_loss_value);
                total_words_epoch += batch_words;
                batches += 1;

                if batches.is_multiple_of(FINE_TUNE_BATCH_METRIC_INTERVAL)
                    || step_idx + 1 == total_steps
                {
                    metrics.emit_batch(
                        epoch + 1,
                        step_idx + 1,
                        total_steps,
                        cur_batch,
                        total_loss_value,
                        ctc_value,
                        mse_value,
                        batch_words,
                        batch_start_time.elapsed().as_secs_f64(),
                    );
                }
            }

            let seconds = epoch_start.elapsed().as_secs_f64();
            #[allow(clippy::cast_precision_loss)]
            let stats = if batches > 0 {
                EpochStats {
                    ctc_loss: (ctc_sum / batches as f64) as f32,
                    mse_loss: (mse_sum / batches as f64) as f32,
                    total_loss: (total_sum / batches as f64) as f32,
                    words: total_words_epoch,
                    seconds,
                }
            } else {
                EpochStats {
                    ctc_loss: f32::NAN,
                    mse_loss: f32::NAN,
                    total_loss: f32::NAN,
                    words: 0,
                    seconds,
                }
            };
            epoch_stats.push(stats);

            detail(&format!(
                "  epoch {}/{} total={:.4} ctc={:.4} mse={:.6} words={} ({seconds:.1}s)",
                epoch + 1,
                args.epochs,
                stats.total_loss,
                stats.ctc_loss,
                stats.mse_loss,
                stats.words,
            ));
            metrics.emit_epoch(epoch + 1, args.epochs, utterances.len(), batches, stats);

            let model_epoch_path = args
                .checkpoint_dir
                .join(format!("speech-aligner-ft-epoch-{}", epoch + 1));
            save_model_checkpoint(&model, &model_epoch_path)?;
            let head_epoch_path = args.checkpoint_dir.join(format!(
                "speech-aligner-ft-scoring-head-epoch-{}",
                epoch + 1
            ));
            save_scoring_head_checkpoint(&scoring_head, &head_epoch_path)?;

            // Inline holdout evaluation at configured interval
            if let Some(ref holdout_data) = holdout {
                let epoch_num = epoch + 1;
                if args.eval_interval > 0 && epoch_num.is_multiple_of(args.eval_interval) {
                    let eval_start = Instant::now();
                    let infer_model: Option<SpeechAligner<InferB>> =
                        (!teacher_as_input).then(|| model.clone().valid());
                    let infer_head: ScoringHead<InferB> = scoring_head.clone().valid();
                    // For single-GPU training, Device::default() = device(0), matching the
                    // training device. Multi-GPU would need explicit device threading.
                    let infer_device = InferB::Device::default();
                    let (rho, pcc, ci, scored) = run_holdout_eval(
                        infer_model.as_ref(),
                        &infer_head,
                        &scoring_config,
                        holdout_data,
                        &infer_device,
                    );
                    let eval_secs = eval_start.elapsed().as_secs_f64();
                    detail(&format!(
                        "  eval epoch {epoch_num}: ρ={rho:.4} PCC={pcc:.4} CI=[{:.4}, {:.4}] \
                         words={scored} ({eval_secs:.1}s)",
                        ci.0, ci.1
                    ));
                    eval_snapshots.push(EvalSnapshot {
                        epoch: epoch_num,
                        rho,
                        pcc,
                        ci,
                        words_scored: scored,
                    });
                    metrics.emit_eval(epoch_num, rho, pcc, ci.0, ci.1, scored);
                }
            }
        }

        let model_infer: SpeechAligner<InferB> = model.valid();
        let scoring_head_infer: ScoringHead<InferB> = scoring_head.valid();
        let final_model_path = args.checkpoint_dir.join("speech-aligner-ft-final");
        let final_head_path = args
            .checkpoint_dir
            .join("speech-aligner-ft-scoring-head-final");
        save_model_checkpoint(&model_infer, &final_model_path)?;
        save_scoring_head_checkpoint(&scoring_head_infer, &final_head_path)?;
        save_scoring_head_config(args)?;

        let first_loss = first_finite_loss(&epoch_stats).unwrap_or(0.0);
        let last_loss = last_finite_loss(&epoch_stats).unwrap_or(first_loss);
        metrics.emit_complete(args.epochs, first_loss, last_loss);

        let report = generate_finetune_report(
            args,
            backend_name,
            model_infer.num_params(),
            scoring_head_infer.num_params(),
            utterances.len(),
            total_words,
            &epoch_stats,
            &eval_snapshots,
            &final_model_path,
            &final_head_path,
        );
        let report_path = args.checkpoint_dir.join("finetune-report.md");
        std::fs::write(&report_path, report)
            .map_err(|err| Error::process(format!("Failed to write report: {err}")))?;
        detail(&format!("  Report: {}", report_path.display()));

        success(&format!(
            "Fine-tuning complete: model={}, scoring_head={}",
            final_model_path.display(),
            final_head_path.display()
        ));

        Ok(())
    }

    fn prepare_utterances(
        args: &FineTuneArgs,
        samples: &[WordSample],
    ) -> Result<Vec<PreparedUtterance>> {
        // Compute per-score weights if enabled
        #[allow(clippy::if_then_some_else_none)] // side-effect: detail() logging
        let bucket_weights = if args.weighted_loss || args.score_balanced_loss {
            let mut counts = std::collections::BTreeMap::<i32, usize>::new();
            for s in samples {
                let bucket = if args.ordinal_loss || args.ordinal_softmax_loss {
                    ordinal_bucket_label(s.human_score_raw)
                } else {
                    s.human_score_raw
                };
                *counts.entry(bucket).or_default() += 1;
            }
            let weights = if args.score_balanced_loss {
                let weights = score_balanced_weights_from_counts(&counts, args.score_balance_beta);
                detail(&format!(
                    "  Score-balanced weights: {} buckets, beta {:.4}, max weight {:.2}",
                    weights.len(),
                    args.score_balance_beta,
                    weights.values().copied().fold(0.0f32, f32::max)
                ));
                weights
            } else {
                let max_count = counts.values().copied().max().unwrap_or(1);
                // Cap at 100x to prevent gradient explosion from tiny buckets
                let max_weight = 100.0f32;
                let weights: std::collections::BTreeMap<i32, f32> = counts
                    .iter()
                    .map(|(&score, &count)| {
                        #[allow(clippy::cast_precision_loss)]
                        let w = (max_count as f32 / count.max(1) as f32).min(max_weight);
                        (score, w)
                    })
                    .collect();
                detail(&format!(
                    "  Inverse-frequency weights: {} buckets, max weight {:.1}",
                    weights.len(),
                    weights.values().copied().fold(0.0f32, f32::max)
                ));
                weights
            };
            Some(weights)
        } else {
            None
        };

        let extractor = MfccExtractor::with_mode(args.feature_mode);
        let dict = CmuDict::load()
            .map_err(|err| Error::config(format!("Failed to load CMUdict: {err}")))?;

        let mut grouped: std::collections::BTreeMap<String, Vec<&WordSample>> =
            std::collections::BTreeMap::new();
        for sample in samples {
            grouped
                .entry(sample.audio_file.clone())
                .or_default()
                .push(sample);
        }

        let mut utterances = Vec::with_capacity(grouped.len());
        let mut skipped = 0usize;

        for (audio_file, word_samples) in grouped {
            let full_path = args.data_root.join(&audio_file);
            let Ok(audio) = load_audio_samples(&full_path) else {
                skipped += 1;
                continue;
            };
            let Ok(feature_frames) = extractor.extract_frames(&audio) else {
                skipped += 1;
                continue;
            };

            let Some(sentence_text) = word_samples
                .first()
                .map(|sample| sample.sentence_text.as_str())
            else {
                skipped += 1;
                continue;
            };
            let num_words = sentence_text.split_whitespace().count();
            if num_words == 0 {
                skipped += 1;
                continue;
            }

            let Ok((ctc_targets, _stats)) = transcript_to_targets(sentence_text, &dict) else {
                skipped += 1;
                continue;
            };
            if feature_frames.len() < ctc_targets.len() {
                skipped += 1;
                continue;
            }

            let mut words: Vec<WordTarget> = word_samples
                .iter()
                .filter(|sample| sample.word_position < num_words)
                .map(|sample| {
                    let weight = bucket_weights
                        .as_ref()
                        .and_then(|w| {
                            let bucket = if args.ordinal_loss || args.ordinal_softmax_loss {
                                ordinal_bucket_label(sample.human_score_raw)
                            } else {
                                sample.human_score_raw
                            };
                            w.get(&bucket).copied()
                        })
                        .unwrap_or(1.0);
                    WordTarget {
                        word_position: sample.word_position,
                        #[allow(clippy::cast_possible_truncation)]
                        human_score: sample.human_score as f32,
                        human_score_raw: sample.human_score_raw,
                        num_words,
                        weight,
                    }
                })
                .collect();

            // Oversample low-score words (Experiment 3C)
            if args.oversample_factor > 1 {
                let originals: Vec<WordTarget> = words.clone();
                for word in &originals {
                    if word.human_score_raw < args.oversample_below {
                        for _ in 1..args.oversample_factor {
                            words.push(*word);
                        }
                    }
                }
            }

            words.sort_by_key(|word| word.word_position);

            if words.is_empty() {
                skipped += 1;
                continue;
            }

            utterances.push(PreparedUtterance {
                feature_frames,
                feature_dim: extractor.feature_dim(),
                ctc_targets,
                words,
                audio_file: audio_file.clone(),
                teacher_features: None,
                teacher_frames: 0,
            });
        }

        detail(&format!(
            "  Skipped utterances during preparation: {skipped}"
        ));

        if utterances.is_empty() {
            return Err(Error::config(
                "No utterances could be prepared from the SO762 dataset",
            ));
        }

        Ok(utterances)
    }

    /// Distillation feature cache: wav2vec2 encoder hidden states per
    /// utterance.
    const DISTILL_MAGIC: &[u8; 4] = b"W2V2";
    const DISTILL_DIM: usize = 1024;

    #[derive(Debug, Clone, serde::Deserialize)]
    struct TeacherFeatureEntry {
        bin: String,
        frames: usize,
    }

    #[derive(Debug, Clone)]
    pub struct TeacherFeatureCache {
        root: PathBuf,
        files: std::collections::HashMap<String, TeacherFeatureEntry>,
    }

    pub fn load_teacher_feature_cache(cache_dir: &Path) -> Result<TeacherFeatureCache> {
        let manifest_path = cache_dir.join("manifest.json");
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| Error::config(format!("Cannot read distillation manifest: {e}")))?;

        #[derive(serde::Deserialize)]
        struct Manifest {
            files: std::collections::HashMap<String, TeacherFeatureEntry>,
        }

        let manifest: Manifest = serde_json::from_str(&raw)
            .map_err(|e| Error::config(format!("Invalid distillation manifest: {e}")))?;

        Ok(TeacherFeatureCache {
            root: cache_dir.to_path_buf(),
            files: manifest.files,
        })
    }

    impl TeacherFeatureCache {
        pub fn load(&self, audio_file: &str) -> Result<Option<(Vec<f32>, usize)>> {
            use std::io::Read as _;

            let Some(entry) = self.files.get(audio_file) else {
                return Ok(None);
            };

            let bin_path = self.root.join(&entry.bin);
            let Ok(mut file) = std::fs::File::open(&bin_path) else {
                tracing::warn!(audio = %audio_file, path = %bin_path.display(), "Teacher cache file missing");
                return Ok(None);
            };

            let mut header = [0u8; 16];
            if let Err(err) = file.read_exact(&mut header) {
                tracing::warn!(audio = %audio_file, error = %err, "Teacher cache header unreadable");
                return Ok(None);
            }

            if &header[..4] != DISTILL_MAGIC {
                tracing::warn!(audio = %audio_file, "Teacher cache magic mismatch");
                return Ok(None);
            }

            let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
            if version != 1 {
                tracing::warn!(
                    audio = %audio_file,
                    version = version,
                    "Teacher cache version mismatch"
                );
                return Ok(None);
            }

            let frames =
                u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
            let dim = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
            if dim != DISTILL_DIM || frames != entry.frames {
                tracing::warn!(
                    audio = %audio_file,
                    expected_dim = DISTILL_DIM,
                    actual_dim = dim,
                    expected_frames = entry.frames,
                    actual_frames = frames,
                    "Teacher cache dim/frame mismatch"
                );
                return Ok(None);
            }

            let num_floats = frames * dim;
            let mut buf = vec![0u8; num_floats * 4];
            if let Err(err) = file.read_exact(&mut buf) {
                tracing::warn!(audio = %audio_file, error = %err, "Teacher cache truncated");
                return Ok(None);
            }

            let features = buf
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(Some((features, frames)))
        }
    }

    /// Load wav2vec2 distillation features from cache and attach to utterances.
    fn load_distillation_features(
        cache: &TeacherFeatureCache,
        utterances: &mut [PreparedUtterance],
    ) -> Result<usize> {
        let mut loaded = 0usize;
        for utt in utterances.iter_mut() {
            if let Some((features, frames)) = cache.load(&utt.audio_file)? {
                utt.teacher_features = Some(features);
                utt.teacher_frames = frames;
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Pre-loaded holdout data for inline evaluation during fine-tuning.
    struct HoldoutUtterance {
        frames: Vec<Vec<f32>>,
        words: Vec<(usize, usize, f64)>,
        teacher_features: Option<Vec<f32>>,
        teacher_frames: usize,
    }

    struct HoldoutData {
        utterances: Vec<HoldoutUtterance>,
        feature_dim: usize,
        scoring_source: ScoringInputSource,
    }

    /// Load the holdout JSON and extract frame features + word metadata.
    fn load_holdout(
        holdout_path: &Path,
        data_root: &Path,
        feature_mode: FeatureMode,
        teacher_feature_cache: Option<&TeacherFeatureCache>,
    ) -> Result<HoldoutData> {
        let raw = std::fs::read_to_string(holdout_path)
            .map_err(|e| Error::config(format!("Cannot read holdout: {e}")))?;
        let dataset: WordDataset = serde_json::from_str(&raw)
            .map_err(|e| Error::config(format!("Invalid holdout JSON: {e}")))?;

        let extractor = MfccExtractor::with_mode(feature_mode);
        let mut grouped: std::collections::BTreeMap<String, Vec<&WordSample>> =
            std::collections::BTreeMap::new();
        for sample in &dataset.samples {
            grouped
                .entry(sample.audio_file.clone())
                .or_default()
                .push(sample);
        }

        let mut utterances = Vec::with_capacity(grouped.len());
        let teacher_mode = teacher_feature_cache.is_some();

        for (audio_file, word_samples) in &grouped {
            let Some(sentence_text) = word_samples.first().map(|s| s.sentence_text.as_str()) else {
                continue;
            };
            let num_words = sentence_text.split_whitespace().count();
            if num_words == 0 {
                continue;
            }

            let words: Vec<(usize, usize, f64)> = word_samples
                .iter()
                .filter(|s| s.word_position < num_words)
                .map(|s| (s.word_position, num_words, s.human_score))
                .collect();
            if words.is_empty() {
                continue;
            }

            let (frames, teacher_features, teacher_frames) =
                if let Some(cache) = teacher_feature_cache {
                    let Some((teacher_features, teacher_frames)) = cache.load(audio_file)? else {
                        continue;
                    };
                    (Vec::new(), Some(teacher_features), teacher_frames)
                } else {
                    let full_path = data_root.join(audio_file);
                    let Ok(audio) = load_audio_samples(&full_path) else {
                        continue;
                    };
                    let Ok(frames) = extractor.extract_frames(&audio) else {
                        continue;
                    };
                    (frames, None, 0)
                };

            utterances.push(HoldoutUtterance {
                frames,
                words,
                teacher_features,
                teacher_frames,
            });
        }

        if teacher_mode && utterances.is_empty() {
            return Err(Error::config(
                "Teacher-feature holdout evaluation matched 0 utterances. Ensure \
                 --distillation-features contains the holdout split, not only the training set.",
            ));
        }

        Ok(HoldoutData {
            utterances,
            feature_dim: extractor.feature_dim(),
            scoring_source: if teacher_mode {
                ScoringInputSource::TeacherCache
            } else {
                ScoringInputSource::Backbone
            },
        })
    }

    /// Run scoring-head inference on the holdout set and return (ρ, PCC, CI).
    fn run_holdout_eval<B: Backend>(
        model: Option<&SpeechAligner<B>>,
        scoring_head: &ScoringHead<B>,
        scoring_config: &ScoringHeadConfig,
        holdout: &HoldoutData,
        device: &B::Device,
    ) -> (f64, f64, (f64, f64), usize) {
        let mut predicted: Vec<f64> = Vec::new();
        let mut human: Vec<f64> = Vec::new();

        for utterance in &holdout.utterances {
            let (frame_features, num_frames, feature_dim) =
                if matches!(holdout.scoring_source, ScoringInputSource::TeacherCache) {
                    let Some(teacher_features) = utterance.teacher_features.as_ref() else {
                        continue;
                    };
                    let teacher_frames = utterance.teacher_frames;
                    if teacher_frames == 0 {
                        continue;
                    }
                    (
                        Tensor::<B, 2>::from_data(
                            TensorData::new(
                                teacher_features.clone(),
                                Shape::new([teacher_frames, DISTILL_DIM]),
                            ),
                            device,
                        ),
                        teacher_frames,
                        DISTILL_DIM,
                    )
                } else {
                    let Some(model) = model else {
                        continue;
                    };
                    let input =
                        frames_to_tensor::<B>(&utterance.frames, holdout.feature_dim, device);
                    let output = model.forward(input);
                    let frame_features = output.frame_features.squeeze_dim::<2>(0);
                    let num_frames = frame_features.dims()[0];
                    let feature_dim = frame_features.dims()[1];
                    (frame_features, num_frames, feature_dim)
                };

            for &(word_position, num_words, human_score) in &utterance.words {
                let Some((start, end)) =
                    proportional_word_frame_span(num_frames, num_words, word_position)
                else {
                    continue;
                };
                let span_features = frame_features.clone().slice([start..end, 0..feature_dim]);
                let logit =
                    scoring_head.forward_word_span(span_features, scoring_config.pooling_mode);
                let score = scoring_predictions_from_logits(logit, scoring_config, device);
                let Ok(score_val) = scalar_from_tensor(score) else {
                    continue;
                };
                predicted.push(f64::from(score_val));
                human.push(human_score);
            }
        }

        let rho = spearman_rho(&predicted, &human);
        let pcc = pearson_r(&predicted, &human);
        // 1000 iterations (vs 2000 in standalone evaluate) — speed tradeoff
        // for inline eval that runs every N epochs during training.
        let ci = bootstrap_ci_95(&predicted, &human, 1000);
        (rho, pcc, ci, predicted.len())
    }

    #[allow(clippy::indexing_slicing)] // Bounded by construction: offset + feature_dim <= flat.len()
    fn frames_to_tensor<B: Backend>(
        frames: &[Vec<f32>],
        feature_dim: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut flat = vec![0.0_f32; frames.len() * feature_dim];
        for (frame_idx, frame) in frames.iter().enumerate() {
            let offset = frame_idx * feature_dim;
            flat[offset..offset + feature_dim].copy_from_slice(frame);
        }

        Tensor::from_data(
            TensorData::new(flat, Shape::new([1, frames.len(), feature_dim])),
            device,
        )
    }

    /// Build a scoring batch: pool frame features per word, collect targets +
    /// weights.
    ///
    /// Returns `(pooled_features [N, D], targets [N], weights [N])`.
    #[cfg(test)]
    pub fn build_scoring_batch<B: Backend>(
        frame_features: &Tensor<B, 2>,
        words: &[WordTarget],
        scoring_head: &ScoringHead<B>,
        pooling_mode: ScoringPoolingMode,
        device: &B::Device,
    ) -> Option<(Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>)> {
        let dims = frame_features.dims();
        let num_frames = dims[0];
        let feature_dim = dims[1];

        let mut pooled_features = Vec::new();
        let mut targets = Vec::new();
        let mut weights = Vec::new();

        for word in words {
            let Some((start, end)) =
                proportional_word_frame_span(num_frames, word.num_words, word.word_position)
            else {
                continue;
            };

            let span_features = frame_features.clone().slice([start..end, 0..feature_dim]);
            let pooled = scoring_head.pool_word_span(span_features, pooling_mode);
            pooled_features.push(pooled);
            targets.push(word.human_score);
            weights.push(word.weight);
        }

        if pooled_features.is_empty() {
            return None;
        }

        let count = targets.len();
        let pooled_features = Tensor::cat(pooled_features, 0);
        let targets = Tensor::from_data(TensorData::new(targets, Shape::new([count])), device);
        let weights = Tensor::from_data(TensorData::new(weights, Shape::new([count])), device);

        Some((pooled_features, targets, weights))
    }

    #[allow(clippy::needless_pass_by_value)] // Tensor::to_data consumes self
    fn scalar_from_tensor<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .map_err(|err| Error::config(format!("Tensor extraction failed: {err}")))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::config("Tensor extraction returned no values"))
    }

    fn load_model_checkpoint<B: Backend>(
        path: &Path,
        config: &SpeechAlignerConfig,
        device: &B::Device,
    ) -> Result<SpeechAligner<B>> {
        let bytes = std::fs::read(path).map_err(|err| {
            Error::config(format!("Cannot read checkpoint {}: {err}", path.display()))
        })?;
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load(bytes, device)
            .map_err(|err| Error::process(format!("Failed to load checkpoint: {err}")))?;
        let model = config.init(device)?;
        Ok(model.load_record(record))
    }

    pub fn load_scoring_head_checkpoint<B: Backend>(
        path: &Path,
        input_dim: usize,
        device: &B::Device,
    ) -> Result<ScoringHead<B>> {
        let config = load_scoring_head_config(path);
        let bytes = std::fs::read(path).map_err(|err| {
            Error::config(format!(
                "Cannot read scoring head {}: {err}",
                path.display()
            ))
        })?;
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let mut failures = Vec::new();
        let output_dim = scoring_head_output_dim_from_config(&config);

        for hidden_dim in scoring_head_hidden_candidates(&config) {
            let head = ScoringHead::with_dims(input_dim, hidden_dim, output_dim, device);
            match catch_unwind_quiet(|| recorder.load(bytes.clone(), device)) {
                Ok(Ok(record)) => match catch_unwind_quiet(|| head.load_record(record)) {
                    Ok(loaded) => return Ok(loaded),
                    Err(()) => failures.push(format!(
                        "new/{hidden_dim}x{output_dim}: panic during load_record"
                    )),
                },
                Ok(Err(err)) => failures.push(format!("new/{hidden_dim}x{output_dim}: {err}")),
                Err(()) => failures.push(format!(
                    "new/{hidden_dim}x{output_dim}: panic during recorder.load"
                )),
            }

            if output_dim != 1 {
                continue;
            }
            let legacy = LegacyScoringHead::with_hidden(input_dim, hidden_dim, device);
            match catch_unwind_quiet(|| recorder.load(bytes.clone(), device)) {
                Ok(Ok(record)) => match catch_unwind_quiet(|| legacy.load_record(record)) {
                    Ok(legacy) => {
                        return Ok(ScoringHead {
                            linear1: legacy.linear1,
                            linear2: legacy.linear2,
                            pooling_attention: LinearConfig::new(input_dim, 1).init(device),
                        });
                    }
                    Err(()) => {
                        failures.push(format!("legacy/{hidden_dim}x1: panic during load_record"));
                    }
                },
                Ok(Err(err)) => failures.push(format!("legacy/{hidden_dim}x1: {err}")),
                Err(()) => {
                    failures.push(format!("legacy/{hidden_dim}x1: panic during recorder.load"));
                }
            }
        }

        Err(Error::process(format!(
            "Failed to load scoring head {}: {}",
            path.display(),
            failures.join("; ")
        )))
    }

    fn save_model_checkpoint<B: Backend>(model: &SpeechAligner<B>, path: &Path) -> Result<()> {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = model.clone().into_record();
        let bytes = recorder
            .record(record, ())
            .map_err(|err| Error::process(format!("Failed to serialize checkpoint: {err}")))?;
        std::fs::write(path, bytes).map_err(|err| {
            Error::process(format!(
                "Failed to write checkpoint {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    pub fn save_scoring_head_checkpoint<B: Backend>(
        scoring_head: &ScoringHead<B>,
        path: &Path,
    ) -> Result<()> {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = scoring_head.clone().into_record();
        let bytes = recorder
            .record(record, ())
            .map_err(|err| Error::process(format!("Failed to serialize scoring head: {err}")))?;
        std::fs::write(path, bytes).map_err(|err| {
            Error::process(format!(
                "Failed to write scoring head {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_finetune_report(
        args: &FineTuneArgs,
        backend_name: &str,
        base_params: usize,
        head_params: usize,
        utterance_count: usize,
        total_words: usize,
        epoch_stats: &[EpochStats],
        eval_snapshots: &[EvalSnapshot],
        final_model_path: &Path,
        final_head_path: &Path,
    ) -> String {
        let backbone_dim = SpeechAlignerConfig::for_feature_mode(args.feature_mode).channels[3];
        let scoring_source = scoring_input_source_from_args(args);
        let scoring_input_dim =
            scoring_input_dim_from_config(&build_scoring_head_config(args), backbone_dim);
        let mut report = String::new();
        report.push_str("# SpeechAligner SO762 Fine-Tuning Report\n\n");
        let _ = writeln!(
            report,
            "**Date**: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        );
        let _ = writeln!(
            report,
            "**Base checkpoint**: `{}`",
            base_checkpoint_label(args)
        );
        let _ = writeln!(report, "**Dataset**: `{}`", args.dataset.display());
        let _ = writeln!(report, "**Backend**: {backend_name}");
        let _ = writeln!(report, "**Feature mode**: {}", args.feature_mode.label());
        let _ = writeln!(report, "**Teacher as input**: {}", args.teacher_as_input);
        let _ = writeln!(
            report,
            "**Scoring input source**: {}",
            scoring_source.label()
        );
        let _ = writeln!(report, "**Scoring input dim**: {scoring_input_dim}");
        let _ = writeln!(report, "**Base model parameters**: {base_params}");
        let _ = writeln!(report, "**Scoring head parameters**: {head_params}");
        let _ = writeln!(report, "**Epochs**: {}", args.epochs);
        let _ = writeln!(report, "**Learning rate**: {}", args.learning_rate);
        let _ = writeln!(report, "**Scoring loss mode**: {}", scoring_loss_mode(args));
        let _ = writeln!(report, "**CTC weight**: {}", effective_ctc_weight(args));
        let _ = writeln!(report, "**Weighted loss**: {}", args.weighted_loss);
        let _ = writeln!(report, "**Balanced MSE**: {}", args.balanced_mse);
        let _ = writeln!(
            report,
            "**Balanced MSE noise sigma**: {}",
            args.balanced_mse_noise_sigma
        );
        let _ = writeln!(
            report,
            "**Score-balanced loss**: {}",
            args.score_balanced_loss
        );
        let _ = writeln!(
            report,
            "**Score-balance beta**: {}",
            args.score_balance_beta
        );
        let _ = writeln!(report, "**Ordinal loss**: {}", args.ordinal_loss);
        let _ = writeln!(
            report,
            "**Ordinal softmax loss**: {}",
            args.ordinal_softmax_loss
        );
        let _ = writeln!(report, "**Pooling mode**: {:?}", args.pooling_mode);
        let _ = writeln!(
            report,
            "**Freeze backbone**: {}",
            effective_freeze_backbone(args)
        );
        let _ = writeln!(report, "**Scoring hidden dim**: {}", args.scoring_hidden);
        let _ = writeln!(
            report,
            "**Rank regularization weight**: {}",
            args.rank_regularization_weight
        );
        let _ = writeln!(
            report,
            "**Rank similarity tau**: {}",
            args.rank_similarity_tau
        );
        let _ = writeln!(
            report,
            "**Feature mixup alpha**: {}",
            args.feature_mixup_alpha
        );
        let _ = writeln!(
            report,
            "**Feature mixup below**: {}",
            args.feature_mixup_below
        );
        let _ = writeln!(
            report,
            "**Pairwise ranking weight**: {}",
            args.pairwise_ranking_weight
        );
        let _ = writeln!(
            report,
            "**Pairwise ranking margin**: {}",
            args.pairwise_ranking_margin
        );
        let _ = writeln!(
            report,
            "**Triplet ranking weight**: {}",
            args.triplet_ranking_weight
        );
        let _ = writeln!(
            report,
            "**Triplet ranking margin**: {}",
            args.triplet_ranking_margin
        );
        let _ = writeln!(report, "**Oversample factor**: {}", args.oversample_factor);
        let _ = writeln!(report, "**Oversample below**: {}", args.oversample_below);
        let _ = writeln!(report, "**Utterances**: {utterance_count}");
        let _ = writeln!(report, "**Words**: {total_words}");
        let _ = writeln!(report, "**Final model**: `{}`", final_model_path.display());
        let _ = writeln!(
            report,
            "**Final scoring head**: `{}`",
            final_head_path.display()
        );
        report.push_str("\n---\n\n## Epoch Metrics\n\n");
        report.push_str("| Epoch | Total Loss | CTC Loss | MSE Loss | Words | Seconds |\n");
        report.push_str("|---|---|---|---|---|---|\n");
        for (index, stats) in epoch_stats.iter().enumerate() {
            let _ = writeln!(
                report,
                "| {} | {:.6} | {:.6} | {:.6} | {} | {:.1} |",
                index + 1,
                stats.total_loss,
                stats.ctc_loss,
                stats.mse_loss,
                stats.words,
                stats.seconds
            );
        }
        if !eval_snapshots.is_empty() {
            report.push_str("\n## Inline Holdout Evaluation\n\n");
            report.push_str("| Epoch | Spearman ρ | Pearson PCC | 95% CI (ρ) | Words |\n");
            report.push_str("|---|---|---|---|---|\n");
            for snapshot in eval_snapshots {
                let _ = writeln!(
                    report,
                    "| {} | {:.4} | {:.4} | [{:.4}, {:.4}] | {} |",
                    snapshot.epoch,
                    snapshot.rho,
                    snapshot.pcc,
                    snapshot.ci.0,
                    snapshot.ci.1,
                    snapshot.words_scored
                );
            }

            if let Some(best_eval) = eval_snapshots
                .iter()
                .max_by(|left, right| left.rho.total_cmp(&right.rho))
            {
                report.push_str("\n## Gate Snapshot\n\n");
                if best_eval.rho >= SUCCESS_GATE {
                    let _ = writeln!(
                        report,
                        "**FULL SUCCESS** — best inline holdout ρ = {:.4} at epoch {} meets the \
                         {:.2} full-success gate.",
                        best_eval.rho, best_eval.epoch, SUCCESS_GATE
                    );
                } else if best_eval.rho >= START_GATE {
                    let _ = writeln!(
                        report,
                        "**START GATE PASS** — best inline holdout ρ = {:.4} at epoch {} meets \
                         the {:.2} start gate, but remains below the {:.2} \
                         full-success gate.",
                        best_eval.rho, best_eval.epoch, START_GATE, SUCCESS_GATE
                    );
                } else {
                    let _ = writeln!(
                        report,
                        "**FAIL** — best inline holdout ρ = {:.4} at epoch {} remains below the \
                         {:.2} start gate.",
                        best_eval.rho, best_eval.epoch, START_GATE
                    );
                }
            }
        } else {
            report.push_str(
                "\n## Inline Holdout Evaluation\n\nNo inline holdout evaluation was recorded for \
                 this run.\n",
            );
        }
        report.push_str("\n## Next Step\n\n");
        if args.teacher_as_input {
            report.push_str(
                "Run the evaluation with --checkpoint <base> --scoring-head \
                 <head> --distillation-features <cache>` against the holdout split for a final \
                 audit, but use the inline holdout table above as the primary training artifact \
                 for gate review.\n",
            );
        } else {
            report.push_str(
                "Run the evaluation with --checkpoint <base> --scoring-head \
                 <head>` against the holdout split for a final audit, but use the inline holdout \
                 table above as the primary training artifact for gate review.\n",
            );
        }
        report
    }

    fn first_finite_loss(epoch_stats: &[EpochStats]) -> Option<f32> {
        epoch_stats
            .iter()
            .find_map(|stats| stats.total_loss.is_finite().then_some(stats.total_loss))
    }

    fn last_finite_loss(epoch_stats: &[EpochStats]) -> Option<f32> {
        epoch_stats
            .iter()
            .rev()
            .find_map(|stats| stats.total_loss.is_finite().then_some(stats.total_loss))
    }
}

pub use inner::{
    execute_finetune, load_scoring_head_checkpoint, load_scoring_head_config,
    load_teacher_feature_cache, scoring_input_dim_from_config, scoring_predictions_from_logits,
    FineTuneArgs, ScoringHead, ScoringHeadConfig, ScoringInputSource, ScoringPoolingMode,
    TeacherFeatureCache,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Distribution, Shape, Tensor, TensorData};
    use serde_json::Value;
    use tempfile::tempdir;

    use super::inner::{
        balanced_mse_loss, build_scoring_batch, load_scoring_head_checkpoint,
        load_scoring_head_config, load_teacher_feature_cache, ordinal_bin_index,
        pairwise_ranking_loss, save_scoring_head_checkpoint, score_balanced_weights_from_counts,
        scoring_input_dim_from_config, scoring_predictions_from_logits, triplet_ranking_loss,
        EpochStats, LegacyScoringHead, MetricsWriter, ScoringHead, ScoringHeadConfig,
        ScoringInputSource, ScoringPoolingMode, WordTarget,
    };

    type B = NdArray<f32>;

    #[test]
    fn test_scoring_head_forward_shape() {
        let device = NdArrayDevice::default();
        let scoring_head = ScoringHead::<B>::new(512, &device);
        let inputs = Tensor::random([4, 512], Distribution::Normal(0.0, 1.0), &device);

        let outputs = scoring_head.forward(inputs);

        assert_eq!(outputs.dims(), [4, 1]);
    }

    #[test]
    fn test_build_scoring_batch_pools_word_features() {
        let device = NdArrayDevice::default();
        let frame_features = Tensor::<B, 2>::ones([10, 512], &device);
        let scoring_head = ScoringHead::<B>::new(512, &device);
        let words = vec![
            WordTarget {
                word_position: 0,
                human_score: 0.4,
                human_score_raw: 4,
                num_words: 3,
                weight: 1.0,
            },
            WordTarget {
                word_position: 2,
                human_score: 0.9,
                human_score_raw: 9,
                num_words: 3,
                weight: 2.5,
            },
        ];

        let (pooled, targets, weights) = build_scoring_batch::<B>(
            &frame_features,
            &words,
            &scoring_head,
            ScoringPoolingMode::Mean,
            &device,
        )
        .expect("batch");

        assert_eq!(pooled.dims(), [2, 512]);
        assert_eq!(targets.dims(), [2]);
        assert_eq!(weights.dims(), [2]);
    }

    #[test]
    fn test_attention_pooling_produces_single_feature_vector() {
        let device = NdArrayDevice::default();
        let scoring_head = ScoringHead::<B>::new(512, &device);
        let span_features = Tensor::random([6, 512], Distribution::Normal(0.0, 1.0), &device);

        let pooled = scoring_head.pool_word_span(span_features, ScoringPoolingMode::Attention);

        assert_eq!(pooled.dims(), [1, 512]);
    }

    #[test]
    fn test_scoring_head_checkpoint_round_trip() {
        let device = NdArrayDevice::default();
        let scoring_head = ScoringHead::<B>::new(512, &device);
        let tempdir = tempdir().expect("tempdir");
        let checkpoint_path = tempdir.path().join("scoring-head");

        save_scoring_head_checkpoint(&scoring_head, &checkpoint_path).expect("save");
        let loaded =
            load_scoring_head_checkpoint::<B>(&checkpoint_path, 512, &device).expect("load");

        assert_eq!(loaded.num_params(), scoring_head.num_params());
    }

    #[test]
    fn test_load_scoring_head_checkpoint_accepts_legacy_shape() {
        let device = NdArrayDevice::default();
        let legacy_head = LegacyScoringHead::<B>::with_hidden(512, 256, &device);
        let tempdir = tempdir().expect("tempdir");
        let checkpoint_path = tempdir.path().join("legacy-scoring-head");

        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder
            .record(legacy_head.into_record(), ())
            .expect("record legacy head");
        std::fs::write(&checkpoint_path, bytes).expect("write legacy head");

        let loaded =
            load_scoring_head_checkpoint::<B>(&checkpoint_path, 512, &device).expect("load");

        assert_eq!(
            loaded
                .forward(Tensor::<B, 2>::ones([1, 512], &device))
                .dims(),
            [1, 1]
        );
    }

    #[test]
    fn test_load_scoring_head_config_preserves_teacher_input_metadata() {
        let tempdir = tempdir().expect("tempdir");
        let scoring_head_path = tempdir.path().join("teacher-scoring-head");
        let config_path = tempdir.path().join("scoring-head-config.json");
        let config = ScoringHeadConfig {
            scoring_input_source: ScoringInputSource::TeacherCache,
            scoring_input_dim: 1024,
            hidden_dim: 384,
            ..ScoringHeadConfig::default()
        };
        let json = serde_json::to_string(&config).expect("serialize config");
        std::fs::write(&config_path, json).expect("write config");

        let loaded = load_scoring_head_config(&scoring_head_path);

        assert_eq!(
            loaded.scoring_input_source,
            ScoringInputSource::TeacherCache
        );
        assert_eq!(loaded.scoring_input_dim, 1024);
        assert_eq!(loaded.hidden_dim, 384);
    }

    #[test]
    fn test_scoring_input_dim_from_config_defaults_legacy_backbone_heads() {
        let config = ScoringHeadConfig::default();

        assert_eq!(scoring_input_dim_from_config(&config, 512), 512);
    }

    #[test]
    fn test_teacher_feature_cache_reads_cached_frames() {
        let tempdir = tempdir().expect("tempdir");
        let cache_dir = tempdir.path();
        let manifest_path = cache_dir.join("manifest.json");
        let bin_path = cache_dir.join("sample.bin");
        let manifest = serde_json::json!({
            "files": {
                "foo.wav": {
                    "bin": "sample.bin",
                    "frames": 2
                }
            }
        });
        std::fs::write(&manifest_path, manifest.to_string()).expect("write manifest");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"W2V2");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1024u32.to_le_bytes());
        for idx in 0..(2 * 1024) {
            bytes.extend_from_slice(&(idx as f32).to_le_bytes());
        }
        std::fs::write(&bin_path, bytes).expect("write features");

        let cache = load_teacher_feature_cache(cache_dir).expect("load cache");
        let loaded = cache.load("foo.wav").expect("cache read").expect("entry");

        assert_eq!(loaded.1, 2);
        assert_eq!(loaded.0.len(), 2 * 1024);
        assert_eq!(loaded.0[0], 0.0);
        assert_eq!(loaded.0[1024], 1024.0);
    }

    #[test]
    fn test_ordinal_bin_index_matches_declared_buckets() {
        assert_eq!(ordinal_bin_index(0), 0);
        assert_eq!(ordinal_bin_index(2), 0);
        assert_eq!(ordinal_bin_index(3), 1);
        assert_eq!(ordinal_bin_index(5), 1);
        assert_eq!(ordinal_bin_index(6), 2);
        assert_eq!(ordinal_bin_index(7), 2);
        assert_eq!(ordinal_bin_index(8), 3);
        assert_eq!(ordinal_bin_index(9), 3);
        assert_eq!(ordinal_bin_index(10), 4);
    }

    #[test]
    fn test_scoring_predictions_support_ordinal_mode() {
        let device = NdArrayDevice::default();
        let config = ScoringHeadConfig {
            ordinal_loss: true,
            ..ScoringHeadConfig::default()
        };
        let logits = Tensor::<B, 2>::zeros([1, 1], &device);

        let predicted = scoring_predictions_from_logits(logits, &config, &device);
        let value = predicted
            .to_data()
            .to_vec::<f32>()
            .expect("values")
            .into_iter()
            .next()
            .expect("first value");

        assert!((0.45..=0.55).contains(&value));
    }

    #[test]
    fn test_scoring_predictions_support_true_ordinal_softmax_mode() {
        let device = NdArrayDevice::default();
        let config = ScoringHeadConfig {
            ordinal_softmax_loss: true,
            output_dim: 5,
            ..ScoringHeadConfig::default()
        };
        let logits = Tensor::<B, 2>::zeros([1, 5], &device);

        let predicted = scoring_predictions_from_logits(logits, &config, &device);
        let value = predicted
            .to_data()
            .to_vec::<f32>()
            .expect("values")
            .into_iter()
            .next()
            .expect("first value");

        assert!((0.58..=0.62).contains(&value));
    }

    #[test]
    fn test_balanced_mse_treats_duplicate_scores_as_shared_positives() {
        let device = NdArrayDevice::default();
        let predicted = Tensor::<B, 1>::ones([2], &device);
        let targets = Tensor::<B, 1>::ones([2], &device);

        let loss = balanced_mse_loss(predicted, targets, &[10, 10], 0.2, &device);
        let value = loss
            .to_data()
            .to_vec::<f32>()
            .expect("values")
            .into_iter()
            .next()
            .expect("first value");

        assert!(
            value < 1e-4,
            "duplicate-positive Balanced MSE should be near zero"
        );
    }

    #[test]
    fn test_score_balanced_weights_upweight_rare_scores_without_exploding() {
        let counts = BTreeMap::from([(10, 90usize), (9, 8usize), (2, 2usize)]);

        let weights = score_balanced_weights_from_counts(&counts, 0.999);
        let weight_2 = weights.get(&2).copied().expect("rare bucket");
        let weight_9 = weights.get(&9).copied().expect("mid bucket");
        let weight_10 = weights.get(&10).copied().expect("common bucket");

        assert!(weight_2 > weight_9);
        assert!(weight_9 > weight_10);
        assert!(weights.values().all(|weight| *weight < 100.0));
    }

    #[test]
    fn test_pairwise_ranking_loss_zero_when_predictions_respect_margin() {
        let device = NdArrayDevice::default();
        let predicted = Tensor::<B, 1>::from_data(TensorData::from([0.9f32, 0.6, 0.2]), &device);

        let loss = pairwise_ranking_loss(predicted, &[10, 7, 2], 0.05, &device);
        let value = loss
            .to_data()
            .to_vec::<f32>()
            .expect("values")
            .into_iter()
            .next()
            .expect("first value");

        assert!(
            value < 1e-6,
            "pairwise loss should vanish when ordering clears the margin"
        );
    }

    #[test]
    fn test_triplet_ranking_loss_zero_when_features_preserve_order() {
        let device = NdArrayDevice::default();
        let pooled = Tensor::<B, 2>::from_data(
            TensorData::new(
                vec![
                    1.0f32, 0.0, // anchor high
                    1.1, 0.0, // close positive
                    4.0, 0.0, // far negative
                ],
                Shape::new([3, 2]),
            ),
            &device,
        );

        let loss = triplet_ranking_loss(pooled, &[10, 9, 2], 0.1, &device);
        let value = loss
            .to_data()
            .to_vec::<f32>()
            .expect("values")
            .into_iter()
            .next()
            .expect("first value");

        assert!(
            value < 1e-6,
            "triplet loss should vanish when positives are already closer"
        );
    }

    #[test]
    fn test_load_scoring_head_config_defaults_when_missing() {
        let tempdir = tempdir().expect("tempdir");
        let config = load_scoring_head_config(&tempdir.path().join("missing-head"));
        assert!(!config.ordinal_loss);
        assert!(!config.ordinal_softmax_loss);
        assert!(!config.balanced_mse);
        assert_eq!(config.pooling_mode, ScoringPoolingMode::Mean);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.output_dim, 1);
    }

    #[test]
    fn test_metrics_writer_epoch_keeps_batches_and_utterances_distinct() {
        let tempdir = tempdir().expect("tempdir");
        let mut metrics = MetricsWriter::new(tempdir.path());
        metrics.emit_epoch(
            2,
            20,
            32,
            4,
            EpochStats {
                ctc_loss: 0.5,
                mse_loss: 0.75,
                total_loss: 1.25,
                words: 96,
                seconds: 8.0,
            },
        );
        drop(metrics);

        let raw = std::fs::read_to_string(tempdir.path().join("training-metrics.jsonl"))
            .expect("read metrics");
        let line = raw.lines().next().expect("epoch line");
        let value: Value = serde_json::from_str(line).expect("epoch json");

        assert_eq!(value.get("utterances").and_then(Value::as_u64), Some(32));
        assert_eq!(value.get("batches").and_then(Value::as_u64), Some(4));
        let throughput = value
            .get("throughput")
            .and_then(Value::as_f64)
            .expect("throughput");
        assert!((throughput - 4.0).abs() < 1e-6);
    }
}
