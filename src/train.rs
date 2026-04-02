//! Real-data training loop for `SpeechAligner` using `LibriSpeech`.
//!
//! Supports both CPU (`NdArray`) and GPU (`Cuda`) backends. The backend is
//! selected at compile time via feature flags:
//! - `ndarray`: CPU training (`NdArray` + `Autodiff`)
//! - `wgpu`: GPU training (`Wgpu` + `Autodiff`) via Vulkan/Metal
//! - `cuda`: GPU training (`Cuda` + `Autodiff`) — requires NVIDIA GPU

mod inner {
    use std::io::{BufRead, BufReader, Write as _};
    use std::path::Path;
    use std::time::Instant;

    use crate::g2p::CmuDict;
    use burn::module::{AutodiffModule, Module};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::{AutodiffBackend, Backend};
    use serde_json::Value;

    use crate::dataset::{
        create_batches, create_dynamic_batches, process_utterance, scan_librispeech,
    };
    use crate::error::{Error, Result};
    use crate::loss::SpeechAlignerLossConfig;
    use crate::mfcc::{FeatureMode, MfccExtractor};
    use crate::model::{SpeechAligner, SpeechAlignerConfig};
    use crate::precompute::PrecomputedManifest;
    use crate::ui::{detail, section, step, success};

    /// Arguments for real-data training.
    pub struct TrainRealArgs {
        pub data_dir: std::path::PathBuf,
        pub split: String,
        pub epochs: usize,
        pub batch_size: usize,
        pub learning_rate: f64,
        pub checkpoint_dir: std::path::PathBuf,
        pub checkpoint_interval: usize,
        pub max_duration_secs: f32,
        pub feature_mode: FeatureMode,
    }

    /// Execute real-data training on `LibriSpeech`.
    ///
    /// Selects the backend based on compile-time feature flags:
    /// - `cuda` enabled → CUDA backend on GPU
    /// - `ndarray` only → `NdArray` backend on CPU
    pub fn execute_train_real(args: &TrainRealArgs) -> Result<()> {
        if args.batch_size == 0 {
            return Err(Error::config("batch_size must be > 0"));
        }

        // Step 1: Load data (backend-independent)
        let samples = load_and_extract(args)?;

        // Phase 4: Train with the appropriate backend
        // Priority: CUDA > WGPU > NdArray (CPU fallback)
        #[cfg(feature = "cuda")]
        {
            use burn::backend::cuda::CudaDevice;
            use burn::backend::{Autodiff, Cuda};

            type TrainB = Autodiff<Cuda>;
            type InferB = Cuda;

            let device = CudaDevice::default();
            detail("  Backend: Autodiff<Cuda> (CUDA GPU)");
            train_loop::<TrainB, InferB>(args, samples, &device, "Autodiff<Cuda> (CUDA GPU)")
        }

        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            use burn::backend::Autodiff;

            type TrainB = Autodiff<Wgpu>;
            type InferB = Wgpu;

            let device = WgpuDevice::default();
            detail("  Backend: Autodiff<Wgpu> (Vulkan/Metal GPU)");
            train_loop::<TrainB, InferB>(
                args,
                samples,
                &device,
                "Autodiff<Wgpu> (Vulkan/Metal GPU)",
            )
        }

        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            use burn::backend::{Autodiff, NdArray};

            type TrainB = Autodiff<NdArray>;
            type InferB = NdArray;

            let device = NdArrayDevice::default();
            detail("  Backend: Autodiff<NdArray> (CPU)");
            train_loop::<TrainB, InferB>(args, samples, &device, "Autodiff<NdArray> (CPU)")
        }
    }

    /// Load data and extract features (backend-independent).
    fn load_and_extract(args: &TrainRealArgs) -> Result<Vec<crate::dataset::TrainingSample>> {
        section("SpeechAligner Real-Data Training");

        step("Loading resources");
        let dict =
            CmuDict::load().map_err(|e| Error::config(format!("Failed to load CMUdict: {e}")))?;
        detail("  CMUdict loaded");

        let extractor = MfccExtractor::with_mode(args.feature_mode);
        detail(&format!(
            "  Feature extractor initialized ({})",
            args.feature_mode.label()
        ));

        step(&format!(
            "Scanning LibriSpeech: {} / {}",
            args.data_dir.display(),
            args.split
        ));
        let utterances = scan_librispeech(&args.data_dir, &args.split, args.max_duration_secs)?;
        detail(&format!("  Found {} utterances", utterances.len()));

        step("Phase 3: Extracting features");
        let mut samples = Vec::new();
        let mut skipped = 0usize;
        let mut total_oov = 0usize;
        let total = utterances.len();

        for (i, meta) in utterances.iter().enumerate() {
            match process_utterance(meta, &extractor, &dict) {
                Ok(sample) => {
                    total_oov += sample.oov_count;
                    samples.push(sample);
                }
                Err(e) => {
                    tracing::debug!("Skipping utterance {}: {e}", meta.audio_path.display());
                    skipped += 1;
                }
            }
            if (i + 1) % 500 == 0 || i + 1 == total {
                detail(&format!(
                    "  Processed {}/{} (skipped: {}, OOV words: {})",
                    i + 1,
                    total,
                    skipped,
                    total_oov
                ));
            }
        }

        if samples.is_empty() {
            return Err(Error::config("No valid training samples after processing"));
        }

        detail(&format!(
            "  Final: {} samples, {} skipped, {} total OOV words",
            samples.len(),
            skipped,
            total_oov
        ));

        Ok(samples)
    }

    // -----------------------------------------------------------------------
    // Structured training metrics (JSON-lines + tracing)
    // -----------------------------------------------------------------------

    /// Writes structured JSON-lines metrics to a file alongside the checkpoint
    /// directory. Each line is a self-contained JSON object parseable by:
    /// - Structured JSON logs for external monitoring
    /// - Log aggregation dashboards (configure per deployment)
    struct MetricsWriter {
        file: Option<std::fs::File>,
        run_start: Instant,
        elapsed_offset_secs: f64,
    }

    impl MetricsWriter {
        fn new(checkpoint_dir: &Path, append: bool, elapsed_offset_secs: f64) -> Self {
            let path = checkpoint_dir.join("training-metrics.jsonl");
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(!append)
                .append(append)
                .open(&path)
                .ok();
            if file.is_some() {
                detail(&format!("  Metrics: {}", path.display()));
            }
            Self {
                file,
                run_start: Instant::now(),
                elapsed_offset_secs,
            }
        }

        fn elapsed_secs(&self) -> f64 {
            self.elapsed_offset_secs + self.run_start.elapsed().as_secs_f64()
        }

        /// Emit a batch-level metric (every N batches).
        #[allow(clippy::too_many_arguments)]
        fn emit_batch(
            &mut self,
            epoch: usize,
            batch: usize,
            total_batches: usize,
            batch_samples: usize,
            loss: f32,
            batch_elapsed_secs: f64,
        ) {
            let elapsed = self.elapsed_secs();
            let samples_per_sec = if batch_elapsed_secs > 0.0 {
                batch_samples as f64 / batch_elapsed_secs
            } else {
                0.0
            };

            // Structured log for monitoring
            tracing::info!(
                target: "training_metrics",
                event = "batch",
                epoch = epoch,
                batch = batch,
                total_batches = total_batches,
                batch_samples = batch_samples,
                loss = loss,
                samples_per_sec = samples_per_sec,
                elapsed_secs = elapsed,
                "training_batch"
            );

            // JSON-lines file for Grafana/offline analysis
            if let Some(ref mut f) = self.file {
                let _ = writeln!(
                    f,
                    r#"{{"event":"batch","epoch":{epoch},"batch":{batch},"total_batches":{total_batches},"batch_samples":{batch_samples},"loss":{loss:.6},"samples_per_sec":{samples_per_sec:.2},"elapsed_secs":{elapsed:.1}}}"#,
                );
                let _ = f.flush();
            }
        }

        /// Emit an epoch-level summary.
        #[allow(clippy::too_many_arguments)]
        fn emit_epoch(
            &mut self,
            epoch: usize,
            total_epochs: usize,
            avg_loss: f32,
            batch_count: usize,
            epoch_secs: f64,
            total_samples: usize,
        ) {
            let elapsed = self.elapsed_secs();
            let remaining_epochs = total_epochs.saturating_sub(epoch);
            #[allow(clippy::cast_precision_loss)]
            let eta_secs = if epoch > 0 {
                (elapsed / epoch as f64) * remaining_epochs as f64
            } else {
                0.0
            };
            #[allow(clippy::cast_precision_loss)]
            let throughput = total_samples as f64 / epoch_secs.max(0.001);

            tracing::info!(
                target: "training_metrics",
                event = "epoch",
                epoch,
                total_epochs,
                avg_loss,
                batch_count,
                epoch_secs,
                throughput_samples_per_sec = throughput,
                elapsed_secs = elapsed,
                eta_secs,
                "training_epoch"
            );

            if let Some(ref mut f) = self.file {
                let _ = writeln!(
                    f,
                    r#"{{"event":"epoch","epoch":{epoch},"total_epochs":{total_epochs},"avg_loss":{avg_loss:.6},"batches":{batch_count},"epoch_secs":{epoch_secs:.1},"throughput":{throughput:.1},"elapsed_secs":{elapsed:.1},"eta_secs":{eta_secs:.1}}}"#,
                );
                let _ = f.flush();
            }

            detail(&format!(
                "  epoch {epoch}/{total_epochs} avg_loss={avg_loss:.6} ({batch_count} batches, \
                 {epoch_secs:.1}s, {throughput:.0} samples/s, ETA {eta_secs:.0}s)"
            ));
        }

        /// Emit training completion summary.
        fn emit_complete(&mut self, epochs: usize, first_loss: f32, last_loss: f32) {
            let elapsed = self.elapsed_secs();
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

            if let Some(ref mut f) = self.file {
                let _ = writeln!(
                    f,
                    r#"{{"event":"complete","epochs":{epochs},"first_loss":{first_loss:.6},"last_loss":{last_loss:.6},"reduction_pct":{reduction_pct:.1},"total_secs":{elapsed:.1}}}"#,
                );
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    pub(super) struct ResumeState {
        pub(super) completed_epochs: usize,
        pub(super) elapsed_offset_secs: f64,
    }

    fn json_usize(value: &Value, key: &str) -> Option<usize> {
        value
            .get(key)?
            .as_u64()
            .and_then(|raw| usize::try_from(raw).ok())
    }

    fn json_f64(value: &Value, key: &str) -> Option<f64> {
        value.get(key)?.as_f64()
    }

    fn checkpoint_epoch_from_name(file_name: &str) -> Option<usize> {
        file_name
            .strip_prefix("speech-aligner-epoch-")?
            .parse::<usize>()
            .ok()
    }

    fn checkpoint_epoch_from_path(path: &Path) -> Option<usize> {
        checkpoint_epoch_from_name(path.file_name()?.to_str()?)
    }

    pub(super) fn collect_resume_state(
        checkpoint_dir: &Path,
        resume_from: Option<&Path>,
    ) -> Result<ResumeState> {
        let mut state = ResumeState::default();

        if let Some(resume_path) = resume_from {
            if let Some(epoch) = checkpoint_epoch_from_path(resume_path) {
                state.completed_epochs = state.completed_epochs.max(epoch);
            }
        }

        if let Ok(entries) = std::fs::read_dir(checkpoint_dir) {
            for entry in entries {
                let entry = entry.map_err(|err| {
                    Error::process(format!(
                        "Failed to read checkpoint dir {}: {err}",
                        checkpoint_dir.display()
                    ))
                })?;
                let file_name = entry.file_name();
                let Some(file_name) = file_name.to_str() else {
                    continue;
                };
                if let Some(epoch) = checkpoint_epoch_from_name(file_name) {
                    state.completed_epochs = state.completed_epochs.max(epoch);
                }
            }
        }

        let metrics_path = checkpoint_dir.join("training-metrics.jsonl");
        if let Ok(file) = std::fs::File::open(&metrics_path) {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|err| {
                    Error::process(format!(
                        "Failed to read metrics file {}: {err}",
                        metrics_path.display()
                    ))
                })?;
                let Ok(value) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };

                if let Some(epoch) = json_usize(&value, "epoch") {
                    state.completed_epochs = state.completed_epochs.max(epoch);
                } else if let Some(epoch) = json_usize(&value, "epochs") {
                    state.completed_epochs = state.completed_epochs.max(epoch);
                }

                if let Some(elapsed_secs) =
                    json_f64(&value, "elapsed_secs").or_else(|| json_f64(&value, "total_secs"))
                {
                    state.elapsed_offset_secs = state.elapsed_offset_secs.max(elapsed_secs);
                }
            }
        }

        Ok(state)
    }

    /// Generic training loop that works with any Burn backend.
    fn train_loop<TrainB, InferB>(
        args: &TrainRealArgs,
        mut samples: Vec<crate::dataset::TrainingSample>,
        device: &TrainB::Device,
        backend_name: &str,
    ) -> Result<()>
    where
        TrainB: AutodiffBackend,
        InferB: Backend<FloatElem = f32>,
        SpeechAligner<TrainB>: AutodiffModule<TrainB, InnerModule = SpeechAligner<InferB>>,
    {
        step(&format!(
            "Phase 4: Training ({} epochs, batch_size={}, lr={}, backend={})",
            args.epochs, args.batch_size, args.learning_rate, backend_name
        ));

        TrainB::seed(device, 42);

        let config = SpeechAlignerConfig::for_feature_mode(args.feature_mode);
        let mut model: SpeechAligner<TrainB> = config.init(device)?;
        let loss_fn = SpeechAlignerLossConfig::init();
        let mut optimizer = AdamConfig::new().init::<TrainB, SpeechAligner<TrainB>>();

        let param_count = model.num_params();
        detail(&format!("  Model parameters: {param_count}"));

        std::fs::create_dir_all(&args.checkpoint_dir).map_err(|e| {
            Error::process(format!(
                "Failed to create checkpoint dir {}: {e}",
                args.checkpoint_dir.display()
            ))
        })?;

        let mut epoch_losses = Vec::new();
        let mut metrics = MetricsWriter::new(&args.checkpoint_dir, false, 0.0);
        let total_samples = samples.len();

        for epoch in 0..args.epochs {
            let epoch_start = Instant::now();
            let mut batches = create_batches::<TrainB>(&mut samples, args.batch_size, device);

            if epoch % 2 == 1 {
                batches.reverse();
            }

            let num_batches = batches.len();
            let mut epoch_loss_sum = 0.0f64;
            let mut batch_count = 0usize;

            for (batch_idx, batch) in batches.into_iter().enumerate() {
                let batch_start = Instant::now();
                let batch_size = batch.input_lengths.dims()[0];
                let output = model.forward_with_pad_mask(batch.inputs, batch.pad_mask);
                let loss_per_sample = loss_fn.forward(
                    output.ctc_log_probs,
                    batch.targets,
                    batch.input_lengths,
                    batch.target_lengths,
                );
                let loss = loss_per_sample.clone().mean();

                let loss_data = loss.clone().to_data();
                let loss_val: f32 = loss_data
                    .to_vec::<f32>()
                    .map_err(|e| Error::config(format!("Loss extraction failed: {e}")))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::config("Empty loss tensor"))?;

                if loss_val.is_finite() {
                    epoch_loss_sum += f64::from(loss_val);
                    batch_count += 1;

                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(args.learning_rate, model, grads);
                } else {
                    tracing::warn!(
                        "Non-finite loss at epoch {}, batch {}: {loss_val} — skipping update",
                        epoch + 1,
                        batch_idx + 1
                    );
                }

                if (batch_idx + 1) % 50 == 0 || batch_idx + 1 == num_batches {
                    let batch_secs = batch_start.elapsed().as_secs_f64();
                    metrics.emit_batch(
                        epoch + 1,
                        batch_idx + 1,
                        num_batches,
                        batch_size,
                        loss_val,
                        batch_secs,
                    );
                }
            }

            #[allow(clippy::cast_precision_loss)]
            let avg_loss = if batch_count > 0 {
                (epoch_loss_sum / batch_count as f64) as f32
            } else {
                f32::NAN
            };
            epoch_losses.push(avg_loss);

            let epoch_secs = epoch_start.elapsed().as_secs_f64();
            metrics.emit_epoch(
                epoch + 1,
                args.epochs,
                avg_loss,
                batch_count,
                epoch_secs,
                total_samples,
            );

            if args.checkpoint_interval > 0 && (epoch + 1) % args.checkpoint_interval == 0 {
                let ckpt_path = args
                    .checkpoint_dir
                    .join(format!("speech-aligner-epoch-{}", epoch + 1));
                save_model_checkpoint(&model, &ckpt_path)?;
                detail(&format!("  Checkpoint saved: {}", ckpt_path.display()));
            }
        }

        // Final checkpoint (inference model)
        let model_infer: SpeechAligner<InferB> = model.valid();
        let final_ckpt = args.checkpoint_dir.join("speech-aligner-final");
        save_model_checkpoint(&model_infer, &final_ckpt)?;

        tracing::info!("");
        success(&format!(
            "Training complete: {} epochs, {} samples, final checkpoint at {}",
            args.epochs,
            total_samples,
            final_ckpt.display()
        ));

        if let (Some(&first), Some(&last)) = (epoch_losses.first(), epoch_losses.last()) {
            if first.is_finite() && last.is_finite() && first.abs() > f32::EPSILON {
                metrics.emit_complete(args.epochs, first, last);
                let pct = ((first - last) / first) * 100.0;
                detail(&format!(
                    "  Loss: {first:.4} -> {last:.4} ({pct:.1}% reduction)"
                ));
            }
        }

        let report = generate_training_report(
            args,
            &epoch_losses,
            param_count,
            total_samples,
            backend_name,
        );
        let report_path = args.checkpoint_dir.join("train-real-report.md");
        std::fs::write(&report_path, &report)
            .map_err(|e| Error::process(format!("Failed to write report: {e}")))?;
        detail(&format!("  Report: {}", report_path.display()));

        Ok(())
    }

    /// Save a model checkpoint (works with both training and inference models).
    fn save_model_checkpoint<B: Backend>(model: &SpeechAligner<B>, path: &Path) -> Result<()> {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = model.clone().into_record();
        let bytes = recorder
            .record(record, ())
            .map_err(|e| Error::process(format!("Failed to serialize checkpoint: {e}")))?;
        std::fs::write(path, &bytes).map_err(|e| {
            Error::process(format!(
                "Failed to write checkpoint to {}: {e}",
                path.display()
            ))
        })?;
        Ok(())
    }

    fn generate_training_report(
        args: &TrainRealArgs,
        losses: &[f32],
        params: usize,
        num_samples: usize,
        backend_name: &str,
    ) -> String {
        use std::fmt::Write as _;

        let mut report = String::new();
        report.push_str("# SpeechAligner Real-Data Training Report\n\n");
        let _ = writeln!(
            report,
            "**Date**: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        );
        report.push_str("**Burn Version**: 0.21.0-pre.2\n");
        let _ = writeln!(report, "**Backend**: {backend_name}");
        report.push_str("**Model**: SpeechAligner (4x ConvSE + Self-Attention + Triple Head)\n");
        let _ = writeln!(report, "**Parameters**: {params}");
        let _ = writeln!(report, "**Dataset**: {}", args.data_dir.display());
        let _ = writeln!(report, "**Split**: {}", args.split);
        let _ = writeln!(report, "**Samples**: {num_samples}");
        let _ = writeln!(report, "**Epochs**: {}", args.epochs);
        let _ = writeln!(report, "**Batch Size**: {}", args.batch_size);
        let _ = writeln!(report, "**Learning Rate**: {}", args.learning_rate);
        let _ = writeln!(
            report,
            "**Checkpoint**: {}\n",
            args.checkpoint_dir.display()
        );
        report.push_str("---\n\n## Loss Curve\n\n");
        report.push_str("| Epoch | Avg Loss |\n|---|---|\n");
        for (i, l) in losses.iter().enumerate() {
            let _ = writeln!(report, "| {} | {l:.6} |", i + 1);
        }
        report.push_str(
            "\n## Specification References\n\n- Data: \
             LibriSpeech\n",
        );
        report
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_precomputed_training_report(
        args: &TrainPrecomputedArgs,
        manifest: &PrecomputedManifest,
        losses: &[f32],
        params: usize,
        num_samples: usize,
        backend_name: &str,
        attention_heads: usize,
    ) -> String {
        use std::fmt::Write as _;

        let mut report = String::new();
        report.push_str("# SpeechAligner Precomputed Training Report\n\n");
        let _ = writeln!(
            report,
            "**Date**: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        );
        report.push_str("**Burn Version**: 0.21.0-pre.2\n");
        let _ = writeln!(report, "**Backend**: {backend_name}");
        report.push_str("**Model**: SpeechAligner (4x ConvSE + Self-Attention + Triple Head)\n");
        let _ = writeln!(report, "**Parameters**: {params}");
        let _ = writeln!(report, "**Cache Dir**: {}", args.cache_dir.display());
        let _ = writeln!(report, "**Source Data**: {}", manifest.data_dir);
        let _ = writeln!(report, "**Split**: {}", manifest.split);
        let duration_cap = manifest.max_duration_secs.map_or_else(
            || "unknown (legacy manifest)".to_owned(),
            |secs| format!("{secs:.1}s"),
        );
        let _ = writeln!(report, "**Duration Cap**: {duration_cap}");
        let _ = writeln!(report, "**Samples**: {num_samples}");
        let _ = writeln!(report, "**Total Frames**: {}", manifest.total_frames);
        let _ = writeln!(report, "**Feature Dim**: {}", manifest.feature_dim);
        let _ = writeln!(report, "**Total OOV**: {}", manifest.total_oov);
        let _ = writeln!(report, "**Epochs**: {}", args.epochs);
        let _ = writeln!(report, "**Attention Heads**: {attention_heads}");
        let _ = writeln!(
            report,
            "**Max Attention Elements**: {}",
            args.max_attn_elements
        );
        let _ = writeln!(
            report,
            "**Max Samples Per Batch**: {}",
            args.max_samples_per_batch
        );
        let _ = writeln!(report, "**Learning Rate**: {}", args.learning_rate);
        let _ = writeln!(report, "**Checkpoint**: {}", args.checkpoint_dir.display());
        if let Some(ref resume_path) = args.resume_from {
            let _ = writeln!(report, "**Resume From**: {}", resume_path.display());
        }
        report.push_str(
            "\n---\n\n## Batching Strategy\n\n- Attention-aware dynamic batching (`B × heads × \
             T²`)\n- Precomputed MFCC cache (no on-the-fly extraction)\n\n## Loss Curve\n\n",
        );
        report.push_str("| Epoch | Avg Loss |\n|---|---|\n");
        for (i, loss) in losses.iter().enumerate() {
            let _ = writeln!(report, "| {} | {loss:.6} |", i + 1);
        }
        report.push_str(
            "\n## Specification References\n\n- Data: \
             LibriSpeech precomputed MFCC cache\n",
        );
        report
    }

    // =====================================================================
    // Precomputed training: loads from binary cache, no MFCC
    // =====================================================================

    /// Arguments for precomputed training (fast ablation mode).
    pub struct TrainPrecomputedArgs {
        /// Directory containing precomputed binary cache + manifest.json.
        pub cache_dir: std::path::PathBuf,
        /// Number of training epochs.
        pub epochs: usize,
        /// Max attention score elements per batch (`B × heads × T²`). Controls
        /// GPU memory: higher → bigger batches → faster. Default 500M is
        /// safe for 80 GB GPUs (~6 GB peak with backward pass).
        pub max_attn_elements: usize,
        /// Hard cap on samples per batch (even if memory budget allows more).
        pub max_samples_per_batch: usize,
        /// Learning rate.
        pub learning_rate: f64,
        /// Directory for checkpoint output.
        pub checkpoint_dir: std::path::PathBuf,
        /// Save checkpoint every N epochs (0 = only final).
        pub checkpoint_interval: usize,
        /// Optional: resume from this checkpoint path.
        pub resume_from: Option<std::path::PathBuf>,
    }

    /// Execute training from precomputed feature cache.
    ///
    /// This is the fast training path: no audio loading, no FFT,
    /// no MFCC extraction. Just tensor loading → GPU training.
    pub fn execute_train_precomputed(args: &TrainPrecomputedArgs) -> Result<()> {
        if args.max_attn_elements == 0 {
            return Err(Error::config("max_attn_elements must be > 0"));
        }
        if args.max_samples_per_batch == 0 {
            return Err(Error::config("max_samples_per_batch must be > 0"));
        }

        section("SpeechAligner Precomputed Training");

        // 1. Load precomputed samples
        step("Loading precomputed features");
        let (manifest, samples) = crate::precompute::load_precomputed(&args.cache_dir)?;
        detail(&format!(
            "  {} samples, {} total frames, feature_dim={}",
            manifest.num_samples, manifest.total_frames, manifest.feature_dim
        ));
        detail(&format!(
            "  Attention-aware batching: max_attn_elements={}, max_samples={}",
            args.max_attn_elements, args.max_samples_per_batch
        ));
        if let Some(max_duration_secs) = manifest.max_duration_secs {
            detail(&format!("  Cache duration cap: {max_duration_secs:.1}s"));
        } else {
            detail("  Cache duration cap: unknown (legacy manifest)");
        }

        // 2. Dispatch to backend
        #[cfg(feature = "cuda")]
        {
            use burn::backend::cuda::CudaDevice;
            use burn::backend::{Autodiff, Cuda};

            type TrainB = Autodiff<Cuda>;
            type InferB = Cuda;

            let device = CudaDevice::default();
            detail("  Backend: Autodiff<Cuda> (CUDA GPU)");
            precomputed_train_loop::<TrainB, InferB>(
                args,
                &manifest,
                samples,
                &device,
                "Autodiff<Cuda> (CUDA GPU)",
            )
        }

        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            use burn::backend::Autodiff;

            type TrainB = Autodiff<Wgpu>;
            type InferB = Wgpu;

            let device = WgpuDevice::default();
            detail("  Backend: Autodiff<Wgpu> (Vulkan/Metal GPU)");
            precomputed_train_loop::<TrainB, InferB>(
                args,
                &manifest,
                samples,
                &device,
                "Autodiff<Wgpu> (Vulkan/Metal GPU)",
            )
        }

        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            use burn::backend::{Autodiff, NdArray};

            type TrainB = Autodiff<NdArray>;
            type InferB = NdArray;

            let device = NdArrayDevice::default();
            detail("  Backend: Autodiff<NdArray> (CPU)");
            precomputed_train_loop::<TrainB, InferB>(
                args,
                &manifest,
                samples,
                &device,
                "Autodiff<NdArray> (CPU)",
            )
        }
    }

    /// Training loop for precomputed data — same core as `train_loop` but
    /// with checkpoint resumption and optimized defaults.
    fn precomputed_train_loop<TrainB, InferB>(
        args: &TrainPrecomputedArgs,
        manifest: &PrecomputedManifest,
        mut samples: Vec<crate::dataset::TrainingSample>,
        device: &TrainB::Device,
        backend_name: &str,
    ) -> Result<()>
    where
        TrainB: AutodiffBackend,
        InferB: Backend<FloatElem = f32>,
        SpeechAligner<TrainB>: AutodiffModule<TrainB, InnerModule = SpeechAligner<InferB>>,
    {
        step(&format!(
            "Training: {} epochs, max_attn_elements={}, max_samples={}, lr={}, backend={}",
            args.epochs,
            args.max_attn_elements,
            args.max_samples_per_batch,
            args.learning_rate,
            backend_name
        ));

        TrainB::seed(device, 42);

        let config = SpeechAlignerConfig::with_input_dim(manifest.feature_dim);
        let attention_heads = config.n_heads;
        let mut model: SpeechAligner<TrainB> = config.init(device)?;

        // Resume from checkpoint if provided
        if let Some(ref resume_path) = args.resume_from {
            step(&format!("Resuming from {}", resume_path.display()));
            let bytes = std::fs::read(resume_path).map_err(|e| {
                Error::config(format!(
                    "Cannot read checkpoint {}: {e}",
                    resume_path.display()
                ))
            })?;
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
            let record = recorder
                .load(bytes, device)
                .map_err(|e| Error::process(format!("Failed to load checkpoint: {e}")))?;
            model = model.load_record(record);
            detail("  Checkpoint loaded — continuing training");
        }

        let loss_fn = SpeechAlignerLossConfig::init();
        let mut optimizer = AdamConfig::new().init::<TrainB, SpeechAligner<TrainB>>();

        let param_count = model.num_params();
        detail(&format!("  Model parameters: {param_count}"));

        std::fs::create_dir_all(&args.checkpoint_dir).map_err(|e| {
            Error::process(format!(
                "Failed to create checkpoint dir {}: {e}",
                args.checkpoint_dir.display()
            ))
        })?;

        let resume_state = collect_resume_state(&args.checkpoint_dir, args.resume_from.as_deref())?;
        if resume_state.completed_epochs > 0 {
            detail(&format!(
                "  Resume offset: {} prior epochs, {:.1}s prior elapsed",
                resume_state.completed_epochs, resume_state.elapsed_offset_secs
            ));
        }

        let mut epoch_losses = Vec::new();
        let mut metrics = MetricsWriter::new(
            &args.checkpoint_dir,
            resume_state.completed_epochs > 0,
            resume_state.elapsed_offset_secs,
        );
        let total_samples = samples.len();
        let total_epochs_planned = resume_state.completed_epochs + args.epochs;

        for epoch in 0..args.epochs {
            let epoch_num = resume_state.completed_epochs + epoch + 1;
            let epoch_start = Instant::now();
            let mut batches = create_dynamic_batches::<TrainB>(
                &mut samples,
                args.max_attn_elements,
                args.max_samples_per_batch,
                attention_heads,
                device,
            );

            if epoch == 0 {
                detail(&format!(
                    "  Dynamic batching: {} batches this epoch",
                    batches.len()
                ));
                if let (Some(first), Some(last)) = (batches.first(), batches.last()) {
                    let first_size = first.input_lengths.dims()[0];
                    let last_size = last.input_lengths.dims()[0];
                    detail(&format!(
                        "  First batch: {first_size} samples (short), last batch: {last_size} \
                         samples (long)"
                    ));
                }
            }

            // Alternate sort direction for regularization
            if epoch % 2 == 1 {
                batches.reverse();
            }

            let num_batches = batches.len();
            let mut epoch_loss_sum = 0.0f64;
            let mut batch_count = 0usize;

            for (batch_idx, batch) in batches.into_iter().enumerate() {
                let batch_start = Instant::now();
                let batch_size = batch.input_lengths.dims()[0];
                let output = model.forward_with_pad_mask(batch.inputs, batch.pad_mask);
                let loss_per_sample = loss_fn.forward(
                    output.ctc_log_probs,
                    batch.targets,
                    batch.input_lengths,
                    batch.target_lengths,
                );
                let loss = loss_per_sample.clone().mean();

                let loss_data = loss.clone().to_data();
                let loss_val: f32 = loss_data
                    .to_vec::<f32>()
                    .map_err(|e| Error::config(format!("Loss extraction failed: {e}")))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::config("Empty loss tensor"))?;

                if loss_val.is_finite() {
                    epoch_loss_sum += f64::from(loss_val);
                    batch_count += 1;

                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(args.learning_rate, model, grads);
                } else {
                    tracing::warn!(
                        "Non-finite loss at epoch {}, batch {}: {loss_val} — skipping",
                        epoch_num,
                        batch_idx + 1
                    );
                }

                if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == num_batches {
                    let batch_secs = batch_start.elapsed().as_secs_f64();
                    metrics.emit_batch(
                        epoch_num,
                        batch_idx + 1,
                        num_batches,
                        batch_size,
                        loss_val,
                        batch_secs,
                    );
                }
            }

            #[allow(clippy::cast_precision_loss)]
            let avg_loss = if batch_count > 0 {
                (epoch_loss_sum / batch_count as f64) as f32
            } else {
                f32::NAN
            };
            epoch_losses.push(avg_loss);

            let epoch_secs = epoch_start.elapsed().as_secs_f64();
            metrics.emit_epoch(
                epoch_num,
                total_epochs_planned,
                avg_loss,
                batch_count,
                epoch_secs,
                total_samples,
            );

            if args.checkpoint_interval > 0 && epoch_num % args.checkpoint_interval == 0 {
                let ckpt_path = args
                    .checkpoint_dir
                    .join(format!("speech-aligner-epoch-{epoch_num}"));
                save_model_checkpoint(&model, &ckpt_path)?;
                detail(&format!("  Checkpoint saved: {}", ckpt_path.display()));
            }
        }

        // Final checkpoint (inference model)
        let model_infer: SpeechAligner<InferB> = model.valid();
        let final_ckpt = args.checkpoint_dir.join("speech-aligner-final");
        save_model_checkpoint(&model_infer, &final_ckpt)?;

        tracing::info!("");
        success(&format!(
            "Training complete: {} total epochs ({} this leg), {} samples, attention-aware \
             batching (max_attn_elements={}), checkpoint at {}",
            total_epochs_planned,
            args.epochs,
            total_samples,
            args.max_attn_elements,
            final_ckpt.display()
        ));

        if let (Some(&first), Some(&last)) = (epoch_losses.first(), epoch_losses.last()) {
            if first.is_finite() && last.is_finite() && first.abs() > f32::EPSILON {
                metrics.emit_complete(total_epochs_planned, first, last);
                let pct = ((first - last) / first) * 100.0;
                detail(&format!(
                    "  Loss: {first:.4} -> {last:.4} ({pct:.1}% reduction)"
                ));
            }
        }

        let report = generate_precomputed_training_report(
            args,
            manifest,
            &epoch_losses,
            param_count,
            total_samples,
            backend_name,
            attention_heads,
        );
        let report_path = args.checkpoint_dir.join("train-precomputed-report.md");
        std::fs::write(&report_path, &report)
            .map_err(|e| Error::process(format!("Failed to write report: {e}")))?;
        detail(&format!("  Report: {}", report_path.display()));

        Ok(())
    }
}

pub use inner::{
    execute_train_precomputed, execute_train_real, TrainPrecomputedArgs, TrainRealArgs,
};

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::inner::{collect_resume_state, ResumeState};

    #[test]
    fn test_collect_resume_state_uses_existing_metrics_and_checkpoints() {
        let tempdir = tempdir().expect("tempdir");
        let checkpoint_dir = tempdir.path();
        std::fs::write(
            checkpoint_dir.join("training-metrics.jsonl"),
            concat!(
                "{\"event\":\"epoch\",\"epoch\":2,\"elapsed_secs\":18.0}\n",
                "{\"event\":\"batch\",\"epoch\":2,\"elapsed_secs\":20.0}\n"
            ),
        )
        .expect("write metrics");
        std::fs::write(checkpoint_dir.join("speech-aligner-epoch-3"), b"")
            .expect("write checkpoint");

        let state = collect_resume_state(
            checkpoint_dir,
            Some(Path::new("/tmp/speech-aligner-epoch-2")),
        )
        .expect("resume state");

        assert_eq!(
            state,
            ResumeState {
                completed_epochs: 3,
                elapsed_offset_secs: 20.0
            }
        );
    }
}
