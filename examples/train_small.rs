//! Quick-start: train SpeechAligner on synthetic data.
//!
//! Demonstrates the training loop on CPU with a fixed synthetic batch and CTC
//! targets. No external data files are needed.
//!
//! ```bash
//! cargo run --example train_small --features ndarray --release
//! ```
//!
//! Expected output: finite loss values and valid inference output shapes.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Shape, Tensor, TensorData};

use burn_speech_training::error::{Error, Result};
use burn_speech_training::loss::SpeechAlignerLossConfig;
use burn_speech_training::model::SpeechAlignerConfig;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn main() -> Result<()> {
    println!("burn-speech-training: quick-start example\n");
    println!("Training SpeechAligner on synthetic data (CPU)...\n");

    let device = NdArrayDevice::default();

    // ── Model setup ──────────────────────────────────────────────────
    //
    // SpeechAligner: CNN+SE+Attention with 3 output heads.
    // Using a tiny config for fast CPU demo (~122K params instead of 1.7M).
    // For full-scale training, use SpeechAlignerConfig::default().
    let config = SpeechAlignerConfig {
        input_dim: 13, // 13-dim MFCC (no deltas)
        num_classes: 42,
        n_heads: 2, // fewer attention heads
        se_reduction: 4,
        channels: [16, 32, 64, 128], // narrow channels (vs default 64-512)
        kernel_size: 3,
    };
    let model = config.init::<TrainBackend>(&device)?;

    let num_params = model.num_params();
    println!(
        "Model: SpeechAligner ({} parameters)",
        format_params(num_params)
    );
    println!(
        "Config: input_dim={}, num_classes={}, heads={}\n",
        config.input_dim, config.num_classes, config.n_heads
    );

    let loss_fn = SpeechAlignerLossConfig::init();
    let mut optimizer = AdamConfig::new().init::<TrainBackend, _>();

    // ── Synthetic training data ──────────────────────────────────────
    //
    // Generate fixed MFCC-like input and CTC target sequences. Model
    // initialization may still change the exact loss values across runs.
    // In real training, the inputs come from LibriSpeech audio + CMU Dict.
    let batch_size = 2;
    let time_steps = 30; // ~0.3s of audio at 100 frames/sec
    let num_classes = config.num_classes as i32;
    let target_len = 6; // typical word has 4-8 phonemes
    let epochs = 5;

    let feature_data = synthetic_features(batch_size, time_steps, config.input_dim, target_len);
    let target_data = synthetic_targets(batch_size, target_len, num_classes);
    let input_lengths_data = vec![time_steps as i32; batch_size];
    let target_lengths_data = vec![target_len as i32; batch_size];

    println!("Synthetic smoke test: fixed batch={batch_size}, time={time_steps}, targets={target_len}/sample");
    println!("Training for {} epochs...\n", epochs);

    // ── Training loop ────────────────────────────────────────────────
    let mut model = model;

    for epoch in 0..epochs {
        let start = std::time::Instant::now();

        let features: Tensor<TrainBackend, 3> = Tensor::from_data(
            TensorData::new(
                feature_data.clone(),
                Shape::new([batch_size, time_steps, config.input_dim]),
            ),
            &device,
        );

        let targets: Tensor<TrainBackend, 2, Int> = Tensor::from_data(
            TensorData::new(target_data.clone(), Shape::new([batch_size, target_len])),
            &device,
        );

        let input_lengths: Tensor<TrainBackend, 1, Int> = Tensor::from_data(
            TensorData::new(input_lengths_data.clone(), Shape::new([batch_size])),
            &device,
        );
        let target_lengths: Tensor<TrainBackend, 1, Int> = Tensor::from_data(
            TensorData::new(target_lengths_data.clone(), Shape::new([batch_size])),
            &device,
        );

        // Forward pass
        let output = model.forward(features);

        // CTC loss on the alignment head
        let loss = loss_fn.forward(output.ctc_log_probs, targets, input_lengths, target_lengths);

        // Extract scalar loss for reporting
        let loss_scalar: f32 = loss
            .clone()
            .mean()
            .into_data()
            .to_vec::<f32>()
            .map_err(|err| Error::training(format!("failed to extract loss: {err}")))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::training("loss tensor was empty"))?;

        // Backward pass + optimizer step
        let grads = loss.mean().backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.001, model, grads);

        let elapsed = start.elapsed();
        println!(
            "  Epoch {}/{}: loss = {:.4}  ({:.1}s)",
            epoch + 1,
            epochs,
            loss_scalar,
            elapsed.as_secs_f32()
        );
    }

    // ── Inference demo ───────────────────────────────────────────────
    println!("\nRunning inference on the trained model...");

    let infer_model = model.valid();
    let test_input: Tensor<InferBackend, 3> = Tensor::from_data(
        TensorData::new(
            synthetic_features(1, time_steps, config.input_dim, target_len),
            Shape::new([1, time_steps, config.input_dim]),
        ),
        &device,
    );

    let output = infer_model.forward(test_input);
    let phoneme_shape = output.phoneme_logits.dims();
    let boundary_shape = output.boundary_logits.dims();
    let ctc_shape = output.ctc_log_probs.dims();

    println!("  Phoneme logits:  {:?}", phoneme_shape);
    println!("  Boundary logits: {:?}", boundary_shape);
    println!("  CTC log-probs:   {:?}", ctc_shape);

    println!("\nDone. This synthetic run checks that the training surface works.");
    println!("For real training, point the library at LibriSpeech data:");
    println!("  See src/train.rs for the full data pipeline.");
    Ok(())
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

fn synthetic_features(
    batch_size: usize,
    time_steps: usize,
    input_dim: usize,
    target_len: usize,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(batch_size * time_steps * input_dim);
    for sample in 0..batch_size {
        for frame in 0..time_steps {
            let target_band = (frame * target_len / time_steps) as f32;
            for dim in 0..input_dim {
                let harmonic = ((frame + dim + sample) % 7) as f32 * 0.01;
                let feature = target_band * 0.05 + dim as f32 * 0.005 + harmonic;
                values.push(feature);
            }
        }
    }
    values
}

fn synthetic_targets(batch_size: usize, target_len: usize, num_classes: i32) -> Vec<i32> {
    (0..batch_size)
        .flat_map(|sample| {
            (0..target_len).map(move |idx| {
                let value = sample + idx + 1;
                (value as i32 % (num_classes - 1)) + 1
            })
        })
        .collect()
}
