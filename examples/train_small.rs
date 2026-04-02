//! Quick-start: train SpeechAligner on synthetic data.
//!
//! Demonstrates the full training loop on CPU with randomly generated
//! input features and CTC targets. No external data files needed.
//!
//! ```bash
//! cargo run --example train_small --features ndarray
//! ```
//!
//! Expected output: loss decreasing over 5 epochs, completing in < 1 second (release mode).

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Distribution, Int, Shape, Tensor, TensorData};

use burn_speech_training::loss::SpeechAlignerLossConfig;
use burn_speech_training::model::SpeechAlignerConfig;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn main() {
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
    let model = config
        .init::<TrainBackend>(&device)
        .expect("model init should succeed");

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
    // Generate random MFCC-like features and CTC target sequences.
    // In real training, these come from LibriSpeech audio + CMU Dict.
    let batch_size = 2;
    let time_steps = 30; // ~0.3s of audio at 100 frames/sec
    let num_classes = config.num_classes as i32;
    let target_len = 6; // typical word has 4-8 phonemes
    let epochs = 5;

    println!(
        "Synthetic data: batch={}, time={}, targets={}/sample",
        batch_size, time_steps, target_len
    );
    println!("Training for {} epochs...\n", epochs);

    // ── Training loop ────────────────────────────────────────────────
    let mut model = model;

    for epoch in 0..epochs {
        let start = std::time::Instant::now();

        // Generate a fresh random batch each epoch (simulates data variation)
        let features: Tensor<TrainBackend, 3> = Tensor::random(
            [batch_size, time_steps, config.input_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Random phoneme targets (1..num_classes, avoiding 0 = CTC blank)
        let target_data: Vec<i32> = (0..batch_size * target_len)
            .map(|i| (i as i32 % (num_classes - 1)) + 1)
            .collect();
        let targets: Tensor<TrainBackend, 2, Int> = Tensor::from_data(
            TensorData::new(target_data, Shape::new([batch_size, target_len])),
            &device,
        );

        let input_lengths: Tensor<TrainBackend, 1, Int> = Tensor::from_data(
            TensorData::new(
                vec![time_steps as i32; batch_size],
                Shape::new([batch_size]),
            ),
            &device,
        );
        let target_lengths: Tensor<TrainBackend, 1, Int> = Tensor::from_data(
            TensorData::new(
                vec![target_len as i32; batch_size],
                Shape::new([batch_size]),
            ),
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
            .expect("loss extraction")
            .into_iter()
            .next()
            .expect("single loss value");

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
    let test_input: Tensor<InferBackend, 3> = Tensor::random(
        [1, time_steps, config.input_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let output = infer_model.forward(test_input);
    let phoneme_shape = output.phoneme_logits.dims();
    let boundary_shape = output.boundary_logits.dims();
    let ctc_shape = output.ctc_log_probs.dims();

    println!("  Phoneme logits:  {:?}", phoneme_shape);
    println!("  Boundary logits: {:?}", boundary_shape);
    println!("  CTC log-probs:   {:?}", ctc_shape);

    println!("\nDone. The model trained on synthetic data — loss should decrease");
    println!("over epochs. For real training, point to LibriSpeech data:");
    println!("  See src/train.rs for the full data pipeline.");
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
