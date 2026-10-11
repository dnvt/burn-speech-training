//! Train a tiny SpeechAligner on one synthetic batch.
//!
//! No data files needed. This checks that the model, CTC loss, backward pass
//! and optimizer are wired together. It says nothing about speech accuracy.
//!
//! ```bash
//! cargo run --example train_small --features ndarray --release
//! ```

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Shape, Tensor, TensorData};

use burn_speech_training::error::Result;
use burn_speech_training::loss::SpeechAlignerLossConfig;
use burn_speech_training::model::SpeechAlignerConfig;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

const BATCH_SIZE: usize = 2;
/// 30 frames is 0.3 s of audio at the 10 ms hop used by the feature extractor.
const FRAMES: usize = 30;
const TARGET_LEN: usize = 6;
const STEPS: usize = 5;
const LEARNING_RATE: f64 = 1e-3;

fn main() -> Result<()> {
    let device = NdArrayDevice::default();

    // A narrow version of the default model (~122K params instead of ~1.7M)
    // so the example runs in seconds on a CPU.
    let config = SpeechAlignerConfig {
        input_dim: 13,
        num_classes: 42,
        n_heads: 2,
        se_reduction: 4,
        channels: [16, 32, 64, 128],
        kernel_size: 3,
    };
    let mut model = config.init::<TrainBackend>(&device)?;
    println!(
        "SpeechAligner: {} parameters, input_dim={}, classes={}",
        model.num_params(),
        config.input_dim,
        config.num_classes
    );

    let features = float_tensor::<TrainBackend, 3>(
        synthetic_features(BATCH_SIZE, FRAMES, config.input_dim),
        [BATCH_SIZE, FRAMES, config.input_dim],
        &device,
    );
    let targets = int_tensor::<TrainBackend, 2>(
        synthetic_targets(BATCH_SIZE, TARGET_LEN, config.num_classes),
        [BATCH_SIZE, TARGET_LEN],
        &device,
    );
    let input_lengths =
        int_tensor::<TrainBackend, 1>(vec![FRAMES as i32; BATCH_SIZE], [BATCH_SIZE], &device);
    let target_lengths =
        int_tensor::<TrainBackend, 1>(vec![TARGET_LEN as i32; BATCH_SIZE], [BATCH_SIZE], &device);

    let loss_fn = SpeechAlignerLossConfig::init();
    let mut optimizer = AdamConfig::new().init::<TrainBackend, _>();

    for step in 1..=STEPS {
        let output = model.forward(features.clone());
        let loss = loss_fn
            .forward(
                output.ctc_log_probs,
                targets.clone(),
                input_lengths.clone(),
                target_lengths.clone(),
            )
            .mean();
        let loss_value: f32 = loss.clone().into_scalar();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);

        println!("step {step}/{STEPS}  ctc_loss={loss_value:.4}");
    }

    let model = model.valid();
    let output = model.forward(float_tensor::<InferBackend, 3>(
        synthetic_features(1, FRAMES, config.input_dim),
        [1, FRAMES, config.input_dim],
        &device,
    ));
    println!("phoneme_logits  {:?}", output.phoneme_logits.dims());
    println!("boundary_logits {:?}", output.boundary_logits.dims());
    println!("ctc_log_probs   {:?}", output.ctc_log_probs.dims());

    Ok(())
}

fn float_tensor<B: burn::tensor::backend::Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(values, Shape::new(shape)), device)
}

fn int_tensor<B: burn::tensor::backend::Backend, const D: usize>(
    values: Vec<i32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D, Int> {
    Tensor::from_data(TensorData::new(values, Shape::new(shape)), device)
}

/// Smooth MFCC-like values whose level steps up once per target phoneme, so
/// there is something for CTC to align against.
fn synthetic_features(batch_size: usize, frames: usize, dim: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(batch_size * frames * dim);
    for sample in 0..batch_size {
        for frame in 0..frames {
            let band = (frame * TARGET_LEN / frames) as f32;
            for d in 0..dim {
                let ripple = ((frame + d + sample) % 7) as f32 * 0.01;
                values.push(band * 0.05 + d as f32 * 0.005 + ripple);
            }
        }
    }
    values
}

/// Non-blank class indices in `1..num_classes`.
fn synthetic_targets(batch_size: usize, target_len: usize, num_classes: usize) -> Vec<i32> {
    (0..batch_size)
        .flat_map(|sample| {
            (0..target_len).map(move |i| ((sample + i + 1) % (num_classes - 1) + 1) as i32)
        })
        .collect()
}
