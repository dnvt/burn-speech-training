//! Train a tiny SpeechAligner on synthetic data with the library's training
//! loop.
//!
//! No data files needed. This checks that batching, the model, CTC loss and
//! checkpointing work end to end. It says nothing about speech accuracy.
//!
//! ```bash
//! cargo run --release --example train_small
//! ```

use burn::module::Module;
use burn::tensor::{Shape, Tensor, TensorData};

use burn_speech_training::backend::{default_device, InferBackend};
use burn_speech_training::checkpoint;
use burn_speech_training::data::{Batching, TrainingSample};
use burn_speech_training::error::Result;
use burn_speech_training::model::SpeechAlignerConfig;
use burn_speech_training::train::{train, TrainConfig};

const FEATURE_DIM: usize = 13;
const NUM_CLASSES: usize = 42;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    // A narrow version of the default model (~122K parameters instead of
    // ~1.7M) so the example runs in seconds on a CPU.
    let model = SpeechAlignerConfig {
        input_dim: FEATURE_DIM,
        num_classes: NUM_CLASSES,
        n_heads: 2,
        se_reduction: 4,
        channels: [16, 32, 64, 128],
        kernel_size: 3,
    };

    let samples: Vec<TrainingSample> = (0..8).map(|i| synthetic_sample(30 + 4 * i, i)).collect();
    let checkpoint_dir = std::env::temp_dir().join("burn-speech-training-example");
    let config = TrainConfig {
        model: model.clone(),
        epochs: 5,
        learning_rate: 1e-3,
        batching: Batching::Fixed { batch_size: 4 },
        checkpoint_dir,
        checkpoint_interval: 0,
        ..TrainConfig::default()
    };

    let summary = train(samples, &config)?;
    println!("\nparameters: {}", summary.num_params);
    for (epoch, loss) in summary.epoch_losses.iter().enumerate() {
        println!("epoch {}  ctc_loss={loss:.4}", epoch + 1);
    }

    // Reload the final checkpoint and run it on a new utterance.
    let device = default_device();
    let trained = checkpoint::load(
        model.init::<InferBackend>(&device)?,
        &summary.final_checkpoint,
        &device,
    )?;
    let sample = synthetic_sample(40, 99);
    let input = Tensor::<InferBackend, 3>::from_data(
        TensorData::new(sample.features, Shape::new([1, 40, FEATURE_DIM])),
        &device,
    );
    let output = trained.forward(input);
    println!("\ncheckpoint: {}", summary.final_checkpoint.display());
    println!(
        "ctc_log_probs  {:?}  [frames, batch, classes]",
        output.ctc_log_probs.dims()
    );
    println!("frame_features {:?}", output.frame_features.dims());
    println!("{} parameters reloaded", trained.num_params());
    Ok(())
}

/// Smooth MFCC-like features whose level steps up once per target phoneme,
/// so there is something for CTC to align.
fn synthetic_sample(frames: usize, seed: usize) -> TrainingSample {
    const TARGETS: usize = 6;
    let targets: Vec<i32> = (0..TARGETS)
        .map(|i| ((seed + i) % (NUM_CLASSES - 2) + 1) as i32)
        .collect();
    let mut features = Vec::with_capacity(frames * FEATURE_DIM);
    for frame in 0..frames {
        let level = targets[frame * TARGETS / frames] as f32 * 0.05;
        for d in 0..FEATURE_DIM {
            features.push(level + d as f32 * 0.005 + ((frame + d) % 7) as f32 * 0.01);
        }
    }
    TrainingSample {
        features,
        feature_dim: FEATURE_DIM,
        targets,
        oov_words: 0,
    }
}
