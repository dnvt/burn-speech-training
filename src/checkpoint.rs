//! Saving and loading module weights.
//!
//! Checkpoints are burnpack files (`.bpk`, appended when the path has no
//! extension), written with Burn's `ModuleRecord`. Model hyperparameters are
//! stored separately as JSON (`model-config.json`) next to the weights, so a
//! checkpoint can be loaded without knowing how it was configured.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::store::ModuleRecord;

use crate::error::{self, Error, Result};
use crate::mfcc::FeatureMode;
use crate::model::SpeechAlignerConfig;

const MODEL_CONFIG_FILE: &str = "model-config.json";

/// The file a checkpoint saved at `path` is written to: `path` itself, or
/// `path.bpk` when it has no extension.
pub fn file_path(path: &Path) -> PathBuf {
    if path.extension().is_some() {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    }
}

/// Serialize a module's weights to `path` and return the file written (see
/// [`file_path`]).
pub fn save<M: Module>(module: &M, path: &Path) -> Result<PathBuf> {
    let file = file_path(path);
    module
        .clone()
        .save_file(&file)
        .map_err(|err| Error::checkpoint(&file, err))?;
    Ok(file)
}

/// Load weights from `path` into an already-initialised module of the same
/// shape. A checkpoint saved from a different architecture is an error.
pub fn load<M: Module>(module: M, path: &Path) -> Result<M> {
    let record = ModuleRecord::load(path).map_err(|err| {
        Error::checkpoint(
            path,
            format!("{err} (checkpoints written before version 0.3 cannot be read)"),
        )
    })?;
    module.try_load_record(record).map_err(|err| {
        Error::checkpoint(
            path,
            format!(
                "{err}; the weights do not match the model architecture (check model-config.json)"
            ),
        )
    })
}

fn model_config_path(checkpoint: &Path) -> PathBuf {
    checkpoint
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(MODEL_CONFIG_FILE)
}

/// Write the model config into `dir` so checkpoints saved there can be
/// reloaded with the right shape.
pub fn save_model_config(dir: &Path, config: &SpeechAlignerConfig) -> Result<()> {
    error::write(
        &dir.join(MODEL_CONFIG_FILE),
        serde_json::to_string_pretty(config)?,
    )
}

/// Error unless the model takes `feature_mode`'s features as input.
pub fn check_feature_mode(config: &SpeechAlignerConfig, feature_mode: FeatureMode) -> Result<()> {
    if config.input_dim == feature_mode.feature_dim() {
        return Ok(());
    }
    let hint = [FeatureMode::Mfcc39, FeatureMode::LogMel80]
        .into_iter()
        .find(|mode| mode.feature_dim() == config.input_dim)
        .map(|mode| format!("; it was probably trained with {}", mode.label()))
        .unwrap_or_default();
    Err(Error::config(format!(
        "the model expects {}-dim input but {} features are {}-dim{hint}",
        config.input_dim,
        feature_mode.label(),
        feature_mode.feature_dim()
    )))
}

/// Model config for a checkpoint: read from `model-config.json` next to it,
/// or the default config for `feature_mode` if that file does not exist
/// (checkpoints written before configs were saved).
pub fn model_config_for(
    checkpoint: &Path,
    feature_mode: FeatureMode,
) -> Result<SpeechAlignerConfig> {
    let path = model_config_path(checkpoint);
    match std::fs::read_to_string(&path) {
        Ok(raw) => Ok(serde_json::from_str(&raw)?),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(SpeechAlignerConfig::for_feature_mode(feature_mode))
        }
        Err(err) => Err(Error::io(path, err)),
    }
}

#[cfg(test)]
mod tests {
    use burn::nn::LinearConfig;
    use burn::tensor::{Device, Tensor};

    use super::*;

    #[test]
    fn feature_mode_must_match_input_dim() {
        let logmel = SpeechAlignerConfig::for_feature_mode(FeatureMode::LogMel80);
        assert!(check_feature_mode(&logmel, FeatureMode::LogMel80).is_ok());
        let err = check_feature_mode(&logmel, FeatureMode::Mfcc39).expect_err("mismatch");
        assert!(err.to_string().contains("logmel80"), "{err}");
    }

    #[test]
    fn mismatched_architecture_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let device = Device::default();
        let path = dir.path().join("lin");
        save(&LinearConfig::new(3, 2).init(&device), &path).expect("save");
        let err = load(LinearConfig::new(4, 2).init(&device), &path).expect_err("shape mismatch");
        assert!(matches!(err, Error::Checkpoint { .. }), "{err}");
    }

    #[test]
    fn missing_checkpoint_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let module = LinearConfig::new(3, 2).init(&Device::default());
        let err = load(module, &dir.path().join("absent")).expect_err("missing");
        assert!(matches!(err, Error::Checkpoint { .. }), "{err}");
    }

    #[test]
    fn roundtrip_of_never_used_module() {
        let dir = tempfile::tempdir().expect("tempdir");
        let device = Device::default();
        let a = LinearConfig::new(3, 2).init(&device);
        let path = dir.path().join("lin");
        save(&a, &path).expect("save");
        let b = load(LinearConfig::new(3, 2).init(&device), &path).expect("load");
        let x = Tensor::<2>::ones([1, 3], &device);
        let ya: Vec<f32> = a.forward(x.clone()).into_data().try_to_vec().unwrap();
        let yb: Vec<f32> = b.forward(x).into_data().try_to_vec().unwrap();
        assert_eq!(ya, yb);
    }
}
