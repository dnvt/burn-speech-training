//! Saving and loading module weights.
//!
//! Checkpoints are Burn records in full precision, written with
//! `BinBytesRecorder`. Model hyperparameters are stored separately as JSON
//! (`model-config.json`) next to the weights, so a checkpoint can be loaded
//! without knowing how it was configured.

use std::path::{Path, PathBuf};

use burn::module::{Module, ModuleVisitor, Param};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::error::{self, Error, Result};
use crate::mfcc::FeatureMode;
use crate::model::SpeechAlignerConfig;

const MODEL_CONFIG_FILE: &str = "model-config.json";

/// Forces lazily-initialised parameters to materialise.
struct Materialize;

impl<B: Backend> ModuleVisitor<B> for Materialize {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let _ = param.val();
    }
}

/// Serialize a module's weights to `path`.
pub fn save<B: Backend, M: Module<B>>(module: &M, path: &Path) -> Result<()> {
    // Burn initialises parameters on first use. Cloning a module that was
    // never used would give the clone its own, different random weights, so
    // materialise them before cloning.
    module.visit(&mut Materialize);
    let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
        .record(module.clone().into_record(), ())
        .map_err(|err| Error::checkpoint(path, err))?;
    error::write(path, bytes)
}

/// Load weights from `path` into an already-initialised module of the same
/// shape. A checkpoint saved from a different architecture is an error.
pub fn load<B: Backend, M: Module<B>>(module: M, path: &Path, device: &B::Device) -> Result<M> {
    let bytes = std::fs::read(path).map_err(|err| Error::io(path, err))?;
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(bytes, device)
        .map_err(|err| Error::checkpoint(path, err))?;
    // Burn does not check shapes when loading a record; a mismatch would only
    // surface as a panic in the first forward pass.
    let expected = param_shapes(&module);
    let loaded = module.load_record(record);
    if param_shapes(&loaded) != expected {
        return Err(Error::checkpoint(
            path,
            "weights do not match the model architecture (check model-config.json)",
        ));
    }
    Ok(loaded)
}

/// Shapes of all float parameters, in visiting order.
fn param_shapes<B: Backend, M: Module<B>>(module: &M) -> Vec<Vec<usize>> {
    struct Shapes(Vec<Vec<usize>>);
    impl<B: Backend> ModuleVisitor<B> for Shapes {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.0.push(param.val().dims().to_vec());
        }
    }
    let mut shapes = Shapes(Vec::new());
    module.visit(&mut shapes);
    shapes.0
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
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;

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
        let device = Default::default();
        let path = dir.path().join("lin");
        save(&LinearConfig::new(3, 2).init::<NdArray>(&device), &path).expect("save");
        let err = load(
            LinearConfig::new(4, 2).init::<NdArray>(&device),
            &path,
            &device,
        )
        .expect_err("shape mismatch");
        assert!(matches!(err, Error::Checkpoint { .. }), "{err}");
    }

    #[test]
    fn roundtrip_of_never_used_module() {
        let dir = tempfile::tempdir().expect("tempdir");
        let device = Default::default();
        let a = LinearConfig::new(3, 2).init::<NdArray>(&device);
        let path = dir.path().join("lin");
        save(&a, &path).expect("save");
        let b = load(
            LinearConfig::new(3, 2).init::<NdArray>(&device),
            &path,
            &device,
        )
        .expect("load");
        let x = Tensor::<NdArray, 2>::ones([1, 3], &device);
        let ya: Vec<f32> = a.forward(x.clone()).into_data().to_vec().unwrap();
        let yb: Vec<f32> = b.forward(x).into_data().to_vec().unwrap();
        assert_eq!(ya, yb);
    }
}
