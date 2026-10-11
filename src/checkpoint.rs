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
/// shape.
pub fn load<B: Backend, M: Module<B>>(module: M, path: &Path, device: &B::Device) -> Result<M> {
    let bytes = std::fs::read(path).map_err(|err| Error::io(path, err))?;
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(bytes, device)
        .map_err(|err| Error::checkpoint(path, err))?;
    Ok(module.load_record(record))
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
