//! Compile-time backend selection.
//!
//! Exactly one backend is used per build. When several backend features are
//! enabled the priority is `cuda` > `wgpu` > `ndarray`.

use burn::backend::Autodiff;

#[cfg(not(any(feature = "ndarray", feature = "wgpu", feature = "cuda")))]
compile_error!("enable one backend feature: `ndarray` (CPU), `wgpu` or `cuda`");

#[cfg(feature = "cuda")]
pub type InferBackend = burn::backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type InferBackend = burn::backend::Wgpu;
#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
pub type InferBackend = burn::backend::NdArray;

/// Backend used for training (inference backend + autodiff).
pub type TrainBackend = Autodiff<InferBackend>;

pub type Device = burn::tensor::Device<InferBackend>;

#[cfg(feature = "cuda")]
pub const BACKEND_NAME: &str = "cuda";
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub const BACKEND_NAME: &str = "wgpu";
#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
pub const BACKEND_NAME: &str = "ndarray (CPU)";

pub fn default_device() -> Device {
    Device::default()
}
