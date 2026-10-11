//! Error type shared by the whole crate.

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The caller passed inconsistent or out-of-range settings.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Input data (audio, transcripts, dataset JSON, feature cache) is missing
    /// or malformed.
    #[error("{0}")]
    Data(String),

    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A model or scoring-head checkpoint could not be saved or loaded.
    #[error("checkpoint {}: {message}", path.display())]
    Checkpoint { path: PathBuf, message: String },

    /// Training produced no usable result (e.g. every batch was skipped).
    #[error("training failed: {0}")]
    Training(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn data(msg: impl Into<String>) -> Self {
        Self::Data(msg.into())
    }

    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    pub fn checkpoint(path: impl AsRef<Path>, message: impl std::fmt::Display) -> Self {
        Self::Checkpoint {
            path: path.as_ref().to_path_buf(),
            message: message.to_string(),
        }
    }

    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Read a file to a string, attaching the path to any error.
pub(crate) fn read_to_string(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|err| Error::io(path, err))
}

/// Write a file, attaching the path to any error.
pub(crate) fn write(path: &Path, contents: impl AsRef<[u8]>) -> Result<()> {
    std::fs::write(path, contents).map_err(|err| Error::io(path, err))
}

/// Create a directory and its parents, attaching the path to any error.
pub(crate) fn create_dir_all(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).map_err(|err| Error::io(path, err))
}
