//! Error types for burn-speech-training.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Config(String),

    #[error("{0}")]
    Training(String),

    #[error("{0}")]
    Dataset(String),

    #[error("{0}")]
    Evaluation(String),

    #[error("{0}")]
    Process(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training(msg.into())
    }

    pub fn dataset(msg: impl Into<String>) -> Self {
        Self::Dataset(msg.into())
    }

    pub fn evaluation(msg: impl Into<String>) -> Self {
        Self::Evaluation(msg.into())
    }

    pub fn process(msg: impl Into<String>) -> Self {
        Self::Process(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
