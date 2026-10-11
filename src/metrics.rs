//! Training metrics as JSON lines.
//!
//! Each event is written as one JSON object per line (`train` uses
//! `training-metrics.jsonl`, `finetune` uses `finetune-metrics.jsonl` in the
//! checkpoint directory) and mirrored to `tracing` at debug level (target
//! `training_metrics`). Every line carries `event` and `elapsed_secs`.

use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use serde_json::{Map, Value};

use crate::error::{Error, Result};

pub struct MetricsWriter {
    file: File,
    started: Instant,
    elapsed_offset_secs: f64,
}

impl MetricsWriter {
    /// Start a new metrics file, replacing any existing one.
    pub fn create(path: &Path) -> Result<Self> {
        Self::open(path, false, 0.0)
    }

    /// Append to an existing metrics file when resuming a run.
    /// `elapsed_offset_secs` is the wall-clock time of the earlier run.
    pub fn append(path: &Path, elapsed_offset_secs: f64) -> Result<Self> {
        Self::open(path, true, elapsed_offset_secs)
    }

    fn open(path: &Path, append: bool, elapsed_offset_secs: f64) -> Result<Self> {
        let file = File::options()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .map_err(|err| Error::io(path, err))?;
        Ok(Self {
            file,
            started: Instant::now(),
            elapsed_offset_secs,
        })
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed_offset_secs + self.started.elapsed().as_secs_f64()
    }

    /// Record one event. `fields` must be a JSON object.
    pub fn log(&mut self, event: &str, fields: Value) {
        let mut line = match fields {
            Value::Object(map) => map,
            other => {
                let mut map = Map::new();
                map.insert("value".into(), other);
                map
            }
        };
        line.insert("event".into(), event.into());
        line.insert("elapsed_secs".into(), round(self.elapsed_secs(), 1).into());
        let line = Value::Object(line);

        tracing::debug!(target: "training_metrics", "{line}");
        if let Err(err) = writeln!(self.file, "{line}").and_then(|()| self.file.flush()) {
            tracing::warn!("failed to write metrics: {err}");
        }
    }
}

fn round(value: f64, decimals: i32) -> f64 {
    let scale = 10f64.powi(decimals);
    (value * scale).round() / scale
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn writes_one_json_object_per_event() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut metrics = MetricsWriter::create(&dir.path().join("m.jsonl")).expect("create");
        metrics.log("epoch", json!({ "epoch": 1, "loss": 0.5 }));
        metrics.log("epoch", json!({ "epoch": 2, "loss": 0.25 }));

        let raw = std::fs::read_to_string(dir.path().join("m.jsonl")).expect("read");
        let lines: Vec<Value> = raw
            .lines()
            .map(|line| serde_json::from_str(line).expect("json"))
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["event"], "epoch");
        assert_eq!(lines[1]["epoch"], 2);
        assert!(lines[1]["elapsed_secs"].is_number());
    }

    #[test]
    fn append_keeps_existing_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        MetricsWriter::create(&dir.path().join("m.jsonl"))
            .expect("create")
            .log("epoch", json!({ "epoch": 1 }));
        MetricsWriter::append(&dir.path().join("m.jsonl"), 10.0)
            .expect("append")
            .log("epoch", json!({ "epoch": 2 }));

        let raw = std::fs::read_to_string(dir.path().join("m.jsonl")).expect("read");
        assert_eq!(raw.lines().count(), 2);
    }
}
