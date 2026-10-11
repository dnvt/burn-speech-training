//! wav2vec2 "teacher" features.
//!
//! The distillation and teacher-input experiments read wav2vec2 encoder
//! hidden states from a cache produced outside this crate (any script that
//! writes the format below will do).
//!
//! # Cache format
//!
//! `manifest.json`:
//!
//! ```json
//! { "files": { "<audio_file as in the SO762 JSON>": { "bin": "000001.bin", "frames": 87 } } }
//! ```
//!
//! Each `.bin` file, little-endian: magic `W2V2`, version `u32` = 1,
//! `frames` `u32`, `dim` `u32` (must be 1024), then `frames × dim` `f32`
//! values. Features are at 50 frames/s (20 ms hop), half the MFCC rate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::nn::Linear;
use burn::tensor::{Device, Shape, Tensor, TensorData};
use serde::Deserialize;

use crate::data::so762;
use crate::data::TrainingSample;
use crate::error::{self, Error, Result};
use crate::finetune::head::{InputSource, ScoringHead};
use crate::finetune::{correlate, data::WordTarget, ScoredUtterance};
use crate::stats::Correlation;

pub const TEACHER_DIM: usize = 1024;
const MAGIC: &[u8; 4] = b"W2V2";

#[derive(Debug, Clone, Deserialize)]
struct Entry {
    bin: String,
    frames: usize,
}

#[derive(Debug, Clone)]
pub struct TeacherCache {
    root: PathBuf,
    files: HashMap<String, Entry>,
}

/// Row-major `[frames, TEACHER_DIM]` features.
#[derive(Debug, Clone)]
pub struct TeacherFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

impl TeacherFeatures {
    pub fn tensor(&self, device: &Device) -> Tensor<2> {
        Tensor::from_data(
            TensorData::new(self.values.clone(), Shape::new([self.frames, TEACHER_DIM])),
            device,
        )
    }
}

impl TeacherCache {
    pub fn open(dir: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct Manifest {
            files: HashMap<String, Entry>,
        }
        let manifest: Manifest =
            serde_json::from_str(&error::read_to_string(&dir.join("manifest.json"))?)?;
        Ok(Self {
            root: dir.to_path_buf(),
            files: manifest.files,
        })
    }

    /// Features for `audio_file`, or `None` if the cache has no entry for it.
    pub fn get(&self, audio_file: &str) -> Result<Option<TeacherFeatures>> {
        let Some(entry) = self.files.get(audio_file) else {
            return Ok(None);
        };
        let path = self.root.join(&entry.bin);
        let data = std::fs::read(&path).map_err(|err| Error::io(&path, err))?;
        let corrupt = |why: String| Error::data(format!("{}: {why}", path.display()));
        let header = |i: usize| -> Result<u32> {
            data.get(i * 4..i * 4 + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or_else(|| corrupt("truncated header".into()))
        };

        if data.get(..4) != Some(MAGIC.as_slice()) {
            return Err(corrupt("bad magic".into()));
        }
        let (version, frames, dim) = (header(1)?, header(2)? as usize, header(3)? as usize);
        if version != 1 || dim != TEACHER_DIM || frames != entry.frames {
            return Err(corrupt(format!(
                "expected version 1, dim {TEACHER_DIM}, {} frames; found version {version}, dim \
                 {dim}, {frames} frames",
                entry.frames
            )));
        }
        let body = &data[16..];
        if body.len() != frames * dim * 4 {
            return Err(corrupt(format!(
                "expected {} bytes of features",
                frames * dim * 4
            )));
        }
        Ok(Some(TeacherFeatures {
            values: body
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            frames,
        }))
    }

    /// Features for each utterance, in order; `None` where the cache has no
    /// entry.
    pub fn for_utterances(
        &self,
        utterances: &[ScoredUtterance],
    ) -> Result<Vec<Option<TeacherFeatures>>> {
        utterances.iter().map(|u| self.get(&u.audio_file)).collect()
    }
}

/// Feature-matching loss between the student's frame features
/// `[frames, dim]` and the projected teacher features.
///
/// The student runs at 100 frames/s and the teacher at 50, so every second
/// student frame is compared. The slicing stays on the autodiff graph, so
/// the backbone receives the gradient. (The runs in the experiment log
/// copied student features to the host first, which detached them; only the
/// projection learned.)
pub fn distillation_loss(
    student: Tensor<2>,
    teacher: &TeacherFeatures,
    projection: &Linear,
) -> Option<Tensor<1>> {
    let [student_frames, dim] = student.dims();
    let common = (student_frames / 2).min(teacher.frames);
    if common == 0 {
        return None;
    }
    let device = student.device();
    let student = student
        .slice([0..common * 2, 0..dim])
        .reshape([common, 2, dim])
        .slice([0..common, 0..1, 0..dim])
        .reshape([common, dim]);
    let teacher = Tensor::<2>::from_data(
        TensorData::new(
            teacher.values[..common * TEACHER_DIM].to_vec(),
            Shape::new([common, TEACHER_DIM]),
        ),
        &device,
    );
    let error = student - projection.forward(teacher);
    Some((error.clone() * error).mean())
}

/// Spearman ρ of a teacher-input scoring head on a word-score file.
pub fn evaluate_head(
    head_path: &Path,
    cache_dir: &Path,
    dataset: &Path,
    max_samples: Option<usize>,
) -> Result<Correlation> {
    let device = Device::default();
    let (head, config) = ScoringHead::load(head_path, &device)?;
    if config.input_source != InputSource::TeacherCache {
        return Err(Error::config(
            "this head reads backbone features; use `evaluate::evaluate`",
        ));
    }
    let cache = TeacherCache::open(cache_dir)?;

    // Only the words and audio file names are needed; no audio is decoded.
    let words = so762::load(dataset, max_samples)?;
    let utterances: Vec<ScoredUtterance> = so762::group_by_audio(&words)
        .into_iter()
        .map(|(audio_file, words)| ScoredUtterance {
            audio_file: audio_file.to_owned(),
            sample: TrainingSample::from_frames(&[], TEACHER_DIM, Vec::new()),
            words: words
                .iter()
                .map(|w| WordTarget {
                    position: w.word_position,
                    num_words: w.num_words(),
                    score: w.human_score as f32,
                    raw_score: w.raw_score(),
                    weight: 1.0,
                })
                .collect(),
        })
        .collect();
    let features = cache.for_utterances(&utterances)?;
    let result = correlate(&head, &config, &utterances, |i| {
        features[i].as_ref().map(|f| f.tensor(&device))
    });
    if result.count < 10 {
        return Err(Error::data(format!(
            "only {} words matched the teacher cache",
            result.count
        )));
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Write a cache with one entry per `(audio_file, frames)`.
    pub(crate) fn write_cache(dir: &Path, entries: &[(&str, usize)]) {
        let mut files = serde_json::Map::new();
        for (i, (audio, frames)) in entries.iter().enumerate() {
            let bin = format!("{i:06}.bin");
            let mut bytes = Vec::new();
            bytes.extend_from_slice(MAGIC);
            for v in [1u32, *frames as u32, TEACHER_DIM as u32] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            for v in 0..frames * TEACHER_DIM {
                bytes.extend_from_slice(&((v % 97) as f32 * 0.01).to_le_bytes());
            }
            std::fs::write(dir.join(&bin), bytes).expect("write bin");
            files.insert(
                (*audio).into(),
                serde_json::json!({ "bin": bin, "frames": frames }),
            );
        }
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::json!({ "files": files }).to_string(),
        )
        .expect("write manifest");
    }

    #[test]
    fn reads_cached_features() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_cache(dir.path(), &[("a.wav", 3)]);
        let cache = TeacherCache::open(dir.path()).expect("open");

        let features = cache.get("a.wav").expect("read").expect("entry");
        assert_eq!(features.frames, 3);
        assert_eq!(features.values.len(), 3 * TEACHER_DIM);
        assert!(cache.get("missing.wav").expect("read").is_none());
    }

    #[test]
    fn distillation_gradient_reaches_the_student() {
        use burn::nn::LinearConfig;

        let device = Device::default().autodiff();
        let student = Tensor::<2>::ones([10, 4], &device).require_grad();
        let teacher = TeacherFeatures {
            values: vec![0.5; 5 * TEACHER_DIM],
            frames: 5,
        };
        let projection = LinearConfig::new(TEACHER_DIM, 4).init(&device);

        let loss = distillation_loss(student.clone(), &teacher, &projection).expect("loss");
        let grads = loss.backward();
        let grad = student.grad(&grads).expect("student gradient");
        let grad: Vec<f32> = grad.into_data().try_to_vec().expect("vec");
        // Even frames are compared, odd ones are not.
        assert!(grad[..4].iter().any(|g| g.abs() > 0.0));
        assert!(grad[4..8].iter().all(|g| *g == 0.0));
    }
}
