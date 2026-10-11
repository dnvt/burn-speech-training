//! Data loading: audio decoding, LibriSpeech, SpeechOcean762, batching and
//! the precomputed feature cache.

pub mod audio;
pub mod batch;
pub mod cache;
pub mod librispeech;
pub mod so762;

pub use batch::{Batch, Batching, TrainingSample};

use crate::error::Result;

/// CTC targets for a transcript, plus the number of words that needed the
/// out-of-vocabulary fallback.
pub(crate) fn transcript_targets(text: &str) -> Result<(Vec<i32>, usize)> {
    let (targets, stats) = crate::phoneme_map::transcript_to_targets(text)?;
    Ok((targets, stats.g2p_hits))
}

/// Minimum number of frames CTC needs to emit `targets`: one per label plus
/// one blank between each pair of identical neighbours.
pub(crate) fn ctc_min_frames(targets: &[i32]) -> usize {
    let repeats = targets.windows(2).filter(|pair| pair[0] == pair[1]).count();
    targets.len() + repeats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_min_frames_counts_repeats() {
        assert_eq!(ctc_min_frames(&[]), 0);
        assert_eq!(ctc_min_frames(&[1, 2, 3]), 3);
        assert_eq!(ctc_min_frames(&[1, 1, 2, 2, 2]), 8);
    }
}
