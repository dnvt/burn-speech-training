//! Audio decoding.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{Error, Result};
use crate::mfcc::SAMPLE_RATE;

/// Decode a WAV or FLAC file into mono `f32` samples.
///
/// The file must already be 16 kHz; nothing is resampled. Multi-channel audio
/// is averaged down to mono.
pub fn load(path: &Path) -> Result<Vec<f32>> {
    let bad = |what: &str, err: &dyn std::fmt::Display| {
        Error::data(format!("{}: {what}: {err}", path.display()))
    };

    let file = std::fs::File::open(path).map_err(|err| Error::io(path, err))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }

    let mut reader = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| bad("unsupported audio", &err))?
        .format;
    let track = reader
        .default_track()
        .ok_or_else(|| Error::data(format!("{}: no audio track", path.display())))?;
    let track_id = track.id;

    if let Some(rate) = track.codec_params.sample_rate {
        if rate as usize != SAMPLE_RATE {
            return Err(Error::data(format!(
                "{}: sample rate is {rate} Hz, expected {SAMPLE_RATE} Hz (resample first)",
                path.display()
            )));
        }
    }

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| bad("cannot create decoder", &err))?;

    let mut samples = Vec::new();
    loop {
        let packet = match reader.next_packet() {
            Ok(packet) => packet,
            Err(DecodeError::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(err) => return Err(bad("read error", &err)),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|err| bad("decode error", &err))?;
        let spec = *decoded.spec();
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);

        let channels = spec.channels.count();
        if channels == 1 {
            samples.extend_from_slice(buffer.samples());
        } else {
            samples.extend(
                buffer
                    .samples()
                    .chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }
    }

    if samples.is_empty() {
        return Err(Error::data(format!("{}: no audio samples", path.display())));
    }
    Ok(samples)
}

/// Write 16-bit mono PCM WAV. Used to build test fixtures.
#[cfg(test)]
pub(crate) fn write_test_wav(path: &Path, samples: &[f32], sample_rate: u32) {
    let data_len = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write wav");
}

/// One second of a 440 Hz tone at 16 kHz.
#[cfg(test)]
pub(crate) fn test_tone(seconds: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE as f32) as usize;
    (0..n)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / SAMPLE_RATE as f32).sin() * 0.5)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_mono_wav() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tone.wav");
        let tone = test_tone(1.0);
        write_test_wav(&path, &tone, 16_000);

        let decoded = load(&path).expect("decode");
        assert_eq!(decoded.len(), tone.len());
        assert!((decoded[100] - tone[100]).abs() < 1e-3);
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tone.wav");
        write_test_wav(&path, &test_tone(0.1), 44_100);

        let err = load(&path).expect_err("44.1 kHz should be rejected");
        assert!(err.to_string().contains("44100"), "{err}");
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = load(Path::new("/nonexistent.wav")).expect_err("missing");
        assert!(matches!(err, Error::Io { .. }));
    }
}
