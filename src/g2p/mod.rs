//! Grapheme-to-Phoneme (G2P) conversion
//!
//! Provides pronunciation dictionary lookup for text-guided phoneme alignment.
//! Currently supports English via `CMUdict` (~126,000 unique words).
//!
//! ## Reference-Guided Phoneme Alignment
//!
//! This module enables correcting word boundaries by comparing acoustic
//! phonemes against expected phoneme sequences from reference text.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use burn_speech_training::g2p::{CmuDict, G2pLookup};
//!
//! let dict = CmuDict::load().expect("load CMU dict");
//! if let Some(phonemes) = dict.lookup("hello") {
//!     // IPA phonemes: h, ʌ, l, oʊ
//!     assert_eq!(phonemes[0].ipa_symbol(), "h");
//!     assert_eq!(phonemes[1].ipa_symbol(), "ʌ");
//! }
//! ```

mod arpabet;
mod cmudict;
pub mod types;

use std::sync::Arc;

pub use arpabet::{arpabet_to_ipa, parse_arpabet_stress};
pub use cmudict::{CmuDict, CmuDictError};
pub use types::Phoneme;

/// G2P lookup trait for language-specific implementations.
///
/// Enables swapping dictionary implementations while maintaining a consistent
/// interface for text-guided alignment.
pub trait G2pLookup: Send + Sync {
    /// Returns the language code (ISO 639-1).
    fn language_code(&self) -> &'static str;

    /// Looks up pronunciation for a word.
    ///
    /// Returns `None` if word not found in dictionary.
    /// Word lookup is case-insensitive.
    fn lookup(&self, word: &str) -> Option<Vec<Phoneme>>;

    /// Returns the number of entries in the dictionary.
    fn dictionary_size(&self) -> usize;
}

impl<T: G2pLookup + ?Sized> G2pLookup for Arc<T> {
    fn language_code(&self) -> &'static str {
        (**self).language_code()
    }

    fn lookup(&self, word: &str) -> Option<Vec<Phoneme>> {
        (**self).lookup(word)
    }

    fn dictionary_size(&self) -> usize {
        (**self).dictionary_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmudict_lookup_this() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");
        let phonemes = dict.lookup("this");

        assert!(phonemes.is_some(), "Expected 'this' to be in dictionary");
        let phonemes = phonemes.unwrap();

        // "this" -> DH IH S (ARPAbet) -> ð ɪ s (IPA)
        assert_eq!(phonemes.len(), 3);
        let ipa: Vec<_> = phonemes.iter().map(|p| p.ipa_symbol()).collect();
        assert_eq!(ipa, vec!["ð", "ɪ", "s"]);
    }

    #[test]
    fn test_cmudict_lookup_hello() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");
        let phonemes = dict.lookup("hello");

        assert!(phonemes.is_some(), "Expected 'hello' to be in dictionary");
        let phonemes = phonemes.unwrap();

        // "hello" -> HH AH L OW (ARPAbet) -> h ʌ l oʊ (IPA)
        assert_eq!(phonemes.len(), 4);
        let ipa: Vec<_> = phonemes.iter().map(|p| p.ipa_symbol()).collect();
        assert_eq!(ipa, vec!["h", "ʌ", "l", "oʊ"]);
    }

    #[test]
    fn test_cmudict_lookup_unknown_word() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");
        let phonemes = dict.lookup("xyzzyplugh");

        assert!(phonemes.is_none(), "Expected unknown word to return None");
    }

    #[test]
    fn test_cmudict_lookup_running() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");
        let phonemes = dict.lookup("running");

        assert!(phonemes.is_some(), "Expected 'running' to be in dictionary");
        let phonemes = phonemes.unwrap();

        // "running" -> R AH N IH NG (ARPAbet) -> r ʌ n ɪ ŋ (IPA)
        assert_eq!(phonemes.len(), 5);
        let ipa: Vec<_> = phonemes.iter().map(|p| p.ipa_symbol()).collect();
        assert_eq!(ipa, vec!["r", "ʌ", "n", "ɪ", "ŋ"]);
    }

    #[test]
    fn test_cmudict_case_insensitive() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");

        let lower = dict.lookup("hello");
        let upper = dict.lookup("HELLO");
        let mixed = dict.lookup("HeLLo");

        assert_eq!(lower, upper);
        assert_eq!(lower, mixed);
    }

    #[test]
    fn test_dictionary_size() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");
        let size = dict.dictionary_size();

        // CMUdict has 135K lines but ~126K unique words after deduplicating variants
        assert!(
            size >= 120_000,
            "Expected at least 120,000 entries, got {size}"
        );
    }

    #[test]
    fn test_language_code() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");
        assert_eq!(dict.language_code(), "en");
    }
}
