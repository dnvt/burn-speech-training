//! `CMUdict` pronunciation dictionary
//!
//! Provides English G2P lookup using the Carnegie Mellon Pronouncing
//! Dictionary. The dictionary is embedded at compile time (~3MB).
//!
//! ## Format
//!
//! `CMUdict` uses `ARPAbet` notation with space-separated phonemes:
//! ```text
//! hello HH AH0 L OW1
//! world W ER1 L D
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::Phoneme;
use super::G2pLookup;

/// `CMUdict` pronunciation dictionary for English.
///
/// Provides O(1) lookup for ~126,000 unique English words.
/// Dictionary is pre-computed at build time and deserialized on load.
#[derive(Debug, Serialize, Deserialize)]
pub struct CmuDict {
    /// Word -> Phoneme sequence mapping (lowercase keys)
    entries: HashMap<String, Vec<Phoneme>>,
}

impl CmuDict {
    /// Load and parse the embedded `CMUdict`.
    ///
    /// # Returns
    ///
    /// `Ok(CmuDict)` with parsed dictionary, or `Err` if parsing fails.
    ///
    /// # Performance
    ///
    /// First load parses ~126K unique entries (~50-100ms).
    /// Consider caching the instance for repeated lookups.
    ///
    /// # Errors
    ///
    /// Returns error if dictionary data is corrupted (should not happen
    /// with embedded data).
    /// Load and deserialize the pre-computed `CMUdict`.
    ///
    /// # Returns
    ///
    /// `Ok(CmuDict)` with parsed dictionary, or `Err` if deserialization fails.
    ///
    /// # Performance
    ///
    /// Fast binary deserialization (~10ms vs ~100ms parsing).
    pub fn load() -> Result<Self, CmuDictError> {
        let data = include_bytes!(concat!(env!("OUT_DIR"), "/cmudict.bin"));
        let entries: HashMap<String, Vec<Phoneme>> =
            bitcode::deserialize(data).map_err(|e| CmuDictError::ParseError(e.to_string()))?;

        Ok(Self { entries })
    }

    /// Get the phoneme sequence for a word.
    ///
    /// Alias for `lookup()` with more intuitive name.
    #[must_use]
    pub fn get(&self, word: &str) -> Option<&[Phoneme]> {
        self.entries.get(&word.to_lowercase()).map(Vec::as_slice)
    }
}

impl G2pLookup for CmuDict {
    fn language_code(&self) -> &'static str {
        "en"
    }

    fn lookup(&self, word: &str) -> Option<Vec<Phoneme>> {
        self.entries.get(&word.to_lowercase()).cloned()
    }

    fn dictionary_size(&self) -> usize {
        self.entries.len()
    }
}

/// Errors from `CMUdict` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CmuDictError {
    /// Dictionary parsing failed
    ParseError(String),
}

impl std::fmt::Display for CmuDictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "CMUdict parse error: {msg}"),
        }
    }
}

impl std::error::Error for CmuDictError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_succeeds() {
        let dict = CmuDict::load();
        assert!(dict.is_ok(), "Failed to load CMUdict: {:?}", dict.err());
    }

    #[test]
    fn test_common_words() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");

        // Test a variety of common words
        let test_words = [
            "the", "a", "is", "are", "hello", "world", "computer", "python",
        ];

        for word in test_words {
            assert!(
                dict.lookup(word).is_some(),
                "Expected '{word}' to be in dictionary"
            );
        }
    }

    #[test]
    fn test_get_alias() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");

        let lookup_result = dict.lookup("hello");
        let get_result = dict.get("hello");

        assert!(lookup_result.is_some());
        assert!(get_result.is_some());
        assert_eq!(lookup_result.as_deref(), get_result);
    }

    #[test]
    fn test_phoneme_format() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");

        // "cat" should be K AE1 T
        let phonemes = dict.lookup("cat").expect("Expected 'cat' in dictionary");
        let ipa_symbols: Vec<String> = phonemes
            .iter()
            .map(|p| p.ipa_symbol().into_owned())
            .collect();

        assert_eq!(ipa_symbols, vec!["k", "æ", "t"]);
    }

    #[test]
    fn test_special_characters_in_words() {
        let dict = CmuDict::load().expect("Failed to load CMUdict");

        // CMUdict has contractions and possessives
        // Note: These may or may not be in the dictionary depending on version
        let contractions = ["don't", "can't", "it's"];

        // Just verify we don't panic on these lookups
        for word in contractions {
            let _ = dict.lookup(word);
        }
    }
}
