//! English pronunciation lookup backed by the CMU Pronouncing Dictionary.
//!
//! The plain-text dictionary (`src/g2p/data/cmudict.dict`) is embedded in the
//! binary and parsed into a map the first time it is used. Lines look like:
//!
//! ```text
//! father F AA1 DH ER0
//! the DH AH0
//! the(2) DH AH1
//! aalborg AO1 L B AO0 R G # place, danish
//! ```
//!
//! Only the first pronunciation of each word is kept (alternates such as
//! `the(2)` are skipped), stress digits are dropped and trailing `#` comments
//! are ignored. Pronunciations are stored as CTC class indices; see
//! [`crate::phoneme_map`] for the inventory.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::phoneme_map::phoneme_class;

const CMUDICT: &str = include_str!("g2p/data/cmudict.dict");

static DICTIONARY: LazyLock<HashMap<&'static str, Box<[u8]>>> = LazyLock::new(|| parse(CMUDICT));

/// Returns the CTC class indices of `word`'s pronunciation, or `None` if the
/// word is not in the dictionary. The lookup is case-insensitive.
#[must_use]
pub fn lookup(word: &str) -> Option<&'static [u8]> {
    DICTIONARY
        .get(word.to_ascii_lowercase().as_str())
        .map(Box::as_ref)
}

/// Number of words in the dictionary.
#[must_use]
pub fn len() -> usize {
    DICTIONARY.len()
}

fn parse(text: &'static str) -> HashMap<&'static str, Box<[u8]>> {
    let mut entries = HashMap::with_capacity(130_000);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(word) = fields.next() else {
            continue;
        };
        if word.contains('(') {
            continue;
        }
        let phonemes: Box<[u8]> = fields
            .take_while(|field| !field.starts_with('#'))
            .map(|symbol| {
                phoneme_class(symbol).unwrap_or_else(|| {
                    panic!("CMUdict entry {word:?} uses unknown phoneme {symbol:?}")
                })
            })
            .collect();
        if !phonemes.is_empty() {
            entries.entry(word).or_insert(phonemes);
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classes(symbols: &str) -> Vec<u8> {
        symbols
            .split_whitespace()
            .map(|symbol| phoneme_class(symbol).expect("known phoneme"))
            .collect()
    }

    #[test]
    fn dictionary_has_all_words() {
        assert!(len() > 120_000, "only {} entries", len());
    }

    #[test]
    fn keys_are_lowercase() {
        assert!(DICTIONARY
            .keys()
            .all(|word| !word.bytes().any(|b| b.is_ascii_uppercase())));
    }

    #[test]
    fn known_words() {
        assert_eq!(lookup("father"), Some(classes("F AA DH ER").as_slice()));
        assert_eq!(lookup("cat"), Some(classes("K AE T").as_slice()));
        assert_eq!(lookup("hello"), Some(classes("HH AH L OW").as_slice()));
    }

    #[test]
    fn keeps_first_pronunciation_only() {
        assert_eq!(lookup("the"), Some(classes("DH AH").as_slice()));
        assert_eq!(lookup("the(2)"), None);
    }

    #[test]
    fn ignores_trailing_comments() {
        assert_eq!(lookup("aalborg"), Some(classes("AO L B AO R G").as_slice()));
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(lookup("HELLO"), lookup("hello"));
        assert_eq!(lookup("HeLLo"), lookup("hello"));
    }

    #[test]
    fn unknown_word() {
        assert_eq!(lookup("xyzzyplugh"), None);
        assert_eq!(lookup(""), None);
    }
}
