//! Phonetic types
//!
//! Core types for phonetic analysis and pronunciation assessment.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Phoneme representation.
///
/// The [`Phoneme`] type is intentionally flexible: it provides strongly typed
/// variants for common English phonemes while also supporting arbitrary IPA
/// symbols from external vocabularies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Phoneme {
    /// /ɑ/ as in "father"
    AH,
    /// /æ/ as in "cat"
    AE,
    /// /ə/ as in "about"
    AX,
    /// /i/ as in "see"
    IY,
    /// /u/ as in "boot"
    UW,
    /// Arbitrary phoneme loaded from runtime vocabulary.
    Custom(String),
}

impl Phoneme {
    /// Create a phoneme from an IPA symbol.
    ///
    /// Known IPA symbols map to the strongly typed variants above while all
    /// other inputs fall back to [`Phoneme::Custom`].
    #[must_use]
    pub fn from_ipa(symbol: &str) -> Self {
        match symbol {
            "ɑ" => Self::AH,
            "æ" => Self::AE,
            "ə" => Self::AX,
            "i" | "iː" => Self::IY,
            "u" | "uː" => Self::UW,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Get IPA symbol for this phoneme.
    #[must_use]
    pub fn ipa_symbol(&self) -> Cow<'_, str> {
        match self {
            Self::AH => Cow::Borrowed("ɑ"),
            Self::AE => Cow::Borrowed("æ"),
            Self::AX => Cow::Borrowed("ə"),
            Self::IY => Cow::Borrowed("i"),
            Self::UW => Cow::Borrowed("u"),
            Self::Custom(symbol) => Cow::Owned(symbol.clone()),
        }
    }

    /// Get `ARPAbet` symbol for this phoneme.
    ///
    /// For `Custom` phonemes containing IPA symbols (e.g., from wav2vec2
    /// models), this performs IPA-to-ARPABET conversion to enable proper
    /// phonetic feature classification in DTW alignment.
    #[must_use]
    pub fn arpabet_symbol(&self) -> Cow<'_, str> {
        match self {
            Self::AH => Cow::Borrowed("AH"),
            Self::AE => Cow::Borrowed("AE"),
            Self::AX => Cow::Borrowed("AX"),
            Self::IY => Cow::Borrowed("IY"),
            Self::UW => Cow::Borrowed("UW"),
            Self::Custom(symbol) => Cow::Owned(ipa_to_arpabet(symbol)),
        }
    }
}

/// Convert IPA symbol to ARPABET code.
///
/// This mapping enables wav2vec2 model output (IPA) to work with DTW alignment
/// and phonetic feature classification that expects ARPABET symbols.
///
/// ## Mapping Coverage (Dec 2025 Audit)
///
/// Extended to cover L2 speaker phoneme patterns commonly seen in wav2vec2
/// output:
/// - Nasalized vowels (L2 influence from Romance/Asian languages)
/// - Central vowels and reduced forms
/// - Flaps, taps, and glottal stops
/// - Length markers (stripped for comparison)
fn ipa_to_arpabet(ipa: &str) -> String {
    // First, strip combining diacritics and length markers for cleaner matching
    let stripped = strip_diacritics(ipa);
    let ipa_clean = stripped.as_str();

    // IPA → ARPABET mapping based on CMUdict conventions
    match ipa_clean {
        // Vowels (full IPA symbols)
        "ɑ" | "ɑː" => "AA".to_owned(),
        "æ" => "AE".to_owned(),
        "ʌ" => "AH".to_owned(),
        "ɔ" | "ɔː" => "AO".to_owned(),
        "aʊ" | "au" => "AW".to_owned(),
        "aɪ" | "ai" => "AY".to_owned(),
        "ɛ" => "EH".to_owned(),
        "ɝ" | "ɜr" | "ɚ" | "ɜ" | "ɜː" => "ER".to_owned(), // Rhotic variants
        "eɪ" | "ei" => "EY".to_owned(),
        "ɪ" => "IH".to_owned(),
        "i" | "iː" => "IY".to_owned(),
        "oʊ" | "ou" => "OW".to_owned(),
        "ɔɪ" | "oi" => "OY".to_owned(),
        "ʊ" => "UH".to_owned(),
        "u" | "uː" => "UW".to_owned(),
        "ə" => "AX".to_owned(), // Schwa (reduced vowel)

        // Nasalized vowels (common in L2 speakers, especially Romance/Asian L1)
        // Map to non-nasalized equivalents for comparison
        "ɑ̃" | "ã" => "AA".to_owned(),
        "ɛ̃" | "ẽ" => "EH".to_owned(),
        "ɔ̃" | "õ" => "AO".to_owned(),
        "œ̃" => "AH".to_owned(),

        // Central vowels (common in reduced syllables)
        "ɨ" => "IH".to_owned(), // Close central unrounded → IH
        "ʉ" => "UW".to_owned(), // Close central rounded → UW
        "ɵ" => "AH".to_owned(), // Close-mid central rounded → AH
        "ɘ" => "AX".to_owned(), // Close-mid central unrounded → schwa
        // Note: "ɜ" is already mapped above in rhotic variants
        "ɞ" => "ER".to_owned(), // Open-mid central rounded → ER
        "ɐ" => "AH".to_owned(), // Near-open central → AH

        // Additional vowel variants
        "ɒ" => "AA".to_owned(), // Open back rounded (British "lot") → AA
        "ʏ" => "IH".to_owned(), // Near-close near-front rounded → IH
        "ø" => "EH".to_owned(), // Close-mid front rounded → EH
        "œ" => "AH".to_owned(), // Open-mid front rounded → AH
        "y" => "IY".to_owned(), // Close front rounded → IY

        // L2-ARCTIC vocabulary uses plain Latin letters for some vowels
        // These are commonly used in wav2vec2 model outputs
        "a" => "AA".to_owned(), // Open front vowel → AA (father)
        "o" => "OW".to_owned(), // Close-mid back vowel → OW (boat)
        "e" => "EH".to_owned(), // Close-mid front vowel → EH (bet)

        // Consonants - Stops
        "b" => "B".to_owned(),
        "d" => "D".to_owned(),
        "g" | "ɡ" => "G".to_owned(), // Both ASCII and IPA g
        "k" => "K".to_owned(),
        "p" => "P".to_owned(),
        "t" => "T".to_owned(),

        // Consonants - Affricates
        "tʃ" | "ʧ" => "CH".to_owned(),
        "dʒ" | "ʤ" => "JH".to_owned(),
        "ts" => "T".to_owned(), // Affricate simplified to stop
        "dz" => "D".to_owned(), // Affricate simplified to stop

        // Consonants - Fricatives
        "f" => "F".to_owned(),
        "v" => "V".to_owned(),
        "θ" => "TH".to_owned(),
        "ð" => "DH".to_owned(),
        "s" => "S".to_owned(),
        "z" => "Z".to_owned(),
        "ʃ" => "SH".to_owned(),
        "ʒ" => "ZH".to_owned(),
        "h" | "ɦ" => "HH".to_owned(), // Include voiced glottal fricative
        "x" => "HH".to_owned(),       // Voiceless velar fricative → HH
        "ç" => "HH".to_owned(),       // Voiceless palatal fricative → HH
        "ɣ" => "G".to_owned(),        // Voiced velar fricative → G
        "β" => "V".to_owned(),        // Voiced bilabial fricative → V
        "ɸ" => "F".to_owned(),        // Voiceless bilabial fricative → F

        // Consonants - Nasals
        "m" => "M".to_owned(),
        "n" => "N".to_owned(),
        "ŋ" => "NG".to_owned(),
        "ɲ" => "N".to_owned(), // Palatal nasal → N
        "ɱ" => "M".to_owned(), // Labiodental nasal → M

        // Consonants - Liquids
        "l" | "ɫ" => "L".to_owned(),       // Both light and dark L
        "r" | "ɹ" | "ɻ" => "R".to_owned(), // Rhotic variants
        "ʁ" => "R".to_owned(),             // Uvular fricative (French R) → R
        "ʀ" => "R".to_owned(),             // Uvular trill → R

        // Flaps and Taps (common in L2 speakers)
        "ɾ" => "D".to_owned(), // Alveolar flap → D (like American "butter")
        "ɽ" => "D".to_owned(), // Retroflex flap → D

        // Glottal stop (common in L2 speakers)
        "ʔ" => "Q".to_owned(), // Glottal stop → Q (extended ARPABET)

        // Consonants - Glides
        "w" => "W".to_owned(),
        "j" => "Y".to_owned(),
        "ʍ" => "W".to_owned(), // Voiceless labial-velar fricative → W
        "ɥ" => "Y".to_owned(), // Labial-palatal approximant → Y

        // Already ARPABET or unknown - return as-is (uppercase for consistency)
        other => {
            let upper = other.to_uppercase();
            if upper.is_empty() {
                "?".to_owned()
            } else {
                upper
            }
        }
    }
}

/// Strip combining diacritics and length markers from IPA symbols
///
/// Common diacritics to remove:
/// - ː (length marker)
/// - ̃ (nasalization)
/// - ̥ (voicelessness)
/// - ʰ (aspiration)
/// - ˈ ˌ (stress markers - these should be stripped before comparison)
fn strip_diacritics(ipa: &str) -> String {
    ipa.chars()
        .filter(|c| {
            !matches!(
                c,
                'ː'        // Length marker
            | '\u{0303}' // Combining tilde (nasalization) - '̃'
            | '\u{0325}' // Combining ring below (voiceless) - '̥'
            | 'ʰ'        // Aspiration
            | 'ˈ'        // Primary stress
            | 'ˌ'        // Secondary stress
            | '\u{0329}' // Combining vertical line below (syllabic) - '̩'
            | '\u{032F}' // Combining inverted breve below (non-syllabic) - '̯'
            | '\u{032A}' // Combining bridge below (dental) - '̪'
            | '\u{033A}' // Combining inverted bridge below (apical) - '̺'
            | '\u{033B}' // Combining square below (laminal) - '̻'
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_ipa_symbol() {
        assert_eq!(Phoneme::AH.ipa_symbol(), "ɑ");
        assert_eq!(Phoneme::IY.ipa_symbol(), "i");
    }

    #[test]
    fn test_phoneme_arpabet_symbol() {
        assert_eq!(Phoneme::AH.arpabet_symbol(), "AH");
        assert_eq!(Phoneme::IY.arpabet_symbol(), "IY");
    }

    #[test]
    fn test_phoneme_equality() {
        assert_eq!(Phoneme::AH, Phoneme::AH);
        assert_ne!(Phoneme::AH, Phoneme::IY);
    }

    #[test]
    fn test_ipa_to_arpabet_vowels() {
        // Test IPA vowel symbols convert to ARPABET
        assert_eq!(ipa_to_arpabet("ɔ"), "AO");
        assert_eq!(ipa_to_arpabet("ɛ"), "EH");
        assert_eq!(ipa_to_arpabet("ɪ"), "IH");
        assert_eq!(ipa_to_arpabet("ʊ"), "UH");
        assert_eq!(ipa_to_arpabet("ʌ"), "AH");

        // Rhotic vowels
        assert_eq!(ipa_to_arpabet("ɚ"), "ER");
        assert_eq!(ipa_to_arpabet("ɝ"), "ER");
        assert_eq!(ipa_to_arpabet("ɜr"), "ER");

        // L2-ARCTIC vocabulary plain Latin vowels
        assert_eq!(ipa_to_arpabet("a"), "AA");
        assert_eq!(ipa_to_arpabet("o"), "OW");
        assert_eq!(ipa_to_arpabet("e"), "EH");
    }

    #[test]
    fn test_ipa_to_arpabet_consonants() {
        // Stops
        assert_eq!(ipa_to_arpabet("b"), "B");
        assert_eq!(ipa_to_arpabet("d"), "D");
        assert_eq!(ipa_to_arpabet("k"), "K");
        assert_eq!(ipa_to_arpabet("p"), "P");
        assert_eq!(ipa_to_arpabet("t"), "T");

        // Fricatives
        assert_eq!(ipa_to_arpabet("s"), "S");
        assert_eq!(ipa_to_arpabet("z"), "Z");
        assert_eq!(ipa_to_arpabet("θ"), "TH");
        assert_eq!(ipa_to_arpabet("ð"), "DH");
        assert_eq!(ipa_to_arpabet("ʃ"), "SH");
        assert_eq!(ipa_to_arpabet("ʒ"), "ZH");

        // Nasals
        assert_eq!(ipa_to_arpabet("m"), "M");
        assert_eq!(ipa_to_arpabet("n"), "N");
        assert_eq!(ipa_to_arpabet("ŋ"), "NG");

        // Liquids
        assert_eq!(ipa_to_arpabet("l"), "L");
        assert_eq!(ipa_to_arpabet("ɫ"), "L"); // Dark L
        assert_eq!(ipa_to_arpabet("r"), "R");
        assert_eq!(ipa_to_arpabet("ɹ"), "R"); // Alveolar approximant
    }

    #[test]
    fn test_custom_phoneme_arpabet_conversion() {
        // Wav2vec2 returns IPA, should convert to ARPABET
        let p = Phoneme::Custom("ɚ".to_owned());
        assert_eq!(p.arpabet_symbol(), "ER");

        let p2 = Phoneme::Custom("ɔ".to_owned());
        assert_eq!(p2.arpabet_symbol(), "AO");

        let p3 = Phoneme::Custom("θ".to_owned());
        assert_eq!(p3.arpabet_symbol(), "TH");
    }

    #[test]
    fn test_ipa_to_arpabet_l2_speaker_patterns() {
        // Central vowels (common in reduced syllables)
        assert_eq!(ipa_to_arpabet("ɨ"), "IH");
        assert_eq!(ipa_to_arpabet("ɐ"), "AH");
        assert_eq!(ipa_to_arpabet("ɵ"), "AH");

        // Additional vowel variants
        assert_eq!(ipa_to_arpabet("ɒ"), "AA"); // British "lot"
        assert_eq!(ipa_to_arpabet("ø"), "EH"); // French "peu"
        assert_eq!(ipa_to_arpabet("y"), "IY"); // French "tu"

        // Flaps and taps (American English patterns)
        assert_eq!(ipa_to_arpabet("ɾ"), "D"); // Alveolar flap
        assert_eq!(ipa_to_arpabet("ɽ"), "D"); // Retroflex flap

        // Glottal stop
        assert_eq!(ipa_to_arpabet("ʔ"), "Q");

        // Additional fricatives
        assert_eq!(ipa_to_arpabet("x"), "HH"); // German "Bach"
        assert_eq!(ipa_to_arpabet("ç"), "HH"); // German "ich"
        assert_eq!(ipa_to_arpabet("ɣ"), "G"); // Voiced velar fricative

        // Additional nasals
        assert_eq!(ipa_to_arpabet("ɲ"), "N"); // Palatal nasal (Spanish "ñ")
        assert_eq!(ipa_to_arpabet("ɱ"), "M"); // Labiodental nasal

        // Uvular sounds (common in French/German L1 speakers)
        assert_eq!(ipa_to_arpabet("ʁ"), "R"); // French R
        assert_eq!(ipa_to_arpabet("ʀ"), "R"); // Uvular trill
    }

    #[test]
    fn test_strip_diacritics() {
        // Length markers
        assert_eq!(strip_diacritics("iː"), "i");
        assert_eq!(strip_diacritics("uː"), "u");

        // Stress markers
        assert_eq!(strip_diacritics("ˈhɛloʊ"), "hɛloʊ");
        assert_eq!(strip_diacritics("ˌsɛkəndˈɛri"), "sɛkəndɛri");

        // Aspiration
        assert_eq!(strip_diacritics("pʰ"), "p");
        assert_eq!(strip_diacritics("tʰ"), "t");

        // Combined
        assert_eq!(strip_diacritics("ˈtʰiːm"), "tim");
    }

    #[test]
    fn test_ipa_to_arpabet_diphthongs() {
        // Standard diphthongs
        assert_eq!(ipa_to_arpabet("aɪ"), "AY");
        assert_eq!(ipa_to_arpabet("aʊ"), "AW");
        assert_eq!(ipa_to_arpabet("eɪ"), "EY");
        assert_eq!(ipa_to_arpabet("oʊ"), "OW");
        assert_eq!(ipa_to_arpabet("ɔɪ"), "OY");

        // ASCII-like diphthongs (some models use these)
        assert_eq!(ipa_to_arpabet("ai"), "AY");
        assert_eq!(ipa_to_arpabet("au"), "AW");
        assert_eq!(ipa_to_arpabet("ei"), "EY");
        assert_eq!(ipa_to_arpabet("ou"), "OW");
        assert_eq!(ipa_to_arpabet("oi"), "OY");
    }
}
