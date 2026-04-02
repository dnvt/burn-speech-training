//! `ARPAbet` phoneme conversion utilities
//!
//! Provides conversion between `ARPAbet` notation (used in `CMUdict`) and IPA.
//!
//! ## `ARPAbet` Format
//!
//! `ARPAbet` uses uppercase ASCII symbols with optional stress markers (0, 1,
//! 2):
//! - `0` = no stress (unstressed)
//! - `1` = primary stress
//! - `2` = secondary stress
//!
//! Example: "AH0" = unstressed schwa, "IH1" = primary-stressed lax-i

/// Parse stress marker from `ARPAbet` symbol.
///
/// Returns `(base_phoneme, stress_level)` where `stress_level` is:
/// - `0` = unstressed
/// - `1` = primary stress
/// - `2` = secondary stress
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(parse_arpabet_stress("AH0"), ("AH", 0));
/// assert_eq!(parse_arpabet_stress("IH1"), ("IH", 1));
/// assert_eq!(parse_arpabet_stress("ER2"), ("ER", 2));
/// assert_eq!(parse_arpabet_stress("K"), ("K", 0)); // consonants have no stress
/// ```
#[must_use]
pub fn parse_arpabet_stress(arpabet: &str) -> (&str, u8) {
    // Check if last character is a stress marker (0, 1, or 2).
    // Empty strings fall through to the default case.
    if let Some(prefix) = arpabet.strip_suffix('0') {
        (prefix, 0)
    } else if let Some(prefix) = arpabet.strip_suffix('1') {
        (prefix, 1)
    } else if let Some(prefix) = arpabet.strip_suffix('2') {
        (prefix, 2)
    } else {
        (arpabet, 0) // No stress marker (consonants)
    }
}

/// Convert `ARPAbet` symbol to IPA.
///
/// Uses the standard `CMUdict` `ARPAbet`-to-IPA mapping.
///
/// # Arguments
///
/// * `arpabet` - `ARPAbet` symbol (without stress marker)
///
/// # Returns
///
/// IPA symbol as a static string, or empty string if not recognized.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(arpabet_to_ipa("DH"), "ð");
/// assert_eq!(arpabet_to_ipa("IH"), "ɪ");
/// assert_eq!(arpabet_to_ipa("TH"), "θ");
/// ```
#[must_use]
pub fn arpabet_to_ipa(arpabet: &str) -> &'static str {
    match arpabet {
        // Vowels
        "AA" => "ɑ",  // odd
        "AE" => "æ",  // at
        "AH" => "ʌ",  // hut
        "AO" => "ɔ",  // ought
        "AW" => "aʊ", // cow
        "AX" => "ə",  // about (schwa)
        "AY" => "aɪ", // hide
        "EH" => "ɛ",  // Ed
        "ER" => "ɜr", // hurt (r-colored)
        "EY" => "eɪ", // ate
        "IH" => "ɪ",  // it
        "IY" => "i",  // eat
        "OW" => "oʊ", // oat
        "OY" => "ɔɪ", // toy
        "UH" => "ʊ",  // hood
        "UW" => "u",  // two

        // Consonants
        "B" => "b",
        "CH" => "tʃ", // cheese
        "D" => "d",
        "DH" => "ð", // thee
        "F" => "f",
        "G" => "g",
        "HH" => "h",
        "JH" => "dʒ", // jee
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ", // sing
        "P" => "p",
        "R" => "r",
        "S" => "s",
        "SH" => "ʃ", // she
        "T" => "t",
        "TH" => "θ", // thin
        "V" => "v",
        "W" => "w",
        "Y" => "j", // yes
        "Z" => "z",
        "ZH" => "ʒ", // measure

        // Unknown - return empty (caller should handle)
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stress_primary() {
        let (base, stress) = parse_arpabet_stress("IH1");
        assert_eq!(base, "IH");
        assert_eq!(stress, 1);
    }

    #[test]
    fn test_parse_stress_unstressed() {
        let (base, stress) = parse_arpabet_stress("AH0");
        assert_eq!(base, "AH");
        assert_eq!(stress, 0);
    }

    #[test]
    fn test_parse_stress_secondary() {
        let (base, stress) = parse_arpabet_stress("ER2");
        assert_eq!(base, "ER");
        assert_eq!(stress, 2);
    }

    #[test]
    fn test_parse_stress_consonant() {
        let (base, stress) = parse_arpabet_stress("K");
        assert_eq!(base, "K");
        assert_eq!(stress, 0);
    }

    #[test]
    fn test_parse_stress_empty() {
        let (base, stress) = parse_arpabet_stress("");
        assert_eq!(base, "");
        assert_eq!(stress, 0);
    }

    #[test]
    fn test_arpabet_to_ipa_vowels() {
        assert_eq!(arpabet_to_ipa("AA"), "ɑ");
        assert_eq!(arpabet_to_ipa("AE"), "æ");
        assert_eq!(arpabet_to_ipa("AH"), "ʌ");
        assert_eq!(arpabet_to_ipa("IH"), "ɪ");
        assert_eq!(arpabet_to_ipa("IY"), "i");
        assert_eq!(arpabet_to_ipa("UW"), "u");
    }

    #[test]
    fn test_arpabet_to_ipa_consonants() {
        assert_eq!(arpabet_to_ipa("DH"), "ð");
        assert_eq!(arpabet_to_ipa("TH"), "θ");
        assert_eq!(arpabet_to_ipa("SH"), "ʃ");
        assert_eq!(arpabet_to_ipa("CH"), "tʃ");
        assert_eq!(arpabet_to_ipa("NG"), "ŋ");
    }

    #[test]
    fn test_arpabet_to_ipa_unknown() {
        assert_eq!(arpabet_to_ipa("XX"), "");
        assert_eq!(arpabet_to_ipa("UNKNOWN"), "");
    }
}
