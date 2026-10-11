//! Phoneme inventory and transcript-to-CTC-target conversion.
//!
//! # Class inventory
//!
//! The model predicts [`NUM_CLASSES`] = 42 classes:
//!
//! - `0`: CTC blank ([`BLANK_IDX`], Burn's `CTCLoss` default)
//! - `1..=40`: ARPABET phonemes in alphabetical order (below)
//! - `41`: `SIL`, the word-boundary marker inserted between words ([`SIL_IDX`])
//!
//! ```text
//!  1 AA    9 CH   17 HH   25 NG   33 TH
//!  2 AE   10 D    18 IH   26 OW   34 UH
//!  3 AH   11 DH   19 IY   27 OY   35 UW
//!  4 AO   12 EH   20 JH   28 P    36 V
//!  5 AW   13 ER   21 K    29 R    37 W
//!  6 AX   14 EY   22 L    30 S    38 Y
//!  7 AY   15 F    23 M    31 SH   39 Z
//!  8 B    16 G    24 N    32 T    40 ZH
//! ```
//!
//! These are the 39 phonemes used by CMUdict plus `AX` (schwa). CMUdict writes
//! schwa as `AH0`, so class 6 never appears in generated targets; it is kept so
//! that class indices stay compatible with existing checkpoints.
//!
//! # Transcripts
//!
//! [`transcript_to_targets`] looks each word up in the embedded CMU dictionary
//! ([`crate::g2p`]). Out-of-vocabulary words go through a chain of
//! morphological and spelling fallbacks, so every word containing a letter
//! yields at least an approximate pronunciation.

use crate::error::{Error, Result};
use crate::g2p;

/// CTC blank class.
pub const BLANK_IDX: i32 = 0;

/// Word-boundary silence class.
pub const SIL_IDX: i32 = 41;

/// Total number of output classes: blank, 40 phonemes and `SIL`.
pub const NUM_CLASSES: usize = 42;

/// Phonemes in class order: `ARPABET_INVENTORY[i]` is class `i + 1`.
/// Must stay sorted (it is binary-searched) and must not change, since trained
/// checkpoints depend on these indices.
const ARPABET_INVENTORY: [&str; 40] = [
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
];

/// Maps an ARPABET symbol, with or without a stress digit, to its CTC class.
#[must_use]
pub fn arpabet_to_idx(symbol: &str) -> Option<i32> {
    phoneme_class(symbol).map(i32::from)
}

pub(crate) fn phoneme_class(symbol: &str) -> Option<u8> {
    let base = symbol.strip_suffix(['0', '1', '2']).unwrap_or(symbol);
    let position = ARPABET_INVENTORY.binary_search(&base).ok()?;
    u8::try_from(position + 1).ok()
}

fn classes<'a>(symbols: &'a [&str]) -> impl Iterator<Item = i32> + 'a {
    symbols.iter().filter_map(|symbol| arpabet_to_idx(symbol))
}

/// Counts of how the words of a transcript were converted to phonemes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolutionStats {
    /// Words containing at least one letter or digit.
    pub total_words: usize,
    /// Found directly in the dictionary.
    pub dict_hits: usize,
    /// Dictionary word plus a possessive or contraction ending (`KETTLE'S`).
    pub possessive_hits: usize,
    /// Hyphenated, with every part in the dictionary (`SEA-CAPTAIN`).
    pub hyphen_hits: usize,
    /// Dictionary stem plus one suffix or prefix (`TIDINESS`).
    pub suffix_hits: usize,
    /// Two dictionary words written together (`SHIPMASTERS`).
    pub compound_hits: usize,
    /// Dictionary stem with both a prefix and a suffix (`UNTIDINESS`).
    pub chained_hits: usize,
    /// Spelled out with letter-to-sound rules.
    pub g2p_hits: usize,
    /// Produced no phonemes (for example digits) and were left out.
    pub unresolved: usize,
}

impl ResolutionStats {
    /// Fraction of words found directly in the dictionary, in `0.0..=1.0`.
    #[must_use]
    pub fn dict_rate(&self) -> f64 {
        if self.total_words == 0 {
            return 0.0;
        }
        self.dict_hits as f64 / self.total_words as f64
    }

    fn record(&mut self, method: ResolutionMethod) {
        let counter = match method {
            ResolutionMethod::Dictionary => &mut self.dict_hits,
            ResolutionMethod::Possessive => &mut self.possessive_hits,
            ResolutionMethod::HyphenSplit => &mut self.hyphen_hits,
            ResolutionMethod::SuffixStrip => &mut self.suffix_hits,
            ResolutionMethod::CompoundSplit => &mut self.compound_hits,
            ResolutionMethod::ChainedAffix => &mut self.chained_hits,
            ResolutionMethod::PossessiveG2p | ResolutionMethod::RuleG2p => &mut self.g2p_hits,
        };
        *counter += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolutionMethod {
    Dictionary,
    Possessive,
    HyphenSplit,
    SuffixStrip,
    CompoundSplit,
    ChainedAffix,
    PossessiveG2p,
    RuleG2p,
}

/// Converts a transcript to a CTC target sequence, with [`SIL_IDX`] between
/// words.
///
/// Characters other than ASCII letters, apostrophes and hyphens are ignored.
/// Words that still produce no phonemes (such as `42`) are left out of the
/// targets and counted in [`ResolutionStats::unresolved`].
///
/// # Errors
///
/// Returns an error if the transcript is empty or yields no phonemes at all.
pub fn transcript_to_targets(text: &str) -> Result<(Vec<i32>, ResolutionStats)> {
    if text.trim().is_empty() {
        return Err(Error::config("Empty transcript"));
    }

    let mut targets = Vec::new();
    let mut stats = ResolutionStats::default();

    for token in text.split_whitespace() {
        if !token.chars().any(char::is_alphanumeric) {
            continue;
        }
        stats.total_words += 1;

        let word: String = token
            .chars()
            .filter(|c| c.is_ascii_alphabetic() || matches!(c, '\'' | '-'))
            .collect();
        let Some((phonemes, method)) = resolve_word(&word) else {
            tracing::debug!(word = token, "no phonemes for word");
            stats.unresolved += 1;
            continue;
        };
        if method != ResolutionMethod::Dictionary {
            tracing::debug!(word = %word, method = ?method, "OOV fallback resolved");
        }
        stats.record(method);

        if !targets.is_empty() {
            targets.push(SIL_IDX);
        }
        targets.extend(phonemes);
    }

    if targets.is_empty() {
        return Err(Error::config("No phonemes resolved from transcript"));
    }
    Ok((targets, stats))
}

type Fallback = fn(&str) -> Option<Vec<i32>>;

/// Converts one word to CTC classes, trying in order:
///
/// 1. the dictionary;
/// 2. a dictionary word plus a possessive or contraction ending (`KETTLE'S`);
/// 3. hyphen-separated parts that are all in the dictionary, joined by `SIL`;
/// 4. a dictionary stem plus one suffix, else one prefix (`QUIETLY`, `REBUILD`);
/// 5. two dictionary words of at least three letters each, joined by `SIL`
///    (`SHIPMASTERS`);
/// 6. a dictionary stem with both a prefix and a suffix (`UNTIDINESS`);
/// 7. letter-to-sound rules on the base of a `'S` possessive, plus `Z`;
/// 8. letter-to-sound rules on the whole word.
///
/// Returns `None` only if the word has no letters.
fn resolve_word(word: &str) -> Option<(Vec<i32>, ResolutionMethod)> {
    const FALLBACKS: [(Fallback, ResolutionMethod); 7] = [
        (dictionary, ResolutionMethod::Dictionary),
        (try_possessive, ResolutionMethod::Possessive),
        (try_hyphen_split, ResolutionMethod::HyphenSplit),
        (try_suffix_strip, ResolutionMethod::SuffixStrip),
        (try_compound_split, ResolutionMethod::CompoundSplit),
        (try_chained_affix, ResolutionMethod::ChainedAffix),
        (try_possessive_g2p, ResolutionMethod::PossessiveG2p),
    ];

    let word = word.to_ascii_uppercase();
    let (phonemes, method) = FALLBACKS
        .iter()
        .find_map(|&(fallback, method)| fallback(&word).map(|phonemes| (phonemes, method)))
        .unwrap_or_else(|| (rule_based_g2p(&word), ResolutionMethod::RuleG2p));
    (!phonemes.is_empty()).then_some((phonemes, method))
}

fn dictionary(word: &str) -> Option<Vec<i32>> {
    g2p::lookup(word).map(|phonemes| phonemes.iter().copied().map(i32::from).collect())
}

/// An affix and the ARPABET phonemes it contributes.
type Affix = (&'static str, &'static [&'static str]);

const POSSESSIVE_SUFFIXES: &[Affix] = &[
    ("'S", &["Z"]),
    ("'D", &["D"]),
    ("'LL", &["L"]),
    ("'VE", &["V"]),
    ("'RE", &["R"]),
    ("'T", &["T"]),
    ("'M", &["M"]),
];

/// Longest first, so that `LESS` is tried before `ES`.
const SUFFIXES: &[Affix] = &[
    ("TION", &["SH", "AH", "N"]),
    ("SION", &["ZH", "AH", "N"]),
    ("NESS", &["N", "AH", "S"]),
    ("MENT", &["M", "AH", "N", "T"]),
    ("ABLE", &["AH", "B", "AH", "L"]),
    ("IBLE", &["AH", "B", "AH", "L"]),
    ("LING", &["L", "IH", "NG"]),
    ("INGS", &["IH", "NG", "Z"]),
    ("ALLY", &["AH", "L", "IY"]),
    ("EOUS", &["IY", "AH", "S"]),
    ("IOUS", &["IY", "AH", "S"]),
    ("ICAL", &["IH", "K", "AH", "L"]),
    ("LESS", &["L", "AH", "S"]),
    ("FUL", &["F", "AH", "L"]),
    ("EST", &["AH", "S", "T"]),
    ("ING", &["IH", "NG"]),
    ("LY", &["L", "IY"]),
    ("ED", &["D"]),
    ("ER", &["ER"]),
    ("EN", &["AH", "N"]),
    ("ES", &["IH", "Z"]),
];

const PREFIXES: &[Affix] = &[
    ("OVER", &["OW", "V", "ER"]),
    ("UNDER", &["AH", "N", "D", "ER"]),
    ("FORE", &["F", "AO", "R"]),
    ("MIS", &["M", "IH", "S"]),
    ("OUT", &["AW", "T"]),
    ("PRE", &["P", "R", "IY"]),
    ("UN", &["AH", "N"]),
    ("RE", &["R", "IY"]),
];

fn try_possessive(word: &str) -> Option<Vec<i32>> {
    POSSESSIVE_SUFFIXES.iter().find_map(|(suffix, phones)| {
        let base = word.strip_suffix(suffix).filter(|base| !base.is_empty())?;
        let mut indices = dictionary(base)?;
        indices.extend(classes(phones));
        Some(indices)
    })
}

fn try_hyphen_split(word: &str) -> Option<Vec<i32>> {
    let parts: Vec<&str> = word.split('-').filter(|part| !part.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    let mut indices = Vec::new();
    for part in parts {
        if !indices.is_empty() {
            indices.push(SIL_IDX);
        }
        indices.extend(dictionary(part)?);
    }
    Some(indices)
}

/// Looks up what is left after removing a suffix: as written, with a final
/// `E` restored (`GLID` → `GLIDE`), or with a final `I` turned back into `Y`
/// (`TIDI` → `TIDY`).
fn lookup_stem(base: &str) -> Option<Vec<i32>> {
    dictionary(base)
        .or_else(|| dictionary(&format!("{base}E")))
        .or_else(|| dictionary(&format!("{}Y", base.strip_suffix('I')?)))
}

fn try_suffix_strip(word: &str) -> Option<Vec<i32>> {
    let suffixed = SUFFIXES.iter().find_map(|(suffix, phones)| {
        let base = word.strip_suffix(suffix).filter(|base| base.len() >= 2)?;
        let mut indices = lookup_stem(base)?;
        indices.extend(classes(phones));
        Some(indices)
    });
    suffixed.or_else(|| {
        PREFIXES.iter().find_map(|(prefix, phones)| {
            let base = word.strip_prefix(prefix).filter(|base| base.len() >= 2)?;
            let base = dictionary(base)?;
            Some(classes(phones).chain(base).collect())
        })
    })
}

fn try_compound_split(word: &str) -> Option<Vec<i32>> {
    if word.len() < 6 {
        return None;
    }
    (3..word.len() - 2).find_map(|split| {
        let (left, right) = word.split_at_checked(split)?;
        let mut indices = dictionary(left)?;
        let right = dictionary(right)?;
        indices.push(SIL_IDX);
        indices.extend(right);
        Some(indices)
    })
}

fn try_chained_affix(word: &str) -> Option<Vec<i32>> {
    PREFIXES.iter().find_map(|(prefix, prefix_phones)| {
        let rest = word.strip_prefix(prefix).filter(|rest| rest.len() >= 4)?;
        SUFFIXES.iter().find_map(|(suffix, suffix_phones)| {
            let base = rest.strip_suffix(suffix).filter(|base| base.len() >= 2)?;
            let base = lookup_stem(base)?;
            Some(
                classes(prefix_phones)
                    .chain(base)
                    .chain(classes(suffix_phones))
                    .collect(),
            )
        })
    })
}

fn try_possessive_g2p(word: &str) -> Option<Vec<i32>> {
    let base = word.strip_suffix("'S").filter(|base| base.len() >= 2)?;
    let mut indices = rule_based_g2p(base);
    if indices.is_empty() {
        return None;
    }
    indices.extend(classes(&["Z"]));
    Some(indices)
}

/// Approximate letter-to-sound rules: a common digraph if one starts here,
/// otherwise one phoneme per letter (`X` gives `K S`). `GH` is silent.
fn rule_based_g2p(word: &str) -> Vec<i32> {
    let letters: String = word
        .chars()
        .filter(char::is_ascii_alphabetic)
        .map(|c| c.to_ascii_uppercase())
        .collect();

    let mut indices = Vec::new();
    let mut rest = letters.as_str();
    while let Some(first) = rest.chars().next() {
        let (phones, consumed) = match rest.get(..2).and_then(digraph_phones) {
            Some(phones) => (phones, 2),
            None => (letter_phones(first), 1),
        };
        indices.extend(classes(phones));
        rest = rest.get(consumed..).unwrap_or_default();
    }
    indices
}

fn digraph_phones(digraph: &str) -> Option<&'static [&'static str]> {
    let phones: &[&str] = match digraph {
        "TH" => &["TH"],
        "SH" => &["SH"],
        "CH" => &["CH"],
        "PH" => &["F"],
        "WH" => &["W"],
        "NG" => &["NG"],
        "CK" => &["K"],
        "GH" => &[],
        "EE" | "EA" | "IE" => &["IY"],
        "OO" => &["UW"],
        "AI" | "AY" | "EI" | "EY" => &["EY"],
        "OI" | "OY" => &["OY"],
        "OU" => &["AW"],
        "OW" => &["OW"],
        "AU" | "AW" => &["AO"],
        _ => return None,
    };
    Some(phones)
}

fn letter_phones(letter: char) -> &'static [&'static str] {
    match letter {
        'A' => &["AE"],
        'E' => &["EH"],
        'I' => &["IH"],
        'O' => &["AA"],
        'U' => &["AH"],
        'Y' => &["IY"],
        'B' => &["B"],
        'C' | 'K' | 'Q' => &["K"],
        'D' => &["D"],
        'F' => &["F"],
        'G' => &["G"],
        'H' => &["HH"],
        'J' => &["JH"],
        'L' => &["L"],
        'M' => &["M"],
        'N' => &["N"],
        'P' => &["P"],
        'R' => &["R"],
        'S' => &["S"],
        'T' => &["T"],
        'V' => &["V"],
        'W' => &["W"],
        'X' => &["K", "S"],
        'Z' => &["Z"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices(symbols: &str) -> Vec<i32> {
        symbols
            .split_whitespace()
            .map(|symbol| arpabet_to_idx(symbol).expect("known phoneme"))
            .collect()
    }

    fn targets(text: &str) -> (Vec<i32>, ResolutionStats) {
        transcript_to_targets(text).expect("targets")
    }

    #[test]
    fn class_indices_are_stable() {
        let expected = "AA AE AH AO AW AX AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG \
                        OW OY P R S SH T TH UH UW V W Y Z ZH";
        for (position, symbol) in expected.split_whitespace().enumerate() {
            assert_eq!(arpabet_to_idx(symbol), i32::try_from(position + 1).ok());
        }
        assert_eq!(ARPABET_INVENTORY.len(), 40);
        assert_eq!(BLANK_IDX, 0);
        assert_eq!(SIL_IDX, 41);
        assert_eq!(NUM_CLASSES, 42);
    }

    #[test]
    fn inventory_is_sorted() {
        assert!(ARPABET_INVENTORY.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn arpabet_to_idx_strips_stress() {
        assert_eq!(arpabet_to_idx("AH0"), Some(3));
        assert_eq!(arpabet_to_idx("IH1"), Some(18));
        assert_eq!(arpabet_to_idx("ER2"), Some(13));
    }

    #[test]
    fn arpabet_to_idx_unknown() {
        assert_eq!(arpabet_to_idx("XX"), None);
        assert_eq!(arpabet_to_idx("ah"), None);
        assert_eq!(arpabet_to_idx(""), None);
    }

    #[test]
    fn dictionary_words() {
        let (targets, stats) = targets("father");
        assert_eq!(targets, [15, 1, 11, 13]);
        assert_eq!(targets, indices("F AA DH ER"));
        assert_eq!(stats.dict_hits, 1);
    }

    #[test]
    fn silence_between_words() {
        let (targets, stats) = targets("The cat");
        assert_eq!(
            targets,
            indices("DH AH")
                .into_iter()
                .chain([SIL_IDX])
                .chain(indices("K AE T"))
                .collect::<Vec<_>>()
        );
        assert_eq!(stats.total_words, 2);
        assert_eq!(stats.dict_hits, 2);
    }

    #[test]
    fn empty_transcript_is_an_error() {
        assert!(transcript_to_targets("").is_err());
        assert!(transcript_to_targets("  -- ").is_err());
        assert!(transcript_to_targets("1984").is_err());
    }

    #[test]
    fn words_without_phonemes_are_counted() {
        let (targets, stats) = targets("hello 42 world");
        assert_eq!(
            targets,
            transcript_to_targets("hello world").expect("targets").0
        );
        assert_eq!(stats.total_words, 3);
        assert_eq!(stats.dict_hits, 2);
        assert_eq!(stats.unresolved, 1);
    }

    #[test]
    fn possessive_of_dictionary_word() {
        let (targets, stats) = targets("KETTLE'S");
        assert_eq!(targets, indices("K EH T AH L Z"));
        assert_eq!(stats.possessive_hits, 1);

        for (word, last) in [
            ("HE'LL", "L"),
            ("I'VE", "V"),
            ("HE'D", "D"),
            ("THEY'RE", "R"),
        ] {
            let indices = try_possessive(word).expect(word);
            assert_eq!(indices.last(), arpabet_to_idx(last).as_ref(), "{word}");
        }
        assert_eq!(try_possessive("XYZZY'S"), None);
    }

    #[test]
    fn hyphenated_word() {
        let (targets, stats) = targets("SEA-CAPTAIN");
        let expected: Vec<i32> = indices("S IY")
            .into_iter()
            .chain([SIL_IDX])
            .chain(indices("K AE P T AH N"))
            .collect();
        assert_eq!(targets, expected);
        assert_eq!(stats.hyphen_hits, 1);

        assert_eq!(try_hyphen_split("WELL-XYZZY"), None);
        assert_eq!(try_hyphen_split("HELLO"), None);
    }

    #[test]
    fn suffix_and_prefix_stripping() {
        assert_eq!(try_suffix_strip("GLIDING"), Some(indices("G L AY D IH NG")));
        assert_eq!(
            try_suffix_strip("TIDINESS"),
            Some(indices("T AY D IY N AH S"))
        );
        for word in ["QUIETLY", "WALKED", "DARKNESS", "UNKIND", "REBUILD"] {
            assert!(try_suffix_strip(word).is_some(), "{word}");
        }
        assert_eq!(try_suffix_strip("XYZZYNESS"), None);
    }

    #[test]
    fn compound_split() {
        let expected: Vec<i32> = indices("SH IH P")
            .into_iter()
            .chain([SIL_IDX])
            .chain(indices("M AE S T ER Z"))
            .collect();
        assert_eq!(try_compound_split("SHIPMASTERS"), Some(expected));
        assert!(try_compound_split("OFTTIMES").is_some());
        assert_eq!(try_compound_split("CAT"), None);
    }

    #[test]
    fn prefix_and_suffix_together() {
        assert!(try_chained_affix("UNKINDNESS").is_some());
        assert_eq!(
            try_chained_affix("UNTIDINESS"),
            Some(indices("AH N T AY D IY N AH S"))
        );
    }

    #[test]
    fn possessive_of_unknown_word() {
        let (targets, stats) = targets("JELLYBY'S");
        assert_eq!(targets.last(), arpabet_to_idx("Z").as_ref());
        assert_eq!(stats.g2p_hits, 1);
        assert_eq!(try_possessive_g2p("HELLO"), None);
    }

    #[test]
    fn letter_to_sound_rules() {
        assert_eq!(rule_based_g2p("CAT"), indices("K AE T"));
        assert_eq!(rule_based_g2p("THINK"), indices("TH IH N K"));
        assert_eq!(rule_based_g2p("BOX"), indices("B AA K S"));
        assert_eq!(rule_based_g2p("PHONE"), indices("F AA N EH"));
    }

    #[test]
    fn gh_is_silent() {
        assert_eq!(rule_based_g2p("NIGHT"), indices("N IH T"));
        assert_eq!(rule_based_g2p("GH"), Vec::<i32>::new());
    }

    #[test]
    fn unknown_words_use_rules() {
        let (_, stats) = targets("HELLO XYZZYPLUGH WORLD");
        assert_eq!(stats.dict_hits, 2);
        assert_eq!(stats.g2p_hits, 1);
        assert!(transcript_to_targets("XYZZY PLUGH").is_ok());
    }
}
