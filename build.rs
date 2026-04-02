#![allow(missing_docs)]

use std::collections::HashMap;
use std::path::Path;
use std::{env, fs};

use serde::{Deserialize, Serialize};

// --- Duplicated types from src/g2p/types.rs ---
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phoneme {
    AH,
    AE,
    AX,
    IY,
    UW,
    Custom(String),
}

impl Phoneme {
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
}

// --- Duplicated logic from src/g2p/arpabet.rs ---
fn parse_arpabet_stress(arpabet: &str) -> (&str, u8) {
    let bytes = arpabet.as_bytes();
    let last_byte = bytes.last().copied().unwrap_or(0);

    match last_byte {
        b'0' => (&arpabet[..arpabet.len() - 1], 0),
        b'1' => (&arpabet[..arpabet.len() - 1], 1),
        b'2' => (&arpabet[..arpabet.len() - 1], 2),
        _ => (arpabet, 0),
    }
}

fn arpabet_to_ipa(arpabet: &str) -> &'static str {
    match arpabet {
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => "ʌ",
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AX" => "ə",
        "AY" => "aɪ",
        "EH" => "ɛ",
        "ER" => "ɜr",
        "EY" => "eɪ",
        "IH" => "ɪ",
        "IY" => "i",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "UH" => "ʊ",
        "UW" => "u",
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "F" => "f",
        "G" => "g",
        "HH" => "h",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "P" => "p",
        "R" => "r",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        _ => "",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var_os("OUT_DIR").ok_or("OUT_DIR not set")?;
    let dest_path = Path::new(&out_dir).join("cmudict.bin");

    // Path to the dictionary file relative to crate root
    let dict_path = "src/g2p/data/cmudict.dict";
    println!("cargo:rerun-if-changed={dict_path}");

    let content = fs::read_to_string(dict_path)
        .map_err(|err| format!("Failed to read cmudict.dict: {err}"))?;

    let mut entries: HashMap<String, Vec<Phoneme>> = HashMap::with_capacity(140_000);

    for line in content.lines() {
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        let Some(word) = parts.first() else {
            continue;
        };
        let word_base = word.split('(').next().unwrap_or(word);
        let word_lower = word_base.to_lowercase();

        if entries.contains_key(&word_lower) {
            continue;
        }

        let phonemes: Vec<Phoneme> = parts
            .iter()
            .skip(1)
            .take_while(|&&p| !p.starts_with('#'))
            .map(|&p| {
                let (base, _stress) = parse_arpabet_stress(p);
                let ipa = arpabet_to_ipa(base);
                if ipa.is_empty() {
                    Phoneme::Custom(base.to_owned())
                } else {
                    Phoneme::from_ipa(ipa)
                }
            })
            .collect();

        if phonemes.is_empty() {
            continue;
        }

        entries.insert(word_lower, phonemes);
    }

    let encoded: Vec<u8> = bitcode::serialize(&entries)
        .map_err(|err| format!("Failed to serialize cmudict.bin: {err}"))?;
    fs::write(&dest_path, encoded).map_err(|err| format!("Failed to write cmudict.bin: {err}"))?;

    Ok(())
}
