use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Result;

#[derive(Clone)]
pub struct Pair {
    pub tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub pad_id: u32,
    pub unk_id: u32,
    pub mask_id: u32,
}

impl Vocab {
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();
        let pad_id = Self::push_token(&mut token_to_id, &mut id_to_token, "<pad>");
        let unk_id = Self::push_token(&mut token_to_id, &mut id_to_token, "<unk>");
        let mask_id = Self::push_token(&mut token_to_id, &mut id_to_token, "<mask>");
        Self {
            token_to_id,
            id_to_token,
            pad_id,
            unk_id,
            mask_id,
        }
    }

    fn push_token(
        token_to_id: &mut HashMap<String, u32>,
        id_to_token: &mut Vec<String>,
        token: &str,
    ) -> u32 {
        let id = id_to_token.len() as u32;
        token_to_id.insert(token.to_string(), id);
        id_to_token.push(token.to_string());
        id
    }

    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(id) = self.token_to_id.get(token) {
            return *id;
        }
        Self::push_token(&mut self.token_to_id, &mut self.id_to_token, token)
    }

    pub fn encode(&self, tokens: &[String]) -> Vec<u32> {
        tokens
            .iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or(&self.unk_id))
            .collect()
    }
}

pub fn vocab_signature(vocab: &Vocab) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for token in &vocab.id_to_token {
        for byte in token.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(b'\n');
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// Load vocab from a file (one token per line).
pub fn load_vocab_from_file(vocab_path: impl AsRef<Path>) -> Result<Vocab> {
    let vocab_text = fs::read_to_string(vocab_path.as_ref())?;
    let mut vocab = Vocab::new();
    for line in vocab_text.lines() {
        if line.is_empty() {
            continue;
        }
        vocab.add_token(line);
    }
    Ok(vocab)
}

pub fn save_vocab_to_file(vocab: &Vocab, vocab_path: impl AsRef<Path>) -> Result<()> {
    let path = vocab_path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut vocab_text = String::new();
    for token in &vocab.id_to_token {
        vocab_text.push_str(token);
        vocab_text.push('\n');
    }
    fs::write(path, vocab_text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{vocab_signature, Vocab};

    #[test]
    fn vocab_signature_is_stable_for_same_vocab() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello");
        vocab.add_token("world");
        let sig_a = vocab_signature(&vocab);
        let sig_b = vocab_signature(&vocab);
        assert_eq!(sig_a, sig_b);
    }

    #[test]
    fn vocab_signature_changes_when_vocab_changes() {
        let mut vocab_a = Vocab::new();
        vocab_a.add_token("hello");
        let mut vocab_b = vocab_a.clone();
        vocab_b.add_token("world");
        assert_ne!(vocab_signature(&vocab_a), vocab_signature(&vocab_b));
    }
}
