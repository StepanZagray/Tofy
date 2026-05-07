use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct Pair {
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedVocab {
    version: u32,
    format: String,
    tokens: Vec<String>,
    merges: Vec<[u32; 3]>,
}

#[derive(Clone)]
pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub pad_id: u32,
    pub unk_id: u32,
    pub mask_id: u32,
    pub merges: Vec<(u32, u32, u32)>,
    merge_ranks: HashMap<(u32, u32), (usize, u32)>,
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
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
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

    pub fn add_merge(&mut self, left: u32, right: u32, token: &str) -> u32 {
        let merged_id = self.add_token(token);
        if !self
            .merges
            .iter()
            .any(|&(l, r, m)| l == left && r == right && m == merged_id)
        {
            let rank = self.merges.len();
            self.merges.push((left, right, merged_id));
            self.merge_ranks.insert((left, right), (rank, merged_id));
        }
        merged_id
    }

    pub fn rebuild_merge_ranks(&mut self) {
        self.merge_ranks.clear();
        for (rank, &(left, right, merged)) in self.merges.iter().enumerate() {
            self.merge_ranks.insert((left, right), (rank, merged));
        }
    }

    pub fn encode(&self, tokens: &[String]) -> Vec<u32> {
        tokens
            .iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or(&self.unk_id))
            .collect()
    }

    pub fn encode_boundless(&self, text: &str) -> Vec<u32> {
        let mut ids = text
            .chars()
            .map(|ch| {
                let token = ch.to_string();
                *self.token_to_id.get(&token).unwrap_or(&self.unk_id)
            })
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return ids;
        }
        loop {
            let mut best: Option<(usize, usize, u32)> = None;
            for idx in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merge_ranks.get(&(ids[idx], ids[idx + 1])) {
                    match best {
                        Some((_, best_rank, _)) if rank >= best_rank => {}
                        _ => best = Some((idx, rank, merged)),
                    }
                }
            }
            let Some((idx, _rank, merged)) = best else {
                break;
            };
            ids[idx] = merged;
            ids.remove(idx + 1);
        }
        ids
    }

    pub fn decode_ids_lossy(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id == self.pad_id || id == self.mask_id {
                    None
                } else {
                    self.id_to_token.get(id as usize).cloned()
                }
            })
            .collect::<String>()
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
    for &(left, right, merged) in &vocab.merges {
        for value in [left, right, merged] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
    }
    format!("{hash:016x}")
}

fn legacy_load_vocab(vocab_text: &str) -> Vocab {
    let mut vocab = Vocab::new();
    for line in vocab_text.lines() {
        if line.is_empty() {
            continue;
        }
        vocab.add_token(line);
    }
    vocab
}

pub fn load_vocab_from_file(vocab_path: impl AsRef<Path>) -> Result<Vocab> {
    let vocab_text = fs::read_to_string(vocab_path.as_ref())?;
    if vocab_text.trim_start().starts_with('{') {
        let serialized: SerializedVocab = serde_json::from_str(&vocab_text)?;
        let mut vocab = Vocab::new();
        for token in serialized.tokens {
            if token != "<pad>" && token != "<unk>" && token != "<mask>" {
                vocab.add_token(&token);
            }
        }
        vocab.merges = serialized
            .merges
            .into_iter()
            .map(|triple| (triple[0], triple[1], triple[2]))
            .collect();
        vocab.rebuild_merge_ranks();
        return Ok(vocab);
    }
    Ok(legacy_load_vocab(&vocab_text))
}

pub fn save_vocab_to_file(vocab: &Vocab, vocab_path: impl AsRef<Path>) -> Result<()> {
    let path = vocab_path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let serialized = SerializedVocab {
        version: 1,
        format: "boundless_bpe".to_string(),
        tokens: vocab.id_to_token.clone(),
        merges: vocab
            .merges
            .iter()
            .map(|&(left, right, merged)| [left, right, merged])
            .collect(),
    };
    fs::write(path, serde_json::to_string_pretty(&serialized)?)?;
    Ok(())
}
