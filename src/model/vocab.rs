use std::collections::HashMap;

#[derive(Clone)]
pub struct Pair {
    pub context: Vec<u32>,
    pub target: Vec<u32>,
}

pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub pad_id: u32,
    pub unk_id: u32,
}

impl Vocab {
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();
        let pad_id = Self::push_token(&mut token_to_id, &mut id_to_token, "<pad>");
        let unk_id = Self::push_token(&mut token_to_id, &mut id_to_token, "<unk>");
        Self {
            token_to_id,
            id_to_token,
            pad_id,
            unk_id,
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
