//! Parsing and formatting of user input for inference: "phrase _ or [MASK] => answer".

/// Parsed user input for inference: "phrase _ more => answer" → tokens, mask position, answer token.
pub struct FormattedInput {
    pub tokens: Vec<String>,
    pub answer_token: String,
}

#[derive(Debug)]
pub enum FormatInputError {
    NoSeparator,
    NoMask,
}

/// Parse a line "phrase _ or [MASK] => answer" into tokens (with <mask>) and answer token (first word, lowercased, backticks stripped).
pub fn format_user_input(line: &str) -> Result<FormattedInput, FormatInputError> {
    let line = line.trim();
    let (phrase_part, answer_part) = line.split_once("=>").ok_or(FormatInputError::NoSeparator)?;
    let phrase_part = phrase_part.trim();
    let answer_part = answer_part.trim();
    let parts: Vec<&str> = phrase_part.split_whitespace().collect();
    let mut tokens = Vec::new();
    let mut pos = None;
    for (i, &p) in parts.iter().enumerate() {
        let w = p.trim_matches('`');
        if w == "_" || w.eq_ignore_ascii_case("[mask]") {
            pos = Some(i);
            tokens.push("<mask>".to_string());
        } else {
            tokens.push(w.to_string());
        }
    }
    pos.ok_or(FormatInputError::NoMask)?;
    let answer_token = answer_part
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches('`')
        .to_lowercase();
    Ok(FormattedInput {
        tokens,
        answer_token,
    })
}
