//! The checkpoint's tokenizer, through the `tokenizers` crate: the same code Python's
//! `tokenizers` package runs, pinned to the version that produced the fixtures.

use std::path::Path;

/// A loaded `tokenizer/tokenizer.json`.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

/// A tokenizer load or encode failure.
#[derive(Debug, thiserror::Error)]
#[error("tokenizer: {0}")]
pub struct TokenizerError(String);

impl Tokenizer {
    /// Loads `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner =
            tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| TokenizerError(format!("{}: {e}", path.as_ref().display())))?;
        Ok(Tokenizer { inner })
    }

    /// Token ids of `text` without special tokens, as `tok(text, add_special_tokens=False)`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        self.append_encode(&mut ids, text);
        ids
    }

    /// Appends `text`'s token ids (no special tokens) to `dst`.
    pub fn append_encode(&self, dst: &mut Vec<u32>, text: &str) {
        if text.is_empty() {
            return;
        }
        // Encoding a &str with no special tokens doesn't fail for this tokenizer; an error
        // would mean a broken tokenizer.json, which from_file already rejected.
        let enc = self.inner.encode_fast(text, false).expect("tokenizer encode");
        dst.extend_from_slice(enc.get_ids());
    }
}
