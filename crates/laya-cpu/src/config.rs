//! The part of `encoder/config.json` and `rl_agent_config.json` the forward pass needs
//! (laya-go's `internal/modernbert/config.go`).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::LoadError;

/// Model dimensions and attention layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `hidden_size`.
    pub hidden: usize,
    /// `num_hidden_layers`.
    pub layers: usize,
    /// `num_attention_heads`.
    pub heads: usize,
    /// `intermediate_size` (the GeGLU width; `Wi` has twice this).
    pub intermediate: usize,
    /// `vocab_size`.
    pub vocab: usize,
    /// `norm_eps`.
    pub norm_eps: f32,
    /// Sliding-window half-width: token `i` sees `j` when `|i − j| ≤ window`.
    pub window: usize,
    /// Per layer: full attention (`true`) or sliding window.
    pub global: Vec<bool>,
    /// RoPE θ for full-attention layers.
    pub global_theta: f64,
    /// RoPE θ for sliding-window layers.
    pub local_theta: f64,
    /// `head_layers` from `rl_agent_config.json`.
    pub head_layers: usize,
    /// `4 · hidden` (`nn.TransformerEncoderLayer` `dim_feedforward`).
    pub head_ff: usize,
    /// `max(1, hidden / 64)`.
    pub head_heads: usize,
    /// `len(act_costs) + 1`.
    pub n_act: usize,
    /// `max_len`.
    pub max_len: usize,
    /// `head_max_len`.
    pub head_max_len: usize,
}

#[derive(Deserialize)]
struct EncoderJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    norm_eps: Option<f64>,
    layer_norm_eps: Option<f64>,
    #[serde(default)]
    local_attention: usize,
    #[serde(default)]
    global_attn_every_n_layers: usize,
    #[serde(default)]
    layer_types: Vec<String>,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: HashMap<String, Rope>,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    hidden_activation: String,
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    head_dim: usize,
}

#[derive(Deserialize)]
struct Rope {
    rope_theta: f64,
    #[serde(default)]
    rope_type: String,
}

#[derive(Deserialize)]
struct AgentJson {
    head_layers: Option<usize>,
    #[serde(default)]
    max_len: usize,
    #[serde(default)]
    head_max_len: usize,
    #[serde(default)]
    act_costs: HashMap<String, f64>,
}

impl Config {
    /// Reads `dir/encoder/config.json` and `dir/rl_agent_config.json`.
    pub fn load(dir: &Path) -> Result<Config, LoadError> {
        let e = std::fs::read(dir.join("encoder").join("config.json"))?;
        let a = std::fs::read(dir.join("rl_agent_config.json"))?;
        Config::parse(&e, &a)
    }

    /// Parses the two config files' contents.
    pub fn parse(encoder_config: &[u8], agent_config: &[u8]) -> Result<Config, LoadError> {
        let bad = |m: String| Err(LoadError::Config(m));
        let e: EncoderJson = serde_json::from_slice(encoder_config).map_err(|err| LoadError::Config(format!("encoder config: {err}")))?;
        let a: AgentJson = serde_json::from_slice(agent_config).map_err(|err| LoadError::Config(format!("rl_agent_config: {err}")))?;
        if !e.model_type.is_empty() && e.model_type != "modernbert" {
            return bad(format!("unsupported encoder model_type {:?}", e.model_type));
        }
        if e.norm_bias || e.attention_bias || e.mlp_bias {
            return bad("encoder biases are not supported".into());
        }
        if e.hidden_activation != "gelu" {
            return bad(format!("unsupported hidden_activation {:?}", e.hidden_activation));
        }
        let (hidden, layers, heads) = (e.hidden_size, e.num_hidden_layers, e.num_attention_heads);
        if hidden == 0 || layers == 0 || heads == 0 || e.intermediate_size == 0 || e.vocab_size == 0 || hidden % heads != 0 {
            return bad("bad encoder dimensions".into());
        }
        if e.head_dim != 0 && e.head_dim != hidden / heads {
            return bad(format!("head_dim {} != hidden/heads", e.head_dim));
        }
        if (hidden / heads) % 2 != 0 {
            return bad("odd head dimension".into());
        }
        let norm_eps = e.norm_eps.or(e.layer_norm_eps).unwrap_or(1e-5) as f32;
        let window = e.local_attention / 2;
        let global: Vec<bool> = if !e.layer_types.is_empty() {
            if e.layer_types.len() != layers {
                return bad(format!("{} layer_types for {layers} layers", e.layer_types.len()));
            }
            let mut g = Vec::with_capacity(layers);
            for t in &e.layer_types {
                match t.as_str() {
                    "full_attention" => g.push(true),
                    "sliding_attention" => g.push(false),
                    _ => return bad(format!("unknown layer type {t:?}")),
                }
            }
            g
        } else {
            let every = e.global_attn_every_n_layers.max(1);
            (0..layers).map(|i| i % every == 0).collect()
        };
        let mut global_theta = e.global_rope_theta.unwrap_or(160000.0);
        let mut local_theta = e.local_rope_theta.unwrap_or(10000.0);
        for (key, theta) in [("full_attention", &mut global_theta), ("sliding_attention", &mut local_theta)] {
            if let Some(r) = e.rope_parameters.get(key) {
                if !r.rope_type.is_empty() && r.rope_type != "default" {
                    return bad(format!("unsupported rope_type {:?}", r.rope_type));
                }
                *theta = r.rope_theta;
            }
        }
        if let Some(i) = global.iter().position(|&g| !g)
            && window == 0
        {
            return bad(format!("layer {i} is sliding but local_attention is {}", e.local_attention));
        }
        Ok(Config {
            hidden,
            layers,
            heads,
            intermediate: e.intermediate_size,
            vocab: e.vocab_size,
            norm_eps,
            window,
            global,
            global_theta,
            local_theta,
            head_layers: a.head_layers.unwrap_or(2),
            head_ff: 4 * hidden,
            head_heads: (hidden / 64).max(1),
            n_act: a.act_costs.len() + 1,
            max_len: if a.max_len == 0 { 512 } else { a.max_len },
            head_max_len: if a.head_max_len == 0 { 192 } else { a.head_max_len },
        })
    }

    /// The per-head width.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}
