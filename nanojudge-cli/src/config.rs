/// Config file loading and creation for nanojudge CLI.
///
/// Config lives at ~/.config/nanojudge/config.toml.
/// All fields are optional — CLI args override config values.
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::bail;

#[derive(Deserialize, Default)]
pub struct NanojudgeConfig {
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub rounds: Option<usize>,
    pub concurrency: Option<usize>,
    pub prompt_template: Option<String>,
    pub temperature: Option<f64>,
    pub temperature_jitter: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub top_p: Option<f64>,
    pub no_logprobs: Option<bool>,
    pub narrow_win: Option<f64>,
    pub analysis_length: Option<String>,
    pub strategy: Option<String>,
    pub top_k: Option<usize>,
    pub retries: Option<usize>,
    pub confidence_level: Option<f64>,
    pub regularization_strength: Option<f64>,
    pub mcmc_iterations: Option<usize>,
    pub mcmc_burn_in: Option<usize>,
    pub bias_prior: Option<f64>,
    pub matchmaking_sharpness: Option<f64>,
    pub min_games: Option<usize>,
    pub prior_tau2: Option<f64>,
    pub sigma2: Option<f64>,
    pub proposal_std: Option<f64>,
    pub bias_prior_tau2: Option<f64>,
    pub bias_proposal_std: Option<f64>,
}

const DEFAULT_CONFIG_TEMPLATE: &str = "\
# nanojudge configuration
# All values here can be overridden by CLI flags.

# OpenAI-compatible API endpoint
# endpoint = \"http://localhost:8000\"

# Model ID
# model = \"model-id\"

# API key: use OPENAI_API_KEY env var or --api-key flag (not stored in config)

# Number of comparison rounds
# rounds = 10

# Max concurrent LLM requests
# concurrency = 32

# LLM sampling temperature (required — each model needs a different value)
# temperature = 0.7

# Temperature jitter: standard deviation of N(1.0, jitter) multiplier.
# 0.0 = no jitter (default). Adds randomness to temperature across calls.
# temperature_jitter = 0.0

# Presence penalty: penalizes repeated tokens. Range: -2.0 to 2.0.
# Not sent to the API unless specified.
# presence_penalty = 1.5

# Top-p (nucleus sampling): only sample from tokens whose cumulative probability
# exceeds this threshold. Range: 0.0 to 1.0. Not sent to the API unless specified.
# top_p = 1.0

# Path to a custom prompt template file.
# The template must contain these variables: $criterion, $option1, $option2, $length
# If not set, the built-in default prompt is used.
# prompt_template = \"/path/to/my-prompt.txt\"

# Disable logprob extraction and use text-based verdict parsing.
# Required for endpoints that do not support logprobs (e.g. Gemini).
# Produces discrete probabilities instead of continuous — may need more rounds.
# no_logprobs = false

# Win probability assigned to a \"narrow win\" verdict (B or D on the likert scale).
# Must be > 0.5 and < 1.0. Default: 0.8. \"Clear win\" (A/E) is always 1.0/0.0.
# narrow_win = 0.8

# How much analysis the LLM should write before its verdict.
# Examples: \"3 sentences\", \"1 paragraph\", \"5 sentences\".
# analysis_length = \"2 paragraphs\"

# Pairing strategy: \"balanced\" or \"top-heavy\".
# Balanced gives equal attention to all items. Top-heavy focuses on contenders.
# strategy = \"balanced\"

# How many top positions to track for the top-heavy strategy.
# Default: sqrt(n) * 3, clamped to n-1. Only used with strategy = \"top-heavy\".
# top_k = 10

# Max retries per comparison on HTTP errors. 0 = no retries. Default: 3.
# retries = 3

# --- Scoring & MCMC hyperparameters ---
# Most users should not need to change these.

# Confidence interval level. Default: 0.95 (95%).
# confidence_level = 0.95

# Ghost player regularization strength. Default: 0.01.
# regularization_strength = 0.01

# Number of post-burn-in MCMC iterations for final scoring. Default: 2000.
# mcmc_iterations = 2000

# MCMC burn-in iterations for final scoring. Default: 500.
# mcmc_burn_in = 500

# Positional bias prior in probability space. 0.5 = no bias (default).
# Values > 0.5 mean the model tends to favor the first-listed item.
# Must be > 0.0 and < 1.0 (exclusive).
# bias_prior = 0.5

# Info-gain exponent for matchmaking. Higher = more exploitation. Default: 1.0.
# matchmaking_sharpness = 1.0

# Minimum games per item before using top-heavy strategy. Default: 3.
# min_games = 3

# Prior variance on log-strengths. Default: 10.0.
# prior_tau2 = 10.0

# Observation noise variance. Default: 1.0.
# sigma2 = 1.0

# MH proposal step size for strengths. Default: 0.3.
# proposal_std = 0.3

# Prior variance on positional bias (logit space). Default: 2.0.
# bias_prior_tau2 = 2.0

# MH proposal step size for bias. Default: 0.15.
# bias_proposal_std = 0.15
";

/// Returns the default config path.
/// Linux: ~/.config/nanojudge/config.toml
/// macOS: ~/Library/Application Support/nanojudge/config.toml
/// Windows: C:\Users\<user>\AppData\Roaming\nanojudge\config.toml
pub fn config_path() -> PathBuf {
    let config_dir = dirs::config_dir().unwrap_or_else(|| bail("Could not determine config directory"));
    config_dir.join("nanojudge").join("config.toml")
}

/// Load config from a file path. Returns default (all None) if file doesn't exist.
pub fn load_config(path: &Path) -> NanojudgeConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            toml::from_str(&content)
                .unwrap_or_else(|e| bail(format!("Failed to parse config at {}: {e}", path.display())))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => NanojudgeConfig::default(),
        Err(e) => bail(format!("Failed to read config at {}: {e}", path.display())),
    }
}

/// Create the default config file. Errors if it already exists.
pub fn create_default_config() -> PathBuf {
    let path = config_path();

    if path.exists() {
        bail(format!("Config file already exists at {}", path.display()));
    }

    // Create parent directories
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| bail(format!("Failed to create directory {}: {e}", parent.display())));
    }

    std::fs::write(&path, DEFAULT_CONFIG_TEMPLATE)
        .unwrap_or_else(|e| bail(format!("Failed to write config to {}: {e}", path.display())));

    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_config_path_ends_correctly() {
        let path = config_path();
        assert!(path.ends_with("nanojudge/config.toml"));
    }

    #[test]
    fn test_load_config_missing_file_returns_defaults() {
        let config = load_config(Path::new("/tmp/nanojudge-test-nonexistent/config.toml"));
        assert!(config.endpoint.is_none());
        assert!(config.model.is_none());
        assert!(config.rounds.is_none());
        assert!(config.concurrency.is_none());
        assert!(config.temperature.is_none());
    }

    #[test]
    fn test_load_config_parses_fields() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
endpoint = "http://localhost:8000"
model = "qwen-4b"
rounds = 10
concurrency = 32
temperature = 0.7
temperature_jitter = 0.05
presence_penalty = 1.5
top_p = 0.95
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.endpoint.unwrap(), "http://localhost:8000");
        assert_eq!(config.model.unwrap(), "qwen-4b");
        assert_eq!(config.rounds.unwrap(), 10);
        assert_eq!(config.concurrency.unwrap(), 32);
        assert_eq!(config.temperature.unwrap(), 0.7);
        assert_eq!(config.temperature_jitter.unwrap(), 0.05);
        assert_eq!(config.presence_penalty.unwrap(), 1.5);
        assert_eq!(config.top_p.unwrap(), 0.95);
    }

    #[test]
    fn test_load_config_partial_fields() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
endpoint = "http://localhost:8000"
rounds = 5
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.endpoint.unwrap(), "http://localhost:8000");
        assert_eq!(config.rounds.unwrap(), 5);
        assert!(config.model.is_none());
        assert!(config.temperature.is_none());
    }

    #[test]
    fn test_default_config_template_parses() {
        // The default template is all comments, so it should parse to all-None
        let config: NanojudgeConfig = toml::from_str(DEFAULT_CONFIG_TEMPLATE).unwrap();
        assert!(config.endpoint.is_none());
        assert!(config.model.is_none());
    }
}
