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
