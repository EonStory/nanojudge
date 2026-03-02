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
}

const DEFAULT_CONFIG_TEMPLATE: &str = "\
# nanojudge configuration
# All values here can be overridden by CLI flags.

# OpenAI-compatible API endpoint
# endpoint = \"http://localhost:8000\"

# Model ID
# model = \"Qwen/Qwen3-4B-Instruct-2507\"

# API key: use OPENAI_API_KEY env var or --api-key flag (not stored in config)

# Number of comparison rounds
# rounds = 10

# Max concurrent LLM requests
# concurrency = 32

# Path to a custom prompt template file.
# The template must contain these variables: $criterion, $option1, $option2, $length
# If not set, the built-in default prompt is used.
# prompt_template = \"/path/to/my-prompt.txt\"
";

/// Returns the default config path: ~/.config/nanojudge/config.toml
pub fn config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| bail("HOME environment variable not set"));
    PathBuf::from(home).join(".config").join("nanojudge").join("config.toml")
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
