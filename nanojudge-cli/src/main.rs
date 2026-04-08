mod benchmark;
mod config;
mod llm;
mod output;
mod parse;
mod prompt;

use clap::Parser;
use nanojudge_core::{
    ComparisonInput, EngineConfig, JudgeInfo, RankingEngine, ScoringOptions, Strategy,
    calculate_total_expected_comparisons, run_scoring, stable_hash,
};
use rand::seq::SliceRandom;
use reqwest::Client;
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::llm::{LlmConfig, compare_pair};

const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_TEMPERATURE_JITTER: f64 = 0.0;
const DEFAULT_MAX_RETRIES: usize = 3;
const DEFAULT_ANALYSIS_LENGTH: &str = "2 paragraphs";
const DEFAULT_BENCHMARK_PAIRS: &str = "100";

pub fn bail(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

/// Merge a CLI value with a config file value. CLI wins.
/// Warns to stderr if both are set and differ.
fn merge_opt<T: PartialEq + std::fmt::Display>(
    cli: Option<T>,
    cfg: Option<T>,
    flag: &str,
) -> Option<T> {
    match (cli, cfg) {
        (Some(c), Some(f)) => {
            if c != f {
                eprintln!("Warning: --{flag} ({c}) overrides config file value ({f})");
            }
            Some(c)
        }
        (c @ Some(_), None) => c,
        (None, f) => f,
    }
}

#[derive(Parser)]
#[command(name = "nanojudge", version, about = "Rank items using LLM pairwise comparisons")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Run pairwise ranking on a list of items
    Rank(RankArgs),
    /// Benchmark an LLM endpoint for throughput, latency, and reliability
    Benchmark(BenchmarkArgs),
    /// Create a default config file at ~/.config/nanojudge/config.toml
    Init,
}

/// Persistent configuration settings — available as both CLI flags and config file entries.
/// These match the config file fields 1:1 (except api_key which is CLI/env only).
#[derive(Parser)]
struct ConfigArgs {
    /// Bearer token for the API (also reads OPENAI_API_KEY env var).
    /// Used as fallback for judges without api_key_env.
    #[arg(long)]
    api_key: Option<String>,

    /// Enable logprob extraction for continuous win probabilities.
    /// Requires an endpoint that supports logprobs.
    #[arg(long)]
    logprobs: bool,

    /// Number of comparison rounds
    #[arg(long)]
    rounds: Option<usize>,

    /// Max concurrent LLM requests
    #[arg(long)]
    concurrency: Option<usize>,

    /// Win probability assigned to a "narrow win" verdict (B or D on the likert scale).
    /// Must be > 0.5 and < 1.0. Default: 0.8. "Clear win" (A/E) is always 1.0/0.0.
    #[arg(long)]
    narrow_win: Option<f64>,

    /// Pairing strategy: "balanced" or "top-heavy"
    #[arg(long)]
    strategy: Option<String>,

    /// How many top positions to track for the top-heavy strategy.
    /// Default: sqrt(n) * 3, clamped to n-1. Only used with --strategy top-heavy.
    #[arg(long)]
    top_k: Option<usize>,

    /// Max retries per comparison on HTTP errors. Default: 3. Set to 0 to disable.
    #[arg(long)]
    retries: Option<usize>,

    /// How much analysis the LLM should write before its verdict.
    /// Default: "2 paragraphs". Examples: "3 sentences", "1 paragraph", "5 sentences".
    #[arg(long)]
    analysis_length: Option<String>,

    /// Path to a custom prompt template file.
    /// The template must contain: $criterion, $option1, $option2, $length
    #[arg(long)]
    prompt_template: Option<PathBuf>,

    /// Confidence interval level (e.g. 0.95 for 95%). Default: 0.95.
    #[arg(long)]
    confidence_level: Option<f64>,

    /// Ghost player regularization strength. Default: 0.01.
    #[arg(long)]
    regularization_strength: Option<f64>,

    /// Number of post-burn-in MCMC iterations for final scoring. Default: 2000.
    #[arg(long)]
    mcmc_iterations: Option<usize>,

    /// MCMC burn-in iterations for final scoring. Default: 500.
    #[arg(long)]
    mcmc_burn_in: Option<usize>,

    /// Positional bias prior in probability space (0.0-1.0 exclusive).
    /// 0.5 = no bias (default). >0.5 = model tends to favor item listed first.
    #[arg(long)]
    bias_prior: Option<f64>,

    /// Info-gain exponent for matchmaking. Higher = more exploitation. Default: 1.0.
    #[arg(long)]
    matchmaking_sharpness: Option<f64>,

    /// Minimum games per item before using top-heavy strategy. Default: 3.
    #[arg(long)]
    min_games: Option<usize>,

    /// Prior variance on log-strengths. Default: 10.0.
    #[arg(long)]
    prior_tau2: Option<f64>,

    /// Observation noise variance. Default: 1.0.
    #[arg(long)]
    sigma2: Option<f64>,

    /// MH proposal step size for strengths. Default: 0.3.
    #[arg(long)]
    proposal_std: Option<f64>,

    /// Prior variance on positional bias (logit space). Default: 2.0.
    #[arg(long)]
    bias_prior_tau2: Option<f64>,

    /// MH proposal step size for bias. Default: 0.15.
    #[arg(long)]
    bias_proposal_std: Option<f64>,

    /// Prior variance on log-decisiveness (logprobs mode). Default: 1.0.
    #[arg(long)]
    decisiveness_prior_tau2: Option<f64>,

    /// MH proposal step size for log-decisiveness (logprobs mode). Default: 0.1.
    #[arg(long)]
    decisiveness_proposal_std: Option<f64>,
}

#[derive(Parser)]
struct BenchmarkArgs {
    #[command(flatten)]
    cfg: ConfigArgs,

    /// Number of comparison pairs to run (each pair runs both directions)
    #[arg(short, long, default_value = DEFAULT_BENCHMARK_PAIRS)]
    num_pairs: usize,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Parser)]
struct RankArgs {
    #[command(flatten)]
    cfg: ConfigArgs,

    /// The comparison criterion (e.g. "Which is more rewatchable?")
    #[arg(long)]
    criterion: String,

    /// File with one item per line, or a directory of text files (each file = one item)
    #[arg(long)]
    items: Option<PathBuf>,

    /// Inline item (repeatable)
    #[arg(long = "item")]
    inline_items: Vec<String>,

    /// Output JSON instead of table
    #[arg(long)]
    json: bool,

    /// Show progress during execution
    #[arg(short, long)]
    verbose: bool,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Save a sample of comparisons to JSONL for inspection.
    /// Integer (e.g. 50) = exact count, float (e.g. 0.3) = fraction of total.
    #[arg(long)]
    save_comparisons: Option<String>,

    /// Output path for saved comparisons (default: comparisons.jsonl)
    #[arg(long)]
    save_comparisons_to: Option<PathBuf>,
}

const TITLE_MAX_LEN: usize = 20;

/// Derive a display title from item text: first 20 chars, hard cut.
fn title_from_text(text: &str) -> String {
    if text.chars().count() <= TITLE_MAX_LEN {
        text.to_string()
    } else {
        text.chars().take(TITLE_MAX_LEN).collect()
    }
}

/// Parse a string as either a JSON array of strings or plain text (one item per line).
fn parse_items_from_str(content: &str) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.starts_with('[') {
        // Try JSON array
        let items: Vec<String> = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| bail(format!("File looks like JSON but failed to parse: {e}")));
        items.into_iter().filter(|s| !s.trim().is_empty()).collect()
    } else {
        // Plain text, one item per line
        trimmed
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// Parse --save-comparisons value: float with '.' → fraction of total, integer → exact count.
fn parse_save_count(value: &str, total: usize) -> usize {
    if value.contains('.') {
        let frac: f64 = value.parse()
            .unwrap_or_else(|_| bail(format!("Invalid fraction for --save-comparisons: \"{value}\"")));
        if !(0.0..=1.0).contains(&frac) {
            bail(format!("--save-comparisons fraction must be between 0.0 and 1.0, got {frac}"));
        }
        (frac * total as f64).round() as usize
    } else {
        let count: usize = value.parse()
            .unwrap_or_else(|_| bail(format!("Invalid count for --save-comparisons: \"{value}\"")));
        count.min(total)
    }
}

/// Assign exact pair counts per judge based on weights, then shuffle the assignments.
/// Guarantees each judge gets exactly floor(total * weight) pairs, with remainders
/// distributed one each to judges in order of largest fractional part.
fn assign_pairs_to_judges(total_pairs: usize, normalized_weights: &[f64], rng: &mut impl rand::Rng) -> Vec<usize> {
    let num_judges = normalized_weights.len();

    // Compute exact fractional counts and floor counts
    let mut counts: Vec<usize> = Vec::with_capacity(num_judges);
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(num_judges);
    let mut assigned = 0usize;

    for (i, &w) in normalized_weights.iter().enumerate() {
        let exact = w * total_pairs as f64;
        let floor = exact.floor() as usize;
        counts.push(floor);
        remainders.push((i, exact - floor as f64));
        assigned += floor;
    }

    // Distribute leftover pairs to judges with the largest fractional remainders
    remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(judge_idx, _) in remainders.iter().take(total_pairs - assigned) {
        counts[judge_idx] += 1;
    }

    // Build assignment array and shuffle
    let mut assignments: Vec<usize> = Vec::with_capacity(total_pairs);
    for (judge_idx, &count) in counts.iter().enumerate() {
        assignments.extend(std::iter::repeat_n(judge_idx, count));
    }
    assignments.shuffle(rng);

    assignments
}

/// Plain text file extensions that we read from directories.
const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "html", "csv", "json", "xml", "rst", "log", "yaml", "yml", "toml",
];

/// Load items from all sources: --items file/dir, --item inline args, or stdin.
/// Returns (titles, texts) where titles are for display and texts are sent to the LLM.
fn load_items(args: &RankArgs) -> (Vec<String>, Vec<String>) {
    let mut titles = Vec::new();
    let mut texts = Vec::new();

    if let Some(ref path) = args.items {
        if path.is_dir() {
            // Directory mode: each file is an item
            let entries = std::fs::read_dir(path)
                .unwrap_or_else(|e| bail(format!("Failed to read directory {}: {e}", path.display())));

            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .collect();

            // Sort by filename for deterministic ordering
            files.sort_by_key(|e| e.file_name());

            let mut skipped = 0usize;
            let total = files.len();

            for entry in &files {
                let file_path = entry.path();
                let ext = file_path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");

                if !TEXT_EXTENSIONS.contains(&ext) {
                    skipped += 1;
                    continue;
                }

                let content = std::fs::read_to_string(&file_path)
                    .unwrap_or_else(|e| bail(format!("Failed to read {}: {e}", file_path.display())));
                let content = content.trim().to_string();

                if content.is_empty() {
                    skipped += 1;
                    continue;
                }

                let stem = file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unnamed")
                    .to_string();

                titles.push(stem);
                texts.push(content);
            }

            let loaded = titles.len();
            eprintln!("Found {total} files, loaded {loaded}, skipped {skipped} (unsupported format or empty)");
        } else {
            // File mode: one item per line or JSON array
            let content = std::fs::read_to_string(path)
                .unwrap_or_else(|e| bail(format!("Failed to read items file {}: {e}", path.display())));
            texts = parse_items_from_str(&content);
            titles = texts.iter().map(|t| title_from_text(t)).collect();
        }
    }

    // From inline --item flags
    for item in &args.inline_items {
        titles.push(title_from_text(item));
        texts.push(item.clone());
    }

    // From stdin (only if no file/dir and no inline items)
    if texts.is_empty() {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            bail("No items provided. Use --items <file|dir>, --item <name>, or pipe items via stdin.");
        }
        let content: String = stdin.lock().lines()
            .map(|l| l.expect("Failed to read from stdin"))
            .collect::<Vec<_>>()
            .join("\n");
        texts = parse_items_from_str(&content);
        titles = texts.iter().map(|t| title_from_text(t)).collect();
    }

    if texts.len() < 2 {
        bail(format!("Need at least 2 items to rank, got {}", texts.len()));
    }
    (titles, texts)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Rank(args) => run_rank(args).await,
        Commands::Benchmark(args) => run_benchmark_cmd(args).await,
        Commands::Init => {
            let path = config::create_default_config();
            println!("Created config at {}", path.display());
            println!("Edit it to set your default endpoint, model, etc.");
        }
    }
}

/// Resolved configuration — CLI args merged with config file values.
/// All required values are concrete (no Options except genuinely optional ones).
struct ResolvedConfig {
    rounds: Option<usize>,
    strategy: Strategy,
    top_k: Option<usize>,
    retries: usize,
    analysis_length: String,
    prompt_template: String,
    confidence_level: f64,
    regularization_strength: f64,
    mcmc_iterations: usize,
    mcmc_burn_in: usize,
    bias_prior_logit: f64,
    matchmaking_sharpness: f64,
    min_games_before_strategy: usize,
    prior_tau2: f64,
    sigma2: f64,
    proposal_std: f64,
    bias_prior_tau2: f64,
    bias_proposal_std: f64,
    decisiveness_prior_tau2: f64,
    decisiveness_proposal_std: f64,
}

/// A resolved judge — all fields concrete, ready to build LlmConfig.
struct ResolvedJudge {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    temperature: f64,
    temperature_jitter: f64,
    presence_penalty: Option<f64>,
    top_p: Option<f64>,
    logprobs: bool,
    concurrency: usize,
    weight: f64,
    narrow_win: f64,
    max_tokens: u32,
    reasoning_effort: Option<String>,
    judge_id: u64,
    display_name: String,
}

/// Resolve judges from [[judge]] blocks in the config file.
/// Errors if no [[judge]] blocks are defined.
fn resolve_judges(
    shared: &ConfigArgs,
    cfg: &config::NanojudgeConfig,
    config_path: &Path,
) -> Vec<ResolvedJudge> {
    let judge_configs = cfg.judge.as_ref().filter(|j| !j.is_empty())
        .unwrap_or_else(|| {
            bail(format!(
                "No [[judge]] blocks defined in {}. At least one judge is required.\n\
                 Example:\n\n  [[judge]]\n  endpoint = \"http://localhost:8000\"\n  model = \"my-model\"\n  temperature = 0.7",
                config_path.display()
            ));
        });

    // Validate: no duplicate endpoint+model
    let mut seen = HashSet::new();
    for jc in judge_configs {
        let key = format!("{}\0{}", jc.endpoint, jc.model);
        if !seen.insert(key) {
            bail(format!(
                "Duplicate judge: endpoint=\"{}\" model=\"{}\". Each judge must have a unique endpoint+model combination.",
                jc.endpoint, jc.model
            ));
        }
    }

    // logprobs is a global setting — all judges must be in the same mode
    let global_logprobs = if shared.logprobs {
        true
    } else {
        cfg.logprobs.unwrap_or(false)
    };

    let global_concurrency = cfg.concurrency.unwrap_or(DEFAULT_CONCURRENCY);

    // CLI --api-key or OPENAI_API_KEY env var
    let cli_api_key = shared.api_key.clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    // Resolve max_tokens: per-judge → average of specified judges → 2048
    let specified_max_tokens: Vec<u32> = judge_configs.iter()
        .filter_map(|jc| jc.max_tokens)
        .collect();
    let default_max_tokens = if specified_max_tokens.is_empty() {
        2048
    } else {
        let sum: u32 = specified_max_tokens.iter().sum();
        sum / specified_max_tokens.len() as u32
    };
    // Print message if some judges are missing max_tokens and we're using the average
    if !specified_max_tokens.is_empty() && specified_max_tokens.len() < judge_configs.len() {
        let missing: Vec<&str> = judge_configs.iter()
            .filter(|jc| jc.max_tokens.is_none())
            .map(|jc| jc.model.as_str())
            .collect();
        let specified: Vec<String> = judge_configs.iter()
            .filter_map(|jc| jc.max_tokens.map(|_| jc.model.clone()))
            .collect();
        eprintln!(
            "max_tokens not set for {}; using {} (average of {})",
            missing.join(", "), default_max_tokens, specified.join(", ")
        );
    }

    // Compute display names: model is default, disambiguate with endpoint host if models collide
    let mut model_counts: HashMap<String, usize> = HashMap::new();
    for jc in judge_configs {
        *model_counts.entry(jc.model.clone()).or_insert(0) += 1;
    }

    let mut judges: Vec<ResolvedJudge> = Vec::with_capacity(judge_configs.len());

    for jc in judge_configs {
        let temperature = jc.temperature
            .unwrap_or_else(|| {
                bail(format!(
                    "No temperature specified for judge {}. Set temperature in the [[judge]] block.",
                    jc.model,
                ));
            });

        let api_key = if let Some(ref env_name) = jc.api_key_env {
            std::env::var(env_name).ok()
        } else {
            cli_api_key.clone()
        };

        let display_name = if model_counts[&jc.model] > 1 {
            let host = jc.endpoint
                .trim_start_matches("http://")
                .trim_start_matches("https://")
                .split('/')
                .next()
                .unwrap_or(&jc.endpoint);
            format!("{} ({})", jc.model, host)
        } else {
            jc.model.clone()
        };

        let judge_id = stable_hash(&format!("{}\0{}", jc.endpoint, jc.model));
        let weight = jc.weight.unwrap_or(1.0);
        if !weight.is_finite() || weight <= 0.0 {
            bail(format!("Judge {} has non-positive weight {}. All weights must be > 0.", jc.model, weight));
        }

        let narrow_win = jc.narrow_win.unwrap_or(parse::DEFAULT_NARROW_WIN);
        if !narrow_win.is_finite() || narrow_win <= 0.5 || narrow_win >= 1.0 {
            bail(format!(
                "Judge {} has narrow_win={}, must be finite, > 0.5 and < 1.0",
                jc.model, narrow_win
            ));
        }

        judges.push(ResolvedJudge {
            endpoint: jc.endpoint.clone(),
            model: jc.model.clone(),
            api_key,
            temperature,
            temperature_jitter: jc.temperature_jitter.unwrap_or(DEFAULT_TEMPERATURE_JITTER),
            presence_penalty: jc.presence_penalty,
            top_p: jc.top_p,
            logprobs: global_logprobs,
            concurrency: jc.concurrency.unwrap_or(global_concurrency),
            weight,
            narrow_win,
            max_tokens: jc.max_tokens.unwrap_or(default_max_tokens),
            reasoning_effort: jc.reasoning_effort.clone(),
            judge_id,
            display_name,
        });
    }

    judges
}

/// Resolve CLI args + config file + defaults into final config.
/// Judge-specific settings (endpoint, model, temperature, etc.) are handled by resolve_judges().
fn resolve_config(shared: &ConfigArgs, cfg: &config::NanojudgeConfig) -> ResolvedConfig {
    let rounds = merge_opt(shared.rounds, cfg.rounds, "rounds");

    let strategy_str = merge_opt(shared.strategy.clone(), cfg.strategy.clone(), "strategy")
        .unwrap_or_else(|| "balanced".to_string());
    let strategy = match strategy_str.as_str() {
        "balanced" => Strategy::Balanced,
        "top-heavy" => Strategy::TopHeavy,
        other => bail(format!("Unknown strategy \"{other}\". Use \"balanced\" or \"top-heavy\".")),
    };

    let top_k = merge_opt(shared.top_k, cfg.top_k, "top-k");
    let retries = merge_opt(shared.retries, cfg.retries, "retries")
        .unwrap_or(DEFAULT_MAX_RETRIES);
    let analysis_length = merge_opt(shared.analysis_length.clone(), cfg.analysis_length.clone(), "analysis-length")
        .unwrap_or_else(|| DEFAULT_ANALYSIS_LENGTH.to_string());

    let confidence_level = merge_opt(shared.confidence_level, cfg.confidence_level, "confidence-level")
        .unwrap_or(0.95);
    let regularization_strength = merge_opt(shared.regularization_strength, cfg.regularization_strength, "regularization-strength")
        .unwrap_or(0.01);
    let mcmc_iterations = merge_opt(shared.mcmc_iterations, cfg.mcmc_iterations, "mcmc-iterations")
        .unwrap_or(2000);
    let mcmc_burn_in = merge_opt(shared.mcmc_burn_in, cfg.mcmc_burn_in, "mcmc-burn-in")
        .unwrap_or(500);
    let matchmaking_sharpness = merge_opt(shared.matchmaking_sharpness, cfg.matchmaking_sharpness, "matchmaking-sharpness")
        .unwrap_or(1.0);
    let min_games_before_strategy = merge_opt(shared.min_games, cfg.min_games, "min-games")
        .unwrap_or(3);
    let prior_tau2 = merge_opt(shared.prior_tau2, cfg.prior_tau2, "prior-tau2")
        .unwrap_or(10.0);
    let sigma2 = merge_opt(shared.sigma2, cfg.sigma2, "sigma2")
        .unwrap_or(1.0);
    let proposal_std = merge_opt(shared.proposal_std, cfg.proposal_std, "proposal-std")
        .unwrap_or(0.3);
    let bias_prior_tau2 = merge_opt(shared.bias_prior_tau2, cfg.bias_prior_tau2, "bias-prior-tau2")
        .unwrap_or(2.0);
    let bias_proposal_std = merge_opt(shared.bias_proposal_std, cfg.bias_proposal_std, "bias-proposal-std")
        .unwrap_or(0.15);
    let decisiveness_prior_tau2 = merge_opt(shared.decisiveness_prior_tau2, cfg.decisiveness_prior_tau2, "decisiveness-prior-tau2")
        .unwrap_or(1.0);
    let decisiveness_proposal_std = merge_opt(shared.decisiveness_proposal_std, cfg.decisiveness_proposal_std, "decisiveness-proposal-std")
        .unwrap_or(0.1);

    // bias_prior: user specifies in probability space, we convert to logit
    let bias_prior = merge_opt(shared.bias_prior, cfg.bias_prior, "bias-prior")
        .unwrap_or(0.5);
    if bias_prior <= 0.0 || bias_prior >= 1.0 {
        bail("--bias-prior must be greater than 0.0 and less than 1.0");
    }
    let bias_prior_logit = (bias_prior / (1.0 - bias_prior)).ln();

    // Prompt template: CLI path > config path > built-in default
    let prompt_template = {
        let cli_path = shared.prompt_template.clone();
        let cfg_path = cfg.prompt_template.as_ref().map(PathBuf::from);

        if let (Some(cp), Some(fp)) = (&cli_path, &cfg_path) {
            if cp != fp {
                eprintln!("Warning: --prompt-template ({}) overrides config file value ({})",
                    cp.display(), fp.display());
            }
        }

        let template_path = cli_path.or(cfg_path);
        match template_path {
            Some(path) => prompt::load_template(&path),
            None => prompt::DEFAULT_TEMPLATE.to_string(),
        }
    };

    ResolvedConfig {
        rounds,
        strategy,
        top_k,
        retries,
        analysis_length,
        prompt_template,
        confidence_level,
        regularization_strength,
        mcmc_iterations,
        mcmc_burn_in,
        bias_prior_logit,
        matchmaking_sharpness,
        min_games_before_strategy,
        prior_tau2,
        sigma2,
        proposal_std,
        bias_prior_tau2,
        bias_proposal_std,
        decisiveness_prior_tau2,
        decisiveness_proposal_std,
    }
}

async fn run_benchmark_cmd(args: BenchmarkArgs) {
    let config_path = args.config.clone().unwrap_or_else(config::config_path);
    let cfg = config::load_config(&config_path);
    let resolved = resolve_config(&args.cfg, &cfg);
    let judges = resolve_judges(&args.cfg, &cfg, &config_path);

    if args.num_pairs == 0 {
        bail("--num-pairs must be at least 1");
    }

    let template = &resolved.prompt_template;

    for judge in &judges {
        let llm_config = LlmConfig {
            endpoint: judge.endpoint.clone(),
            model: judge.model.clone(),
            api_key: judge.api_key.clone(),
            temperature: judge.temperature,
            temperature_jitter: judge.temperature_jitter,
            presence_penalty: judge.presence_penalty,
            top_p: judge.top_p,
            logprobs: judge.logprobs,
            max_tokens: judge.max_tokens,
            reasoning_effort: judge.reasoning_effort.clone(),
        };

        benchmark::run_benchmark(
            &llm_config,
            &judge.display_name,
            args.num_pairs,
            judge.concurrency,
            judge.narrow_win,
            template,
        ).await;

        if judges.len() > 1 {
            eprintln!();
        }
    }
}

async fn run_rank(args: RankArgs) {
    let config_path = args.config.clone().unwrap_or_else(config::config_path);
    let cfg = config::load_config(&config_path);
    let resolved = resolve_config(&args.cfg, &cfg);

    let rounds = resolved.rounds.unwrap_or_else(|| {
        bail(format!("No rounds specified. Pass --rounds or set it in {}", config_path.display()));
    });

    // Resolve judges from [[judge]] blocks
    let judges = resolve_judges(&args.cfg, &cfg, &config_path);
    let logprobs_mode = judges[0].logprobs;

    if !logprobs_mode {
        eprintln!("Warning: Running without logprobs. Requires more comparisons to reach equivalent accuracy as when using logprobs.");
    }

    let (titles, texts) = load_items(&args);
    let item_ids: Vec<i64> = (0..texts.len() as i64).collect();

    // Build JudgeInfo for the core engine
    let judge_ids: Vec<u64> = judges.iter().map(|j| j.judge_id).collect();
    let judge_info = JudgeInfo {
        judge_ids: judge_ids.clone(),
        logprobs_mode,
    };

    // Build per-judge LlmConfigs and semaphores
    let judge_llm_configs: Vec<Arc<LlmConfig>> = judges.iter().map(|j| {
        Arc::new(LlmConfig {
            endpoint: j.endpoint.clone(),
            model: j.model.clone(),
            api_key: j.api_key.clone(),
            temperature: j.temperature,
            temperature_jitter: j.temperature_jitter,
            presence_penalty: j.presence_penalty,
            top_p: j.top_p,
            logprobs: j.logprobs,
            max_tokens: j.max_tokens,
            reasoning_effort: j.reasoning_effort.clone(),
        })
    }).collect();

    let judge_semaphores: Vec<Arc<tokio::sync::Semaphore>> = judges.iter()
        .map(|j| Arc::new(tokio::sync::Semaphore::new(j.concurrency)))
        .collect();

    // Compute normalized weights for pair assignment
    let total_weight: f64 = judges.iter().map(|j| j.weight).sum();
    let normalized_weights: Vec<f64> = judges.iter().map(|j| j.weight / total_weight).collect();
    // Per-judge narrow_win values
    let judge_narrow_wins: Vec<f64> = judges.iter().map(|j| j.narrow_win).collect();

    let prompt_template = Arc::new(resolved.prompt_template.clone());

    let client = Client::new();
    let titles = Arc::new(titles);
    let texts = Arc::new(texts);

    let total_planned = calculate_total_expected_comparisons(texts.len(), rounds);

    if args.verbose {
        eprintln!(
            "Ranking {} items across {} rounds ({} comparisons planned)",
            texts.len(),
            rounds,
            total_planned,
        );
        eprintln!("Criterion: \"{}\"", args.criterion);

        if judges.len() == 1 {
            eprintln!("Endpoint: {} | Model: {}", judges[0].endpoint, judges[0].model);
        } else {
            eprintln!("Judge panel ({} judges):", judges.len());
            for j in &judges {
                eprintln!(
                    "  {} — {} (concurrency: {}, weight: {:.0}%)",
                    j.display_name,
                    j.endpoint,
                    j.concurrency,
                    j.weight / total_weight * 100.0,
                );
            }
        }
    }

    // Set up comparison saving if requested
    let save_file = if let Some(ref save_value) = args.save_comparisons {
        let save_count = parse_save_count(save_value, total_planned);
        let save_path = args.save_comparisons_to.clone()
            .unwrap_or_else(|| PathBuf::from("comparisons.jsonl"));

        let save_indices: HashSet<usize> = if save_count >= total_planned {
            (0..total_planned).collect()
        } else {
            use rand::seq::index::sample;
            let mut rng = rand::rng();
            sample(&mut rng, total_planned, save_count).into_iter().collect()
        };

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&save_path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", save_path.display())));

        if args.verbose {
            eprintln!("Saving {} comparisons to {}", save_count, save_path.display());
        }

        Some((std::sync::Mutex::new(file), save_indices))
    } else {
        None
    };

    let mut global_idx: usize = 0;

    let strategy = resolved.strategy;

    if resolved.top_k.is_some() && matches!(strategy, Strategy::Balanced) {
        eprintln!("Warning: --top-k has no effect with the balanced strategy. It only applies to --strategy top-heavy.");
    }

    // Pure heuristic — no empirical basis. Just a guess at how many top
    // positions users typically care about for a given list size.
    let top_k = resolved.top_k.unwrap_or_else(|| {
        ((texts.len() as f64).sqrt() * 3.0) as usize
    }).min(texts.len() - 1);

    let engine_config = EngineConfig {
        strategy,
        matchmaking_sharpness: resolved.matchmaking_sharpness,
        min_games_before_strategy: resolved.min_games_before_strategy,
        number_of_rounds: Some(rounds),
    };
    let mut engine = RankingEngine::new(&item_ids, engine_config);

    let analysis_length = resolved.analysis_length.clone();
    let max_retries = resolved.retries;

    // Judge display names (Arc for sharing across tasks)
    let judge_display_names: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.display_name.clone()).collect());
    let judge_models: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.model.clone()).collect());
    let judge_endpoints: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.endpoint.clone()).collect());

    let mut total_comparisons: usize = 0;
    let mut total_retries: usize = 0;
    let mut failed_http: usize = 0;
    let mut failed_parse: usize = 0;
    let mut judge_input_tokens: Vec<u64> = vec![0; judges.len()];
    let mut judge_output_tokens: Vec<u64> = vec![0; judges.len()];
    let mut judge_max_tokens_hits: Vec<usize> = vec![0; judges.len()];
    let mut judge_total_responses: Vec<usize> = vec![0; judges.len()];
    let mut judge_wall_time_sums: Vec<f64> = vec![0.0; judges.len()];
    let mut judge_round_counts: Vec<usize> = vec![0; judges.len()];

    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let cancelled = cancelled.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            cancelled.store(true, Ordering::Relaxed);
        });
    }

    let mut interim_warm_start: Option<nanojudge_core::WarmStartState> = None;
    for round in 0..rounds {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let pairs = engine.generate_pairs_for_round(round);
        let round_start = std::time::Instant::now();

        if args.verbose {
            eprintln!("Round {}/{}: {} pairs", round + 1, rounds, pairs.len());
        }

        // Assign exact counts per judge based on weights, then shuffle which pairs go where
        let mut rng = rand::rng();
        let pair_assignments = assign_pairs_to_judges(pairs.len(), &normalized_weights, &mut rng);

        let mut handles = Vec::with_capacity(pairs.len());

        for (pair_idx, (id_a, id_b)) in pairs.iter().enumerate() {
            let judge_idx = pair_assignments[pair_idx];
            let sem = judge_semaphores[judge_idx].clone();
            let client = client.clone();
            let llm_config = judge_llm_configs[judge_idx].clone();
            let texts = texts.clone();
            let criterion = args.criterion.clone();
            let analysis_length = analysis_length.clone();
            let template = prompt_template.clone();
            let id_a = *id_a;
            let id_b = *id_b;
            let narrow_win = judge_narrow_wins[judge_idx];
            let assigned_judge_id = judge_ids[judge_idx];
            let judge_name = judge_display_names[judge_idx].clone();

            let verbose = args.verbose;
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let result = compare_pair(
                    &client,
                    &llm_config,
                    &template,
                    &criterion,
                    &texts[id_a as usize],
                    &texts[id_b as usize],
                    id_a,
                    id_b,
                    narrow_win,
                    &analysis_length,
                    max_retries,
                    verbose,
                    &judge_name,
                )
                .await;
                (result, assigned_judge_id, judge_idx, std::time::Instant::now())
            });

            handles.push((handle, judge_idx));
        }

        // Collect results
        let mut round_results: Vec<ComparisonInput> = Vec::new();
        let mut judge_last_finish: Vec<Option<std::time::Instant>> = vec![None; judges.len()];
        let mut judge_aborted: Vec<usize> = vec![0; judges.len()];

        for (handle, handle_judge_idx) in handles {
            let this_idx = global_idx;
            global_idx += 1;
            if cancelled.load(Ordering::Relaxed) {
                handle.abort();
                judge_aborted[handle_judge_idx] += 1;
                continue;
            }
            let cancelled_ref = &cancelled;
            let result = tokio::select! {
                r = handle => r,
                _ = async { while !cancelled_ref.load(Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } } => {
                    judge_aborted[handle_judge_idx] += 1;
                    continue;
                }
            };
            match result {
                Ok((Ok(result), assigned_judge_id, judge_idx, finished_at)) => {
                    // Track latest finish time per judge for this round
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    total_retries += result.retries_used;
                    judge_total_responses[judge_idx] += 1;
                    if result.hit_max_tokens {
                        judge_max_tokens_hits[judge_idx] += 1;
                    }
                    if let Some(usage) = &result.usage {
                        judge_input_tokens[judge_idx] += usage.prompt_tokens;
                        judge_output_tokens[judge_idx] += usage.completion_tokens;
                    }
                    if let Some(p) = result.parse_result.item1_win_probability {
                        // Save to JSONL if this index was selected
                        if let Some((ref file_mutex, ref indices)) = save_file {
                            if indices.contains(&this_idx) {
                                let line = serde_json::json!({
                                    "round": round + 1,
                                    "item1": titles[result.item1_id as usize],
                                    "item2": titles[result.item2_id as usize],
                                    "probability": p,
                                    "judge_model": judge_models[judge_idx],
                                    "judge_endpoint": judge_endpoints[judge_idx],
                                    "response": result.response_text,
                                });
                                let mut f = file_mutex.lock().unwrap();
                                let _ = writeln!(f, "{}", line);
                                let _ = f.flush();
                            }
                        }

                        round_results.push(ComparisonInput {
                            item1: result.item1_id,
                            item2: result.item2_id,
                            item1_win_probability: p,
                            judge_id: assigned_judge_id,
                        });
                    } else {
                        failed_parse += 1;
                        if args.verbose {
                            if judges.len() > 1 {
                                eprintln!(
                                    "  Warning: unparseable response for {} vs {} [{}], skipping",
                                    titles[result.item1_id as usize],
                                    titles[result.item2_id as usize],
                                    judge_display_names[judge_idx],
                                );
                            } else {
                                eprintln!(
                                    "  Warning: unparseable response for {} vs {}, skipping",
                                    titles[result.item1_id as usize],
                                    titles[result.item2_id as usize],
                                );
                            }
                        }
                    }
                }
                Ok((Err(e), _judge_id, judge_idx, finished_at)) => {
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    failed_http += 1;
                    if args.verbose {
                        eprintln!(
                            "  Error [{}] (after exhausting {} retries): {e}",
                            judge_display_names[judge_idx], max_retries,
                        );
                    }
                }
                Err(e) => {
                    failed_http += 1;
                    if args.verbose {
                        eprintln!("  Task panicked: {e}");
                    }
                }
            }
        }

        if cancelled.load(Ordering::Relaxed) {
            // Print which judges had in-flight requests when cancelled
            for (i, judge) in judges.iter().enumerate() {
                if judge_aborted[i] > 0 {
                    eprintln!("  {} had {} in-flight requests", judge.display_name, judge_aborted[i]);
                }
            }
            break;
        }

        // Accumulate per-judge wall time for this round
        for (j, finish) in judge_last_finish.iter().enumerate() {
            if let Some(t) = finish {
                judge_wall_time_sums[j] += t.duration_since(round_start).as_secs_f64();
                judge_round_counts[j] += 1;
            }
        }

        total_comparisons += round_results.len();

        let round_failed = pairs.len() - round_results.len();
        if args.verbose {
            eprintln!(
                "  Completed: {} successful, {} failed",
                round_results.len(),
                round_failed,
            );
        }

        if round_failed == pairs.len() {
            eprintln!(
                "Warning: all {} comparisons in round {} failed. \
                 If your endpoint requires an API key, ensure it is set via \
                 --api-key or api_key_env in your config.",
                pairs.len(),
                round + 1,
            );
        }

        engine.record_results(&round_results);
        engine.update_current_ratings();

        // TopHeavy needs interim MCMC scoring to guide next round's pairing
        if matches!(strategy, Strategy::TopHeavy) && !engine.completed_comparisons.is_empty() {
            let interim = run_scoring(
                &item_ids,
                &engine.completed_comparisons,
                &ScoringOptions {
                    iterations: 200,
                    burn_in: if interim_warm_start.is_some() { 0 } else { 100 },
                    confidence_level: resolved.confidence_level,
                    top_k,
                    warm_start: interim_warm_start.take(),
                    regularization_strength: resolved.regularization_strength,
                    prior_tau2: resolved.prior_tau2,
                    sigma2: resolved.sigma2,
                    proposal_std: resolved.proposal_std,
                    bias_prior_tau2: resolved.bias_prior_tau2,
                    bias_proposal_std: resolved.bias_proposal_std,
                    bias_prior_logit: resolved.bias_prior_logit,
                    decisiveness_prior_tau2: resolved.decisiveness_prior_tau2,
                    decisiveness_proposal_std: resolved.decisiveness_proposal_std,
                },
                &judge_info,
            );
            engine.mcmc_top_k_probs = interim.top_k_probs;
            engine.mcmc_sample_means = interim.sample_means;
            interim_warm_start = Some(interim.warm_start_state);
        }
    }

    if cancelled.load(Ordering::Relaxed) {
        eprintln!("\nCancelled. {} comparisons completed before interrupt.", total_comparisons);
    }

    if total_comparisons == 0 {
        bail("All comparisons failed. No results to score.");
    }

    if args.verbose {
        eprintln!("Running final MCMC scoring ({total_comparisons} comparisons)...");
    }

    // Final scoring with full MCMC
    let scoring_result = run_scoring(
        &item_ids,
        &engine.completed_comparisons,
        &ScoringOptions {
            iterations: resolved.mcmc_iterations,
            burn_in: resolved.mcmc_burn_in,
            confidence_level: resolved.confidence_level,
            top_k: 0,
            warm_start: None,
            regularization_strength: resolved.regularization_strength,
            prior_tau2: resolved.prior_tau2,
            sigma2: resolved.sigma2,
            proposal_std: resolved.proposal_std,
            bias_prior_tau2: resolved.bias_prior_tau2,
            bias_proposal_std: resolved.bias_proposal_std,
            bias_prior_logit: resolved.bias_prior_logit,
            decisiveness_prior_tau2: resolved.decisiveness_prior_tau2,
            decisiveness_proposal_std: resolved.decisiveness_proposal_std,
        },
        &judge_info,
    );

    if args.verbose {
        if total_retries > 0 {
            eprintln!("HTTP retries: {total_retries}");
        }
        if failed_http > 0 {
            eprintln!("HTTP failures (after exhausting retries): {failed_http}");
        }
        if failed_parse > 0 {
            eprintln!("Unparseable responses: {failed_parse}");
        }
    }

    // Print max_tokens warnings (always, not just verbose)
    let mut any_max_tokens_hit = false;
    for (i, judge) in judges.iter().enumerate() {
        if judge_max_tokens_hits[i] > 0 {
            any_max_tokens_hit = true;
            eprintln!(
                "Warning: {} hit max_tokens on {}/{} responses.",
                judge.display_name, judge_max_tokens_hits[i], judge_total_responses[i],
            );
        }
    }
    if any_max_tokens_hit {
        eprintln!("Consider increasing max_tokens or adjusting the length instruction in the prompt.");
    }

    // Build judge_id → display_name and token count maps for output
    let judge_names: HashMap<u64, String> = judges.iter()
        .map(|j| (j.judge_id, j.display_name.clone()))
        .collect();
    let judge_tokens: HashMap<u64, (u64, u64)> = judges.iter().enumerate()
        .map(|(i, j)| (j.judge_id, (judge_input_tokens[i], judge_output_tokens[i])))
        .collect();
    let judge_avg_wall_time: HashMap<u64, f64> = judges.iter().enumerate()
        .map(|(i, j)| {
            let avg = if judge_round_counts[i] > 0 {
                judge_wall_time_sums[i] / judge_round_counts[i] as f64
            } else {
                0.0
            };
            (j.judge_id, avg)
        })
        .collect();

    if args.json {
        output::print_json(&scoring_result.rankings, &titles, rounds, total_comparisons, &scoring_result.judge_analytics);
    } else {
        output::print_table(
            &scoring_result.rankings,
            &titles,
            &engine.games_played,
            rounds,
            total_comparisons,
            resolved.confidence_level,
            &scoring_result.judge_analytics,
            &judge_names,
            &judge_tokens,
            &judge_avg_wall_time,
        );
    }
}
