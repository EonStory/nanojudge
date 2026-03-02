/// OpenAI-compatible API client for pairwise comparisons.
use crate::parse::{LogprobContent, ParseResult, parse_response};
use crate::prompt::build_prompt;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Configuration for the LLM endpoint.
pub struct LlmConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: f64,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: u32,
    logprobs: bool,
    top_logprobs: u8,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageContent,
    logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChoiceLogprobs {
    content: Option<Vec<LogprobContent>>,
}

/// Result of a single LLM comparison call.
pub struct ComparisonResult {
    pub item1_id: i64,
    pub item2_id: i64,
    pub parse_result: ParseResult,
    pub response_text: String,
    pub retries_used: usize,
}

/// Apply normal jitter to temperature: N(1.0, 0.1) clamped to [0.8, 1.2].
/// Uses Box-Muller transform to avoid an extra crate dependency.
fn jittered_temperature(base: f64) -> f64 {
    let mut rng = rand::rng();
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let multiplier = (1.0 + 0.1 * z).clamp(0.8, 1.2);
    base * multiplier
}

/// Send one HTTP request to the LLM and parse the response.
/// Returns Ok on any successful HTTP response (even if verdict is unparseable).
/// Returns Err only on HTTP/network failures.
async fn send_comparison_request(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
    narrow_win: f64,
) -> Result<(ParseResult, String), String> {
    let request = ChatCompletionRequest {
        model: config.model.clone(),
        messages: vec![ChatMessage {
            role: "user",
            content: prompt.to_string(),
        }],
        temperature: jittered_temperature(config.temperature),
        max_tokens: 4000,
        logprobs: true,
        top_logprobs: 10,
    };

    let url = format!("{}/v1/chat/completions", config.endpoint.trim_end_matches('/'));

    let mut req_builder = client.post(&url).json(&request);
    if let Some(ref key) = config.api_key {
        req_builder = req_builder.bearer_auth(key);
    }

    let resp = req_builder.send().await.map_err(|e| format!("HTTP request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("LLM API returned {status}: {}", &body[..body.len().min(200)]));
    }

    let data: ChatCompletionResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse LLM response JSON: {e}"))?;

    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or("No choices in LLM response")?;

    let content = choice.message.content.unwrap_or_default();
    let logprobs = choice
        .logprobs
        .and_then(|lp| lp.content)
        .unwrap_or_default();

    let parse_result = parse_response(&content, &logprobs, narrow_win);
    Ok((parse_result, content))
}

/// Call the LLM to compare two items, with retries on HTTP errors.
///
/// Retries up to `max_retries` times with a 1-second delay between attempts.
/// Only HTTP/network errors trigger retries — unparseable verdicts do not.
pub async fn compare_pair(
    client: &Client,
    config: &LlmConfig,
    template: &str,
    criterion: &str,
    item1_name: &str,
    item2_name: &str,
    item1_id: i64,
    item2_id: i64,
    narrow_win: f64,
    analysis_length: &str,
    max_retries: usize,
    verbose: bool,
) -> Result<ComparisonResult, String> {
    let prompt = build_prompt(template, criterion, item1_name, item2_name, analysis_length);

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        match send_comparison_request(client, config, &prompt, narrow_win).await {
            Ok((parse_result, content)) => {
                return Ok(ComparisonResult {
                    item1_id,
                    item2_id,
                    parse_result,
                    response_text: content,
                    retries_used: attempt,
                });
            }
            Err(e) => {
                last_err = e;
                if attempt < max_retries {
                    if verbose {
                        eprintln!(
                            "  Retry {}/{} for {} vs {}: {}",
                            attempt + 1, max_retries, item1_name, item2_name, last_err
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    }

    Err(last_err)
}
