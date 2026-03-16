/// Output formatting: terminal table and JSON.
use nanojudge_core::RankedItem;
use serde::Serialize;

#[derive(Serialize)]
struct JsonRankedItem {
    rank: usize,
    id: i64,
    name: String,
    score: f64,
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Serialize)]
struct JsonOutput {
    items: Vec<JsonRankedItem>,
    total_comparisons: usize,
    rounds: usize,
    positional_bias: f64,
    positional_bias_ci_low: f64,
    positional_bias_ci_high: f64,
}

/// Print results as a formatted terminal table.
pub fn print_table(rankings: &[RankedItem], names: &[String], games_played: &[usize], rounds: usize, total_comparisons: usize, positional_bias: f64, positional_bias_confidence_interval: (f64, f64)) {
    // Find the widest item name for padding
    let name_width = rankings.iter()
        .map(|r| names[r.item as usize].len())
        .max()
        .unwrap_or(4)
        .max(4); // at least "Item"

    // Header
    println!(" # | {:<name_width$} |   Score | 95% CI Low | 95% CI High | Comparisons | ID", "Item");
    println!("---|-{}-|---------|------------|-------------|-------------|----", "-".repeat(name_width));

    // Rows
    for (i, r) in rankings.iter().enumerate() {
        let name = &names[r.item as usize];
        let games = games_played[r.item as usize];
        println!(
            "{:>2} | {:<name_width$} | {:>7.4} | {:>10.2} | {:>11.2} | {:>11} | {:>2}",
            i + 1, name, r.score, r.lower_bound, r.upper_bound, games, r.item,
        );
    }

    println!(
        "\n{} items ranked across {} rounds ({} comparisons)",
        rankings.len(),
        rounds,
        total_comparisons,
    );
    println!(
        "Position bias — estimated: {:.3} [{:.3}, {:.3}] (corrected for in scores, 0.5 = no bias)",
        positional_bias, positional_bias_confidence_interval.0, positional_bias_confidence_interval.1,
    );
}

/// Build JSON output string.
fn build_json(rankings: &[RankedItem], names: &[String], rounds: usize, total_comparisons: usize, positional_bias: f64, positional_bias_confidence_interval: (f64, f64)) -> String {
    let items: Vec<JsonRankedItem> = rankings
        .iter()
        .enumerate()
        .map(|(i, r)| JsonRankedItem {
            rank: i + 1,
            id: r.item,
            name: names[r.item as usize].clone(),
            score: r.score,
            lower_bound: r.lower_bound,
            upper_bound: r.upper_bound,
        })
        .collect();

    let output = JsonOutput {
        items,
        total_comparisons,
        rounds,
        positional_bias,
        positional_bias_ci_low: positional_bias_confidence_interval.0,
        positional_bias_ci_high: positional_bias_confidence_interval.1,
    };

    serde_json::to_string_pretty(&output).unwrap()
}

/// Print results as JSON.
pub fn print_json(rankings: &[RankedItem], names: &[String], rounds: usize, total_comparisons: usize, positional_bias: f64, positional_bias_confidence_interval: (f64, f64)) {
    println!("{}", build_json(rankings, names, rounds, total_comparisons, positional_bias, positional_bias_confidence_interval));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rankings() -> (Vec<RankedItem>, Vec<String>) {
        let rankings = vec![
            RankedItem { item: 2, score: 1.58, lower_bound: 1.20, upper_bound: 1.97 },
            RankedItem { item: 0, score: 0.75, lower_bound: 0.45, upper_bound: 1.05 },
            RankedItem { item: 1, score: 0.42, lower_bound: 0.12, upper_bound: 0.68 },
        ];
        let names = vec!["Apple".to_string(), "Banana".to_string(), "Mango".to_string()];
        (rankings, names)
    }

    #[test]
    fn test_json_contains_all_fields() {
        let (rankings, names) = sample_rankings();
        let json = build_json(&rankings, &names, 10, 30, 0.523, (0.481, 0.567));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["rounds"], 10);
        assert_eq!(parsed["total_comparisons"], 30);
        assert_eq!(parsed["positional_bias"], 0.523);
        assert_eq!(parsed["positional_bias_ci_low"], 0.481);
        assert_eq!(parsed["positional_bias_ci_high"], 0.567);
    }

    #[test]
    fn test_json_items_structure() {
        let (rankings, names) = sample_rankings();
        let json = build_json(&rankings, &names, 10, 30, 0.5, (0.48, 0.52));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        let items = parsed["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);

        // First item should be rank 1, id 2 (Mango)
        assert_eq!(items[0]["rank"], 1);
        assert_eq!(items[0]["id"], 2);
        assert_eq!(items[0]["name"], "Mango");
        assert_eq!(items[0]["score"], 1.58);
        assert_eq!(items[0]["lower_bound"], 1.20);
        assert_eq!(items[0]["upper_bound"], 1.97);

        // Last item should be rank 3
        assert_eq!(items[2]["rank"], 3);
        assert_eq!(items[2]["name"], "Banana");
    }

    #[test]
    fn test_json_is_valid() {
        let (rankings, names) = sample_rankings();
        let json = build_json(&rankings, &names, 5, 15, 0.5, (0.5, 0.5));
        // Should parse without error
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }
}
