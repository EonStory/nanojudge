/// Prompt building for pairwise comparisons.
///
/// Replicates the format from fleet-worker.py `build_prompt()`.

/// Build a comparison prompt for two items.
///
/// The prompt asks the LLM to analyze and then give a verdict on a 5-point
/// likert scale (A-E). This format is designed to produce a "Verdict:" marker
/// followed by a single letter, which we can extract probabilities from via
/// logprobs.
pub fn build_prompt(criterion: &str, option1: &str, option2: &str, analysis_length: &str) -> String {
    format!(
        "{criterion}\n\n\
         Option 1:\n{option1}\n\n\
         Option 2:\n{option2}\n\n\
         Instructions:\n\
         Conduct an analysis. Write roughly {analysis_length}. \
         Then write \"Verdict:\" on its own line, followed by exactly one of \
         these letters and its label:\n\n\
         A: Option 1 clearly wins\n\
         B: Option 1 narrowly wins\n\
         C: Draw\n\
         D: Option 2 narrowly wins\n\
         E: Option 2 clearly wins\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_contains_all_parts() {
        let prompt = build_prompt("Which is tastier?", "Pizza", "Sushi", "2 paragraphs");
        assert!(prompt.starts_with("Which is tastier?"));
        assert!(prompt.contains("Option 1:\nPizza"));
        assert!(prompt.contains("Option 2:\nSushi"));
        assert!(prompt.contains("Verdict:"));
        assert!(prompt.contains("A: Option 1 clearly wins"));
        assert!(prompt.contains("E: Option 2 clearly wins"));
    }
}
