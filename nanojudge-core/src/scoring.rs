/// Unified MCMC scoring wrapper.
///
/// One function, one options struct. Pure function — no IO, no state.
/// Items are identified by caller-provided `i64` IDs.
use std::collections::HashMap;

use crate::gaussian_bt::GaussianBT;
use crate::types::{
    ComparisonInput, IdMap, JudgeAnalytics, JudgeInfo, ScoringOptions, ScoringResult,
    WarmStartState,
};

/// Compute confidence interval from sorted samples.
fn ci_from_sorted(samples: &[f64], confidence_level: f64) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let alpha = 1.0 - confidence_level;
    let n = samples.len();
    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = upper_idx.saturating_sub(1).max(lower_idx);
    (samples[lower_idx], samples[upper_idx])
}

/// Convert logit-space samples to probability-space CI.
fn logit_to_prob_ci(sorted_logit_samples: &[f64], mean_logit: f64, confidence_level: f64) -> (f64, (f64, f64)) {
    let prob = 1.0 / (1.0 + (-mean_logit).exp());
    let (lower_logit, upper_logit) = ci_from_sorted(sorted_logit_samples, confidence_level);
    let lower = 1.0 / (1.0 + (-lower_logit).exp());
    let upper = 1.0 / (1.0 + (-upper_logit).exp());
    (prob, (lower, upper))
}

/// Run MCMC scoring on pairwise comparison data.
///
/// `item_ids` is the full list of item IDs being ranked. The returned state,
/// `top_k_probs`, and `sample_means` are in the same order as `item_ids`.
pub fn run_scoring(
    item_ids: &[i64],
    comparisons: &[ComparisonInput],
    options: &ScoringOptions,
    judge_info: &JudgeInfo,
) -> ScoringResult {
    let id_map = IdMap::from_ids(item_ids);
    let num_items = id_map.len();

    // Build judge_id -> internal index mapping
    let mut judge_id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(judge_info.judge_ids.len());
    for (idx, &id) in judge_info.judge_ids.iter().enumerate() {
        judge_id_to_idx.insert(id, idx);
    }

    let indexed = id_map.convert_comparisons(comparisons, &judge_id_to_idx);

    let mut mcmc = GaussianBT::new(
        num_items,
        &indexed,
        options,
        judge_info,
    );

    let samples_result = if let Some(ref warm_start) = options.warm_start {
        assert_eq!(
            warm_start.item_strengths.len(), num_items,
            "warm_start item_strengths length ({}) must match num_items ({})",
            warm_start.item_strengths.len(), num_items
        );
        mcmc.calculate_incremental_with_samples(
            &warm_start.item_strengths,
            &warm_start.judge_biases,
            &warm_start.judge_log_decisiveness,
            &judge_id_to_idx,
            options.iterations,
            options.burn_in,
            options.top_k,
        )
    } else {
        mcmc.calculate_with_samples(options.iterations, options.burn_in, options.top_k)
    };

    // Compute confidence intervals; returned items use index-as-i64, map back to real IDs
    let mut rankings = GaussianBT::compute_confidence_intervals_from_sorted_samples(
        &samples_result.sorted_samples,
        &samples_result.means,
        options.confidence_level,
    );

    for r in &mut rankings {
        r.item = id_map.to_id(r.item as usize);
    }

    // Build per-judge analytics
    let mut judge_analytics = Vec::with_capacity(judge_info.judge_ids.len());
    for (j, &judge_id) in judge_info.judge_ids.iter().enumerate() {
        let (bias_prob, bias_ci) = logit_to_prob_ci(
            &samples_result.bias_logit_samples[j],
            samples_result.bias_logit_means[j],
            options.confidence_level,
        );

        let (decisiveness, decisiveness_ci) = if judge_info.logprobs_mode {
            let log_d_samples = &samples_result.log_decisiveness_samples[j];
            let mean_d = samples_result.log_decisiveness_means[j].exp();
            let (lower_log, upper_log) = ci_from_sorted(log_d_samples, options.confidence_level);
            (Some(mean_d), Some((lower_log.exp(), upper_log.exp())))
        } else {
            (None, None)
        };

        judge_analytics.push(JudgeAnalytics {
            judge_id,
            positional_bias: bias_prob,
            positional_bias_ci: bias_ci,
            decisiveness,
            decisiveness_ci,
            num_comparisons: samples_result.comparisons_per_judge[j],
        });
    }

    // Build warm start state
    let warm_start_state = WarmStartState {
        item_strengths: mcmc.get_current_state(),
        judge_biases: mcmc.get_current_biases(judge_info),
        judge_log_decisiveness: mcmc.get_current_log_decisiveness(judge_info),
    };

    ScoringResult {
        rankings,
        top_k_probs: if options.top_k > 0 { samples_result.top_k_probs } else { None },
        sample_means: if options.top_k > 0 { Some(samples_result.means) } else { None },
        warm_start_state,
        sample_size: options.iterations,
        judge_analytics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_judge_info() -> JudgeInfo {
        JudgeInfo {
            judge_ids: vec![42],
            logprobs_mode: true,
        }
    }

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(id1: i64, id2: i64, prob: f64) -> [ComparisonInput; 2] {
        [
            ComparisonInput { item1: id1, item2: id2, item1_win_probability: prob, judge_id: 42 },
            ComparisonInput { item1: id2, item2: id1, item1_win_probability: 1.0 - prob, judge_id: 42 },
        ]
    }

    fn default_scoring_options() -> ScoringOptions {
        ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            sigma2: 1.0,
            proposal_std: 0.3,
            bias_prior_tau2: 2.0,
            bias_proposal_std: 0.15,
            bias_prior_logit: 0.0,
            decisiveness_prior_tau2: 1.0,
            decisiveness_proposal_std: 0.1,
        }
    }

    #[test]
    fn test_cold_start_scoring() {
        let item_ids = vec![100, 200, 300];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(100, 300, 0.8),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].item, 100);
        assert!(result.top_k_probs.is_none());
        assert_eq!(result.warm_start_state.item_strengths.len(), 3);
        assert_eq!(result.sample_size, 200);
        assert_eq!(result.judge_analytics.len(), 1);
        assert_eq!(result.judge_analytics[0].judge_id, 42);
        assert!(result.judge_analytics[0].decisiveness.is_some());
    }

    #[test]
    fn test_warm_start_scoring() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
            make_pair(20, 30, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result1 = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let mut opts2 = default_scoring_options();
        opts2.warm_start = Some(result1.warm_start_state);
        opts2.burn_in = 0;

        let result2 = run_scoring(&item_ids, &comparisons, &opts2, &ji);

        assert_eq!(result2.rankings.len(), 3);
        assert_eq!(result2.warm_start_state.item_strengths.len(), 3);
    }

    #[test]
    #[should_panic(expected = "warm_start item_strengths length (2) must match num_items (3)")]
    fn test_warm_start_wrong_length_panics() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let mut opts = default_scoring_options();
        opts.warm_start = Some(WarmStartState {
            item_strengths: vec![1.0, 1.0], // Wrong length: 2 instead of 3
            judge_biases: vec![],
            judge_log_decisiveness: vec![],
        });

        run_scoring(&item_ids, &comparisons, &opts, &ji);
    }

    #[test]
    fn test_scoring_with_top_k() {
        let item_ids = vec![1, 2, 3, 4];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(1, 2, 0.9),
            make_pair(1, 3, 0.85),
            make_pair(1, 4, 0.9),
            make_pair(2, 3, 0.7),
            make_pair(2, 4, 0.75),
            make_pair(3, 4, 0.6),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let mut opts = default_scoring_options();
        opts.top_k = 2;

        let result = run_scoring(&item_ids, &comparisons, &opts, &ji);

        assert!(result.top_k_probs.is_some());
        assert_eq!(result.top_k_probs.as_ref().unwrap().len(), 4);
        assert!(result.sample_means.is_some());
    }

    #[test]
    fn test_scoring_with_arbitrary_ids() {
        let item_ids = vec![999, 42, 7777];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(999, 42, 0.8),
            make_pair(42, 7777, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let ranked_ids: Vec<i64> = result.rankings.iter().map(|r| r.item).collect();
        assert!(ranked_ids.contains(&999));
        assert!(ranked_ids.contains(&42));
        assert!(ranked_ids.contains(&7777));
    }

    #[test]
    #[should_panic(expected = "Unknown item ID")]
    fn test_scoring_unknown_id_panics() {
        let item_ids = vec![1, 2, 3];
        let comparisons = vec![
            ComparisonInput { item1: 1, item2: 99, item1_win_probability: 0.8, judge_id: 42 },
        ];

        let ji = single_judge_info();
        run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);
    }

    #[test]
    #[should_panic(expected = "Duplicate item ID")]
    fn test_scoring_duplicate_ids_panics() {
        let item_ids = vec![1, 2, 1];

        let ji = single_judge_info();
        run_scoring(&item_ids, &[], &default_scoring_options(), &ji);
    }

    #[test]
    fn test_multi_judge_scoring() {
        let item_ids = vec![100, 200, 300];
        let judge_a = 111;
        let judge_b = 222;

        let comparisons = vec![
            ComparisonInput { item1: 100, item2: 200, item1_win_probability: 0.9, judge_id: judge_a },
            ComparisonInput { item1: 200, item2: 100, item1_win_probability: 0.1, judge_id: judge_a },
            ComparisonInput { item1: 100, item2: 300, item1_win_probability: 0.8, judge_id: judge_b },
            ComparisonInput { item1: 300, item2: 100, item1_win_probability: 0.2, judge_id: judge_b },
            ComparisonInput { item1: 200, item2: 300, item1_win_probability: 0.7, judge_id: judge_a },
            ComparisonInput { item1: 300, item2: 200, item1_win_probability: 0.3, judge_id: judge_b },
        ];

        let ji = JudgeInfo {
            judge_ids: vec![judge_a, judge_b],
            logprobs_mode: true,
        };

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.judge_analytics.len(), 2);
        assert_eq!(result.judge_analytics[0].judge_id, judge_a);
        assert_eq!(result.judge_analytics[1].judge_id, judge_b);
        assert!(result.judge_analytics[0].decisiveness.is_some());
        assert!(result.judge_analytics[1].decisiveness.is_some());
        assert_eq!(result.judge_analytics[0].num_comparisons + result.judge_analytics[1].num_comparisons, 6);
    }

    #[test]
    fn test_no_logprobs_scoring() {
        let item_ids = vec![100, 200, 300];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = JudgeInfo {
            judge_ids: vec![42],
            logprobs_mode: false,
        };

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.judge_analytics.len(), 1);
        assert!(result.judge_analytics[0].decisiveness.is_none());
        assert!(result.judge_analytics[0].decisiveness_ci.is_none());
    }

    /// Generate ground-truth BT strengths and comparisons.
    fn generate_ground_truth(n: usize, seed: u64) -> (Vec<i64>, Vec<f64>, Vec<ComparisonInput>) {
        use rand::{SeedableRng, Rng, rngs::SmallRng};

        let item_ids: Vec<i64> = (1..=n as i64).collect();
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut true_strengths = Vec::with_capacity(n);
        for _ in 0..n {
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            true_strengths.push(2.0_f64.powf(z));
        }

        let log_mean = true_strengths.iter().map(|s| s.ln()).sum::<f64>() / n as f64;
        for s in &mut true_strengths {
            *s /= log_mean.exp();
        }

        let mut comparisons = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let prob = true_strengths[i] / (true_strengths[i] + true_strengths[j]);
                comparisons.extend(make_pair(item_ids[i], item_ids[j], prob));
            }
        }

        (item_ids, true_strengths, comparisons)
    }

    fn log_rmse(rankings: &[crate::types::RankedItem], true_by_id: &std::collections::HashMap<i64, f64>) -> f64 {
        let n = rankings.len();
        let sum_sq: f64 = rankings.iter().map(|r| {
            let diff = true_by_id[&r.item].ln() - r.score.ln();
            diff * diff
        }).sum();
        (sum_sq / n as f64).sqrt()
    }

    #[test]
    fn test_bt_warm_start_vs_cold_start_accuracy() {
        let num_trials = 50;
        let n = 10;
        let iterations = 2000;

        let ji = single_judge_info();

        let mut cold_rmses = Vec::with_capacity(num_trials);
        let mut warm50_rmses = Vec::with_capacity(num_trials);
        let mut warm0_rmses = Vec::with_capacity(num_trials);

        for trial in 0..num_trials {
            let seed = 1000 + trial as u64 * 7;
            let (item_ids, true_strengths, comparisons) = generate_ground_truth(n, seed);

            let true_by_id: std::collections::HashMap<i64, f64> = item_ids.iter()
                .zip(true_strengths.iter())
                .map(|(&id, &s)| (id, s))
                .collect();

            let mut opts = default_scoring_options();
            opts.iterations = iterations;
            opts.burn_in = 500;

            let cold = run_scoring(&item_ids, &comparisons, &opts, &ji);

            let id_map = IdMap::from_ids(&item_ids);
            let judge_id_to_idx: HashMap<u64, usize> = ji.judge_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let indexed = id_map.convert_comparisons(&comparisons, &judge_id_to_idx);
            // Strip judge_idx for BT MLE (it only uses first 3 tuple elements)
            let mut bt = crate::bradley_terry::BradleyTerry::new(n, &indexed, 0.01);
            bt.calculate_scores(30);
            let bt_scores: Vec<f64> = (0..n).map(|i| bt.get_score(i)).collect();

            let mut opts_warm50 = default_scoring_options();
            opts_warm50.iterations = iterations;
            opts_warm50.burn_in = 50;
            opts_warm50.warm_start = Some(WarmStartState {
                item_strengths: bt_scores.clone(),
                judge_biases: vec![],
                judge_log_decisiveness: vec![],
            });

            let warm50 = run_scoring(&item_ids, &comparisons, &opts_warm50, &ji);

            let mut opts_warm0 = default_scoring_options();
            opts_warm0.iterations = iterations;
            opts_warm0.burn_in = 0;
            opts_warm0.warm_start = Some(WarmStartState {
                item_strengths: bt_scores,
                judge_biases: vec![],
                judge_log_decisiveness: vec![],
            });

            let warm0 = run_scoring(&item_ids, &comparisons, &opts_warm0, &ji);

            cold_rmses.push(log_rmse(&cold.rankings, &true_by_id));
            warm50_rmses.push(log_rmse(&warm50.rankings, &true_by_id));
            warm0_rmses.push(log_rmse(&warm0.rankings, &true_by_id));
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let cold_mean = mean(&cold_rmses);
        let warm50_mean = mean(&warm50_rmses);
        let warm0_mean = mean(&warm0_rmses);

        eprintln!("\n=== BT warm-start accuracy benchmark ({num_trials} trials, {n} items, {iterations} MCMC iter) ===");
        eprintln!("Cold   (500 burn-in): mean RMSE = {cold_mean:.4}");
        eprintln!("Warm50  (50 burn-in): mean RMSE = {warm50_mean:.4}");
        eprintln!("Warm0    (0 burn-in): mean RMSE = {warm0_mean:.4}");

        assert!(cold_mean < 0.2, "Cold mean RMSE {cold_mean:.4} too high");
        assert!(warm50_mean < 0.2, "Warm50 mean RMSE {warm50_mean:.4} too high");
        assert!(warm0_mean < 0.2, "Warm0 mean RMSE {warm0_mean:.4} too high");
    }
}
