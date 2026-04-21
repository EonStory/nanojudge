/// Pairing strategies for pairwise comparison tournaments.
///
/// Public functions accept `item_ids: &[i64]` and return `Pair` (i64, i64).
/// Internal functions use `usize` indices for efficient array indexing.
use rand::Rng;

use crate::constants::OPPONENT_WINDOW_SIZE;
use crate::engine::calculate_pairs_for_round;
use crate::types::{IndexedPair, Pair};

/// Pairing strategy enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Strategy {
    Balanced,
    TopHeavy,
}

/// Calculate information gain for a matchup between two items.
pub fn calculate_info_gain(rating_a: f64, rating_b: f64, sharpness: f64) -> f64 {
    let p = 1.0 / (1.0 + (rating_b - rating_a).exp());
    let info_gain = p * (1.0 - p);
    info_gain.powf(sharpness)
}

/// Determine the effective strategy to use for this round.
///
/// Three stages:
///   Stage 1 (Bootstrap): Balanced until every item has >= min_games.
///   Stage 2 (P(top K)): Top-heavy — the main phase.
///   Stage 3 (Smoothing): Last round reverts to balanced.
pub fn get_effective_strategy(
    user_strategy: Strategy,
    num_items: usize,
    games_played: &[usize],
    current_round_number: usize,
    min_games_before_strategy: usize,
    number_of_rounds: Option<usize>,
) -> Strategy {
    if user_strategy == Strategy::Balanced {
        return Strategy::Balanced;
    }

    // Stage 1: Bootstrap
    for i in 0..num_items {
        if games_played[i] < min_games_before_strategy {
            return Strategy::Balanced;
        }
    }

    // Stage 3: Smoothing — last round
    if let Some(total_rounds) = number_of_rounds {
        if current_round_number >= total_rounds - 1 {
            return Strategy::Balanced;
        }
    }

    // Stage 2
    Strategy::TopHeavy
}

// ---------------------------------------------------------------------------
// Public pairing functions (work with i64 IDs)
// ---------------------------------------------------------------------------

/// Generate balanced pairings for a round.
///
/// `current_ratings[i]` is the rating for `item_ids[i]`.
/// Returns pairs of item IDs.
pub fn generate_balanced_pairings(
    item_ids: &[i64],
    round_number: usize,
    current_ratings: &[f64],
    sharpness: f64,
) -> Vec<Pair> {
    let index_pairs = generate_balanced_pairings_indexed(
        item_ids.len(),
        round_number,
        current_ratings,
        sharpness,
    );
    index_pairs.into_iter().map(|(a, b)| (item_ids[a], item_ids[b])).collect()
}

/// Generate top-heavy pairings for a round.
///
/// `top_k_probs[i]` and `sample_means[i]` correspond to `item_ids[i]`.
/// Returns pairs of item IDs.
pub fn generate_top_heavy_pairings(
    item_ids: &[i64],
    round_number: usize,
    top_k_probs: &[f64],
    sample_means: &[f64],
    sharpness: f64,
) -> Vec<Pair> {
    let index_pairs = generate_top_heavy_pairings_indexed(
        item_ids.len(),
        round_number,
        top_k_probs,
        sample_means,
        sharpness,
    );
    index_pairs.into_iter().map(|(a, b)| (item_ids[a], item_ids[b])).collect()
}

// ---------------------------------------------------------------------------
// Internal indexed pairing functions (work with usize indices)
// ---------------------------------------------------------------------------

pub(crate) fn generate_balanced_pairings_indexed(
    num_items: usize,
    round_number: usize,
    current_ratings: &[f64],
    sharpness: f64,
) -> Vec<IndexedPair> {
    let mut rng = rand::rng();
    let pairs_target = calculate_pairs_for_round(num_items, round_number + 1);

    let mut pairings: Vec<IndexedPair> = Vec::with_capacity(pairs_target);

    let items_per_iteration = num_items / 2;
    if items_per_iteration == 0 {
        return pairings;
    }
    let full_iterations = pairs_target / items_per_iteration;
    let remaining_pairs = pairs_target % items_per_iteration;

    for _ in 0..full_iterations {
        generate_balanced_iteration(num_items, current_ratings, sharpness, items_per_iteration, &mut pairings, &mut rng);
    }

    if remaining_pairs > 0 {
        generate_balanced_iteration(num_items, current_ratings, sharpness, remaining_pairs, &mut pairings, &mut rng);
    }

    pairings
}

fn generate_balanced_iteration(
    num_items: usize,
    current_ratings: &[f64],
    sharpness: f64,
    max_pairs: usize,
    pairings: &mut Vec<IndexedPair>,
    rng: &mut impl Rng,
) {
    // Sort items by rating ascending so we can pick opponents from a narrow
    // rating-window around each item1. sorted_pool is immutable for the rest
    // of the function; "removal" is done via tombstones so the remaining
    // entries keep their sorted positions (and the window math stays valid).
    let mut sorted_pool: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, current_ratings[i]))
        .collect();
    sorted_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_pool.len();
    // alive[p] = false means sorted_pool[p] has already been paired this iteration.
    let mut alive = vec![true; n];
    // live_positions holds the sorted-pool indices still available. Picking
    // item1 is a swap_remove on this list (O(1) random removal). live_idx_of
    // is the inverse map: live_idx_of[p] = index of p inside live_positions,
    // or None if tombstoned. It lets us also O(1)-remove item2 once we know
    // its sorted-pool position.
    let mut live_positions: Vec<usize> = (0..n).collect();
    let mut live_idx_of: Vec<Option<usize>> = (0..n).map(Some).collect();

    let half_w = OPPONENT_WINDOW_SIZE / 2;
    let mut weights: Vec<f64> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 1);
    let mut candidates: Vec<usize> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 1);

    for _ in 0..max_pairs {
        if live_positions.len() < 2 {
            break;
        }

        // Pick item1: random live entry, removed via swap_remove.
        let live_idx1 = rng.random_range(0..live_positions.len());
        let pos1 = swap_remove_live(&mut live_positions, &mut live_idx_of, live_idx1);
        alive[pos1] = false;
        let (item1, item1_rating) = sorted_pool[pos1];

        // Collect live candidates in the rating window around pos1.
        let window_start = pos1.saturating_sub(half_w);
        let window_end = (pos1 + half_w + 1).min(n);
        candidates.clear();
        weights.clear();
        for p in window_start..window_end {
            if alive[p] {
                candidates.push(p);
                weights.push(calculate_info_gain(item1_rating, sorted_pool[p].1, sharpness));
            }
        }
        if candidates.is_empty() {
            // Window exhausted — skip this item1 for the rest of this iteration.
            continue;
        }

        let total_weight: f64 = weights.iter().sum();
        let selected = if total_weight == 0.0 {
            rng.random_range(0..candidates.len())
        } else {
            weighted_random_select(&weights, total_weight, rng)
        };

        let pos2 = candidates[selected];
        let live_idx2 = live_idx_of[pos2].expect("candidate must be alive");
        swap_remove_live(&mut live_positions, &mut live_idx_of, live_idx2);
        alive[pos2] = false;
        let (item2, _) = sorted_pool[pos2];

        if rng.random::<f64>() < 0.5 {
            pairings.push((item1, item2));
        } else {
            pairings.push((item2, item1));
        }
    }
}

/// Remove the entry at `live_idx` from `live_positions` in O(1) using
/// swap_remove, keeping `live_idx_of` in sync. Returns the sorted-pool
/// position that was removed.
fn swap_remove_live(
    live_positions: &mut Vec<usize>,
    live_idx_of: &mut [Option<usize>],
    live_idx: usize,
) -> usize {
    let removed_pos = live_positions.swap_remove(live_idx);
    live_idx_of[removed_pos] = None;
    // If we didn't remove the last entry, the old last entry now sits at live_idx.
    if live_idx < live_positions.len() {
        let moved_pos = live_positions[live_idx];
        live_idx_of[moved_pos] = Some(live_idx);
    }
    removed_pos
}

pub(crate) fn generate_top_heavy_pairings_indexed(
    num_items: usize,
    round_number: usize,
    top_k_probs: &[f64],
    sample_means: &[f64],
    sharpness: f64,
) -> Vec<IndexedPair> {
    if num_items < 2 {
        return Vec::new();
    }

    let mut rng = rand::rng();
    let pairs_target = calculate_pairs_for_round(num_items, round_number + 1);

    let total_item1_weight: f64 = top_k_probs.iter().sum();

    // Pre-sort items by sample_means for windowed opponent selection.
    // sorted_by_mean[i] = (original_index, mean), sorted by mean ascending.
    let mut sorted_by_mean: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, sample_means[i]))
        .collect();
    sorted_by_mean.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Reverse lookup: original_index -> position in sorted order.
    let mut sorted_pos = vec![0usize; num_items];
    for (pos, &(orig_idx, _)) in sorted_by_mean.iter().enumerate() {
        sorted_pos[orig_idx] = pos;
    }

    let mut pairs: Vec<IndexedPair> = Vec::with_capacity(pairs_target);

    for _ in 0..pairs_target {
        // Item 1: weighted sample from P(top K)
        let item1 = if total_item1_weight <= 0.0 {
            rng.random_range(0..num_items)
        } else {
            weighted_random_select(top_k_probs, total_item1_weight, &mut rng)
        };

        // Item 2: info-gain weighted from a window of nearby items in rating order.
        let item1_rating = sample_means[item1];
        let center = sorted_pos[item1];

        let half_w = OPPONENT_WINDOW_SIZE / 2;
        let window_start = center.saturating_sub(half_w);
        let window_end = (center + half_w + 1).min(num_items);

        let mut opponents: Vec<usize> = Vec::with_capacity(window_end - window_start);
        let mut opp_weights: Vec<f64> = Vec::with_capacity(window_end - window_start);

        for &(orig_idx, opp_rating) in &sorted_by_mean[window_start..window_end] {
            if orig_idx == item1 { continue; }
            opponents.push(orig_idx);
            opp_weights.push(calculate_info_gain(item1_rating, opp_rating, sharpness));
        }

        if opponents.is_empty() {
            break;
        }

        let total_opp_weight: f64 = opp_weights.iter().sum();
        let item2_local_idx = if total_opp_weight <= 0.0 {
            rng.random_range(0..opponents.len())
        } else {
            weighted_random_select(&opp_weights, total_opp_weight, &mut rng)
        };

        let item2 = opponents[item2_local_idx];

        if rng.random::<f64>() < 0.5 {
            pairs.push((item1, item2));
        } else {
            pairs.push((item2, item1));
        }
    }

    pairs
}

fn weighted_random_select(weights: &[f64], total_weight: f64, rng: &mut impl Rng) -> usize {
    let mut r = rng.random::<f64>() * total_weight;
    for (j, &w) in weights.iter().enumerate() {
        r -= w;
        if r < 1e-10 {
            return j;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info_gain_equal_ratings() {
        let gain = calculate_info_gain(1.0, 1.0, 1.0);
        assert!((gain - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_info_gain_unequal_ratings() {
        let gain = calculate_info_gain(100.0, 1.0, 1.0);
        assert!(gain < 0.01);
    }

    #[test]
    fn test_info_gain_sharpness() {
        let gain_low = calculate_info_gain(1.0, 2.0, 0.5);
        let gain_high = calculate_info_gain(1.0, 2.0, 2.0);
        assert!(gain_high < gain_low);
    }

    #[test]
    fn test_effective_strategy_balanced_user_choice() {
        let games = vec![10, 10];
        let result = get_effective_strategy(Strategy::Balanced, 2, &games, 5, 3, Some(10));
        assert_eq!(result, Strategy::Balanced);
    }

    #[test]
    fn test_effective_strategy_bootstrap_stage() {
        let games = vec![1, 10]; // Item 0 below minimum
        let result = get_effective_strategy(Strategy::TopHeavy, 2, &games, 5, 3, Some(10));
        assert_eq!(result, Strategy::Balanced);
    }

    #[test]
    fn test_effective_strategy_smoothing_stage() {
        let games = vec![10, 10];
        let result = get_effective_strategy(Strategy::TopHeavy, 2, &games, 9, 3, Some(10));
        assert_eq!(result, Strategy::Balanced);
    }

    #[test]
    fn test_effective_strategy_main_phase() {
        let games = vec![10, 10];
        let result = get_effective_strategy(Strategy::TopHeavy, 2, &games, 5, 3, Some(10));
        assert_eq!(result, Strategy::TopHeavy);
    }

    #[test]
    fn test_balanced_pairings_coverage() {
        let item_ids: Vec<i64> = (100..110).collect(); // IDs 100-109
        let ratings = vec![1.0; 10];
        let pairs = generate_balanced_pairings(&item_ids, 0, &ratings, 1.0);

        assert_eq!(pairs.len(), 5); // floor(10/2)

        // All pairs should use IDs from item_ids, not indices
        for (a, b) in &pairs {
            assert!(*a >= 100 && *a <= 109, "ID {} not in range", a);
            assert!(*b >= 100 && *b <= 109, "ID {} not in range", b);
        }
    }

    #[test]
    fn test_top_heavy_pairings() {
        let item_ids: Vec<i64> = (0..10).collect();
        let sample_means: Vec<f64> = (0..10).map(|i| 10.0 - i as f64).collect();
        let top_k_probs: Vec<f64> = (0..10).map(|i| if i < 3 { 0.8 } else { 0.05 }).collect();

        let pairs = generate_top_heavy_pairings(&item_ids, 0, &top_k_probs, &sample_means, 1.0);
        assert!(!pairs.is_empty());
    }
}
