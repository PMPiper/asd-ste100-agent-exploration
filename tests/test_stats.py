"""Tests for the summary statistics and embedding-space analyses."""

import math

import numpy as np
import pytest
import pandas as pd

from ste_dial.analyze import geometry, stats


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def should_give_a_ci_that_contains_the_sample_mean():
    low, high = stats.bootstrap_mean_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), _rng())

    assert low < 3.0 < high


def should_give_the_same_ci_for_the_same_seed():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert stats.bootstrap_mean_ci(values, _rng()) == stats.bootstrap_mean_ci(
        values, _rng()
    )


def should_give_no_ci_for_a_single_value():
    low, high = stats.bootstrap_mean_ci(np.array([1.0]), _rng())

    assert math.isnan(low) and math.isnan(high)


def should_describe_median_quartiles_and_n():
    result = stats.describe(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), _rng())

    assert (result["n"], result["median"], result["q1"], result["q3"]) == (5, 3, 2, 4)


def should_report_perfect_negative_rank_correlation():
    result = stats.spearman_with_ci(
        np.array([0, 25, 50, 80, 100]), np.array([9.0, 7.0, 6.0, 2.0, 1.0]), _rng()
    )

    assert result["rho"] == pytest.approx(-1.0)
    assert result["n"] == 5


def should_report_no_correlation_for_a_constant_metric():
    result = stats.spearman_with_ci(
        np.array([0, 25, 50]), np.array([1.0, 1.0, 1.0]), _rng()
    )

    assert math.isnan(result["rho"])


def should_report_small_p_value_for_clearly_separated_groups():
    result = stats.kruskal_across_groups(
        [np.array([1.0, 2.0, 3.0, 4.0]), np.array([11.0, 12.0, 13.0, 14.0])]
    )

    assert result["p"] < 0.05


def should_report_no_test_when_all_values_are_equal():
    result = stats.kruskal_across_groups([np.array([1.0, 1.0]), np.array([1.0, 1.0])])

    assert math.isnan(result["p"])


RUNS = pd.DataFrame(
    {
        "run_id": ["a", "b", "c", "d"],
        "subsystem": ["coders", "coders", "coders", "coders"],
        "level": [0, 0, 100, 100],
    }
)
EAST = [1.0, 0.0]
NORTH = [0.0, 1.0]


def should_compare_a_baseline_run_with_the_other_baseline_runs_only():
    vectors = np.array([EAST, NORTH, EAST, EAST])

    drift = geometry.drift_from_baseline(vectors, RUNS)

    assert drift[0] == 0.0


def should_report_similarity_one_for_a_run_at_the_baseline_centroid():
    vectors = np.array([EAST, EAST, EAST, NORTH])

    drift = geometry.drift_from_baseline(vectors, RUNS)

    assert drift[2] == 1.0
    assert drift[3] == 0.0


def should_report_pairwise_similarity_of_baseline_runs_as_noise_floor():
    vectors = np.array([EAST, NORTH, EAST, EAST])

    noise = geometry.baseline_pairwise(vectors, RUNS)

    assert noise.tolist() == [0.0]


def should_report_mean_pairwise_similarity_within_each_cell():
    vectors = np.array([EAST, NORTH, EAST, EAST])

    cells = geometry.cell_consistency(vectors, RUNS, _rng())

    assert cells["mean_similarity"].tolist() == [0.0, 1.0]
    assert cells["pairs"].tolist() == [1, 1]


def should_score_well_separated_groups_near_one():
    vectors = np.array([EAST, EAST, NORTH, NORTH])

    score = geometry.silhouette(vectors, np.array([0, 0, 100, 100]))

    assert score == 1.0


def should_report_no_silhouette_when_every_run_is_its_own_group():
    vectors = np.array([EAST, NORTH])

    score = geometry.silhouette(vectors, np.array([0, 100]))

    assert math.isnan(score)


def should_pick_the_run_nearest_the_cell_centroid():
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b", "c"],
            "subsystem": ["coders", "coders", "coders"],
            "level": [0, 0, 0],
        }
    )
    vectors = np.array([EAST, [0.6, 0.8], NORTH])

    chosen = geometry.closest_to_centroid(vectors, runs)

    assert chosen[("coders", 0)] == "b"
