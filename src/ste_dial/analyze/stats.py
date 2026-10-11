"""Summary statistics with bootstrap confidence intervals."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

N_RESAMPLES = 10_000


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_resamples: int = N_RESAMPLES
) -> tuple[float, float]:
    """Returns the bootstrap 95% confidence interval of the mean.

    Returns NaN bounds for fewer than two values.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def describe(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Returns n, median, quartiles, mean, and the bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        nan = float("nan")
        return dict(n=0, median=nan, q1=nan, q3=nan, mean=nan, ci_low=nan, ci_high=nan)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    low, high = bootstrap_mean_ci(values, rng)
    return dict(
        n=len(values),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        mean=float(values.mean()),
        ci_low=low,
        ci_high=high,
    )


def aggregate(
    frame: pd.DataFrame, metrics: list[str], by: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Describes every metric within every group.

    Args:
        frame: One row per run.
        metrics: The metric columns.
        by: The grouping columns.
        rng: Seeded generator for the bootstrap.

    Returns:
        One row per group and metric.
    """
    rows = []
    for key, group in frame.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            row = dict(zip(by, key)) | {"metric": metric}
            rows.append(row | describe(group[metric].to_numpy(), rng))
    return pd.DataFrame(rows)


def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = N_RESAMPLES,
) -> dict:
    """Returns Spearman's rho, its p-value, and a bootstrap 95% CI.

    The interval resamples (x, y) pairs. Resamples in which x or y is
    constant are dropped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    nan = float("nan")
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=len(x), rho=nan, p=nan, ci_low=nan, ci_high=nan)
    result = scipy_stats.spearmanr(x, y)
    indices = rng.integers(0, len(x), size=(n_resamples, len(x)))
    ranks_x = scipy_stats.rankdata(x[indices], axis=1)
    ranks_y = scipy_stats.rankdata(y[indices], axis=1)
    ranks_x -= ranks_x.mean(axis=1, keepdims=True)
    ranks_y -= ranks_y.mean(axis=1, keepdims=True)
    denominator = np.sqrt((ranks_x**2).sum(axis=1) * (ranks_y**2).sum(axis=1))
    valid = denominator > 0
    rhos = (ranks_x * ranks_y).sum(axis=1)[valid] / denominator[valid]
    low, high = np.percentile(rhos, [2.5, 97.5])
    return dict(
        n=len(x),
        rho=float(result.statistic),
        p=float(result.pvalue),
        ci_low=float(low),
        ci_high=float(high),
    )


def kruskal_across_groups(groups: list[np.ndarray]) -> dict:
    """Returns the Kruskal-Wallis H statistic and p-value across groups.

    Returns NaN when fewer than two groups have data or all values are equal.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    nan = float("nan")
    if len(groups) < 2 or np.ptp(np.concatenate(groups)) == 0:
        return dict(h=nan, p=nan)
    result = scipy_stats.kruskal(*groups)
    return dict(h=float(result.statistic), p=float(result.pvalue))


def trend_tests(
    frame: pd.DataFrame, metrics: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Tests every metric against the STE level.

    Returns:
        One row per metric with Spearman's rho and its CI, and the
        Kruskal-Wallis test across levels.
    """
    rows = []
    for metric in metrics:
        spearman = spearman_with_ci(
            frame["level"].to_numpy(), frame[metric].to_numpy(), rng
        )
        groups = [group[metric].to_numpy() for _, group in frame.groupby("level")]
        kruskal = kruskal_across_groups(groups)
        rows.append(
            {
                "metric": metric,
                "n": spearman["n"],
                "spearman_rho": spearman["rho"],
                "spearman_ci_low": spearman["ci_low"],
                "spearman_ci_high": spearman["ci_high"],
                "spearman_p": spearman["p"],
                "kruskal_h": kruskal["h"],
                "kruskal_p": kruskal["p"],
            }
        )
    return pd.DataFrame(rows)
