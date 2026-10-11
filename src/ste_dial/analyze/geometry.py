"""Analyses of the embedding space: clusters, drift, and consistency."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap import UMAP

from ste_dial.analyze import stats


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def project_pca(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the 2D PCA projection and its explained variance ratios."""
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors), pca.explained_variance_ratio_


def project_tsne(vectors: np.ndarray, seed: int) -> np.ndarray:
    """Returns a 2D t-SNE projection on cosine distances."""
    perplexity = min(30.0, max(2.0, (len(vectors) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        init="random",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(vectors)


def project_umap(vectors: np.ndarray, seed: int) -> np.ndarray:
    """Returns a 2D UMAP projection on cosine distances."""
    neighbors = min(15, len(vectors) - 1)
    # With few points the default spectral start fails; start at random.
    init = "spectral" if len(vectors) > 20 else "random"
    reducer = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=neighbors,
        init=init,
        random_state=seed,
    )
    return reducer.fit_transform(vectors)


def silhouette(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Returns the cosine silhouette score, or NaN if it is not defined.

    The score needs at least two groups and fewer groups than points.
    """
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    return float(silhouette_score(vectors, labels, metric="cosine"))


def silhouette_table(vectors: np.ndarray, runs: pd.DataFrame) -> pd.DataFrame:
    """Scores the grouping by subsystem and by level, overall and per subsystem.

    Args:
        vectors: One unit vector per run, in the row order of `runs`.
        runs: One row per run with `subsystem` and `level`.

    Returns:
        One row per grouping and scope, with the number of runs.
    """
    subsystem = runs["subsystem"].to_numpy()
    level = runs["level"].to_numpy()
    rows = [
        dict(
            grouping="subsystem",
            scope="all runs",
            n=len(runs),
            silhouette=silhouette(vectors, subsystem),
        ),
        dict(
            grouping="level",
            scope="all runs",
            n=len(runs),
            silhouette=silhouette(vectors, level),
        ),
    ]
    for name in dict.fromkeys(subsystem.tolist()):
        mask = subsystem == name
        rows.append(
            dict(
                grouping="level",
                scope=f"within {name}",
                n=int(mask.sum()),
                silhouette=silhouette(vectors[mask], level[mask]),
            )
        )
    return pd.DataFrame(rows)


def drift_from_baseline(vectors: np.ndarray, runs: pd.DataFrame) -> np.ndarray:
    """Returns each run's cosine similarity to its subsystem's 0% centroid.

    A 0% run is compared with the centroid of the other 0% runs. The value is
    NaN when there is no other 0% run.
    """
    subsystem = runs["subsystem"].to_numpy()
    level = runs["level"].to_numpy()
    similarity = np.full(len(runs), np.nan)
    for index in range(len(runs)):
        baseline = (subsystem == subsystem[index]) & (level == 0)
        baseline[index] = False
        if baseline.any():
            centroid = _unit(vectors[baseline].mean(axis=0))
            similarity[index] = float(vectors[index] @ centroid)
    return similarity


def baseline_pairwise(vectors: np.ndarray, runs: pd.DataFrame) -> np.ndarray:
    """Returns the cosine similarity of every pair of 0% runs of a subsystem.

    This is the noise floor: how alike two explanations are when the prompt
    is the same.
    """
    values: list[float] = []
    baseline = runs[runs["level"] == 0]
    for _, group in baseline.groupby("subsystem"):
        positions = runs.index.get_indexer(group.index)
        block = vectors[positions] @ vectors[positions].T
        values.extend(block[np.triu_indices(len(positions), k=1)].tolist())
    return np.array(values)


def cell_consistency(
    vectors: np.ndarray, runs: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Describes the pairwise cosine similarity within each cell.

    Returns:
        One row per subsystem and level with the number of runs and pairs,
        the mean pairwise similarity, and a bootstrap CI over the pairs.
    """
    rows = []
    for (subsystem, level), group in runs.groupby(["subsystem", "level"], sort=True):
        positions = runs.index.get_indexer(group.index)
        block = vectors[positions] @ vectors[positions].T
        pairs = block[np.triu_indices(len(positions), k=1)]
        low, high = stats.bootstrap_mean_ci(pairs, rng)
        rows.append(
            dict(
                subsystem=subsystem,
                level=level,
                n=len(positions),
                pairs=len(pairs),
                mean_similarity=float(pairs.mean()) if len(pairs) else float("nan"),
                ci_low=low,
                ci_high=high,
            )
        )
    return pd.DataFrame(rows)


def cell_similarity(
    vectors: np.ndarray, runs: pd.DataFrame
) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Returns the mean cosine similarity between every pair of cells.

    The diagonal is the mean over pairs of different runs of the same cell,
    and NaN for a cell with one run.
    """
    groups = {
        key: runs.index.get_indexer(group.index)
        for key, group in runs.groupby(["subsystem", "level"], sort=True)
    }
    cells = list(groups)
    matrix = np.full((len(cells), len(cells)), np.nan)
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            block = vectors[groups[a]] @ vectors[groups[b]].T
            if i == j:
                pairs = block[np.triu_indices(len(groups[a]), k=1)]
                matrix[i, j] = pairs.mean() if len(pairs) else np.nan
            else:
                matrix[i, j] = block.mean()
    return cells, matrix


def closest_to_centroid(vectors: np.ndarray, runs: pd.DataFrame) -> pd.Series:
    """Returns, per subsystem and level, the run_id nearest the cell centroid."""
    chosen = {}
    for key, group in runs.groupby(["subsystem", "level"], sort=True):
        positions = runs.index.get_indexer(group.index)
        centroid = _unit(vectors[positions].mean(axis=0))
        best = positions[int(np.argmax(vectors[positions] @ centroid))]
        chosen[key] = runs["run_id"].iloc[best]
    return pd.Series(chosen)
