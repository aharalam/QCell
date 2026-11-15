"""Fold-safe feature pipeline: raw counts -> 8 rotation angles in [0, pi].

Every statistic (which genes vary most, scaling, PCA, angle range) is
estimated from the cells passed to fit() and nothing else. The benchmark
fits one of these per outer training fold, so held-out cells never shape
the features they are scored on.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

DESCRIPTION = (
    "per-cell library-size normalisation to 10,000 and log1p (no fitted statistics); "
    "then, fitted on the training fold only: top {n_hvg} genes by variance, z-scoring "
    "clipped at +/-10, PCA to {n_components} components, and min-max scaling of each "
    "component to [0, pi] (held-out values clipped to that range)"
)


def log_normalize(counts: np.ndarray) -> np.ndarray:
    """Per-cell normalisation. Uses only the cell's own counts, so nothing is fitted."""
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(counts / totals * 1e4)


class FoldFeatures:
    def __init__(self, n_hvg: int = 500, n_components: int = 8, seed: int = 0):
        self.n_hvg = n_hvg
        self.n_components = n_components
        self.seed = seed

    def describe(self) -> str:
        return DESCRIPTION.format(n_hvg=self.n_hvg, n_components=self.n_components)

    def fit(self, counts: np.ndarray) -> "FoldFeatures":
        logged = log_normalize(np.asarray(counts, dtype=float))
        variance = logged.var(axis=0)
        keep = min(self.n_hvg, logged.shape[1])
        self.gene_index_ = np.sort(np.argsort(-variance, kind="stable")[:keep])
        selected = logged[:, self.gene_index_]
        self.mean_ = selected.mean(axis=0)
        self.std_ = selected.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.pca_ = PCA(n_components=self.n_components, svd_solver="full", random_state=self.seed)
        components = self.pca_.fit_transform(self._scale(selected))
        self.low_ = components.min(axis=0)
        span = components.max(axis=0) - self.low_
        span[span == 0] = 1.0
        self.span_ = span
        return self

    def _scale(self, selected: np.ndarray) -> np.ndarray:
        return np.clip((selected - self.mean_) / self.std_, -10, 10)

    def transform(self, counts: np.ndarray) -> np.ndarray:
        logged = log_normalize(np.asarray(counts, dtype=float))
        components = self.pca_.transform(self._scale(logged[:, self.gene_index_]))
        return np.clip((components - self.low_) / self.span_, 0.0, 1.0) * np.pi
