"""Data for the kernel-collapse explorer.

Shows what happens to the 8-qubit quantum kernel as the input scale changes.
At scale 1.0 nearly every pair of cells has similarity close to 1/256, so the
similarity heatmap is a bare diagonal and a classifier has nothing to learn
from. Shrinking the inputs brings the structure back.

Two things are produced for each scale:
  - the kernel matrix on a fixed subset of cells (for the heatmap), and
  - the held-out accuracy of a QSVM forced to use that scale (for the readout).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from benchmark import run_nested_cv

SCALES = (1.0, 0.1, 0.05)
REPS = 2  # the project's original circuit depth
MAX_CELLS = 60
MATRIX_NOTE = (
    "Display only. Kernel matrices are computed on a fixed subset of the benchmark cells using "
    "features from a pipeline fitted on all benchmark cells. They are not used for any score."
)
ACCURACY_NOTE = (
    "Held-out accuracy of a quantum-kernel SVM forced to use this input scale (reps=2), on all "
    "benchmark cells with the same outer folds and fold-fitted features as the main benchmark. "
    "Only C is tuned, on inner folds."
)


def pick_subset(y: np.ndarray, cell_ids: np.ndarray, max_cells: int = MAX_CELLS, seed: int = 0) -> np.ndarray:
    """Row indices of the cells to show: both classes, rare class last, ordered by id."""
    y = np.asarray(y)
    ids = np.asarray(cell_ids).astype(str)
    chosen = np.arange(len(y))
    if len(y) > max_cells:
        rng = np.random.default_rng(seed)
        rare, others = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_rare = min(len(rare), max_cells // 3)
        chosen = np.concatenate([
            rng.choice(rare, size=n_rare, replace=False),
            rng.choice(others, size=min(len(others), max_cells - n_rare), replace=False),
        ])
    return np.array(sorted(chosen.tolist(), key=lambda i: (int(y[i]), ids[i])), dtype=int)


def collapse_matrices(features: np.ndarray, kernel, scales=SCALES) -> list:
    """One entry per scale: the kernel matrix and how much off-diagonal similarity is left."""
    out = []
    off_diagonal = ~np.eye(len(features), dtype=bool)
    for scale in scales:
        states = kernel.states(features, scale=scale)
        matrix = kernel.kernel(states, states)
        matrix = (matrix + matrix.T) / 2
        values = matrix[off_diagonal]
        out.append({
            "scale": float(scale),
            "matrix": np.round(matrix, 5).tolist(),
            "off_diagonal_mean": round(float(values.mean()), 6),
            "off_diagonal_median": round(float(np.median(values)), 6),
            "off_diagonal_min": round(float(values.min()), 6),
            "off_diagonal_max": round(float(values.max()), 6),
        })
    return out


def accuracy_at_scale(y, folds, features_for_fold, kernel, seed: int, inner_folds: int) -> dict:
    """Held-out scores of a QSVM whose kernel offers a single input scale."""
    y = np.asarray(y)
    pred = run_nested_cv(y, folds, features_for_fold, seed, inner_folds,
                         quantum=kernel, classical=False)["qsvm"].pred

    def mean(metric):
        return round(float(np.mean([metric(y[test], pred[test]) for _, test in folds])), 6)

    return {
        "balanced_accuracy": mean(balanced_accuracy_score),
        "macro_f1": mean(lambda a, b: f1_score(a, b, average="macro", zero_division=0)),
        "rare_recall": mean(lambda a, b: recall_score(a, b, pos_label=1, zero_division=0)),
        "predicted_positive": int(pred.sum()),
        "n_cells": int(len(y)),
    }
