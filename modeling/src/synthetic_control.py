"""Synthetic planted-interaction control.

Two planted values per cell are drawn from a seeded random generator, and the
label is a fixed rule on them. Nothing here is estimated from data: the
thresholds are the constant 0, so there is nothing to leak across folds.
This target is engineered. A result on it says how a model handles a planted
two-feature interaction, nothing about real cells.
"""

from __future__ import annotations

import numpy as np

ANGLE_COLUMNS = (0, 1)  # where the planted values sit in the 8-feature matrix
PLANTED_NAMES = ("planted_x1", "planted_x2")
RULE_DESCRIPTION = (
    "Two planted values x1 and x2 are drawn independently and uniformly from [-1, 1] "
    "with a fixed seed. A cell is in the planted interaction state (label 1) when "
    "exactly one of x1 > 0 and x2 > 0 holds (an XOR), otherwise background (label 0). "
    "Neither value predicts the label alone. The planted values are mapped to "
    "[0, pi] by the fixed formula (x + 1) / 2 * pi and replace the first two of the "
    "8 model features; the other six are the cell's real expression components 3-8."
)


def planted_control(n_cells: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    planted = rng.uniform(-1.0, 1.0, size=(n_cells, 2))
    labels = ((planted[:, 0] > 0) != (planted[:, 1] > 0)).astype(int)
    return planted, labels


def synthetic_features(planted: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Planted values as fixed angles, followed by the nuisance features unchanged."""
    return np.hstack([(planted + 1.0) / 2.0 * np.pi, nuisance])
