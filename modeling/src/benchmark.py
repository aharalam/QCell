"""Nested cross-validation shared by every model.

Outer folds give each cell exactly one held-out prediction. Inside each outer
training fold, hyperparameters are chosen by an inner stratified grid search
scored by macro F1. Held-out labels are never read during fitting or tuning,
and all models see the same folds and the same features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

C_VALUES = [0.01, 0.1, 1, 10, 100]
GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1, 1, 10]
SELECTION_METRIC = "f1_macro"
QUANTUM_SCALE_VALUES = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005]  # input-angle scale (bandwidth)
QUANTUM_REPS_VALUES = [1, 2, 3]  # feature-map repetitions (circuit depth)
CLASSICAL_MODELS = ("linear", "poly2", "rbf")
DISPLAY_NAMES = {
    "linear": "Linear SVM",
    "poly2": "Degree-2 polynomial SVM",
    "rbf": "RBF SVM",
    "qsvm": "Quantum-kernel SVM (QSVM)",
}
_GRIDS = {
    "linear": (dict(kernel="linear"), {"C": C_VALUES}),
    "poly2": (dict(kernel="poly", degree=2), {"C": C_VALUES, "gamma": GAMMA_VALUES}),
    "rbf": (dict(kernel="rbf"), {"C": C_VALUES, "gamma": GAMMA_VALUES}),
    "qsvm": (dict(kernel="precomputed"), {"C": C_VALUES}),
}


@dataclass
class ModelOutcome:
    pred: np.ndarray  # out-of-fold prediction per cell (-1 where no fold covered it)
    score: np.ndarray  # uncalibrated SVM decision value per cell
    fit_seconds: dict = field(default_factory=dict)
    predict_seconds: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)


def model_config(name: str) -> dict:
    fixed, grid = _GRIDS[name]
    if name == "qsvm":
        grid = {**grid, "scale": QUANTUM_SCALE_VALUES, "reps": QUANTUM_REPS_VALUES}
    return {**fixed, "class_weight": "balanced", "grid": grid, "selection_metric": SELECTION_METRIC}


def make_folds(y: np.ndarray, n_folds: int, seed: int) -> list:
    y = np.asarray(y)
    rare = int(min(np.sum(y == 0), np.sum(y == 1)))
    if rare < n_folds:
        raise ValueError(
            f"only {rare} rare-class cells for {n_folds} folds; every fold needs at least one"
        )
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.zeros(len(y)), y)]


def fold_features(counts: np.ndarray, train: np.ndarray, test: np.ndarray, pipeline_factory: Callable):
    """Fit the feature pipeline on the training cells only, then transform both sets."""
    pipeline = pipeline_factory().fit(counts[train])
    return pipeline.transform(counts[train]), pipeline.transform(counts[test])


def _search(name: str, seed: int, inner_folds: int) -> GridSearchCV:
    fixed, grid = _GRIDS[name]
    return GridSearchCV(
        SVC(class_weight="balanced", **fixed),
        grid,
        scoring=SELECTION_METRIC,
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed),
        refit=True,
    )


def _plain(params: dict) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def run_nested_cv(
    y: np.ndarray,
    folds: Sequence,
    features_for_fold: Callable,
    seed: int,
    inner_folds: int = 3,
    quantum=None,
    classical: bool = True,
    on_progress: Callable | None = None,
) -> dict:
    """Return {model name: ModelOutcome}.

    features_for_fold(train_idx, test_idx) -> (x_train, x_test).
    quantum, if given, needs states(features) and kernel(states_a, states_b);
    it is run as a precomputed-kernel SVM on exactly the same folds. If it has a
    `scales` attribute, states(features, scale=...) is tried for each one. A list of
    such kernels (for example one per circuit depth) is searched the same way.
    """
    y = np.asarray(y)
    names = list(CLASSICAL_MODELS if classical else ()) + (["qsvm"] if quantum is not None else [])
    out = {
        name: ModelOutcome(pred=np.full(len(y), -1, dtype=int), score=np.full(len(y), np.nan))
        for name in names
    }
    for k, (train, test) in enumerate(folds):
        x_train, x_test = features_for_fold(train, test)
        for name in names:
            result = out[name]
            scale = None
            start = time.perf_counter()
            if name == "qsvm":
                # Circuit depth and input scale are tuned like any other hyperparameter: by
                # inner-fold score on the training cells only. Ties keep the earlier
                # candidate, so the shallower circuit and the larger scale win a tie.
                search = None
                for candidate_kernel in (quantum if isinstance(quantum, (list, tuple)) else [quantum]):
                    for candidate in getattr(candidate_kernel, "scales", None) or [None]:
                        states = (candidate_kernel.states(x_train) if candidate is None
                                  else candidate_kernel.states(x_train, scale=candidate))
                        fitted = _search(name, seed, inner_folds).fit(candidate_kernel.kernel(states, states), y[train])
                        if search is None or fitted.best_score_ > search.best_score_:
                            search, scale, train_states, chosen = fitted, candidate, states, candidate_kernel
            else:
                search = _search(name, seed, inner_folds).fit(x_train, y[train])
            result.fit_seconds[k] = time.perf_counter() - start

            start = time.perf_counter()
            if name == "qsvm":
                test_states = chosen.states(x_test) if scale is None else chosen.states(x_test, scale=scale)
                held_out = chosen.kernel(test_states, train_states)
            else:
                held_out = x_test
            result.pred[test] = search.predict(held_out)
            result.predict_seconds[k] = time.perf_counter() - start
            result.score[test] = search.decision_function(held_out)
            result.best_params[k] = _plain(search.best_params_)
            if scale is not None:
                result.best_params[k]["scale"] = scale
            if name == "qsvm":
                result.best_params[k].update(getattr(chosen, "params", {}))
            if on_progress:
                on_progress(name, k, result.fit_seconds[k] + result.predict_seconds[k])
    return out
