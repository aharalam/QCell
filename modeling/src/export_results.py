"""Write the dashboard bundle that the Base44 app uploads.

The bundle is four JSON files (run_manifest.json, results.json, cells.json,
kernels.json) whose shapes are fixed by the DATA CONTRACT block in Prompt 1 of
base44_prompts.md. This module turns the arrays your pipeline already has into
those files, and refuses to write a bundle the app would reject.

What you give it (per target): cell ids, true labels, the outer fold each cell
was held out in, each model's out-of-fold predictions and measured timings,
display coordinates, and the three display kernel matrices.

What it works out for you: per-fold and aggregate metrics (scikit-learn, exactly
the calls in the spec), pooled confusion matrices, bootstrap intervals, and
Kernel-PCA coordinates from the kernel matrices.

What it never does: invent a number. Anything it cannot derive from your arrays
must be passed in, and a missing timing or prediction is an error, not a default.

Typical use from runner.py:

    from export_results import ModelRun, QsvmAttempt, TargetInputs, export_bundle

    report = export_bundle(
        out_dir="artifacts/<run_id>/dashboard_bundle",
        manifest=manifest_dict,
        targets={"biological": bio_inputs, "synthetic": syn_inputs},
        circuit_image="artifacts/<run_id>/circuit.png",   # optional
    )

Command line:

    python export_results.py validate <bundle_dir>
    python export_results.py placeholder <out_dir>      # fake, flagged bundle
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "1.0"
TARGET_KEYS = ("biological", "synthetic")  # always present
OPTIONAL_TARGET_KEYS = ("quantum_structured",)  # exported and validated when present
MODEL_KEYS = ("linear", "poly2", "rbf", "qsvm")
KERNEL_KEYS = ("rbf", "poly2", "qsvm")
METRIC_KEYS = ("balanced_accuracy", "macro_f1", "rare_recall", "rare_precision")
COORD_KEYS = ("umap3", "pca3")
STAGE_STATUSES = ("succeeded", "failed", "skipped", "not_run")
BUNDLE_FILES = {
    "manifest": "run_manifest.json",
    "results": "results.json",
    "cells": "cells.json",
    "kernels": "kernels.json",
}
METRIC_TOLERANCE = 0.005
SECRET_KEY_PATTERN = re.compile(r"(api[_-]?key|secret|token|password|credential)", re.I)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class QsvmAttempt:
    """One quantum configuration that was actually run (for example reps=2)."""

    reps: int
    is_final: bool
    pred: Any  # out-of-fold predictions, one 0/1 per cell
    total_runtime_seconds: float


@dataclass
class ModelRun:
    """One model's measured out-of-fold results for one target."""

    display_name: str
    pred: Any  # out-of-fold predictions, one 0/1 per cell, in cell_ids order
    fit_seconds: Mapping[int, float]  # outer fold -> measured seconds
    predict_seconds: Mapping[int, float]  # outer fold -> measured seconds
    config: dict = field(default_factory=dict)
    best_params: Mapping[int, dict] = field(default_factory=dict)
    score: Any = None  # optional uncalibrated SVM decision values, one per cell
    total_runtime_seconds: float | None = None  # default: sum of fold timings
    attempts: Sequence[QsvmAttempt] = ()  # qsvm only


@dataclass
class TargetInputs:
    """Everything the dashboard shows for one target, in one cell order."""

    positive_label_name: str
    negative_label_name: str
    cell_ids: Sequence[str]
    y_true: Any  # 0/1 per cell, 1 is the rare class
    fold: Any  # outer fold in which each cell was held out
    annotation: Sequence[str]  # cell type name, or "planted"/"background"
    feature_names: Sequence[str]
    features: Any  # (n_cells, n_features), as transformed for the held-out fold
    coords: Mapping[str, Any]  # "umap3" and "pca3", each (n_cells, 3)
    models: Mapping[str, ModelRun]  # linear, poly2, rbf, qsvm
    kernels: Mapping[str, Any]  # rbf, poly2, qsvm; each (n_cells, n_cells)
    kernel_params: Mapping[str, dict]
    kernel_fit_scope: str  # say these are display-only fits, not scoring fits
    planted: Any = None  # synthetic only: (n_cells, 2) planted x1, x2
    decision_regions: dict | None = None  # synthetic only, optional
    extra: Mapping[str, Any] | None = None  # optional per-cell values, name -> one number per cell


class BundleError(ValueError):
    """The inputs cannot produce a bundle the dashboard would accept."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} problem(s) with the bundle:\n  - "
            + "\n  - ".join(self.problems)
        )


# --------------------------------------------------------------------------
# Metrics (scikit-learn, as written in the spec)
# --------------------------------------------------------------------------


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rare_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "rare_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1])
        .astype(int)
        .tolist(),
    }


def _aggregate(per_fold: Sequence[dict]) -> dict:
    out = {}
    for key in METRIC_KEYS:
        values = np.array([f[key] for f in per_fold], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[key] = {"mean": _r(values.mean()), "std": _r(std)}
    return out


def bootstrap_ci95(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, seed: int
) -> dict | None:
    """Percentile interval from resampling the pooled out-of-fold predictions."""
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    yt, yp = y_true[idx], y_pred[idx]
    tp = ((yt == 1) & (yp == 1)).sum(axis=1).astype(float)
    fn = ((yt == 1) & (yp == 0)).sum(axis=1).astype(float)
    fp = ((yt == 0) & (yp == 1)).sum(axis=1).astype(float)
    tn = ((yt == 0) & (yp == 0)).sum(axis=1).astype(float)
    keep = ((tp + fn) > 0) & ((tn + fp) > 0)  # both classes drawn
    if keep.sum() < max(100, n_boot // 2):
        return None
    tp, fn, fp, tn = tp[keep], fn[keep], fp[keep], tn[keep]
    balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
    macro_f1 = (f1_pos + f1_neg) / 2
    return {
        "balanced_accuracy": [_r(np.percentile(balanced, 2.5)), _r(np.percentile(balanced, 97.5))],
        "macro_f1": [_r(np.percentile(macro_f1, 2.5)), _r(np.percentile(macro_f1, 97.5))],
    }


# --------------------------------------------------------------------------
# Kernel-PCA
# --------------------------------------------------------------------------


def kernel_pca_3d(kernel: np.ndarray) -> tuple[np.ndarray, str | None]:
    """First three Kernel-PCA components of a precomputed kernel.

    Returns (coords, note). The note is set when the centered kernel has a
    clearly negative eigenvalue (a noisy quantum kernel can be slightly
    non-positive-semidefinite), so the reader knows the projection only uses
    its top non-negative components. scikit-learn does not report this itself.
    """
    n = kernel.shape[0]
    one = np.full((n, n), 1.0 / n)
    centered = kernel - one @ kernel - kernel @ one + one @ kernel @ one
    values, vectors = np.linalg.eigh((centered + centered.T) / 2)
    note = None
    if values.min() < -1e-8 * max(abs(values.max()), 1e-12):
        note = (
            f"not positive semidefinite (smallest eigenvalue {values.min():.3g}); "
            "Kernel-PCA uses only its top non-negative components"
        )
    try:
        from sklearn.decomposition import KernelPCA

        projected = KernelPCA(n_components=min(3, n - 1), kernel="precomputed").fit_transform(kernel)
        if not np.all(np.isfinite(projected)):
            raise ValueError("non-finite projection")
    except ValueError:
        order = np.argsort(values)[::-1][:3]
        projected = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
    coords = np.zeros((n, 3))
    coords[:, : projected.shape[1]] = projected
    for j in range(3):  # fix the arbitrary sign so reruns give identical output
        column = coords[:, j]
        if column[np.argmax(np.abs(column))] < 0:
            coords[:, j] = -column
    return coords, note


def classical_display_kernels(
    features: np.ndarray,
    rbf_gamma: float,
    poly_gamma: float,
    poly_coef0: float = 0.0,
) -> dict:
    """RBF and degree-2 polynomial kernels on one display feature matrix.

    Use the same matrix (same cells, same order) that the quantum display
    kernel was computed on. coef0 defaults to scikit-learn SVC's default.
    """
    from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

    return {
        "rbf": rbf_kernel(features, gamma=rbf_gamma),
        "poly2": polynomial_kernel(features, degree=2, gamma=poly_gamma, coef0=poly_coef0),
    }


# --------------------------------------------------------------------------
# Building the bundle
# --------------------------------------------------------------------------


def _r(value: Any, digits: int = 6) -> float:
    return float(round(float(value), digits))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _labels(values: Any, n: int, name: str, problems: list) -> np.ndarray | None:
    arr = np.asarray(values)
    if arr.shape != (n,):
        problems.append(f"{name}: expected {n} values, got shape {arr.shape}")
        return None
    if not np.all(np.isin(arr, (0, 1))):
        problems.append(f"{name}: every value must be 0 or 1")
        return None
    return arr.astype(int)


def _matrix(values: Any, shape: tuple, name: str, problems: list) -> np.ndarray | None:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        problems.append(f"{name}: not numeric")
        return None
    if arr.shape != shape:
        problems.append(f"{name}: expected shape {shape}, got {arr.shape}")
        return None
    if not np.all(np.isfinite(arr)):
        problems.append(f"{name}: contains NaN or infinity")
        return None
    return arr


def _model_block(
    name: str, run: ModelRun, y: np.ndarray, fold: np.ndarray, folds: list,
    n_boot: int, seed: int, where: str, problems: list,
) -> tuple[dict | None, np.ndarray | None]:
    pred = _labels(run.pred, len(y), f"{where}.{name}.pred", problems)
    if pred is None:
        return None, None
    per_fold = []
    for k in folds:
        if k not in run.fit_seconds or k not in run.predict_seconds:
            problems.append(
                f"{where}.{name}: no measured fit_seconds/predict_seconds for fold {k} "
                "(time them with time.perf_counter; they are never estimated here)"
            )
            continue
        mask = fold == k
        metrics = fold_metrics(y[mask], pred[mask])
        per_fold.append(
            {
                "fold": int(k),
                **{key: _r(metrics[key]) for key in METRIC_KEYS},
                "confusion_matrix": metrics["confusion_matrix"],
                "best_params": _jsonable(dict(run.best_params.get(k, {}))),
                "fit_seconds": _r(run.fit_seconds[k], 4),
                "predict_seconds": _r(run.predict_seconds[k], 4),
            }
        )
    if len(per_fold) != len(folds):
        return None, pred
    total = run.total_runtime_seconds
    if total is None:
        total = sum(f["fit_seconds"] + f["predict_seconds"] for f in per_fold)
    block = {
        "display_name": run.display_name,
        "config": _jsonable(run.config),
        "aggregate": _aggregate(per_fold),
        "pooled_confusion_matrix": fold_metrics(y, pred)["confusion_matrix"],
        "folds": per_fold,
        "total_runtime_seconds": _r(total, 4),
    }
    interval = bootstrap_ci95(y, pred, n_boot, seed)
    if interval:
        block["ci95"] = interval
    if run.attempts:
        if name != "qsvm":
            problems.append(f"{where}.{name}: attempts are only valid on qsvm")
        finals = [a for a in run.attempts if a.is_final]
        if len(finals) != 1:
            problems.append(f"{where}.qsvm.attempts: exactly one attempt must be final")
        attempts = []
        for attempt in run.attempts:
            a_pred = _labels(attempt.pred, len(y), f"{where}.qsvm.attempts[reps={attempt.reps}].pred", problems)
            if a_pred is None:
                continue
            if attempt.is_final and not np.array_equal(a_pred, pred):
                problems.append(
                    f"{where}.qsvm: the final attempt's predictions differ from qsvm.pred"
                )
            a_folds = [fold_metrics(y[fold == k], a_pred[fold == k]) for k in folds]
            attempts.append(
                {
                    "reps": int(attempt.reps),
                    "is_final": bool(attempt.is_final),
                    "aggregate": _aggregate(a_folds),
                    "total_runtime_seconds": _r(attempt.total_runtime_seconds, 4),
                }
            )
        block["attempts"] = attempts
    return block, pred


def _target_blocks(
    key: str, t: TargetInputs, n_boot: int, seed: int, problems: list, notes: list
) -> tuple[dict, dict, dict] | None:
    before = len(problems)
    ids = [str(c) for c in t.cell_ids]
    n = len(ids)
    if len(set(ids)) != n:
        problems.append(f"{key}: cell_ids are not unique")
    if len(t.annotation) != n:
        problems.append(f"{key}: annotation has {len(t.annotation)} entries for {n} cells")
    y = _labels(t.y_true, n, f"{key}.y_true", problems)
    fold = np.asarray(t.fold)
    if fold.shape != (n,):
        problems.append(f"{key}.fold: expected {n} values, got shape {fold.shape}")
        fold = None
    features = _matrix(t.features, (n, len(t.feature_names)), f"{key}.features", problems)
    coords = {}
    for name in COORD_KEYS:
        if name not in t.coords:
            problems.append(f"{key}.coords: missing '{name}'")
        else:
            coords[name] = _matrix(t.coords[name], (n, 3), f"{key}.coords.{name}", problems)
    for name in MODEL_KEYS:
        if name not in t.models:
            problems.append(f"{key}.models: missing '{name}'")
    kernels = {}
    for name in KERNEL_KEYS:
        if name not in t.kernels:
            problems.append(f"{key}.kernels: missing '{name}'")
            continue
        kernels[name] = _matrix(t.kernels[name], (n, n), f"{key}.kernels.{name}", problems)
        if name not in t.kernel_params:
            problems.append(f"{key}.kernel_params: missing '{name}'")
    planted = None
    if key == "synthetic":
        if t.planted is None:
            problems.append("synthetic.planted: the two planted dimensions are required")
        else:
            planted = _matrix(t.planted, (n, 2), "synthetic.planted", problems)
    extras = {}
    for extra_name, values in (t.extra or {}).items():
        extras[str(extra_name)] = _matrix(values, (n,), f"{key}.extra.{extra_name}", problems)
    if not str(t.kernel_fit_scope).strip():
        problems.append(f"{key}.kernel_fit_scope: say how the display kernels were fitted")
    if len(problems) > before or y is None or fold is None:
        return None

    fold = fold.astype(int)
    folds = sorted(int(k) for k in np.unique(fold))
    for k in folds:
        present = set(np.unique(y[fold == k]).tolist())
        if present != {0, 1}:
            problems.append(
                f"{key}: outer fold {k} does not contain both classes; "
                "use stratified folds with enough rare cells"
            )
    if len(problems) > before:
        return None

    model_blocks, preds = {}, {}
    for name in MODEL_KEYS:
        block, pred = _model_block(
            name, t.models[name], y, fold, folds, n_boot, seed, key, problems
        )
        if block is not None:
            model_blocks[name], preds[name] = block, pred
    scores = {}
    provided = [m for m in MODEL_KEYS if t.models[m].score is not None]
    if len(provided) == len(MODEL_KEYS):
        for name in MODEL_KEYS:
            arr = _matrix(t.models[name].score, (n,), f"{key}.{name}.score", problems)
            if arr is not None:
                scores[name] = arr
    elif provided:
        notes.append(
            f"{key}: decision scores given for {provided} only; scores are left out "
            "unless all four models provide them"
        )
    if len(problems) > before:
        return None

    # Kernels: rare class last, so every heatmap shows the same class blocks.
    order = sorted(range(n), key=lambda i: (int(y[i]), ids[i]))
    kpca, matrices = {}, {}
    for name in KERNEL_KEYS:
        k_matrix = kernels[name]
        asymmetry = float(np.max(np.abs(k_matrix - k_matrix.T)))
        if asymmetry > 1e-6:
            notes.append(f"{key}.kernels.{name}: asymmetric by up to {asymmetry:.2e}; exported as given")
        else:
            k_matrix = (k_matrix + k_matrix.T) / 2
        kpca[name], note = kernel_pca_3d(k_matrix)
        if note:
            notes.append(f"{key}.kernels.{name}: {note}")
        matrices[name] = np.round(k_matrix[np.ix_(order, order)], 5).tolist()

    cells = []
    for i in range(n):
        cell = {
            "id": ids[i],
            "fold": int(fold[i]),
            "y_true": int(y[i]),
            "annotation": str(t.annotation[i]),
            "coords": {name: np.round(coords[name][i], 5).tolist() for name in COORD_KEYS},
            "kpca": {name: np.round(kpca[name][i], 5).tolist() for name in KERNEL_KEYS},
            "features": np.round(features[i], 5).tolist(),
            "pred": {name: int(preds[name][i]) for name in MODEL_KEYS},
        }
        if scores:
            cell["score"] = {name: _r(scores[name][i], 5) for name in MODEL_KEYS}
        if planted is not None:
            cell["planted"] = {"x1": _r(planted[i, 0], 5), "x2": _r(planted[i, 1], 5)}
        if extras:
            cell["extra"] = {extra_name: _r(values[i], 6) for extra_name, values in extras.items()}
        cells.append(cell)

    cells_block = {"feature_names": [str(f) for f in t.feature_names], "cells": cells}
    if t.decision_regions is not None:
        if key != "synthetic":
            problems.append(f"{key}: decision_regions are only valid on the synthetic target")
        else:
            _check_decision_regions(t.decision_regions, problems)
            cells_block["decision_regions"] = _jsonable(t.decision_regions)

    results_block = {
        "positive_label_name": t.positive_label_name,
        "negative_label_name": t.negative_label_name,
        "n_cells": n,
        "n_positive": int(y.sum()),
        "models": model_blocks,
    }
    kernels_block = {
        "cell_ids": [ids[i] for i in order],
        "fit_scope": str(t.kernel_fit_scope),
        "params": {name: _jsonable(dict(t.kernel_params[name])) for name in KERNEL_KEYS},
        "matrices": matrices,
    }
    return results_block, cells_block, kernels_block


def _check_decision_regions(regions: Mapping, problems: list) -> None:
    resolution = regions.get("resolution")
    if not isinstance(resolution, int) or resolution < 2:
        problems.append("synthetic.decision_regions.resolution: must be an integer >= 2")
        return
    for axis in ("x1_range", "x2_range"):
        if len(regions.get(axis, [])) != 2:
            problems.append(f"synthetic.decision_regions.{axis}: must be [min, max]")
    for name in MODEL_KEYS:
        grid = np.asarray(regions.get("grids", {}).get(name, []))
        if grid.shape != (resolution, resolution) or not np.all(np.isin(grid, (0, 1))):
            problems.append(
                f"synthetic.decision_regions.grids.{name}: must be {resolution}x{resolution} of 0/1"
            )


def _find_secrets(obj: Any, path: str, found: list) -> None:
    secret_value = os.environ.get("QUANTUM_RINGS_API_KEY")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if SECRET_KEY_PATTERN.search(str(k)):
                found.append(f"{path}{k}")
            _find_secrets(v, f"{path}{k}.", found)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find_secrets(v, f"{path}{i}.", found)
    elif isinstance(obj, str) and secret_value and secret_value in obj:
        found.append(f"{path.rstrip('.')} (contains the Quantum Rings API key)")


def _finish_manifest(manifest: Mapping, targets: Mapping, results: Mapping, problems: list) -> dict:
    m = copy.deepcopy(_jsonable(manifest))
    found: list = []
    _find_secrets(m, "", found)
    for path in found:
        problems.append(f"manifest.{path}: looks like a credential; remove it")
    m["schema_version"] = SCHEMA_VERSION
    m.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    m.setdefault("is_placeholder", False)

    def fill(section: str, key: str, measured: Any) -> None:
        block = m.setdefault(section, {})
        if key in block and block[key] != measured:
            problems.append(
                f"manifest.{section}.{key} is {block[key]!r} but the data says {measured!r}"
            )
        block[key] = measured

    bio = results.get("biological")
    if bio:
        fill("sampling", "n_benchmark_cells", bio["n_cells"])
        fill("sampling", "n_positive", bio["n_positive"])
        fill("preprocessing", "n_features", len(targets["biological"].feature_names))
        fill("cv", "outer_folds", len(bio["models"]["qsvm"]["folds"]) if "qsvm" in bio["models"] else None)
        attempts = bio["models"].get("qsvm", {}).get("attempts")
        declared = m.get("quantum", {}).get("attempts")
        if attempts and isinstance(declared, list):
            measured = sorted((a["reps"], a["is_final"]) for a in attempts)
            stated = sorted((a.get("reps"), a.get("is_final")) for a in declared)
            if measured != stated:
                problems.append(
                    f"manifest.quantum.attempts {stated} does not match the attempts "
                    f"that were exported {measured}; every attempt run must be declared"
                )
    return m


def build_bundle(
    manifest: Mapping, targets: Mapping[str, TargetInputs], n_bootstrap: int = 2000
) -> tuple[dict, list]:
    """Return ({manifest, results, cells, kernels}, notes) or raise BundleError."""
    problems: list = []
    notes: list = []
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise BundleError(["manifest.run_id: required"])
    for key in TARGET_KEYS:
        if key not in targets:
            problems.append(f"targets: missing '{key}'")
    for key in targets:
        if key not in TARGET_KEYS + OPTIONAL_TARGET_KEYS:
            problems.append(f"targets: unknown target '{key}'")
    if problems:
        raise BundleError(problems)
    seed = int(manifest.get("cv", {}).get("seed", 0) or 0)

    results_t, cells_t, kernels_t = {}, {}, {}
    for key in (*TARGET_KEYS, *(k for k in OPTIONAL_TARGET_KEYS if k in targets)):
        blocks = _target_blocks(key, targets[key], n_bootstrap, seed, problems, notes)
        if blocks:
            results_t[key], cells_t[key], kernels_t[key] = blocks
    full_manifest = _finish_manifest(manifest, targets, results_t, problems)
    if problems:
        raise BundleError(problems)

    head = {"schema_version": SCHEMA_VERSION, "run_id": run_id}
    bundle = {
        "manifest": full_manifest,
        "results": {
            **head,
            "notes": {
                "std": "sample standard deviation across outer folds (ddof=1)",
                "ci95": "percentile bootstrap over pooled out-of-fold predictions"
                if n_bootstrap > 0 else "not computed",
            },
            "targets": results_t,
        },
        "cells": {**head, "targets": cells_t},
        "kernels": {**head, "targets": kernels_t},
    }
    return bundle, notes


# --------------------------------------------------------------------------
# Validation: the same checks the dashboard runs on upload (E1-E7, W1-W3),
# recomputed here in plain Python so they do not share code with the exporter.
# --------------------------------------------------------------------------

_MANIFEST_FIELDS = {
    "schema_version": str, "run_id": str, "created_at": str, "is_placeholder": bool,
    "dataset.name": str, "dataset.source_url": str, "dataset.n_cells_raw": int,
    "dataset.n_genes_raw": int, "dataset.n_cells_after_qc": int, "dataset.donor_status": str,
    "annotation.method": str, "annotation.positive_class_name": str, "annotation.limitations": str,
    "sampling.strategy": str, "sampling.n_benchmark_cells": int, "sampling.n_positive": int,
    "sampling.seed": int,
    "preprocessing.feature_pipeline": str, "preprocessing.n_features": int,
    "preprocessing.fitted_within_training_folds_only": bool,
    "cv.outer_folds": int, "cv.inner_folds": int, "cv.selection_metric": str, "cv.seed": int,
    "quantum.sdk": str, "quantum.sdk_version": str, "quantum.backend": str,
    "quantum.n_qubits": int, "quantum.feature_map": str, "quantum.entanglement": str,
    "quantum.attempts": list,
    "synthetic_control.rule_description": str, "synthetic_control.features_used": list,
    "synthetic_control.seed": int,
    "pipeline_stages": list, "software_versions": dict,
}


def _is(value: Any, kind: type) -> bool:
    if kind is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, kind)


def _lookup(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return _MISSING
        obj = obj[part]
    return obj


_MISSING = object()


def _confusion(y: Sequence[int], p: Sequence[int]) -> list:
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    return [[tn, fp], [fn, tp]]


def _metrics_from_confusion(cm: Sequence[Sequence[int]]) -> dict:
    (tn, fp), (fn, tp) = cm
    recalls = []
    if tn + fp:
        recalls.append(tn / (tn + fp))
    if tp + fn:
        recalls.append(tp / (tp + fn))
    f1s = []
    if tn + fp + fn:  # class 0 appears in the truth or the predictions
        f1s.append(2 * tn / (2 * tn + fp + fn))
    if tp + fp + fn:
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return {
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "rare_recall": tp / (tp + fn) if tp + fn else 0.0,
        "rare_precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def _is_confusion(cm: Any) -> bool:
    return (
        isinstance(cm, list) and len(cm) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in cm)
        and all(_is(v, int) and v >= 0 for row in cm for v in row)
    )


def validate_bundle(manifest: Any, results: Any, cells: Any, kernels: Any) -> dict:
    errors: list = []
    warnings: list = []
    passed: list = []

    def check(code: str, title: str, found: list, warn: bool = False) -> None:
        if found:
            (warnings if warn else errors).extend(f"{code} {msg}" for msg in found)
        else:
            passed.append(f"{code} {title}")

    files = {"run_manifest": manifest, "results": results, "cells": cells, "kernels": kernels}

    e1 = [f"{name}.json is not a JSON object" for name, doc in files.items() if not isinstance(doc, dict)]
    if e1:
        check("E1", "", e1)
        return _report(errors, warnings, passed)
    run_ids = {name: doc.get("run_id") for name, doc in files.items()}
    if len(set(run_ids.values())) != 1 or not _is(manifest.get("run_id"), str):
        e1.append(f"run_id differs between files: {run_ids}")
    for name, doc in files.items():
        if doc.get("schema_version") != SCHEMA_VERSION:
            e1.append(f"{name}.json schema_version is {doc.get('schema_version')!r}, expected '{SCHEMA_VERSION}'")
    check("E1", "one run_id and schema version across all four files", e1)

    e2: list = []
    for dotted, kind in _MANIFEST_FIELDS.items():
        value = _lookup(manifest, dotted)
        if value is _MISSING:
            e2.append(f"run_manifest.{dotted} is missing")
        elif not _is(value, kind):
            e2.append(f"run_manifest.{dotted} should be {kind.__name__}")
    for i, stage in enumerate(manifest.get("pipeline_stages") or []):
        if not isinstance(stage, dict) or stage.get("status") not in STAGE_STATUSES:
            e2.append(f"run_manifest.pipeline_stages[{i}].status must be one of {STAGE_STATUSES}")
    declared_attempts = _lookup(manifest, "quantum.attempts")
    for i, attempt in enumerate(declared_attempts if isinstance(declared_attempts, list) else []):
        if not (isinstance(attempt, dict) and _is(attempt.get("reps"), int)
                and _is(attempt.get("is_final"), bool) and "reason" in attempt):
            e2.append(f"run_manifest.quantum.attempts[{i}] needs reps, is_final and reason")

    e3: list = []
    e4: list = []
    e5: list = []
    e6: list = []
    e7: list = []
    w1: list = []
    w3: list = []

    optional_present = [
        key for key in OPTIONAL_TARGET_KEYS
        if any(isinstance(_lookup(doc, f"targets.{key}"), dict) for doc in (results, cells, kernels))
    ]
    for target in (*TARGET_KEYS, *optional_present):
        r = _lookup(results, f"targets.{target}")
        c = _lookup(cells, f"targets.{target}")
        k = _lookup(kernels, f"targets.{target}")
        for name, block in (("results", r), ("cells", c), ("kernels", k)):
            if not isinstance(block, dict):
                e2.append(f"{name}.targets.{target} is missing")
        if not all(isinstance(b, dict) for b in (r, c, k)):
            continue

        for key, kind in (("positive_label_name", str), ("negative_label_name", str),
                          ("n_cells", int), ("n_positive", int), ("models", dict)):
            if not _is(r.get(key), kind):
                e2.append(f"results.{target}.{key} is missing or not {kind.__name__}")
        cell_list = c.get("cells")
        names = c.get("feature_names")
        if not isinstance(cell_list, list) or not isinstance(names, list):
            e2.append(f"cells.{target} needs feature_names and cells")
            continue

        ids = [cell.get("id") if isinstance(cell, dict) else None for cell in cell_list]
        if len(set(ids)) != len(ids) or None in ids:
            e4.append(f"{target}: cell ids are missing or not unique")
        usable = True
        for cell in cell_list:
            cid = cell.get("id") if isinstance(cell, dict) else "?"
            where = f"cells.{target}[{cid}]"
            if not isinstance(cell, dict):
                e2.append(f"{where} is not an object")
                usable = False
                continue
            if not (_is(cell.get("fold"), int) and cell.get("y_true") in (0, 1)
                    and _is(cell.get("annotation"), str)):
                e2.append(f"{where} needs fold, y_true (0/1) and annotation")
                usable = False
            pred = cell.get("pred")
            if not isinstance(pred, dict) or any(pred.get(m) not in (0, 1) for m in MODEL_KEYS):
                e4.append(f"{where} needs a 0/1 prediction for each of {MODEL_KEYS}")
                usable = False
            for group, keys in (("coords", COORD_KEYS), ("kpca", KERNEL_KEYS)):
                block = cell.get(group)
                if not isinstance(block, dict) or any(
                    not (isinstance(block.get(key), list) and len(block[key]) == 3
                         and all(_is(v, float) for v in block[key])) for key in keys
                ):
                    e2.append(f"{where}.{group} needs [x, y, z] for each of {keys}")
            feats = cell.get("features")
            if not (isinstance(feats, list) and len(feats) == len(names) and all(_is(v, float) for v in feats)):
                e2.append(f"{where}.features must have one number per feature name")
            if target == "synthetic":
                planted = cell.get("planted")
                if not (isinstance(planted, dict) and _is(planted.get("x1"), float) and _is(planted.get("x2"), float)):
                    e2.append(f"{where}.planted needs x1 and x2")
        if r.get("n_cells") != len(cell_list):
            e4.append(f"{target}: results say {r.get('n_cells')} cells, cells.json has {len(cell_list)}")
        if usable and r.get("n_positive") != sum(cell["y_true"] for cell in cell_list):
            e4.append(f"{target}: n_positive does not match the labels in cells.json")

        fold_ids = sorted({cell["fold"] for cell in cell_list}) if usable else []
        for k_id in fold_ids:
            n_pos = sum(1 for cell in cell_list if cell["fold"] == k_id and cell["y_true"] == 1)
            if n_pos < 3:
                w3.append(f"{target}: fold {k_id} has only {n_pos} rare-class cell(s)")

        for model in MODEL_KEYS:
            m = (r.get("models") or {}).get(model)
            where = f"results.{target}.{model}"
            if not isinstance(m, dict):
                e2.append(f"{where} is missing")
                continue
            if not _is(m.get("display_name"), str) or not _is(m.get("total_runtime_seconds"), float):
                e2.append(f"{where} needs display_name and total_runtime_seconds")
            for metric in METRIC_KEYS:
                agg = (m.get("aggregate") or {}).get(metric)
                if not (isinstance(agg, dict) and _is(agg.get("mean"), float) and _is(agg.get("std"), float)):
                    e2.append(f"{where}.aggregate.{metric} needs mean and std")
                elif not (0 <= agg["mean"] <= 1 and agg["std"] >= 0):
                    e3.append(f"{where}.aggregate.{metric} is out of range")
            for metric, interval in (m.get("ci95") or {}).items():
                if not (isinstance(interval, list) and len(interval) == 2
                        and all(_is(v, float) and 0 <= v <= 1 for v in interval)):
                    e3.append(f"{where}.ci95.{metric} must be [low, high] within 0..1")
            if not _is_confusion(m.get("pooled_confusion_matrix")):
                e3.append(f"{where}.pooled_confusion_matrix must be 2x2 non-negative integers")
            elif usable:
                recomputed = _confusion([c_["y_true"] for c_ in cell_list], [c_["pred"][model] for c_ in cell_list])
                if recomputed != m["pooled_confusion_matrix"]:
                    e5.append(f"{where}: pooled confusion matrix {m['pooled_confusion_matrix']} != {recomputed} from cells.json")
            folds = m.get("folds")
            if not isinstance(folds, list) or not folds:
                e2.append(f"{where}.folds is missing")
                continue
            listed = sorted(f["fold"] for f in folds if isinstance(f, dict) and _is(f.get("fold"), int))
            if usable and listed != fold_ids:
                e5.append(f"{where}: folds do not match the folds in cells.json {fold_ids}")
            for f in folds:
                if not isinstance(f, dict) or not _is(f.get("fold"), int):
                    e2.append(f"{where}.folds has an entry without a fold number")
                    continue
                fw = f"{where}.fold{f['fold']}"
                if not (isinstance(f.get("best_params"), dict) and _is(f.get("fit_seconds"), float)
                        and _is(f.get("predict_seconds"), float)):
                    e2.append(f"{fw} needs best_params, fit_seconds and predict_seconds")
                bad_metric = [mt for mt in METRIC_KEYS if not _is(f.get(mt), float)]
                if bad_metric:
                    e2.append(f"{fw} is missing {bad_metric}")
                elif any(not 0 <= f[mt] <= 1 for mt in METRIC_KEYS):
                    e3.append(f"{fw} has a metric outside 0..1")
                if not _is_confusion(f.get("confusion_matrix")):
                    e3.append(f"{fw}.confusion_matrix must be 2x2 non-negative integers")
                    continue
                if not usable or bad_metric:
                    continue
                members = [c_ for c_ in cell_list if c_["fold"] == f["fold"]]
                recomputed = _confusion([c_["y_true"] for c_ in members], [c_["pred"][model] for c_ in members])
                if recomputed != f["confusion_matrix"]:
                    e5.append(f"{fw}: confusion matrix {f['confusion_matrix']} != {recomputed} from cells.json")
                expected = _metrics_from_confusion(recomputed)
                for mt in METRIC_KEYS:
                    if abs(expected[mt] - f[mt]) > METRIC_TOLERANCE:
                        e6.append(f"{fw}.{mt} is {f[mt]:.4f} but cells.json gives {expected[mt]:.4f}")

        order = k.get("cell_ids")
        matrices = k.get("matrices")
        if not (isinstance(order, list) and isinstance(matrices, dict)
                and _is(k.get("fit_scope"), str) and isinstance(k.get("params"), dict)):
            e2.append(f"kernels.{target} needs cell_ids, fit_scope, params and matrices")
            continue
        unknown = set(order) - set(ids)
        if unknown or len(set(order)) != len(order):
            e7.append(f"{target}: kernel cell_ids are duplicated or not in cells.json ({len(unknown)} unknown)")
        for kernel in KERNEL_KEYS:
            matrix = matrices.get(kernel)
            if not (isinstance(matrix, list) and len(matrix) == len(order)
                    and all(isinstance(row, list) and len(row) == len(order) for row in matrix)):
                e7.append(f"{target}.{kernel}: matrix is not {len(order)}x{len(order)}")
                continue
            if not all(_is(v, float) for row in matrix for v in row):
                e7.append(f"{target}.{kernel}: matrix has a non-numeric value")
                continue
            size = len(order)
            if any(abs(matrix[i][j] - matrix[j][i]) > 1e-6 for i in range(size) for j in range(i)):
                w1.append(f"{target}.{kernel}: matrix is not symmetric")
            if kernel in ("rbf", "qsvm") and any(abs(matrix[i][i] - 1) > 0.01 for i in range(size)):
                w1.append(f"{target}.{kernel}: diagonal is not close to 1")

    check("E2", "every required field is present with the right type", e2)
    check("E3", "metrics are within 0..1 and confusion matrices are well formed", e3)
    check("E4", "cell ids are unique and every cell has four predictions", e4)
    check("E5", "confusion matrices match the per-cell predictions", e5)
    check("E6", "fold metrics match the per-cell predictions", e6)
    check("E7", "kernel matrices are square and use known cells", e7)
    check("W1", "kernel matrices are symmetric with a unit diagonal", w1, warn=True)
    leak = _lookup(manifest, "preprocessing.fitted_within_training_folds_only") is False
    check("W2", "feature transforms were fitted inside training folds only",
          ["preprocessing was NOT fitted within training folds only; the benchmark may leak"] if leak else [],
          warn=True)
    check("W3", "every fold has at least 3 rare-class cells", w3, warn=True)
    return _report(errors, warnings, passed)


def _report(errors: list, warnings: list, passed: list) -> dict:
    return {
        "errors": errors,
        "warnings": warnings,
        "passed": passed,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------
# Writing, reading, and the one-call entry point
# --------------------------------------------------------------------------


def write_bundle(
    out_dir: str | os.PathLike, bundle: Mapping, report: Mapping,
    circuit_image: str | os.PathLike | None = None,
    seqera_screenshot: str | os.PathLike | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, filename in BUNDLE_FILES.items():
        compact = key in ("cells", "kernels")
        text = json.dumps(
            bundle[key], allow_nan=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        (out / filename).write_text(text + "\n", encoding="utf-8")
    (out / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for source, stem in ((circuit_image, "circuit"), (seqera_screenshot, "seqera_run")):
        if source is not None:
            source = Path(source)
            shutil.copyfile(source, out / f"{stem}{source.suffix.lower()}")
    return out


def load_bundle(bundle_dir: str | os.PathLike) -> dict:
    folder = Path(bundle_dir)
    return {
        key: json.loads((folder / filename).read_text(encoding="utf-8"))
        for key, filename in BUNDLE_FILES.items()
    }


def export_bundle(
    out_dir: str | os.PathLike,
    manifest: Mapping,
    targets: Mapping[str, TargetInputs],
    circuit_image: str | os.PathLike | None = None,
    seqera_screenshot: str | os.PathLike | None = None,
    n_bootstrap: int = 2000,
) -> dict:
    """Build, validate and write the bundle. Nothing is written if it is invalid."""
    bundle, notes = build_bundle(manifest, targets, n_bootstrap=n_bootstrap)
    report = validate_bundle(bundle["manifest"], bundle["results"], bundle["cells"], bundle["kernels"])
    report["exporter_notes"] = notes
    if report["errors"]:
        raise BundleError(report["errors"])
    write_bundle(out_dir, bundle, report, circuit_image, seqera_screenshot)
    return report


def collect_software_versions(
    packages: Sequence[str] = (
        "numpy", "scipy", "scikit-learn", "scanpy", "anndata", "pandas",
        "qiskit", "QuantumRingsLib", "quantumrings-toolkit-qiskit",
    ),
) -> dict:
    """Installed versions for manifest['software_versions']; absent ones are skipped."""
    from importlib import metadata

    versions = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def save_circuit_image(circuit: Any, path: str | os.PathLike, fold: int = -1) -> Path:
    """Draw a Qiskit circuit (for example the decomposed ZZFeatureMap) to an image."""
    figure = circuit.draw(output="mpl", fold=fold)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    return Path(path)


# --------------------------------------------------------------------------
# Placeholder bundle: random, flagged data for building the dashboard before
# real results exist. Never mix this with a real run.
# --------------------------------------------------------------------------


def make_placeholder_inputs(seed: int = 0, n_cells: int = 60, n_folds: int = 5) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n_pos = max(3 * n_folds, n_cells // 4)

    def target(key: str) -> TargetInputs:
        y = np.zeros(n_cells, dtype=int)
        y[:n_pos] = 1
        fold = np.empty(n_cells, dtype=int)
        fold[:n_pos] = np.arange(n_pos) % n_folds
        fold[n_pos:] = np.arange(n_cells - n_pos) % n_folds
        features = rng.normal(size=(n_cells, 8)) + y[:, None] * 0.8
        sq = ((features[:, None, :] - features[None, :, :]) ** 2).sum(-1)
        models = {}
        for name, flip in (("linear", 0.25), ("poly2", 0.15), ("rbf", 0.1), ("qsvm", 0.12)):
            pred = np.where(rng.random(n_cells) < flip, 1 - y, y)
            models[name] = ModelRun(
                display_name=f"PLACEHOLDER {name}",
                pred=pred,
                fit_seconds={k: float(rng.uniform(0.01, 0.5)) for k in range(n_folds)},
                predict_seconds={k: float(rng.uniform(0.001, 0.05)) for k in range(n_folds)},
                config={"placeholder": True},
                best_params={k: {"C": 1} for k in range(n_folds)},
            )
        models["qsvm"].attempts = [
            QsvmAttempt(2, False, np.where(rng.random(n_cells) < 0.2, 1 - y, y), 1.0),
            QsvmAttempt(3, True, models["qsvm"].pred, 2.0),
        ]
        return TargetInputs(
            positive_label_name="PLACEHOLDER rare" if key == "biological" else "PLACEHOLDER planted",
            negative_label_name="PLACEHOLDER other" if key == "biological" else "PLACEHOLDER background",
            cell_ids=[f"PLACEHOLDER-{key[:3]}-{i:03d}" for i in range(n_cells)],
            y_true=y,
            fold=fold,
            annotation=["placeholder rare" if v else "placeholder other" for v in y],
            feature_names=[f"placeholder_f{i}" for i in range(8)],
            features=features,
            coords={"umap3": features[:, :3], "pca3": features[:, 3:6]},
            models=models,
            kernels={
                "rbf": np.exp(-0.1 * sq),
                "poly2": (0.1 * features @ features.T + 1.0) ** 2,
                "qsvm": np.exp(-0.3 * sq),  # NOT a quantum kernel: placeholder only
            },
            kernel_params={"rbf": {"gamma": 0.1}, "poly2": {"gamma": 0.1, "coef0": 1.0, "degree": 2},
                           "qsvm": {"placeholder": True}},
            kernel_fit_scope="PLACEHOLDER: random numbers, not a fitted kernel",
            planted=features[:, :2] if key == "synthetic" else None,
        )

    manifest = {
        "run_id": f"PLACEHOLDER-{seed}",
        "is_placeholder": True,
        "dataset": {"name": "PLACEHOLDER (random numbers)", "source_url": "", "n_cells_raw": 0,
                    "n_genes_raw": 0, "n_cells_after_qc": 0, "donor_status": "placeholder"},
        "annotation": {"method": "placeholder", "positive_class_name": "PLACEHOLDER rare",
                       "limitations": "Not real data."},
        "sampling": {"strategy": "placeholder", "seed": seed},
        "preprocessing": {"feature_pipeline": "placeholder", "fitted_within_training_folds_only": True},
        "cv": {"inner_folds": 0, "selection_metric": "f1_macro", "seed": seed},
        "quantum": {"sdk": "placeholder", "sdk_version": "0", "backend": "none", "n_qubits": 8,
                    "feature_map": "placeholder", "entanglement": "placeholder",
                    "attempts": [{"reps": 2, "is_final": False, "reason": "placeholder"},
                                 {"reps": 3, "is_final": True, "reason": "placeholder"}]},
        "synthetic_control": {"rule_description": "placeholder", "features_used": ["placeholder_f0", "placeholder_f1"],
                              "seed": seed},
        "pipeline_stages": [{"id": "placeholder", "name": "Placeholder", "tool": "none", "status": "not_run",
                             "started_at": None, "duration_seconds": 0.0, "artifact": "", "link": ""}],
        "software_versions": collect_software_versions(),
    }
    return manifest, {key: target(key) for key in TARGET_KEYS}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _print_report(report: Mapping) -> None:
    for line in report["passed"]:
        print(f"  pass  {line}")
    for line in report["warnings"]:
        print(f"  WARN  {line}")
    for line in report["errors"]:
        print(f"  FAIL  {line}")
    for line in report.get("exporter_notes", []):
        print(f"  note  {line}")
    print(f"{len(report['errors'])} error(s), {len(report['warnings'])} warning(s)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_validate = sub.add_parser("validate", help="check a bundle folder the way the dashboard will")
    p_validate.add_argument("bundle_dir")
    p_placeholder = sub.add_parser("placeholder", help="write a random bundle flagged is_placeholder")
    p_placeholder.add_argument("out_dir")
    p_placeholder.add_argument("--seed", type=int, default=0)
    p_placeholder.add_argument("--cells", type=int, default=60)
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            bundle = load_bundle(args.bundle_dir)
            report = validate_bundle(bundle["manifest"], bundle["results"], bundle["cells"], bundle["kernels"])
        else:
            manifest, targets = make_placeholder_inputs(seed=args.seed, n_cells=args.cells)
            report = export_bundle(args.out_dir, manifest, targets)
            print(f"Wrote placeholder bundle to {args.out_dir}")
    except BundleError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read the bundle: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
