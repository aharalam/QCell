"""Tests for export_results.py. All inputs here are random placeholder data."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import export_results as er  # noqa: E402


@pytest.fixture()
def inputs():
    return er.make_placeholder_inputs(seed=7, n_cells=60)


@pytest.fixture()
def bundle(inputs):
    manifest, targets = inputs
    built, _ = er.build_bundle(manifest, targets, n_bootstrap=200)
    return built


def validate(b):
    return er.validate_bundle(b["manifest"], b["results"], b["cells"], b["kernels"])


def codes(messages):
    return {m.split()[0] for m in messages}


# ---- the happy path -------------------------------------------------------


def test_export_writes_a_bundle_that_passes_every_check(inputs, tmp_path):
    manifest, targets = inputs
    report = er.export_bundle(tmp_path / "bundle", manifest, targets, n_bootstrap=200)
    assert report["errors"] == []
    for filename in er.BUNDLE_FILES.values():
        assert (tmp_path / "bundle" / filename).is_file()
    reloaded = er.load_bundle(tmp_path / "bundle")
    assert validate(reloaded)["errors"] == []
    assert reloaded["manifest"]["is_placeholder"] is True


def test_fold_metrics_are_exactly_what_scikit_learn_reports(inputs, bundle):
    from sklearn.metrics import balanced_accuracy_score, f1_score

    _, targets = inputs
    for key, target in targets.items():
        y, fold = np.asarray(target.y_true), np.asarray(target.fold)
        for name in er.MODEL_KEYS:
            pred = np.asarray(target.models[name].pred)
            for entry in bundle["results"]["targets"][key]["models"][name]["folds"]:
                mask = fold == entry["fold"]
                assert entry["balanced_accuracy"] == pytest.approx(
                    balanced_accuracy_score(y[mask], pred[mask]), abs=1e-6)
                assert entry["macro_f1"] == pytest.approx(
                    f1_score(y[mask], pred[mask], average="macro", zero_division=0), abs=1e-6)


def test_aggregate_is_mean_and_sample_std_of_the_folds(bundle):
    model = bundle["results"]["targets"]["biological"]["models"]["rbf"]
    values = [f["macro_f1"] for f in model["folds"]]
    assert model["aggregate"]["macro_f1"]["mean"] == pytest.approx(np.mean(values), abs=1e-5)
    assert model["aggregate"]["macro_f1"]["std"] == pytest.approx(np.std(values, ddof=1), abs=1e-5)


def test_predictions_and_labels_reach_cells_json_unchanged(inputs, bundle):
    _, targets = inputs
    for key, target in targets.items():
        cells = bundle["cells"]["targets"][key]["cells"]
        assert [c["id"] for c in cells] == list(target.cell_ids)
        assert [c["y_true"] for c in cells] == list(np.asarray(target.y_true))
        for name in er.MODEL_KEYS:
            assert [c["pred"][name] for c in cells] == list(np.asarray(target.models[name].pred))


def test_kernels_are_reordered_consistently_with_rare_class_last(inputs, bundle):
    _, targets = inputs
    target = targets["biological"]
    block = bundle["kernels"]["targets"]["biological"]
    position = {cid: i for i, cid in enumerate(target.cell_ids)}
    labels = [int(np.asarray(target.y_true)[position[cid]]) for cid in block["cell_ids"]]
    assert labels == sorted(labels)
    original = np.asarray(target.kernels["qsvm"])
    for a in (0, 5, len(labels) - 1):
        for b in (1, 17, len(labels) - 2):
            i, j = position[block["cell_ids"][a]], position[block["cell_ids"][b]]
            assert block["matrices"]["qsvm"][a][b] == pytest.approx(original[i, j], abs=1e-5)


def test_kernel_pca_reproduces_the_centered_kernel_geometry():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 3))  # a linear kernel on 3-D points has rank 3
    coords, note = er.kernel_pca_3d(points @ points.T)
    assert note is None
    centered = points - points.mean(axis=0)
    assert coords @ coords.T == pytest.approx(centered @ centered.T, abs=1e-8)


def test_kernel_pca_survives_a_kernel_that_is_not_positive_semidefinite():
    rng = np.random.default_rng(1)
    noisy = rng.normal(size=(25, 25))
    coords, note = er.kernel_pca_3d((noisy + noisy.T) / 2)
    assert coords.shape == (25, 3) and np.all(np.isfinite(coords))
    assert note is not None


def test_every_attempt_is_exported_including_the_one_that_was_not_final(bundle):
    attempts = bundle["results"]["targets"]["synthetic"]["models"]["qsvm"]["attempts"]
    assert sorted((a["reps"], a["is_final"]) for a in attempts) == [(2, False), (3, True)]


def test_bootstrap_interval_brackets_the_pooled_metric(bundle):
    model = bundle["results"]["targets"]["biological"]["models"]["rbf"]
    (tn, fp), (fn, tp) = model["pooled_confusion_matrix"]
    pooled = (tp / (tp + fn) + tn / (tn + fp)) / 2
    low, high = model["ci95"]["balanced_accuracy"]
    assert 0 <= low <= pooled <= high <= 1


def test_measured_counts_are_written_into_the_manifest(bundle):
    manifest = bundle["manifest"]
    assert manifest["sampling"]["n_benchmark_cells"] == 60
    assert manifest["sampling"]["n_positive"] == 15
    assert manifest["cv"]["outer_folds"] == 5
    assert manifest["preprocessing"]["n_features"] == 8


# ---- the exporter refuses bad inputs --------------------------------------


def expect_problem(manifest, targets, fragment):
    with pytest.raises(er.BundleError) as caught:
        er.build_bundle(manifest, targets, n_bootstrap=0)
    assert any(fragment in p for p in caught.value.problems), caught.value.problems


def test_missing_timing_is_an_error_not_a_default(inputs):
    manifest, targets = inputs
    del targets["biological"].models["rbf"].fit_seconds[2]
    expect_problem(manifest, targets, "no measured fit_seconds")


def test_a_fold_without_rare_cells_is_rejected(inputs):
    manifest, targets = inputs
    fold = np.asarray(targets["biological"].fold).copy()
    y = np.asarray(targets["biological"].y_true)
    fold[(fold == 0) & (y == 1)] = 1
    targets["biological"].fold = fold
    expect_problem(manifest, targets, "does not contain both classes")


def test_non_binary_predictions_are_rejected(inputs):
    manifest, targets = inputs
    pred = np.asarray(targets["synthetic"].models["qsvm"].pred).copy()
    pred[0] = 2
    targets["synthetic"].models["qsvm"].pred = pred
    expect_problem(manifest, targets, "must be 0 or 1")


def test_nan_in_a_kernel_is_rejected(inputs):
    manifest, targets = inputs
    kernel = np.asarray(targets["biological"].kernels["qsvm"]).copy()
    kernel[0, 0] = np.nan
    targets["biological"].kernels["qsvm"] = kernel
    expect_problem(manifest, targets, "NaN or infinity")


def test_final_attempt_must_be_the_reported_qsvm(inputs):
    manifest, targets = inputs
    qsvm = targets["biological"].models["qsvm"]
    qsvm.attempts[1].pred = 1 - np.asarray(qsvm.pred)
    expect_problem(manifest, targets, "final attempt's predictions differ")


def test_hiding_an_attempt_from_the_manifest_is_rejected(inputs):
    manifest, targets = inputs
    manifest["quantum"]["attempts"] = [{"reps": 3, "is_final": True, "reason": "only one shown"}]
    expect_problem(manifest, targets, "every attempt run must be declared")


def test_manifest_counts_that_disagree_with_the_data_are_rejected(inputs):
    manifest, targets = inputs
    manifest["sampling"]["n_positive"] = 99
    expect_problem(manifest, targets, "the data says 15")


def test_credentials_in_the_manifest_are_rejected(inputs, monkeypatch):
    manifest, targets = inputs
    manifest["quantum"]["api_key"] = "abc"
    expect_problem(manifest, targets, "looks like a credential")
    del manifest["quantum"]["api_key"]
    monkeypatch.setenv("QUANTUM_RINGS_API_KEY", "qr-secret-value-123")
    manifest["quantum"]["backend"] = "scarlet qr-secret-value-123"
    expect_problem(manifest, targets, "contains the Quantum Rings API key")


def test_nothing_is_written_when_the_bundle_is_invalid(inputs, tmp_path):
    manifest, targets = inputs
    del manifest["dataset"]["name"]
    with pytest.raises(er.BundleError):
        er.export_bundle(tmp_path / "bundle", manifest, targets, n_bootstrap=0)
    assert not (tmp_path / "bundle").exists()


# ---- the validator catches tampering after the fact ------------------------


def test_editing_a_prediction_after_export_is_caught(bundle):
    tampered = copy.deepcopy(bundle)
    cell = tampered["cells"]["targets"]["biological"]["cells"][0]
    cell["pred"]["qsvm"] = 1 - cell["pred"]["qsvm"]
    assert "E5" in codes(validate(tampered)["errors"])


def test_inflating_a_fold_score_is_caught(bundle):
    tampered = copy.deepcopy(bundle)
    fold = tampered["results"]["targets"]["biological"]["models"]["qsvm"]["folds"][0]
    fold["macro_f1"] = min(1.0, fold["macro_f1"] + 0.05) if fold["macro_f1"] < 0.95 else fold["macro_f1"] - 0.05
    assert "E6" in codes(validate(tampered)["errors"])


def test_mismatched_run_ids_are_caught(bundle):
    tampered = copy.deepcopy(bundle)
    tampered["kernels"]["run_id"] = "another-run"
    assert "E1" in codes(validate(tampered)["errors"])


def test_out_of_range_metric_is_caught(bundle):
    tampered = copy.deepcopy(bundle)
    tampered["results"]["targets"]["synthetic"]["models"]["linear"]["aggregate"]["macro_f1"]["mean"] = 1.2
    assert "E3" in codes(validate(tampered)["errors"])


def test_dropping_a_models_prediction_is_caught(bundle):
    tampered = copy.deepcopy(bundle)
    del tampered["cells"]["targets"]["synthetic"]["cells"][3]["pred"]["poly2"]
    assert "E4" in codes(validate(tampered)["errors"])


def test_truncated_kernel_is_caught(bundle):
    tampered = copy.deepcopy(bundle)
    tampered["kernels"]["targets"]["biological"]["matrices"]["qsvm"].pop()
    assert "E7" in codes(validate(tampered)["errors"])


def test_leakage_flag_and_small_folds_raise_warnings_not_errors(inputs):
    manifest, targets = inputs
    manifest["preprocessing"]["fitted_within_training_folds_only"] = False
    built, _ = er.build_bundle(manifest, targets, n_bootstrap=0)
    report = validate(built)
    assert report["errors"] == []
    assert "W2" in codes(report["warnings"])

    manifest2, targets2 = er.make_placeholder_inputs(seed=3, n_cells=60)
    y = np.asarray(targets2["biological"].y_true).copy()
    fold = np.asarray(targets2["biological"].fold)
    rare_in_fold0 = np.where((y == 1) & (fold == 0))[0]
    y[rare_in_fold0[:2]] = 0  # leave one rare cell in fold 0
    targets2["biological"].y_true = y
    report2 = validate(er.build_bundle(manifest2, targets2, n_bootstrap=0)[0])
    assert report2["errors"] == []
    assert "W3" in codes(report2["warnings"])


def test_asymmetric_kernel_is_exported_as_given_and_warned_about(inputs):
    manifest, targets = inputs
    kernel = np.asarray(targets["biological"].kernels["rbf"]).copy()
    kernel[0, 1] += 0.01
    targets["biological"].kernels["rbf"] = kernel
    built, notes = er.build_bundle(manifest, targets, n_bootstrap=0)
    assert any("asymmetric" in n for n in notes)
    assert "W1" in codes(validate(built)["warnings"])


# ---- command line -----------------------------------------------------------


def test_cli_placeholder_then_validate(tmp_path, capsys):
    out = tmp_path / "bundle"
    assert er.main(["placeholder", str(out), "--seed", "1"]) == 0
    assert er.main(["validate", str(out)]) == 0
    assert "0 error(s)" in capsys.readouterr().out

    cells = json.loads((out / "cells.json").read_text())
    cells["targets"]["biological"]["cells"][0]["y_true"] ^= 1
    (out / "cells.json").write_text(json.dumps(cells))
    assert er.main(["validate", str(out)]) == 1


# ---- optional third target ---------------------------------------------------


def third_target(inputs):
    """A copy of the biological placeholder target with 3 features and an extra per-cell value."""
    import copy as _copy

    _, targets = inputs
    extra = _copy.deepcopy(targets["biological"])
    extra.feature_names = ["a", "b", "c"]
    extra.features = np.asarray(extra.features)[:, :3]
    extra.extra = {"expectation": np.linspace(-1, 1, len(extra.cell_ids))}
    return extra


def test_a_third_target_is_exported_and_validated_like_the_others(inputs):
    manifest, targets = inputs
    targets["quantum_structured"] = third_target(inputs)
    built, _ = er.build_bundle(manifest, targets, n_bootstrap=0)
    assert set(built["results"]["targets"]) == {"biological", "synthetic", "quantum_structured"}
    cells = built["cells"]["targets"]["quantum_structured"]
    assert cells["feature_names"] == ["a", "b", "c"] and len(cells["cells"][0]["features"]) == 3
    assert cells["cells"][0]["extra"]["expectation"] == pytest.approx(-1.0)
    assert "extra" not in built["cells"]["targets"]["biological"]["cells"][0]
    assert validate(built)["errors"] == []


def test_tampering_with_the_third_target_is_caught(inputs):
    manifest, targets = inputs
    targets["quantum_structured"] = third_target(inputs)
    built, _ = er.build_bundle(manifest, targets, n_bootstrap=0)
    cell = built["cells"]["targets"]["quantum_structured"]["cells"][0]
    cell["pred"]["qsvm"] = 1 - cell["pred"]["qsvm"]
    assert "E5" in codes(validate(built)["errors"])


def test_an_unknown_target_name_is_refused(inputs):
    manifest, targets = inputs
    targets["mystery"] = targets["biological"]
    expect_problem(manifest, targets, "unknown target")
