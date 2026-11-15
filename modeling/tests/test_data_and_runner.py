import json
from pathlib import Path

import numpy as np
import pytest

import export_results as er
from data import POSITIVE_ANNOTATION, load_pbmc, stratified_sample

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
needs_data = pytest.mark.skipif(
    not (DATA_DIR / "pbmc3k_raw.h5ad").is_file(), reason="PBMC 3k has not been downloaded yet"
)


def test_sample_keeps_every_rare_cell_and_hits_the_requested_size():
    y = np.array([1] * 12 + [0] * 300)
    picked = stratified_sample(y, n_total=80, seed=4)
    assert len(picked) == 80 and len(set(picked.tolist())) == 80
    assert y[picked].sum() == 12
    assert np.array_equal(picked, np.sort(picked))


def test_sample_is_reproducible_and_changes_with_the_seed():
    y = np.array([1] * 12 + [0] * 300)
    assert np.array_equal(stratified_sample(y, 80, seed=4), stratified_sample(y, 80, seed=4))
    assert not np.array_equal(stratified_sample(y, 80, seed=4), stratified_sample(y, 80, seed=5))


def test_sample_smaller_than_the_rare_class_is_refused():
    with pytest.raises(ValueError, match="rare"):
        stratified_sample(np.array([1] * 12 + [0] * 300), n_total=10, seed=0)


@needs_data
def test_pbmc_counts_are_raw_and_labels_follow_the_barcodes():
    pbmc = load_pbmc(DATA_DIR)
    assert pbmc.counts.shape[0] == len(pbmc.barcodes) == len(pbmc.annotation) == 2638
    assert pbmc.stats["n_cells_raw"] == 2700 and pbmc.stats["n_genes_raw"] == 32738
    assert int(np.sum(pbmc.annotation == POSITIVE_ANNOTATION)) == 37
    block = pbmc.counts[:50].toarray()
    assert block.min() >= 0 and np.array_equal(block, np.round(block))  # raw integer counts
    assert pbmc.counts.shape[1] == len(pbmc.genes) and (pbmc.counts.getnnz(axis=0) >= 3).all()


@needs_data
def test_runner_writes_a_bundle_the_dashboard_accepts(tmp_path, monkeypatch):
    import runner

    monkeypatch.setenv("QUANTUM_RINGS_API_KEY", "not-a-real-key-0123456789")
    run_dir = runner.main([
        "--engine", "reference", "--n-cells", "70", "--n-hvg", "200",
        "--outer-folds", "3", "--inner-folds", "2", "--data-dir", str(DATA_DIR),
        "--out", str(tmp_path), "--run-id", "test-run", "--no-umap",
    ])
    bundle_dir = Path(run_dir) / "dashboard_bundle"
    bundle = er.load_bundle(bundle_dir)
    report = er.validate_bundle(bundle["manifest"], bundle["results"], bundle["cells"], bundle["kernels"])
    assert report["errors"] == []

    manifest = bundle["manifest"]
    assert manifest["is_placeholder"] is False
    assert manifest["sampling"]["n_benchmark_cells"] == 70 and manifest["sampling"]["n_positive"] == 37
    assert manifest["preprocessing"]["fitted_within_training_folds_only"] is True
    assert "NOT Quantum Rings" in manifest["quantum"]["backend"]  # reference engine is labelled as such
    assert "not-a-real-key" not in json.dumps(bundle)

    # Repetitions and input scale are tuned per outer fold on inner folds: there is no
    # retry on held-out scores, so results carry no "attempts" and each fold records its choice.
    quantum = manifest["quantum"]
    assert quantum["reps_grid"] == [1, 2, 3] and 1.0 in quantum["input_scale_grid"]
    assert [a["is_final"] for a in quantum["attempts"]] == [True]
    for target in ("biological", "synthetic"):
        qsvm = bundle["results"]["targets"][target]["models"]["qsvm"]
        assert "attempts" not in qsvm
        for fold in qsvm["folds"]:
            assert fold["best_params"]["reps"] in quantum["reps_grid"]
            assert fold["best_params"]["scale"] in quantum["input_scale_grid"]
        assert qsvm["config"]["grid"]["reps"] == [1, 2, 3]

    synthetic = bundle["cells"]["targets"]["synthetic"]["cells"]
    for cell in synthetic:
        x1, x2 = cell["planted"]["x1"], cell["planted"]["x2"]
        assert cell["y_true"] == int((x1 > 0) != (x2 > 0))
    assert (bundle_dir / "circuit.png").is_file()
