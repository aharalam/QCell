import json

import numpy as np
import pytest

from collapse_explorer import collapse_matrices, pick_subset
from quantum_kernel import QuantumKernel, ReferenceEngine


def test_subset_keeps_both_classes_with_the_rare_class_last():
    y = np.array([1] * 30 + [0] * 170)
    ids = np.array([f"c{i:03d}" for i in range(200)])
    picked = pick_subset(y, ids, max_cells=40, seed=3)
    assert len(picked) == 40 and len(set(picked.tolist())) == 40
    labels = y[picked]
    assert 0 < labels.sum() < 40 and list(labels) == sorted(labels)
    assert np.array_equal(picked, pick_subset(y, ids, max_cells=40, seed=3))


def test_subset_is_everything_when_there_are_fewer_cells_than_the_limit():
    y = np.array([1, 0, 0, 1, 0])
    picked = pick_subset(y, np.array(list("edcba")), max_cells=40, seed=0)
    assert sorted(picked.tolist()) == [0, 1, 2, 3, 4] and list(y[picked]) == [0, 0, 0, 1, 1]


def test_similarity_washes_out_as_the_input_scale_grows():
    x = np.random.default_rng(1).uniform(0, np.pi, size=(14, 8))
    kernel = QuantumKernel(8, reps=2, engine=ReferenceEngine())
    out = collapse_matrices(x, kernel, scales=(1.0, 0.1, 0.05))
    assert [entry["scale"] for entry in out] == [1.0, 0.1, 0.05]
    means = []
    for entry in out:
        matrix = np.array(entry["matrix"])
        assert matrix.shape == (14, 14)
        assert np.diag(matrix) == pytest.approx(np.ones(14), abs=1e-4)
        assert matrix == pytest.approx(matrix.T, abs=1e-9)
        off = matrix[~np.eye(14, dtype=bool)]
        assert entry["off_diagonal_mean"] == pytest.approx(off.mean(), abs=1e-5)
        assert entry["off_diagonal_max"] == pytest.approx(off.max(), abs=1e-5)
        means.append(entry["off_diagonal_mean"])
    assert means[0] < 0.05 and means[0] < means[1] < means[2]
    json.dumps(out)  # plain numbers and lists only
