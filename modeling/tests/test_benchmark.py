import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

from benchmark import CLASSICAL_MODELS, fold_features, make_folds, run_nested_cv


def toy(kind, n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 2))
    y = (x[:, 0] > 0).astype(int) if kind == "linear" else ((x[:, 0] > 0) != (x[:, 1] > 0)).astype(int)
    return x, y


def precomputed(x):
    """Stands in for per-fold feature construction when features are already final."""
    return lambda train, test: (x[train], x[test])


class FixedRbf:
    """A 'quantum' kernel that is really an RBF, to check the precomputed-kernel wiring."""

    def states(self, features):
        return features

    def kernel(self, a, b):
        return rbf_kernel(a, b, gamma=1.0)


def test_folds_are_stratified_reproducible_and_cover_every_cell_once():
    y = np.array([1] * 15 + [0] * 85)
    folds = make_folds(y, n_folds=5, seed=3)
    assert [sorted(np.concatenate([a, b])) for a, b in folds] == [list(range(100))] * 5
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(100))
    assert all(y[test].sum() == 3 for _, test in folds)
    again = make_folds(y, n_folds=5, seed=3)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_feature_pipeline_is_fitted_on_training_cells_only():
    seen = []

    class Spy:
        def fit(self, rows):
            seen.append(rows.copy())
            return self

        def transform(self, rows):
            return rows[:, :2]

    counts = np.arange(40, dtype=float).reshape(10, 4)
    train, test = np.array([0, 1, 2, 3, 4, 5, 6]), np.array([7, 8, 9])
    x_train, x_test = fold_features(counts, train, test, Spy)
    assert len(seen) == 1 and np.array_equal(seen[0], counts[train])
    assert np.array_equal(x_train, counts[train, :2]) and np.array_equal(x_test, counts[test, :2])


def test_every_cell_gets_one_out_of_fold_prediction_and_measured_timings():
    x, y = toy("linear")
    folds = make_folds(y, n_folds=4, seed=0)
    out = run_nested_cv(y, folds, precomputed(x), seed=0, inner_folds=3)
    assert set(out) == set(CLASSICAL_MODELS)
    for result in out.values():
        assert result.pred.shape == y.shape and set(np.unique(result.pred)) <= {0, 1}
        assert result.score.shape == y.shape
        assert sorted(result.fit_seconds) == sorted(result.predict_seconds) == [0, 1, 2, 3]
        assert all(v > 0 for v in result.fit_seconds.values())
        assert sorted(result.best_params) == [0, 1, 2, 3]


def test_held_out_labels_cannot_influence_tuning_or_predictions():
    x, y = toy("xor", seed=1)
    folds = make_folds(y, n_folds=4, seed=0)
    honest = run_nested_cv(y, folds, precomputed(x), seed=0, inner_folds=3)
    train, test = folds[0]
    corrupted = y.copy()
    corrupted[test] = 1 - corrupted[test]
    tampered = run_nested_cv(corrupted, [folds[0]], precomputed(x), seed=0, inner_folds=3)
    for name in CLASSICAL_MODELS:
        assert np.array_equal(tampered[name].pred[test], honest[name].pred[test])
        assert tampered[name].best_params[0] == honest[name].best_params[0]


def test_curved_kernels_solve_an_interaction_a_straight_line_cannot():
    x, y = toy("xor", n=200, seed=2)
    out = run_nested_cv(y, make_folds(y, 5, seed=0), precomputed(x), seed=0, inner_folds=3)
    score = {name: balanced_accuracy_score(y, result.pred) for name, result in out.items()}
    assert score["linear"] < 0.7
    assert score["rbf"] > 0.85 and score["poly2"] > 0.85


def test_quantum_kernel_is_used_as_a_precomputed_kernel_on_the_same_folds():
    x, y = toy("xor", n=120, seed=3)
    folds = make_folds(y, 4, seed=0)
    out = run_nested_cv(y, folds, precomputed(x), seed=0, inner_folds=3, quantum=FixedRbf())
    assert "qsvm" in out
    assert balanced_accuracy_score(y, out["qsvm"].pred) > 0.85
    assert sorted(out["qsvm"].fit_seconds) == [0, 1, 2, 3]
    assert set(out["qsvm"].best_params[0]) == {"C"}


def test_class_weights_are_balanced_so_a_rare_class_is_not_ignored():
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal(0, 1, size=(180, 2)), rng.normal(2.2, 1, size=(20, 2))])
    y = np.array([0] * 180 + [1] * 20)
    out = run_nested_cv(y, make_folds(y, 5, seed=0), precomputed(x), seed=0, inner_folds=3)
    recall = np.mean(out["linear"].pred[y == 1])
    assert recall > 0.7


def test_a_fold_missing_a_class_is_refused():
    y = np.array([0] * 20 + [1] * 2)
    with pytest.raises(ValueError, match="rare"):
        make_folds(y, n_folds=5, seed=0)


class ScaledRbf:
    """Kernel whose usefulness depends on the input scale, like a quantum bandwidth."""

    scales = (1.0, 0.03, 0.0001)

    def states(self, features, scale=1.0):
        return features * scale

    def kernel(self, a, b):
        return rbf_kernel(a, b, gamma=2000.0)


def test_quantum_input_scale_is_chosen_on_inner_folds_and_recorded():
    # At scale 1.0 this kernel is the identity (every cell unlike every other) and at
    # 0.0001 it is constant; only the middle scale carries the XOR structure.
    x, y = toy("xor", n=160, seed=4)
    folds = make_folds(y, 4, seed=0)
    out = run_nested_cv(y, folds, precomputed(x), seed=0, inner_folds=3, quantum=ScaledRbf(), classical=False)
    assert all(params["scale"] == 0.03 for params in out["qsvm"].best_params.values())
    assert all(set(params) == {"C", "scale"} for params in out["qsvm"].best_params.values())
    assert balanced_accuracy_score(y, out["qsvm"].pred) > 0.85


def test_quantum_input_scale_choice_ignores_held_out_labels():
    x, y = toy("xor", n=160, seed=5)
    folds = make_folds(y, 4, seed=0)
    honest = run_nested_cv(y, folds, precomputed(x), seed=0, inner_folds=3, quantum=ScaledRbf(), classical=False)
    train, test = folds[0]
    corrupted = y.copy()
    corrupted[test] = 1 - corrupted[test]
    tampered = run_nested_cv(corrupted, [folds[0]], precomputed(x), seed=0, inner_folds=3,
                             quantum=ScaledRbf(), classical=False)
    assert tampered["qsvm"].best_params[0] == honest["qsvm"].best_params[0]
    assert np.array_equal(tampered["qsvm"].pred[test], honest["qsvm"].pred[test])
