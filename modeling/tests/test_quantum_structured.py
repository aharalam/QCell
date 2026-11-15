import numpy as np
import pytest

from quantum_structured import N_QUBITS, REPS, SCALE, circuit_structured_labels


def angles(n=80, seed=0):
    return np.random.default_rng(seed).uniform(0, np.pi, size=(n, N_QUBITS))


def test_the_generating_circuit_is_the_original_spec_setting_on_three_qubits():
    assert (N_QUBITS, REPS, SCALE) == (3, 2, 1.0)


def test_label_is_one_exactly_when_the_measured_value_is_above_the_median():
    labels, expectation = circuit_structured_labels(angles(), seed=5)
    assert expectation.shape == (80,) and np.all(np.abs(expectation) <= 1 + 1e-9)
    assert np.array_equal(labels, (expectation > np.median(expectation)).astype(int))
    assert labels.sum() == 40


def test_same_seed_gives_the_same_measurement_and_a_different_seed_does_not():
    x = angles(seed=1)
    assert np.array_equal(circuit_structured_labels(x, seed=5)[1], circuit_structured_labels(x, seed=5)[1])
    assert not np.allclose(circuit_structured_labels(x, seed=5)[1], circuit_structured_labels(x, seed=6)[1])


def test_a_cells_measured_value_depends_only_on_its_own_features():
    x = angles(seed=2)
    order = np.random.default_rng(0).permutation(len(x))
    full = circuit_structured_labels(x, seed=5)[1]
    assert circuit_structured_labels(x[order], seed=5)[1] == pytest.approx(full[order])
    assert circuit_structured_labels(x[:10], seed=5)[1] == pytest.approx(full[:10])


def test_wrong_number_of_features_is_rejected():
    with pytest.raises(ValueError, match="3 features"):
        circuit_structured_labels(np.zeros((5, 8)), seed=0)
