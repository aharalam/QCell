import os

import numpy as np
import pytest

from quantum_kernel import (
    QuantumKernel,
    ReferenceEngine,
    build_feature_map,
    fidelity_kernel,
    load_env_file,
    quantum_rings_engine_from_env,
)


def angles(n_cells, n_features, seed=0):
    return np.random.default_rng(seed).uniform(0, np.pi, size=(n_cells, n_features))


def test_feature_map_has_one_qubit_and_one_parameter_per_feature():
    circuit = build_feature_map(n_qubits=8, reps=2, entanglement="full")
    assert circuit.num_qubits == 8 and circuit.num_parameters == 8


def test_more_repetitions_make_a_deeper_circuit():
    shallow, deep = build_feature_map(4, reps=1), build_feature_map(4, reps=3)
    assert deep.decompose().size() > shallow.decompose().size()


def test_two_qubit_kernel_matches_the_closed_form():
    # Two qubits, one repetition. After the Hadamards every basis state has amplitude
    # 1/2, and the map only adds phases: 2*x0 when qubit 0 is 1, 2*x1 when qubit 1 is 1,
    # and 2*(pi - x0)*(pi - x1) when exactly one of them is 1.
    def phases(x0, x1):
        zz = 2 * (np.pi - x0) * (np.pi - x1)
        return np.array([0.0, 2 * x0 + zz, 2 * x1 + zz, 2 * x0 + 2 * x1])

    x = np.array([[0.3, 2.0], [1.1, 0.4], [2.4, 2.9]])
    expected = np.array(
        [[abs(np.exp(1j * (phases(*a) - phases(*b))).sum() / 4) ** 2 for b in x] for a in x]
    )
    kernel = QuantumKernel(n_qubits=2, reps=1, engine=ReferenceEngine())
    states = kernel.states(x)
    assert kernel.kernel(states, states) == pytest.approx(expected, abs=1e-10)


def test_kernel_is_a_valid_similarity_matrix():
    kernel = QuantumKernel(n_qubits=4, reps=2, engine=ReferenceEngine())
    states = kernel.states(angles(12, 4))
    matrix = kernel.kernel(states, states)
    assert np.diag(matrix) == pytest.approx(np.ones(12), abs=1e-10)
    assert matrix == pytest.approx(matrix.T, abs=1e-12)
    assert matrix.min() >= -1e-12 and matrix.max() <= 1 + 1e-10
    assert np.linalg.eigvalsh(matrix).min() > -1e-9


def test_held_out_kernel_has_one_row_per_held_out_cell_and_one_column_per_training_cell():
    kernel = QuantumKernel(n_qubits=3, reps=2, engine=ReferenceEngine())
    train, test = kernel.states(angles(7, 3, seed=1)), kernel.states(angles(4, 3, seed=2))
    assert kernel.kernel(test, train).shape == (4, 7)
    assert kernel.kernel(test, train) == pytest.approx(kernel.kernel(train, test).T)


def test_repetitions_change_the_kernel():
    x = angles(6, 3, seed=3)
    two = QuantumKernel(3, reps=2, engine=ReferenceEngine())
    three = QuantumKernel(3, reps=3, engine=ReferenceEngine())
    assert not np.allclose(two.kernel(two.states(x), two.states(x)), three.kernel(three.states(x), three.states(x)))


def test_wrong_number_of_features_is_rejected():
    kernel = QuantumKernel(n_qubits=4, reps=2, engine=ReferenceEngine())
    with pytest.raises(ValueError, match="4 features"):
        kernel.states(angles(3, 5))


def test_circuits_run_are_counted():
    kernel = QuantumKernel(n_qubits=2, reps=1, engine=ReferenceEngine())
    kernel.states(angles(5, 2))
    kernel.states(angles(3, 2))
    assert kernel.circuits_run == 8


def test_fidelity_kernel_ignores_global_phase():
    states = QuantumKernel(2, reps=2, engine=ReferenceEngine()).states(angles(4, 2))
    rotated = states * np.exp(1j * np.array([0.3, 1.2, 2.5, 4.0]))[:, None]
    assert fidelity_kernel(rotated, states) == pytest.approx(fidelity_kernel(states, states))


def test_env_file_supplies_credentials_without_overriding_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nQUANTUM_RINGS_API_KEY = from-file \nQUANTUM_RINGS_ACCOUNT='me@example.org'\n")
    monkeypatch.delenv("QUANTUM_RINGS_API_KEY", raising=False)
    monkeypatch.setenv("QUANTUM_RINGS_ACCOUNT", "already-set")
    load_env_file(env)
    assert os.environ["QUANTUM_RINGS_API_KEY"] == "from-file"
    assert os.environ["QUANTUM_RINGS_ACCOUNT"] == "already-set"


def test_missing_credentials_give_a_message_that_names_the_variables(monkeypatch):
    monkeypatch.delenv("QUANTUM_RINGS_API_KEY", raising=False)
    monkeypatch.delenv("QUANTUM_RINGS_ACCOUNT", raising=False)
    with pytest.raises(RuntimeError) as caught:
        quantum_rings_engine_from_env()
    assert "QUANTUM_RINGS_API_KEY" in str(caught.value) and "QUANTUM_RINGS_ACCOUNT" in str(caught.value)


@pytest.mark.skipif(
    not (os.environ.get("QUANTUM_RINGS_API_KEY") and os.environ.get("QUANTUM_RINGS_ACCOUNT")),
    reason="needs Quantum Rings credentials in the environment",
)
@pytest.mark.parametrize("n_qubits,reps", [(2, 1), (3, 2), (8, 2), (8, 3)])
def test_quantum_rings_states_match_the_reference_simulation(n_qubits, reps):
    x = angles(4, n_qubits, seed=n_qubits + reps)
    reference = QuantumKernel(n_qubits, reps, engine=ReferenceEngine()).states(x)
    measured = QuantumKernel(n_qubits, reps, engine=quantum_rings_engine_from_env()).states(x)
    overlap = np.abs(np.sum(np.conj(reference) * measured, axis=1)) ** 2
    assert overlap == pytest.approx(np.ones(4), abs=1e-6)


def test_input_scale_shrinks_the_angles_before_they_are_loaded():
    kernel = QuantumKernel(n_qubits=3, reps=2, engine=ReferenceEngine())
    x = angles(4, 3, seed=9)
    assert kernel.states(x, scale=0.25) == pytest.approx(kernel.states(0.25 * x))
    assert kernel.states(x) == pytest.approx(kernel.states(x, scale=1.0))


def test_smaller_input_scale_undoes_kernel_concentration():
    kernel = QuantumKernel(n_qubits=8, reps=2, engine=ReferenceEngine())
    x = angles(10, 8, seed=10)
    off_diagonal = ~np.eye(10, dtype=bool)
    wide = kernel.kernel(kernel.states(x), kernel.states(x))[off_diagonal].mean()
    narrow = kernel.kernel(kernel.states(x, scale=0.05), kernel.states(x, scale=0.05))[off_diagonal].mean()
    assert wide < 0.02 and narrow > 10 * wide and narrow > 0.15


def test_feature_map_qubits_belong_to_a_named_register():
    # The Quantum Rings translator looks qubits up by register name; Qiskit's
    # zz_feature_map returns register-less qubits, which it cannot translate.
    circuit = build_feature_map(n_qubits=4, reps=2)
    assert len(circuit.qregs) == 1 and circuit.qregs[0].size == 4
    assert all(circuit.find_bit(q).registers for q in circuit.qubits)


def test_parallel_workers_return_the_same_states_in_the_same_order():
    from quantum_kernel import ParallelQuantumKernel

    x = angles(23, 4, seed=21)
    serial = QuantumKernel(4, reps=2, engine=ReferenceEngine())
    with ParallelQuantumKernel(4, reps=2, engine_kind="reference", workers=3, scales=(1.0, 0.1)) as parallel:
        assert parallel.scales == (1.0, 0.1)
        assert parallel.states(x, scale=0.1) == pytest.approx(serial.states(x, scale=0.1), abs=1e-12)
        assert parallel.circuits_run == 23
        assert parallel.kernel(parallel.states(x), parallel.states(x)) == pytest.approx(
            serial.kernel(serial.states(x), serial.states(x)), abs=1e-10)
        with pytest.raises(ValueError, match="4 features"):
            parallel.states(angles(3, 5))
