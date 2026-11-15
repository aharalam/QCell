"""Quantum fidelity kernel from a ZZ feature map.

Each cell's 8 features are loaded into an 8-qubit circuit as rotation angles.
The simulator returns the resulting quantum state, and the kernel entry for
two cells is the squared overlap of their states, |<psi(x)|psi(y)>|^2.

Two engines produce the states:
  - QuantumRingsEngine: the Quantum Rings SDK simulator (used for results).
  - ReferenceEngine: Qiskit's exact statevector (used only to check the above).

The kernel here is exact state overlap from a simulator, with no shot noise
and no quantum hardware. One circuit is run per cell, not per pair of cells.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

KEY_VARIABLE = "QUANTUM_RINGS_API_KEY"
ACCOUNT_VARIABLE = "QUANTUM_RINGS_ACCOUNT"


def build_feature_map(n_qubits: int, reps: int, entanglement: str = "full") -> QuantumCircuit:
    """ZZ feature map on a circuit with a named qubit register.

    Qiskit's zz_feature_map returns qubits that belong to no register, which the
    Quantum Rings translator cannot handle, so the gates are copied onto a
    standard registered circuit. The gates and parameters are unchanged.
    """
    circuit = QuantumCircuit(n_qubits, name="zz_feature_map")
    circuit.compose(zz_feature_map(n_qubits, reps=reps, entanglement=entanglement), inplace=True)
    return circuit


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """|<a_i|b_j>|^2 for every pair: rows follow states_a, columns states_b."""
    return np.abs(np.conj(states_a) @ states_b.T) ** 2


class ReferenceEngine:
    """Exact local simulation with Qiskit. For verification, not for reported results."""

    name = "qiskit-statevector-reference"

    def prepare(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return circuit

    def statevector(self, bound: QuantumCircuit) -> np.ndarray:
        return np.asarray(Statevector(bound).data)


class QuantumRingsEngine:
    """Runs each bound circuit on the Quantum Rings simulator and reads its state."""

    def __init__(self, token: str, account: str, precision: str = "double",
                 performance: str = "HIGHESTACCURACY"):
        import QuantumRingsLib
        from quantumrings.toolkit.qiskit import QrBackendV2, QrTranslator

        self._lib = QuantumRingsLib
        self._translator = QrTranslator
        provider = QuantumRingsLib.QuantumRingsProvider(token=token, name=account)
        self._backend = QrBackendV2(provider, precision=precision)
        self._native = self._backend._qr_backend
        self.precision = precision
        self.performance = performance
        self.name = f"quantum-rings:{getattr(self._backend, 'name', 'scarlet_quantum_rings')}"

    def prepare(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return transpile(circuit, basis_gates=self._native._basis_gates, optimization_level=1)

    def statevector(self, bound: QuantumCircuit) -> np.ndarray:
        n = bound.num_qubits
        native = self._lib.QuantumCircuit(
            self._lib.QuantumRegister(n, "q"),
            self._lib.ClassicalRegister(n, "meas"),
            name=bound.name,
            global_phase=float(bound.global_phase),
        )
        self._translator().translate_quantum_circuit(bound, native, False)
        job = self._native.run(
            native, shots=1, mode="sync", performance=self.performance,
            precision=self.precision, quiet=True, defaults=True, transfer_to_cpu=True,
        )
        # mode="sync" returns a finished job, so no polling wait is needed (verified to
        # give identical states and about a third less time per circuit).
        return np.asarray(job.result().get_statevector()).reshape(-1)


class QuantumKernel:
    def __init__(self, n_qubits: int, reps: int, entanglement: str = "full", engine=None,
                 scales=(1.0,)):
        self.scales = tuple(scales)  # candidate input scales, chosen on inner folds
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.engine = engine if engine is not None else ReferenceEngine()
        self.feature_map = build_feature_map(n_qubits, reps, entanglement)
        self._parameters = list(self.feature_map.parameters)
        self._template = self.engine.prepare(self.feature_map)
        self.circuits_run = 0

    @property
    def params(self) -> dict:
        """What identifies this kernel among the candidates tuned on inner folds."""
        return {"reps": self.reps}

    def states(self, features: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """One simulated state per cell. `scale` multiplies the angles before loading:
        it is the kernel's bandwidth, the counterpart of gamma for an RBF kernel."""
        features = np.asarray(features, dtype=float) * scale
        if features.ndim != 2 or features.shape[1] != self.n_qubits:
            raise ValueError(
                f"expected {self.n_qubits} features per cell, got shape {features.shape}"
            )
        out = []
        for row in features:
            bound = self._template.assign_parameters(dict(zip(self._parameters, row.tolist())))
            out.append(self.engine.statevector(bound))
            self.circuits_run += 1
        return np.vstack(out)

    def kernel(self, states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
        return fidelity_kernel(states_a, states_b)


_WORKER_KERNELS: dict = {}


def _worker_states(task):
    """Runs in a worker process: one engine per process, reused across calls."""
    engine_kind, n_qubits, reps, entanglement, rows = task
    key = (engine_kind, n_qubits, reps, entanglement)
    if key not in _WORKER_KERNELS:
        engine = quantum_rings_engine_from_env() if engine_kind == "quantum_rings" else ReferenceEngine()
        _WORKER_KERNELS[key] = QuantumKernel(n_qubits, reps, entanglement, engine)
    return _WORKER_KERNELS[key].states(rows)


class ParallelQuantumKernel:
    """Same interface as QuantumKernel, with circuits spread over worker processes.

    Each worker opens its own simulator session. Credentials reach the workers
    through the environment, never through arguments.
    """

    def __init__(self, n_qubits: int, reps: int, entanglement: str = "full",
                 engine_kind: str = "quantum_rings", scales=(1.0,), workers: int = 4, pool=None):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self.n_qubits, self.reps, self.entanglement = n_qubits, reps, entanglement
        self.engine_kind, self.scales, self.workers = engine_kind, tuple(scales), workers
        self.feature_map = build_feature_map(n_qubits, reps, entanglement)
        self.circuits_run = 0
        self._owns_pool = pool is None
        self._pool = pool or ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))

    @property
    def params(self) -> dict:
        return {"reps": self.reps}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._owns_pool:
            self._pool.shutdown()

    def states(self, features: np.ndarray, scale: float = 1.0) -> np.ndarray:
        features = np.asarray(features, dtype=float) * scale
        if features.ndim != 2 or features.shape[1] != self.n_qubits:
            raise ValueError(
                f"expected {self.n_qubits} features per cell, got shape {features.shape}"
            )
        chunks = [c for c in np.array_split(features, self.workers) if len(c)]
        tasks = [(self.engine_kind, self.n_qubits, self.reps, self.entanglement, c) for c in chunks]
        self.circuits_run += len(features)
        return np.vstack(list(self._pool.map(_worker_states, tasks)))  # map keeps input order

    def kernel(self, states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
        return fidelity_kernel(states_a, states_b)


def load_env_file(path: str | os.PathLike = ".env") -> None:
    """Read KEY=VALUE lines into the environment. Existing variables win. Never prints."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def quantum_rings_engine_from_env() -> QuantumRingsEngine:
    token, account = os.environ.get(KEY_VARIABLE), os.environ.get(ACCOUNT_VARIABLE)
    if not token or not account:
        raise RuntimeError(
            f"Quantum Rings credentials are missing. Set {KEY_VARIABLE} and "
            f"{ACCOUNT_VARIABLE} in the environment or in a .env file next to where you run this."
        )
    return QuantumRingsEngine(token=token, account=account)
