"""Engineered target with the structure of the quantum circuit itself.

Each cell's three input features are loaded into the 3-qubit ZZ feature map at
the project's original setting (reps=2, input scale 1.0). A fixed random
measurement is applied to the resulting state, and cells whose measured value
is above the median are labelled positive.

The features are real; the signal is planted by us. A result on this target
shows the kind of structure a quantum kernel detects and classical kernels
miss at this sample size. It says nothing about real cell types, and it does
not show that such signals exist in biology.

Labels are always computed with Qiskit's exact statevector, so they are the
same whichever simulator the benchmark itself runs on.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import random_unitary

from quantum_kernel import QuantumKernel, ReferenceEngine

N_QUBITS, REPS, SCALE = 3, 2, 1.0
EVIDENCE_TYPE = "Engineered signal with quantum-circuit structure, planted in real cell features"
RULE_DESCRIPTION = (
    "Each cell's first three display components are loaded as rotation angles into a 3-qubit "
    "ZZ feature map (reps=2, full entanglement, input scale 1.0). A fixed random unitary, drawn "
    "once from a seeded generator, is applied to the resulting state, and the parity of the "
    "three qubits is measured. Cells whose expected parity is above the median are labelled "
    "positive. The label is therefore an exact function of the cell's three input features."
)
FEATURE_NOTE = (
    "The three inputs are the first three components of a feature pipeline fitted once on all "
    "benchmark cells, without any labels, and are then treated as fixed inputs for every model. "
    "They are not refitted inside each fold, because the labels are defined from them. No label "
    "information enters any fitted statistic, and all four models see the same inputs."
)


def circuit_structured_labels(features: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, expected parity per cell)."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != N_QUBITS:
        raise ValueError(f"expected {N_QUBITS} features per cell, got shape {features.shape}")
    states = QuantumKernel(N_QUBITS, REPS, engine=ReferenceEngine()).states(features, scale=SCALE)
    measurement = random_unitary(2**N_QUBITS, seed=seed).data
    parity = np.array([(-1) ** bin(i).count("1") for i in range(2**N_QUBITS)])
    expectation = (np.abs(states @ measurement.T) ** 2) @ parity
    labels = (expectation > np.median(expectation)).astype(int)
    return labels, expectation
