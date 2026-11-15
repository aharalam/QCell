# QCell

Quantum vs. classical rare-cell classification: a reproducible benchmark of three classical SVMs (linear, degree-2 polynomial, RBF) against a quantum-kernel SVM (QSVM) on single-cell RNA data.

Built at Hack The Nest 2026. This is a research and education benchmark on healthy-donor data. It is not a clinical or diagnostic tool, and it makes no claim of quantum advantage: it reports what was measured.

## The question

Does a quantum feature map give any measurable benefit on rare-cell classification, after fair, leakage-free tuning and held-out evaluation?

Three targets are scored. Within each target, every model sees the same folds, cells and features:

| Target | Positive class | Type |
|---|---|---|
| Biological | Dendritic cell vs other PBMCs | Real data, labels inferred from marker-gene clustering |
| Synthetic control | Planted two-feature XOR vs background | Engineered; says nothing about real cells |
| Circuit-structured signal | Label measured from each cell's own 3-qubit state | Engineered to have quantum-circuit structure; shows what a quantum kernel can detect, not that such signals exist in cells |

## How it works

1. **Data.** 10x Genomics PBMC 3k raw counts from Scanpy. Cell-type names come from Scanpy's processed tutorial object and are joined by barcode. They are inferred annotations, not validated ground truth.
2. **Sample.** All 37 dendritic cells plus randomly drawn others (200 cells by default). The rare class is deliberately enriched, so precision is higher than at natural prevalence.
3. **Features.** Per-cell normalisation, then variable-gene selection, scaling and PCA to 8 components. Every fitted statistic comes from the training fold only.
4. **Classical models.** scikit-learn `SVC` with `class_weight="balanced"`, tuned by an inner stratified grid search scored by macro F1.
5. **Quantum model.** Each cell's 8 features are loaded into an 8-qubit ZZ feature map as rotation angles. The Quantum Rings simulator returns the resulting state, and the kernel entry for two cells is the squared overlap of their states. A classical SVM then uses that kernel. The simulator computes exact state overlaps: there are no shots and no quantum hardware.
6. **Export.** Results are written as a JSON bundle that the Base44 dashboard uploads and validates.

Two method notes that matter when reading results:

- **Input scale.** Unscaled, the 8-qubit kernel concentrates: nearly every pair of cells has similarity near 1/256, and the SVM predicts a single class. The quantum model therefore has an input scale, tuned on inner folds only, the same way gamma is tuned for RBF.
- **Repetitions.** The number of feature-map repetitions (1, 2 or 3) is tuned on inner folds too, alongside the input scale and C. The project spec's original rule, retry at `reps=3` if `reps=2` loses to RBF on the synthetic control, was dropped because it picks a setting using held-out scores. No quantum setting is chosen that way here. Each fold's choice is recorded in `results.json`.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install numpy scikit-learn scanpy qiskit QuantumRingsLib quantumrings-toolkit-qiskit matplotlib pylatexenc pytest
```

Quantum Rings credentials go in a `.env` file at the repo root. Copy `.env.example` to `.env` and fill in both values. `.env` is git-ignored; never commit it.

| Variable | Value |
|---|---|
| `QUANTUM_RINGS_API_KEY` | Your SDK key from quantumrings.com |
| `QUANTUM_RINGS_ACCOUNT` | The email on your Quantum Rings account |

## Run

Full benchmark on the Quantum Rings simulator. PBMC 3k is downloaded into `data/` on first use.

```bash
.venv/bin/python modeling/src/runner.py
```

Classical models only (no account needed, no bundle written):

```bash
.venv/bin/python modeling/src/runner.py --classical-only
```

Local check with Qiskit's reference simulator in place of Quantum Rings. The bundle is labelled as not being a Quantum Rings run:

```bash
.venv/bin/python modeling/src/runner.py --engine reference
```

Useful options: `--n-cells`, `--outer-folds`, `--inner-folds`, `--seed`, `--workers`, `--run-id`, `--seqera-run-url`.

Each run writes to `artifacts/<run_id>/`:

| File | Contents |
|---|---|
| `dashboard_bundle/run_manifest.json` | Dataset, label provenance, seeds, settings, stage timings, software versions |
| `dashboard_bundle/results.json` | Per-fold and aggregate metrics, confusion matrices, runtimes, tuned parameters |
| `dashboard_bundle/cells.json` | Per-cell labels, out-of-fold predictions, features and display coordinates |
| `dashboard_bundle/kernels.json` | RBF, polynomial and quantum kernel matrices in one cell order |
| `dashboard_bundle/circuit.png` | The feature-map circuit |
| `dashboard_bundle/collapse_explorer.json` | 8-qubit quantum kernel on a subset of cells at input scales 1.0, 0.1 and 0.05, with held-out accuracy at each |
| `dashboard_bundle/circuit_quantum_structured.png` | The 3-qubit circuit that generates the engineered target's labels |
| `features.npz`, `kernel_quantum.npz` | Arrays for reuse outside the dashboard |

Check any bundle the way the dashboard will:

```bash
.venv/bin/python modeling/src/export_results.py validate artifacts/<run_id>/dashboard_bundle
```

Replace `<run_id>` with the folder name of the run.

## Tests

```bash
.venv/bin/python -m pytest modeling/tests -q
```

The tests cover training-only feature fitting, immunity of tuning to held-out labels, the kernel's closed form on two qubits, and bundle validation. Tests that talk to Quantum Rings run only when `.env` is filled in, and they check the simulator against Qiskit's reference at 2, 3 and 8 qubits.

## Dashboard

The dashboard is built in Base44 from the prompts in `base44_prompts.md`. It computes nothing itself: an admin uploads a bundle, the app validates it and displays it.

## Layout

```text
base44_prompts.md          prompts and data contract for the Base44 dashboard
modeling/src/
  data.py                  PBMC 3k loading, label join, rare-class-preserving sample
  features.py              fold-safe 8-feature pipeline
  synthetic_control.py     planted XOR target
  benchmark.py             nested cross-validation for all models
  quantum_kernel.py        ZZ feature map, fidelity kernel, Quantum Rings engine
  runner.py                end-to-end run
  export_results.py        bundle writer and validator
modeling/tests/            test suite
```

## Limits

- One dataset, one run, a few hundred cells. Differences between models are often within fold-to-fold spread.
- Labels are inferred from clustering of the same dataset, and only 37 cells are dendritic.
- The quantum kernel is simulated exactly. Nothing here measures speed or behaviour on quantum hardware.
- Plots of cells are low-dimensional projections. They do not show that any model separated the cells.
