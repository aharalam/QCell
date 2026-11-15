"""Run the whole benchmark and write the dashboard bundle.

    python modeling/src/runner.py                      # Quantum Rings, 200 cells
    python modeling/src/runner.py --engine reference   # local Qiskit check, no account
    python modeling/src/runner.py --classical-only     # no quantum step, no bundle

Steps: load PBMC 3k and its tutorial labels, draw a rare-class-preserving
sample, build the synthetic control, run nested cross-validation for the three
classical SVMs and the quantum-kernel SVM on identical folds, then export.
Credentials are read from the environment or a .env file and never written out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score

import collapse_explorer as ce
import data
import export_results as er
import quantum_structured as qs
import synthetic_control as sc
from benchmark import (
    CLASSICAL_MODELS, DISPLAY_NAMES, QUANTUM_REPS_VALUES, QUANTUM_SCALE_VALUES, SELECTION_METRIC, fold_features, make_folds,
    model_config, run_nested_cv,
)
from features import FoldFeatures
from quantum_kernel import (
    ParallelQuantumKernel, QuantumKernel, ReferenceEngine, build_feature_map, load_env_file,
    quantum_rings_engine_from_env,
)

N_QUBITS = 8
ENTANGLEMENT = "full"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stages:
    """Records what actually ran and how long it took, for the manifest."""

    def __init__(self):
        self.items = []

    def add(self, id_, name, tool, status, started_at=None, seconds=0.0, artifact="", link=""):
        self.items.append({
            "id": id_, "name": name, "tool": tool, "status": status, "started_at": started_at,
            "duration_seconds": round(float(seconds), 3), "artifact": artifact, "link": link,
        })

    def timed(self, id_, name, tool, artifact=""):
        stages = self

        class _Timer:
            def __enter__(self):
                self.started, self.t0 = _now(), time.perf_counter()
                print(f"[{name}] ...", flush=True)

            def __exit__(self, exc_type, *_):
                seconds = time.perf_counter() - self.t0
                status = "failed" if exc_type else "succeeded"
                stages.add(id_, name, tool, status, self.started, seconds, artifact)
                print(f"[{name}] {status} in {seconds:.1f}s", flush=True)

        return _Timer()


def mean_fold_score(y, folds, pred, metric="macro_f1") -> float:
    scores = []
    for _, test in folds:
        if metric == "macro_f1":
            scores.append(f1_score(y[test], pred[test], average="macro", zero_division=0))
        else:
            scores.append(balanced_accuracy_score(y[test], pred[test]))
    return float(np.mean(scores))


class Target:
    """One benchmark target: labels, folds, and per-fold features fitted on training cells."""

    def __init__(self, key, y, counts, n_folds, seed, make_pipeline, planted=None, fixed_features=None):
        # fixed_features: inputs that are given, not fitted per fold (engineered targets
        # whose labels are defined from those same inputs).
        self.key, self.y, self.planted, self.fixed = key, np.asarray(y), planted, fixed_features
        self.n_qubits = N_QUBITS if fixed_features is None else fixed_features.shape[1]
        self.folds = make_folds(self.y, n_folds, seed)
        self.fold_of = np.empty(len(self.y), dtype=int)
        for k, (_, test) in enumerate(self.folds):
            self.fold_of[test] = k
        self.held_out_features = np.full((len(self.y), self.n_qubits), np.nan)
        self._counts, self._make_pipeline, self._cache = counts, make_pipeline, {}

    def features_for_fold(self, train, test):
        key = test.tobytes()
        if key not in self._cache and self.fixed is not None:
            self.held_out_features[test] = self.fixed[test]
            self._cache[key] = (self.fixed[train], self.fixed[test])
        if key not in self._cache:
            x_train, x_test = fold_features(self._counts, train, test, self._make_pipeline)
            if self.planted is not None:
                x_train = sc.synthetic_features(self.planted[train], x_train[:, 2:])
                x_test = sc.synthetic_features(self.planted[test], x_test[:, 2:])
            self.held_out_features[test] = x_test
            self._cache[key] = (x_train, x_test)
        return self._cache[key]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", choices=["quantum_rings", "reference"], default="quantum_rings")
    p.add_argument("--classical-only", action="store_true")
    p.add_argument("--n-cells", type=int, default=200)
    p.add_argument("--n-hvg", type=int, default=500)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-umap", action="store_true", help="skip UMAP (tests only)")
    p.add_argument("--workers", type=int, default=6, help="parallel simulator processes (Quantum Rings only)")
    p.add_argument("--seqera-run-url", default="")
    p.add_argument("--seqera-pipeline", default="")
    p.add_argument("--seqera-status", default="")
    return p.parse_args(argv)


def smoke_test(engine, seed) -> dict:
    """Prove the simulator runs this exact feature map before the large job."""
    rng = np.random.default_rng(seed)
    report = {}
    for reps in QUANTUM_REPS_VALUES:
        x = rng.uniform(0, np.pi, size=(3, N_QUBITS))
        measured = QuantumKernel(N_QUBITS, reps, ENTANGLEMENT, engine).states(x, scale=0.25)
        reference = QuantumKernel(N_QUBITS, reps, ENTANGLEMENT, ReferenceEngine()).states(x, scale=0.25)
        overlap = np.abs(np.sum(np.conj(reference) * measured, axis=1)) ** 2
        report[f"reps_{reps}_min_overlap_with_qiskit_reference"] = float(overlap.min())
        if overlap.min() < 1 - 1e-6:
            raise RuntimeError(
                f"Simulator states differ from the reference at reps={reps} "
                f"(overlap {overlap.min():.6f}); stopping before the full run."
            )
    return report


def display_coordinates(matrix, seed, use_umap):
    pcs = PCA(n_components=min(30, matrix.shape[1], matrix.shape[0] - 1), random_state=seed).fit_transform(matrix)
    pca3 = pcs[:, :3]
    if not use_umap:
        return pca3, pcs[:, 3:6] if pcs.shape[1] >= 6 else pca3
    import umap

    return pca3, umap.UMAP(n_components=3, random_state=seed).fit_transform(pcs)


def main(argv=None) -> str:
    args = parse_args(argv)
    load_env_file(".env")
    run_id = args.run_id or f"qcell-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = Stages()
    stages.add("raw", "10x PBMC 3k raw counts", "scanpy.datasets.pbmc3k", "succeeded",
               artifact="pbmc3k_raw.h5ad", link=data.SOURCE_URL)
    stages.add("seqera", "Seqera / nf-core scrnaseq", "Seqera Platform",
               "succeeded" if args.seqera_status.lower() == "succeeded" else "skipped",
               artifact="" if args.seqera_run_url else "not part of this run: started from the count matrix",
               link=args.seqera_run_url)

    def make_pipeline():
        return FoldFeatures(n_hvg=args.n_hvg, n_components=N_QUBITS, seed=args.seed)

    with stages.timed("scanpy_qc", "Scanpy load, QC and labels", "Scanpy", "pbmc3k_processed.h5ad"):
        pbmc = data.load_pbmc(args.data_dir)
        y_all = (pbmc.annotation == data.POSITIVE_ANNOTATION).astype(int)
        picked = data.stratified_sample(y_all, args.n_cells, args.seed)
        counts = pbmc.counts[picked].toarray().astype(float)
        ids, annotation, y_bio = pbmc.barcodes[picked], pbmc.annotation[picked], y_all[picked]
        planted, y_syn = sc.planted_control(len(picked), args.seed + 1)

    with stages.timed("features", "Split-aware features", "scikit-learn", "features.npz"):
        targets = {
            "biological": Target("biological", y_bio, counts, args.outer_folds, args.seed, make_pipeline),
            "synthetic": Target("synthetic", y_syn, counts, args.outer_folds, args.seed, make_pipeline, planted),
        }
        # Third target: labels measured from each cell's own 3-qubit state (engineered signal).
        display = make_pipeline().fit(counts)
        x_display_bio = display.transform(counts)
        x_structured = x_display_bio[:, :qs.N_QUBITS]
        y_structured, expectation = qs.circuit_structured_labels(x_structured, args.seed + 2)
        targets["quantum_structured"] = Target(
            "quantum_structured", y_structured, counts, args.outer_folds, args.seed, make_pipeline,
            fixed_features=x_structured)
        for target in targets.values():
            for train, test in target.folds:
                target.features_for_fold(train, test)

    def progress(name, fold, seconds):
        print(f"    {name:7s} fold {fold}: {seconds:6.2f}s", flush=True)

    outcomes = {}
    with stages.timed("classical", "Classical SVM benchmarks", "scikit-learn"):
        for key, target in targets.items():
            print(f"  {key}", flush=True)
            outcomes[key] = run_nested_cv(target.y, target.folds, target.features_for_fold,
                                          args.seed, args.inner_folds, on_progress=progress)

    def table(model_names):
        lines = []
        for key, target in targets.items():
            for name in model_names:
                pred = outcomes[key][name].pred
                lines.append(f"  {key:10s} {name:7s} macro F1 {mean_fold_score(target.y, target.folds, pred):.3f}"
                             f"  balanced acc {mean_fold_score(target.y, target.folds, pred, 'ba'):.3f}")
        return "\n".join(lines)

    if args.classical_only:
        print("\nClassical results (mean over outer folds):\n" + table(CLASSICAL_MODELS))
        print("\nNo bundle written: the dashboard needs the quantum model as well.")
        return str(run_dir)

    if args.engine == "quantum_rings":
        engine = quantum_rings_engine_from_env()
        sdk = "Quantum Rings SDK (QuantumRingsLib) through quantumrings-toolkit-qiskit"
        sdk_version = f"QuantumRingsLib {metadata.version('QuantumRingsLib')}"
        backend = f"{engine.name} simulator, exact state overlap (no shots, no hardware)"
    else:
        engine = ReferenceEngine()
        sdk, sdk_version = "Qiskit reference statevector", f"qiskit {metadata.version('qiskit')}"
        backend = f"{engine.name} (NOT Quantum Rings): local check run"

    pool = None
    if args.engine == "quantum_rings" and args.workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))

    def make_kernel(reps, n_qubits=N_QUBITS, scales=QUANTUM_SCALE_VALUES):
        if pool is not None:
            return ParallelQuantumKernel(n_qubits, reps, ENTANGLEMENT, "quantum_rings",
                                         scales, args.workers, pool=pool)
        return QuantumKernel(n_qubits, reps, ENTANGLEMENT, engine, scales=scales)

    with stages.timed("quantum", "Quantum feature map and kernel", sdk, "kernel_quantum.npz"):
        check = smoke_test(engine, args.seed) if args.engine == "quantum_rings" else {}
        # One kernel per circuit depth. Depth, input scale and C are all chosen on inner
        # folds, so no quantum setting is ever picked by looking at held-out scores.
        kernels_for = {
            n: {reps: make_kernel(reps, n) for reps in QUANTUM_REPS_VALUES}
            for n in sorted({target.n_qubits for target in targets.values()})
        }
        for key, target in targets.items():
            print(f"  {key}", flush=True)
            outcomes[key]["qsvm"] = run_nested_cv(
                target.y, target.folds, target.features_for_fold, args.seed, args.inner_folds,
                quantum=list(kernels_for[target.n_qubits].values()), classical=False,
                on_progress=progress)["qsvm"]

    with stages.timed("collapse_explorer", "Kernel-collapse explorer data", sdk, "collapse_explorer.json"):
        bio = targets["biological"]
        subset = ce.pick_subset(bio.y, ids, ce.MAX_CELLS, args.seed)
        explorer_scales = ce.collapse_matrices(
            x_display_bio[subset], make_kernel(ce.REPS, N_QUBITS, scales=(1.0,)), ce.SCALES)
        for entry in explorer_scales:
            print(f"  accuracy at input scale {entry['scale']}", flush=True)
            entry["accuracy"] = ce.accuracy_at_scale(
                bio.y, bio.folds, bio.features_for_fold,
                make_kernel(ce.REPS, N_QUBITS, scales=(entry["scale"],)), args.seed, args.inner_folds)

    with stages.timed("evaluation", "Evaluation and display projections", "scikit-learn", "results.json"):
        x_display = {
            "biological": x_display_bio,
            "synthetic": sc.synthetic_features(planted, x_display_bio[:, 2:]),
            "quantum_structured": x_structured,
        }
        coords = {
            "biological": display_coordinates(display.variable_genes(counts), args.seed, not args.no_umap),
            "synthetic": display_coordinates(x_display["synthetic"], args.seed, not args.no_umap),
            "quantum_structured": display_coordinates(x_structured, args.seed, not args.no_umap),
        }
        naming = {  # positive label, negative label, per-cell annotation, feature names
            "biological": ("Dendritic cell", "Other PBMC", list(annotation), [f"PC{i}" for i in range(1, 9)]),
            "synthetic": ("Planted interaction", "Background", None,
                          [*sc.PLANTED_NAMES, *[f"PC{i}" for i in range(3, 9)]]),
            "quantum_structured": ("Circuit-structured signal (engineered)", "Background", None,
                                   [f"display PC{i}" for i in range(1, qs.N_QUBITS + 1)]),
        }
        inputs, quantum_kernels, display_choice = {}, {}, {}
        for key, target in targets.items():
            x = x_display[key]
            gamma = float(1.0 / (x.shape[1] * x.var()))
            kernels = er.classical_display_kernels(x, rbf_gamma=gamma, poly_gamma=gamma, poly_coef0=0.0)
            chosen = [(params["reps"], params["scale"]) for params in outcomes[key]["qsvm"].best_params.values()]
            display_reps, display_scale = max(set(chosen), key=lambda c: (chosen.count(c), -c[0], c[1]))
            display_choice[key] = display_reps
            final_kernel = kernels_for[target.n_qubits][display_reps]
            states = final_kernel.states(x, scale=display_scale)
            kernels["qsvm"] = quantum_kernels[key] = final_kernel.kernel(states, states)
            models = {}
            for name in (*CLASSICAL_MODELS, "qsvm"):
                o = outcomes[key][name]
                config = model_config(name)
                if name == "qsvm":
                    config.update(feature_map="zz_feature_map", n_qubits=target.n_qubits,
                                  entanglement=ENTANGLEMENT, kernel="fidelity (exact state overlap)")
                models[name] = er.ModelRun(
                    display_name=DISPLAY_NAMES[name], pred=o.pred, fit_seconds=o.fit_seconds,
                    predict_seconds=o.predict_seconds, config=config, best_params=o.best_params,
                    score=o.score,
                )
            positive_name, negative_name, cell_annotation, feature_names = naming[key]
            signal_word = "planted" if key == "synthetic" else "signal"
            inputs[key] = er.TargetInputs(
                positive_label_name=positive_name,
                negative_label_name=negative_name,
                cell_ids=[str(i) for i in ids],
                y_true=target.y,
                fold=target.fold_of,
                annotation=cell_annotation or [signal_word if v else "background" for v in target.y],
                feature_names=feature_names,
                features=target.held_out_features,
                coords={"pca3": coords[key][0], "umap3": coords[key][1]},
                models=models,
                kernels=kernels,
                kernel_params={
                    "rbf": {"gamma": gamma, "gamma_source": "scikit-learn 'scale' on the display features"},
                    "poly2": {"gamma": gamma, "degree": 2, "coef0": 0.0},
                    "qsvm": {"reps": display_reps, "entanglement": ENTANGLEMENT, "n_qubits": target.n_qubits,
                             "scale": display_scale,
                             "scale_source": "repetitions and input scale selected most often across the outer folds"},
                },
                kernel_fit_scope=(
                    f"Display only. The feature pipeline was fitted on all {len(ids)} benchmark cells and "
                    "each kernel computed between all of them, so the three heatmaps and the Kernel-PCA "
                    "coordinates share one cell order. None of this was used for any score: scoring used "
                    "pipelines and kernels fitted inside each training fold."
                ),
                planted=planted if key == "synthetic" else None,
                extra={"expectation": expectation} if key == "quantum_structured" else None,
            )

    if pool is not None:
        pool.shutdown()
    n_pos, n_all = int(y_bio.sum()), len(y_bio)
    manifest = {
        "run_id": run_id,
        "created_at": _now(),
        "is_placeholder": False,
        "dataset": {"name": "10x Genomics PBMC 3k", "source_url": data.SOURCE_URL,
                    "donor_status": "healthy donor", "qc": data.QC_DESCRIPTION, **pbmc.stats},
        "annotation": {"method": data.ANNOTATION_METHOD, "positive_class_name": "Dendritic cell",
                       "limitations": data.ANNOTATION_LIMITATIONS},
        "sampling": {
            "strategy": (
                f"All {n_pos} dendritic cells plus {n_all - n_pos} other cells drawn at random without "
                f"replacement. The rare class is deliberately enriched: {100 * n_pos / n_all:.1f}% of this "
                f"sample versus {100 * y_all.mean():.1f}% of the QC'd dataset, so precision here is higher "
                "than it would be at natural prevalence."
            ),
            "seed": args.seed,
        },
        "preprocessing": {"feature_pipeline": make_pipeline().describe(),
                          "fitted_within_training_folds_only": True},
        "cv": {"inner_folds": args.inner_folds, "selection_metric": SELECTION_METRIC, "seed": args.seed},
        "quantum": {
            "sdk": sdk, "sdk_version": sdk_version, "backend": backend, "n_qubits": N_QUBITS,
            "feature_map": "ZZ feature map (qiskit.circuit.library.zz_feature_map)",
            "entanglement": ENTANGLEMENT,
            "attempts": [{
                "reps": display_choice["biological"], "is_final": True,
                "reason": (
                    f"Repetitions were not fixed in advance: each outer fold chose among {QUANTUM_REPS_VALUES} "
                    "on its inner folds, together with the input scale and C. The value shown is the one "
                    "chosen most often for the biological target; every fold's choice is in results.json. "
                    "No quantum setting was selected using held-out scores."
                ),
            }],
            "reps_grid": QUANTUM_REPS_VALUES,
            "circuits_run_note": "one circuit per cell per fold per repetition count per input scale; kernel entries are state overlaps",
            "simulator_settings": "performance HIGHESTACCURACY, double precision" if args.engine == "quantum_rings" else "exact",
            "parallel_workers": args.workers if pool is not None else 1,
            "input_scale_grid": QUANTUM_SCALE_VALUES,
            "input_scale_note": (
                "Features are multiplied by an input scale before being loaded as rotation angles. "
                "Unscaled (1.0), this 8-qubit kernel concentrates: nearly every pair of cells has "
                "similarity close to 1/256, which leaves the SVM nothing to learn from. The scale is "
                "therefore tuned on inner folds only, together with the number of repetitions, exactly "
                "as gamma is tuned for the RBF kernel."
            ),
            "smoke_test": check,
        },
        "synthetic_control": {"rule_description": sc.RULE_DESCRIPTION,
                              "features_used": list(sc.PLANTED_NAMES), "seed": args.seed + 1},
        "quantum_structured": {
            "evidence_type": qs.EVIDENCE_TYPE,
            "rule_description": qs.RULE_DESCRIPTION,
            "feature_note": qs.FEATURE_NOTE,
            "n_qubits": qs.N_QUBITS, "reps": qs.REPS, "input_scale": qs.SCALE,
            "entanglement": ENTANGLEMENT, "seed": args.seed + 2,
            "label_engine": "Qiskit exact statevector (labels are identical on any simulator)",
            "per_cell_value": "cells.json: extra.expectation is the measured value; label is 1 above the median",
            "caution": (
                "This signal is planted by us. A result here shows what kind of structure a quantum "
                "kernel detects, not that such structure exists in real cells."
            ),
        },
        "display": {"umap3": "UMAP of the display PCA" if not args.no_umap
                    else "UMAP disabled for this run: holds PCA components 4-6"},
        "software_versions": er.collect_software_versions(),
    }
    if args.seqera_run_url:
        manifest["seqera"] = {"run_url": args.seqera_run_url, "pipeline": args.seqera_pipeline,
                              "status": args.seqera_status}

    np.savez_compressed(
        run_dir / "features.npz", cell_ids=np.asarray(ids, dtype=str),
        **{f"{k}_{name}": getattr(t, name) for k, t in targets.items()
           for name in ("y", "fold_of", "held_out_features")})
    np.savez_compressed(run_dir / "kernel_quantum.npz", cell_ids=np.asarray(ids, dtype=str), **quantum_kernels)
    circuit_png = er.save_circuit_image(build_feature_map(N_QUBITS, display_choice["biological"], ENTANGLEMENT),
                                        run_dir / "circuit.png", fold=40)

    started, t0 = _now(), time.perf_counter()
    stages.add("bundle", "Dashboard bundle", "export_results.py", "succeeded", started, 0.0, "dashboard_bundle/")
    manifest["pipeline_stages"] = stages.items
    report = er.export_bundle(run_dir / "dashboard_bundle", manifest, inputs, circuit_image=circuit_png)
    bundle_dir = run_dir / "dashboard_bundle"
    explorer = {
        "schema_version": er.SCHEMA_VERSION,
        "run_id": run_id,
        "target": "biological",
        "n_qubits": N_QUBITS,
        "reps": ce.REPS,
        "entanglement": ENTANGLEMENT,
        "uniform_similarity": 1 / 2**N_QUBITS,  # what a fully collapsed kernel gives every pair
        "positive_label_name": "Dendritic cell",
        "negative_label_name": "Other PBMC",
        "cell_ids": [str(ids[i]) for i in subset],
        "y_true": [int(bio.y[i]) for i in subset],
        "annotation": [str(annotation[i]) for i in subset],
        "scales": explorer_scales,
        "matrix_note": ce.MATRIX_NOTE,
        "accuracy_note": ce.ACCURACY_NOTE,
    }
    (bundle_dir / "collapse_explorer.json").write_text(json.dumps(explorer, allow_nan=False) + "\n")
    np.savez_compressed(
        run_dir / "collapse_explorer.npz", cell_ids=np.asarray(explorer["cell_ids"]),
        y_true=np.asarray(explorer["y_true"]),
        **{f"scale_{entry['scale']}": np.asarray(entry["matrix"]) for entry in explorer_scales})
    er.save_circuit_image(build_feature_map(qs.N_QUBITS, qs.REPS, ENTANGLEMENT),
                          bundle_dir / "circuit_quantum_structured.png", fold=40)
    bundle_seconds = time.perf_counter() - t0
    manifest_path = run_dir / "dashboard_bundle" / "run_manifest.json"
    written = json.loads(manifest_path.read_text())
    written["pipeline_stages"][-1]["duration_seconds"] = round(bundle_seconds, 3)
    manifest_path.write_text(json.dumps(written, indent=2) + "\n")

    print("\nResults (mean over outer folds):\n" + table((*CLASSICAL_MODELS, "qsvm")))
    for key in targets:
        picks = [(p["reps"], p["scale"], p["C"]) for p in outcomes[key]["qsvm"].best_params.values()]
        print(f"  QSVM {key} chose (reps, scale, C) per fold: {picks}")
    print(f"\nValidation: {len(report['errors'])} error(s), {len(report['warnings'])} warning(s)")
    for line in report["warnings"] + report.get("exporter_notes", []):
        print(f"  note: {line}")
    print(f"Bundle: {run_dir / 'dashboard_bundle'}")
    return str(run_dir)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, er.BundleError) as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
