"""Load 10x Genomics PBMC 3k raw counts with tutorial-derived cell-type labels.

Counts come from scanpy.datasets.pbmc3k() (raw). Cell-type names come from
scanpy.datasets.pbmc3k_processed(), where Louvain clusters were named by
marker genes in the Scanpy/Seurat tutorial. The two are joined by barcode.
These labels are inferred annotations, not externally validated ground truth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

POSITIVE_ANNOTATION = "Dendritic cells"
SOURCE_URL = "https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.datasets.pbmc3k.html"
ANNOTATION_METHOD = (
    "Cell-type names taken from Scanpy's processed PBMC 3k tutorial object "
    "(scanpy.datasets.pbmc3k_processed, obs['louvain']): Louvain clusters named by "
    "marker genes, joined to the raw count matrix by cell barcode."
)
ANNOTATION_LIMITATIONS = (
    "Inferred from unsupervised clustering of this same dataset and not externally "
    "validated. The clustering used every cell, so the labels are not independent of "
    "the expression structure the models learn from. Only 37 cells carry the "
    "dendritic label. The donor is healthy; no disease state is represented."
)
QC_DESCRIPTION = (
    "Cells kept are those retained by the Scanpy tutorial's QC (at least 200 genes, "
    "fewer than 2,500 genes, under 5% mitochondrial counts). Genes detected in fewer "
    "than 3 of those cells are dropped."
)


@dataclass
class PbmcData:
    counts: object  # scipy CSR, cells x genes, raw integer counts
    barcodes: np.ndarray
    genes: np.ndarray
    annotation: np.ndarray  # cell-type name per cell
    stats: dict


def load_pbmc(data_dir) -> PbmcData:
    import scanpy as sc

    sc.settings.datasetdir = str(data_dir)
    sc.settings.verbosity = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = sc.datasets.pbmc3k()
        processed = sc.datasets.pbmc3k_processed()
    raw.var_names_make_unique()
    kept = raw[processed.obs_names].copy()  # tutorial QC, in the processed object's order
    counts = kept.X.tocsr()
    expressed = np.asarray(counts.getnnz(axis=0)).ravel() >= 3
    return PbmcData(
        counts=counts[:, expressed],
        barcodes=np.asarray(kept.obs_names),
        genes=np.asarray(kept.var_names)[expressed],
        annotation=np.asarray(processed.obs["louvain"].astype(str)),
        stats={
            "n_cells_raw": int(raw.n_obs),
            "n_genes_raw": int(raw.n_vars),
            "n_cells_after_qc": int(kept.n_obs),
            "n_genes_after_qc": int(expressed.sum()),
        },
    )


def stratified_sample(y: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    """Every rare-class cell plus randomly drawn others, as sorted row indices."""
    y = np.asarray(y)
    rare = np.flatnonzero(y == 1)
    if n_total < len(rare) + 1:
        raise ValueError(f"n_total={n_total} cannot hold all {len(rare)} rare cells plus others")
    others = np.flatnonzero(y == 0)
    n_others = min(n_total - len(rare), len(others))
    drawn = np.random.default_rng(seed).choice(others, size=n_others, replace=False)
    return np.sort(np.concatenate([rare, drawn]))
