"""
02_fit.py — Per-gene distribution fitting across all cells.

For every post-QC gene whose mean count exceeds MIN_MEAN_COUNT_FOR_FIT, we
fit Poisson, NB, ZIP, and ZINB to the raw UMI counts under two modes:

    1. mode="noff"   — no size-factor offset (legacy; biased by depth).
    2. mode="offset" — per-cell log(s_i) offset, where s_i = total_counts /
                       median(total_counts).

Both result tables are saved; 03_select.py consumes the offset version for
the main report and uses the noff version for the ablation figure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from joblib import Parallel, delayed

from config import (
    SEED, PROCESSED_H5AD, FIT_ALL_CELLS_FMT, DISTRIBUTIONS,
    MIN_MEAN_COUNT_FOR_FIT, SIZE_FACTOR_MODES, compute_size_factors,
)
from utils import fit_gene, run_sanity_test

np.random.seed(SEED)

N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def select_genes_to_fit(adata, min_mean=MIN_MEAN_COUNT_FOR_FIT):
    """Return gene names whose mean count across all cells exceeds min_mean."""
    counts = adata.layers["counts"]
    if issparse(counts):
        means = np.asarray(counts.mean(axis=0)).ravel()
    else:
        means = counts.mean(axis=0)
    keep = means > min_mean
    genes = adata.var_names[keep].tolist()
    print(f"  Gene filter: {keep.sum()}/{len(adata.var_names)} genes "
          f"with mean count > {min_mean}")
    return genes


def _fit_one(gene, counts, distributions, size_factors):
    """Worker for joblib: returns a list of per-distribution result dicts."""
    results = fit_gene(counts, distributions, size_factors=size_factors)
    for r in results:
        r["gene"] = gene
        r["params_json"] = json.dumps(r.pop("params"), default=str)
    return results


def fit_all_genes(counts_matrix, gene_list, size_factors, distributions, label,
                  n_jobs=N_JOBS):
    """
    Parallel per-gene fitting via joblib.
    counts_matrix: dense (n_cells, n_genes) integer array aligned with gene_list.
    size_factors:  length n_cells, or None for the no-offset mode.
    """
    n_genes = len(gene_list)
    t0 = time.time()
    print(f"  [{label}] fitting {n_genes} genes on {n_jobs} workers...")

    chunked = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_fit_one)(gene, counts_matrix[:, i], distributions, size_factors)
        for i, gene in enumerate(gene_list)
    )

    all_results = [r for chunk in chunked for r in chunk]
    dt = time.time() - t0
    print(f"  [{label}] done in {dt/60:.1f} min "
          f"({n_genes/max(dt, 1e-6):.1f} genes/s wall-clock)")
    return pd.DataFrame(all_results)


def main():
    print("=" * 60)
    print("Step 2: Per-Gene Distribution Fitting (All Cells)")
    print("=" * 60)

    print("\nRunning sanity test...")
    run_sanity_test()

    print("Loading processed data...")
    adata = sc.read_h5ad(PROCESSED_H5AD)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # ---------- Select genes by mean count, not HVG ----------
    genes = select_genes_to_fit(adata)
    counts_matrix = adata[:, genes].layers["counts"]
    if issparse(counts_matrix):
        counts_matrix = counts_matrix.toarray()
    counts_matrix = counts_matrix.astype(int)

    # ---------- Size factors from total counts (per-cell library size) ----------
    size_factors = compute_size_factors(adata.obs["total_counts"].values)
    print(f"  Size factors: median={np.median(size_factors):.2f}  "
          f"min={size_factors.min():.2f}  max={size_factors.max():.2f}")

    # ---------- Fit each mode ----------
    for mode in SIZE_FACTOR_MODES:
        print(f"\n--- Fitting mode: {mode} ---")
        sf = size_factors if mode == "offset" else None
        df = fit_all_genes(counts_matrix, genes, sf, DISTRIBUTIONS, label=mode)
        conv = df.groupby("distribution")["converged"].mean()
        print("  Convergence: " +
              "  ".join(f"{d}={r:.1%}" for d, r in conv.items()))
        out_path = FIT_ALL_CELLS_FMT.format(mode=mode)
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
