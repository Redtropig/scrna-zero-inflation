"""
04_heterogeneity.py — Per-cluster distribution re-fitting.

For each Leiden cluster with at least MIN_CELLS_PER_CLUSTER cells, we re-fit
Poisson / NB / ZIP / ZINB to every post-QC gene passing the mean-count floor
*within the cluster*, under two modes:

    * noff   — no offset, so results are directly comparable to Kim et al.
               (2020)'s demonstration that apparent dropout vanishes once
               cell-type heterogeneity is removed.
    * offset — per-cluster size-factor offset, so what remains after depth
               correction truly reflects the biological Poisson regime of
               Pan et al. (2023).

Within-cluster size factors are recomputed from that cluster's cells only so
the median of s_i remains 1 for each cluster.
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
    SEED, PROCESSED_H5AD,
    FIT_PER_CLUSTER_FMT, MODEL_SEL_CLUSTER_FMT,
    MODEL_SEL_ALL_FMT, DISTRIBUTIONS, MIN_CELLS_PER_CLUSTER,
    BIC_DELTA_THRESHOLD, MIN_MEAN_COUNT_FOR_FIT,
    SIZE_FACTOR_MODES, compute_size_factors,
)
from utils import fit_gene

np.random.seed(SEED)

N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def _fit_one(gene, counts, distributions, size_factors, cluster, cell_type, n_cells):
    results = fit_gene(counts, distributions, size_factors=size_factors)
    for r in results:
        r["gene"] = gene
        r["cluster"] = cluster
        r["cell_type"] = cell_type
        r["n_cells"] = n_cells
        r["params_json"] = json.dumps(r.pop("params"), default=str)
    return results


def fit_per_cluster(adata, mode):
    """Fit every gene in every qualifying cluster under the given mode."""
    clusters = sorted(adata.obs["leiden"].unique(), key=int)
    all_rows = []

    for cluster in clusters:
        mask = adata.obs["leiden"] == cluster
        n_cells = int(mask.sum())
        cell_type = str(adata.obs.loc[mask, "cell_type"].iloc[0])
        if n_cells < MIN_CELLS_PER_CLUSTER:
            print(f"  Cluster {cluster} ({cell_type}): {n_cells} cells SKIPPED")
            continue

        sub = adata[mask].copy()
        counts_matrix = sub.layers["counts"]
        if issparse(counts_matrix):
            counts_matrix = counts_matrix.toarray()
        counts_matrix = counts_matrix.astype(int)

        # Within-cluster gene filter by mean count
        means = counts_matrix.mean(axis=0)
        keep_idx = np.where(means > MIN_MEAN_COUNT_FOR_FIT)[0]
        kept_genes = sub.var_names[keep_idx].tolist()

        # Within-cluster size factors
        sf_cluster = compute_size_factors(
            sub.obs["total_counts"].values
        ) if mode == "offset" else None

        print(f"  [{mode}] Cluster {cluster} ({cell_type}): "
              f"{n_cells} cells x {len(kept_genes)} genes")

        t0 = time.time()
        chunked = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
            delayed(_fit_one)(
                gene, counts_matrix[:, keep_idx[i]], DISTRIBUTIONS,
                sf_cluster, cluster, cell_type, n_cells
            )
            for i, gene in enumerate(kept_genes)
        )
        dt = time.time() - t0
        flat = [r for chunk in chunked for r in chunk]
        print(f"    done in {dt/60:.1f} min ({len(kept_genes)/max(dt, 1e-6):.1f} genes/s)")
        all_rows.extend(flat)

    return pd.DataFrame(all_rows)


def run_selection(fit_df):
    """Per gene-cluster best BIC model and zero-inflation verdict."""
    fit_df = fit_df.copy()
    fit_df["params"] = fit_df["params_json"].apply(json.loads)

    sel_rows = []
    for (cluster, gene), group in fit_df.groupby(["cluster", "gene"]):
        converged = group[group["converged"] == True]
        if not len(converged):
            continue
        best_bic_row = converged.loc[converged["bic"].idxmin()]
        zi = converged[converged["distribution"].isin(["zip", "zinb"])]
        nonzi = converged[converged["distribution"].isin(["poisson", "nb"])]
        if len(zi) and len(nonzi):
            delta_bic = float(nonzi["bic"].min() - zi["bic"].min())
        else:
            delta_bic = np.nan
        sel_rows.append({
            "cluster": cluster,
            "cell_type": best_bic_row["cell_type"],
            "gene": gene,
            "n_cells": int(best_bic_row["n_cells"]),
            "best_bic": best_bic_row["distribution"],
            "delta_bic_zi": delta_bic,
            "is_zero_inflated": bool(delta_bic > BIC_DELTA_THRESHOLD)
                                 if np.isfinite(delta_bic) else False,
        })
    return pd.DataFrame(sel_rows)


def print_summary(sel_df, mode):
    total = len(sel_df)
    print(f"\n--- Per-cluster BIC summary [{mode}] ---")
    for dist, count in sel_df["best_bic"].value_counts().items():
        print(f"  {dist:6s}: {count:5d} ({count/total:.1%})")
    n_zi = int(sel_df["is_zero_inflated"].sum())
    print(f"  ZI gene-cluster pairs: {n_zi}/{total} ({n_zi/total:.1%})")


def main():
    print("=" * 60)
    print("Step 4: Heterogeneity Analysis (Per-Cluster Fitting)")
    print("=" * 60)

    print("\nLoading processed data...")
    adata = sc.read_h5ad(PROCESSED_H5AD)

    for mode in SIZE_FACTOR_MODES:
        print(f"\n====== Mode: {mode} ======")
        fit_df = fit_per_cluster(adata, mode)
        fit_out = FIT_PER_CLUSTER_FMT.format(mode=mode)
        fit_df.to_csv(fit_out, index=False)
        print(f"  Saved {len(fit_df)} fit rows -> {fit_out}")

        sel_df = run_selection(fit_df)
        sel_out = MODEL_SEL_CLUSTER_FMT.format(mode=mode)
        sel_df.to_csv(sel_out, index=False)
        print_summary(sel_df, mode)

        # Comparison vs all-cells (same mode)
        all_sel_path = MODEL_SEL_ALL_FMT.format(mode=mode)
        if os.path.exists(all_sel_path):
            all_sel = pd.read_csv(all_sel_path)
            n_zi_all = int(all_sel["is_zero_inflated"].sum())
            n_genes_all = len(all_sel)
            n_zi_clu = int(sel_df["is_zero_inflated"].sum())
            n_tests = len(sel_df)
            print(f"  All-cells ZI ({mode}): "
                  f"{n_zi_all}/{n_genes_all} ({n_zi_all/n_genes_all:.1%})")
            print(f"  Per-cluster ZI ({mode}): "
                  f"{n_zi_clu}/{n_tests} ({n_zi_clu/n_tests:.1%})")


if __name__ == "__main__":
    main()
