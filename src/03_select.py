"""
03_select.py — Model selection and zero-fraction analysis.

For both fit modes (with and without library-size offset) we:
  (1) select the best-fitting distribution per gene by BIC/AIC,
  (2) compute observed and expected zero fractions,
  (3) assign each gene to an expression decile so 06_figures.py can plot a
      stratified view of "which distribution wins at which expression level".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from config import (
    PROCESSED_H5AD, FIT_ALL_CELLS_FMT, MODEL_SEL_ALL_FMT,
    BIC_DELTA_THRESHOLD, SIZE_FACTOR_MODES, compute_size_factors,
)
from utils import expected_zero_fraction


def run_model_selection(fit_df, adata, size_factors, mode, gene_col="gene"):
    """
    Given a fit-results DataFrame, select the best model per gene and compute
    observed vs expected zero fractions.
    """
    fit_df = fit_df.copy()
    fit_df["params"] = fit_df["params_json"].apply(json.loads)

    genes = fit_df[gene_col].unique().tolist()
    counts_matrix = adata[:, genes].layers["counts"]
    if issparse(counts_matrix):
        counts_matrix = counts_matrix.toarray()
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Pre-compute expected-zero size-factor context
    sf_for_exp = size_factors if mode == "offset" else None

    results = []
    for gene in genes:
        gene_fits = fit_df[fit_df[gene_col] == gene]
        gene_fits = gene_fits[gene_fits["converged"] == True]
        if len(gene_fits) == 0:
            continue

        idx = gene_to_idx[gene]
        counts_vec = counts_matrix[:, idx]
        obs_zeros = float((counts_vec == 0).mean())
        mean_expr = float(counts_vec.mean())
        var_expr = float(counts_vec.var())

        best_bic_row = gene_fits.loc[gene_fits["bic"].idxmin()]
        best_aic_row = gene_fits.loc[gene_fits["aic"].idxmin()]

        zi_fits = gene_fits[gene_fits["distribution"].isin(["zip", "zinb"])]
        nonzi_fits = gene_fits[gene_fits["distribution"].isin(["poisson", "nb"])]
        if len(zi_fits) and len(nonzi_fits):
            delta_bic = nonzi_fits["bic"].min() - zi_fits["bic"].min()
        else:
            delta_bic = np.nan

        exp_zeros = {}
        for _, row in gene_fits.iterrows():
            dist = row["distribution"]
            exp_zeros[f"exp_zero_{dist}"] = expected_zero_fraction(
                dist, row["params"], size_factors=sf_for_exp
            )

        result = {
            "gene": gene,
            "mode": mode,
            "obs_zero_frac": obs_zeros,
            "best_bic": best_bic_row["distribution"],
            "best_aic": best_aic_row["distribution"],
            "bic_best_value": float(best_bic_row["bic"]),
            "delta_bic_zi": float(delta_bic) if np.isfinite(delta_bic) else np.nan,
            "is_zero_inflated": bool(delta_bic > BIC_DELTA_THRESHOLD)
                                if np.isfinite(delta_bic) else False,
            "mean_expr": mean_expr,
            "var_expr": var_expr,
        }
        result.update(exp_zeros)
        results.append(result)

    out = pd.DataFrame(results)

    # Expression deciles (only for genes with mean>0)
    pos = out[out["mean_expr"] > 0].copy()
    if len(pos):
        pos["expr_decile"] = pd.qcut(
            pos["mean_expr"].rank(method="first"),
            q=10, labels=list(range(1, 11))
        ).astype(int)
        out = out.merge(
            pos[["gene", "expr_decile"]], on="gene", how="left"
        )
    else:
        out["expr_decile"] = np.nan
    return out


def summarize(sel_df, mode):
    print(f"\n--- Mode: {mode} ---")
    print("BIC winners:")
    bic_counts = sel_df["best_bic"].value_counts()
    total = len(sel_df)
    for dist, count in bic_counts.items():
        print(f"  {dist:6s}: {count:5d} genes ({count/total:.1%})")
    n_zi = int(sel_df["is_zero_inflated"].sum())
    print(f"ZI genes (delta-BIC > {BIC_DELTA_THRESHOLD}): "
          f"{n_zi}/{total} ({n_zi/total:.1%})")


def main():
    print("=" * 60)
    print("Step 3: Model Selection & Zero-Fraction Analysis")
    print("=" * 60)

    print("\nLoading processed data...")
    adata = sc.read_h5ad(PROCESSED_H5AD)
    size_factors = compute_size_factors(adata.obs["total_counts"].values)

    for mode in SIZE_FACTOR_MODES:
        fit_path = FIT_ALL_CELLS_FMT.format(mode=mode)
        if not os.path.exists(fit_path):
            print(f"Skipping {mode}: {fit_path} not found.")
            continue
        print(f"\nReading {fit_path}...")
        fit_df = pd.read_csv(fit_path)
        sel_df = run_model_selection(fit_df, adata, size_factors, mode)
        summarize(sel_df, mode)

        out_path = MODEL_SEL_ALL_FMT.format(mode=mode)
        sel_df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
