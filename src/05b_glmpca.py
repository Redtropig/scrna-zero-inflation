"""
05b_glmpca.py — GLM-PCA (Townes et al. 2019) multinomial comparison.

scVI's gene_likelihood does not include a multinomial option, but the
underlying Townes et al. (2019) argument — that scRNA-seq counts are better
viewed as a multinomial over total UMIs per cell — can be tested directly
with GLM-PCA.  We fit GLM-PCA latent embeddings under three likelihoods:

    * poi   — Poisson deviance (equivalent to multinomial conditional on
              row sums when a per-cell offset is included)
    * nb    — Negative Binomial deviance
    * mult  — explicit binomial/multinomial approximation

We compare (1) final deviance, (2) ARI/NMI agreement with the Leiden cell
type labels computed on the log-normalised HVG PCA, and (3) the UMAP
layouts of each embedding.  Results are written to GLMPCA_DIR and
consumed by 06_figures.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from glmpca import glmpca
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import (
    SEED, PROCESSED_H5AD, GLMPCA_DIR,
    SCVI_N_LATENT,
)

np.random.seed(SEED)

FAMILIES = ["mult", "poi", "nb"]
FAMILY_LABELS = {
    "mult": "Multinomial (GLM-PCA)",
    "poi":  "Poisson (GLM-PCA)",
    "nb":   "Negative Binomial (GLM-PCA)",
}


def leiden_on_embedding(emb, n_neighbors=15, resolution=1.0):
    """Run Leiden on a pre-computed embedding and return cluster labels."""
    tmp = sc.AnnData(X=emb.astype(float))
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=n_neighbors,
                    random_state=SEED)
    sc.tl.leiden(tmp, resolution=resolution, random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
    return tmp.obs["leiden"].astype(int).values


def umap_embedding(emb):
    tmp = sc.AnnData(X=emb.astype(float))
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=SEED)
    sc.tl.umap(tmp, random_state=SEED)
    return tmp.obsm["X_umap"]


def main():
    print("=" * 60)
    print("Step 5b: GLM-PCA Multinomial / Poisson / NB Comparison")
    print("=" * 60)

    print("\nLoading processed data...")
    adata = sc.read_h5ad(PROCESSED_H5AD)

    # Use HVGs to keep GLM-PCA fast and comparable to scVI's HVG inputs.
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    counts = adata_hvg.layers["counts"]
    if issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(int)

    # GLM-PCA expects (features x cells)
    Y = counts.T
    n_genes, n_cells = Y.shape
    print(f"  Y shape: {n_genes} genes x {n_cells} cells")
    print(f"  Latent dims: L = {SCVI_N_LATENT}")

    # Labels for external evaluation
    labels = adata.obs["cell_type"].astype("category").cat.codes.values

    rows = []
    embeddings = {}

    for fam in FAMILIES:
        print(f"\n--- GLM-PCA family: {fam} ---")
        t0 = time.time()
        try:
            res = glmpca.glmpca(
                Y, SCVI_N_LATENT, fam=fam,
                ctl={"maxIter": 200, "eps": 1e-4},
                penalty=1, verbose=False,
            )
            factors = np.asarray(res["factors"])  # n_cells x L
            final_dev = float(res["dev"][-1])
            converged = bool(res.get("dev_converged", True))
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s  final deviance={final_dev:.1f}")

        # External cluster agreement with annotated cell types
        try:
            leiden_labels = leiden_on_embedding(factors)
            ari = float(adjusted_rand_score(labels, leiden_labels))
            nmi = float(normalized_mutual_info_score(labels, leiden_labels))
        except Exception as e:
            print(f"  Leiden-eval failed: {e}")
            ari = nmi = np.nan

        # UMAP for plotting
        try:
            umap = umap_embedding(factors)
        except Exception as e:
            print(f"  UMAP failed: {e}")
            umap = None

        rows.append({
            "family": fam,
            "final_deviance": final_dev,
            "converged": converged,
            "n_iter": len(res["dev"]),
            "runtime_sec": dt,
            "leiden_ari": ari,
            "leiden_nmi": nmi,
        })

        np.save(os.path.join(GLMPCA_DIR, f"factors_{fam}.npy"), factors)
        if umap is not None:
            np.save(os.path.join(GLMPCA_DIR, f"umap_{fam}.npy"), umap)
        embeddings[fam] = factors
        print(f"  ARI vs cell type: {ari:.3f}   NMI: {nmi:.3f}")

    # Save summary + labels (needed by figure generator)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(GLMPCA_DIR, "metrics_summary.csv"), index=False)
    np.save(os.path.join(GLMPCA_DIR, "cell_type_codes.npy"), labels)
    adata.obs[["cell_type", "leiden"]].to_csv(
        os.path.join(GLMPCA_DIR, "obs_metadata.csv"))

    print("\n" + "=" * 60)
    print("GLM-PCA Comparison Summary")
    print("=" * 60)
    print(metrics.to_string(index=False))
    print(f"\nResults in {GLMPCA_DIR}")


if __name__ == "__main__":
    main()
