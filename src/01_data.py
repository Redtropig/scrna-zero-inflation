"""
01_data.py — Data acquisition, quality control, preprocessing, and clustering.

Downloads the PBMC 3k dataset, performs standard QC, clusters cells,
annotates cell types, and saves the processed AnnData object.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import anndata
import scanpy as sc

anndata.settings.allow_write_nullable_strings = True

from config import (
    SEED, PROCESSED_H5AD,
    MIN_GENES_PER_CELL, MAX_GENES_PER_CELL, MAX_PCT_MITO,
    MIN_CELLS_PER_GENE, N_HIGHLY_VARIABLE, LEIDEN_RESOLUTION,
    CELL_TYPE_MARKERS,
)

np.random.seed(SEED)


def main():
    print("=" * 60)
    print("Step 1: Data Acquisition and Preprocessing")
    print("=" * 60)

    ### ---------- Download PBMC 3k ----------
    print("\n[1/7] Downloading PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()
    print(f"  Raw: {adata.n_obs} cells x {adata.n_vars} genes")

    # Make gene names unique
    adata.var_names_make_unique()

    ### ---------- QC metrics ----------
    print("[2/7] Computing QC metrics...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    adata = adata[adata.obs["n_genes_by_counts"] < MAX_GENES_PER_CELL, :].copy()
    adata = adata[adata.obs["pct_counts_mt"] < MAX_PCT_MITO, :].copy()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"  After QC: {adata.n_obs} cells x {adata.n_vars} genes "
          f"(removed {n_before - adata.n_obs} cells)")

    ### ---------- Preserve raw counts ----------
    print("[3/7] Saving raw counts to adata.layers['counts']...")
    adata.layers["counts"] = adata.X.copy()

    ### ---------- Normalize + log-transform (for clustering only) ----------
    print("[4/7] Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized in .raw for plotting later
    adata.raw = adata

    ### ---------- HVG selection + PCA ----------
    print(f"[5/7] Selecting top {N_HIGHLY_VARIABLE} HVGs, running PCA...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HIGHLY_VARIABLE, flavor="seurat_v3",
                                layer="counts")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", random_state=SEED)

    ### ---------- Neighbors, UMAP, Leiden ----------
    print(f"[6/7] Computing neighbors, UMAP, Leiden (resolution={LEIDEN_RESOLUTION})...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Found {n_clusters} Leiden clusters")

    ### ---------- Cell type annotation ----------
    print("[7/7] Annotating cell types...")
    cell_type_map = annotate_cell_types(adata)
    adata.obs["cell_type"] = adata.obs["leiden"].map(cell_type_map)
    print("  Cluster -> Cell type mapping:")
    for cluster, ctype in sorted(cell_type_map.items(), key=lambda x: int(x[0])):
        n = (adata.obs["leiden"] == cluster).sum()
        print(f"    Cluster {cluster}: {ctype} ({n} cells)")

    ### ---------- Save ----------
    print(f"\nSaving to {PROCESSED_H5AD}...")
    adata.write(PROCESSED_H5AD)
    print(f"Done! AnnData shape: {adata.shape}")

    return adata


def annotate_cell_types(adata):
    """
    Assign cell types to Leiden clusters based on marker gene expression.
    Uses the highest mean expression of known markers per cluster, with a
    deduplication step to resolve ties when shared markers (e.g. NKG7 in
    both NK and CD8 T) cause multiple clusters to receive the same label.
    """
    # Use raw (normalized, log-transformed) data for marker scoring
    marker_genes = {}
    for ct, genes in CELL_TYPE_MARKERS.items():
        for g in genes:
            if g in adata.raw.var_names:
                marker_genes.setdefault(ct, []).append(g)

    cluster_ids = adata.obs["leiden"].unique()

    # Compute score matrix: cluster -> {cell_type: score}
    cluster_scores = {}
    for cluster in cluster_ids:
        mask = adata.obs["leiden"] == cluster
        scores = {}
        for ct, genes in marker_genes.items():
            expr = np.array(adata.raw[mask, genes].X.mean(axis=0)).flatten()
            scores[ct] = expr.mean()
        cluster_scores[cluster] = scores

    # Initial assignment: each cluster gets its highest-scoring cell type
    cluster_map = {c: max(scores, key=scores.get)
                   for c, scores in cluster_scores.items()}

    # Deduplication: when a cell type is assigned to multiple clusters,
    # keep it for the best-scoring cluster and reassign others to their
    # next-best unused cell type.  Skip cell types expected to span
    # multiple clusters (CD4 T commonly has naive + memory subsets).
    multi_cluster_types = {"CD4 T"}
    from collections import Counter
    ct_counts = Counter(cluster_map.values())
    for ct, count in ct_counts.items():
        if count > 1 and ct not in multi_cluster_types:
            dup_clusters = [c for c, t in cluster_map.items() if t == ct]
            best_c = max(dup_clusters, key=lambda c: cluster_scores[c][ct])
            for c in dup_clusters:
                if c != best_c:
                    ranked = sorted(cluster_scores[c],
                                    key=cluster_scores[c].get, reverse=True)
                    for alt in ranked:
                        if alt != ct:
                            cluster_map[c] = alt
                            break

    return cluster_map


if __name__ == "__main__":
    main()
