"""
06_figures.py — Generate all publication-quality figures for the report.

Reads saved intermediate results and produces Figures 1-8 for the revised
report (all-genes + offset-aware fits + GLM-PCA multinomial comparison).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc

from config import (
    PROCESSED_H5AD, FIG_DIR, SCVI_DIR, GLMPCA_DIR,
    FIT_ALL_CELLS_FMT, MODEL_SEL_ALL_FMT, MODEL_SEL_CLUSTER_FMT,
)
from utils import setup_style, save_fig

setup_style()

DIST_COLORS = {
    "poisson": "#4C72B0",
    "nb":      "#DD8452",
    "zip":     "#55A868",
    "zinb":    "#C44E52",
}
DIST_LABELS = {
    "poisson": "Poisson",
    "nb":      "Negative Binomial",
    "zip":     "Zero-Inflated Poisson",
    "zinb":    "Zero-Inflated NB",
}


# ============================================================
#  Helpers
# ============================================================
def _load_selection(mode):
    p = MODEL_SEL_ALL_FMT.format(mode=mode)
    return pd.read_csv(p) if os.path.exists(p) else None


def _load_selection_cluster(mode):
    p = MODEL_SEL_CLUSTER_FMT.format(mode=mode)
    return pd.read_csv(p) if os.path.exists(p) else None


def _bar_of_bic_winners(ax, df, title):
    counts = df["best_bic"].value_counts()
    total = len(df)
    dists = ["poisson", "nb", "zip", "zinb"]
    pcts = [counts.get(d, 0) / total * 100 for d in dists]
    colors = [DIST_COLORS[d] for d in dists]
    labels = [DIST_LABELS[d] for d in dists]
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        if pct > 2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("% of Genes")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


# ============================================================
#  Figure 1 — Dataset overview (unchanged)
# ============================================================
def figure1_dataset_overview(adata):
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    ax1 = fig.add_subplot(gs[0])
    cell_types = sorted(adata.obs["cell_type"].unique())
    palette = sns.color_palette("tab10", len(cell_types))
    ct_colors = dict(zip(cell_types, palette))
    for ct in cell_types:
        mask = adata.obs["cell_type"] == ct
        ax1.scatter(
            adata.obsm["X_umap"][mask, 0],
            adata.obsm["X_umap"][mask, 1],
            s=3, alpha=0.6, label=ct,
            color=ct_colors[ct], edgecolors="none",
        )
    ax1.legend(fontsize=7, markerscale=3, frameon=False, loc="best")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_title("(a) PBMC 3k — Cell Types")

    ax2 = fig.add_subplot(gs[1])
    data_plot = pd.DataFrame({
        "Cell Type": adata.obs["cell_type"].values,
        "Total UMIs per Cell": adata.obs["total_counts"].values,
    })
    sns.boxplot(data=data_plot, x="Cell Type", y="Total UMIs per Cell",
                hue="Cell Type", legend=False,
                ax=ax2, fliersize=1, linewidth=0.8,
                order=cell_types, palette=ct_colors)
    ax2.set_xticks(range(len(cell_types)))
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right",
                        fontsize=7)
    ax2.set_title("(b) Sequencing Depth per Cell")

    save_fig(fig, "fig1_dataset_overview")
    plt.close(fig)
    print("  Saved fig1_dataset_overview")


# ============================================================
#  Figure 2 — Offset ablation on all-cells selection
# ============================================================
def figure2_offset_ablation(sel_noff, sel_offset):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar_of_bic_winners(
        axes[0], sel_noff,
        "(a) All Cells — No Offset\n(legacy, confounded by depth)",
    )
    _bar_of_bic_winners(
        axes[1], sel_offset,
        "(b) All Cells — With Offset\n(log library size)",
    )
    save_fig(fig, "fig2_offset_ablation")
    plt.close(fig)
    print("  Saved fig2_offset_ablation")


# ============================================================
#  Figure 3 — BIC winners stratified by expression decile
# ============================================================
def figure3_by_expression(sel_offset):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    dists = ["poisson", "nb", "zip", "zinb"]
    deciles = sorted(sel_offset["expr_decile"].dropna().unique().astype(int))
    mat = np.zeros((len(dists), len(deciles)))
    for j, d in enumerate(deciles):
        sub = sel_offset[sel_offset["expr_decile"] == d]
        total = len(sub)
        if total == 0:
            continue
        counts = sub["best_bic"].value_counts()
        for i, dist in enumerate(dists):
            mat[i, j] = counts.get(dist, 0) / total * 100

    bottom = np.zeros(len(deciles))
    for i, dist in enumerate(dists):
        ax.bar(deciles, mat[i], bottom=bottom, label=DIST_LABELS[dist],
               color=DIST_COLORS[dist], edgecolor="white", linewidth=0.4)
        bottom += mat[i]
    ax.set_xlabel("Expression decile (1 = lowest, 10 = highest mean count)")
    ax.set_ylabel("% of Genes")
    ax.set_title("BIC winners stratified by expression level "
                 "(all genes, with offset)")
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)
    save_fig(fig, "fig3_by_expression")
    plt.close(fig)
    print("  Saved fig3_by_expression")


# ============================================================
#  Figure 4 — Mean-variance and zero-fraction diagnostics
# ============================================================
def figure4_diagnostics(sel_offset, fit_df_offset):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Observed vs NB-expected zero fraction (with offset)
    ax = axes[0]
    mask = sel_offset["exp_zero_nb"].notna()
    df = sel_offset[mask]
    colors = ["#C44E52" if zi else "#4C72B0" for zi in df["is_zero_inflated"]]
    ax.scatter(df["obs_zero_frac"], df["exp_zero_nb"],
               s=5, alpha=0.4, c=colors, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Observed zero fraction")
    ax.set_ylabel("NB expected zero fraction (offset)")
    ax.set_title("(a) Offset-aware zero-fraction calibration")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0',
               markersize=6, label='Non-ZI genes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C44E52',
               markersize=6, label='ZI by BIC'),
    ], fontsize=8, frameon=False)

    # (b) Mean-variance with Poisson / NB reference curves
    ax = axes[1]
    m = (sel_offset["mean_expr"] > 0) & (sel_offset["var_expr"] > 0)
    df = sel_offset[m]
    ax.scatter(df["mean_expr"], df["var_expr"],
               s=4, alpha=0.35, c="#4C72B0", edgecolors="none")
    x = np.logspace(np.log10(df["mean_expr"].min()),
                    np.log10(df["mean_expr"].max()), 100)
    ax.plot(x, x, "k--", lw=1.2, label="Poisson")
    # Median phi from NB fits
    nb_phis = (fit_df_offset[fit_df_offset["distribution"] == "nb"]
               ["params_json"].apply(lambda s: json.loads(s).get("phi", np.nan)))
    median_phi = float(nb_phis.median())
    ax.plot(x, x + x ** 2 / median_phi, "r-", lw=1.2,
            label=f"NB (median $\\phi$={median_phi:.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gene mean (UMI)")
    ax.set_ylabel("Gene variance (UMI)")
    ax.set_title("(b) Mean-variance after depth correction")
    ax.legend(fontsize=9, frameon=False)

    save_fig(fig, "fig4_diagnostics")
    plt.close(fig)
    print("  Saved fig4_diagnostics")


# ============================================================
#  Figure 5 — All cells vs per-cluster, both with offset
# ============================================================
def figure5_heterogeneity(sel_all, sel_clu):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar_of_bic_winners(axes[0], sel_all,
                        "(a) All Cells\n(with library-size offset)")
    _bar_of_bic_winners(axes[1], sel_clu,
                        "(b) Per Cell-Type\n(with offset, within cluster)")
    save_fig(fig, "fig5_heterogeneity")
    plt.close(fig)
    print("  Saved fig5_heterogeneity")


# ============================================================
#  Figure 6 — scVI comparison (existing, still valid)
# ============================================================
def figure6_scvi_comparison():
    metrics_path = os.path.join(SCVI_DIR, "metrics_summary.csv")
    adata_path = os.path.join(SCVI_DIR, "adata_with_scvi.h5ad")
    if not os.path.exists(metrics_path) or not os.path.exists(adata_path):
        print("  SKIPPED fig6 (scVI results not found)")
        return

    metrics = pd.read_csv(metrics_path)
    adata = sc.read_h5ad(adata_path)

    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8])
    likelihoods = ["zinb", "nb", "poisson"]
    titles = ["(a) ZINB", "(b) NB", "(c) Poisson"]
    cell_types = sorted(adata.obs["cell_type"].unique())
    palette = sns.color_palette("tab10", len(cell_types))
    ct_colors = dict(zip(cell_types, palette))

    for i, (ll, title) in enumerate(zip(likelihoods, titles)):
        ax = fig.add_subplot(gs[i])
        umap_key = f"X_umap_scvi_{ll}"
        if umap_key not in adata.obsm:
            ax.set_title(f"{title} (no data)")
            continue
        for ct in cell_types:
            mask = adata.obs["cell_type"] == ct
            ax.scatter(
                adata.obsm[umap_key][mask, 0],
                adata.obsm[umap_key][mask, 1],
                s=2, alpha=0.5, color=ct_colors[ct],
                label=ct if i == 0 else None, edgecolors="none",
            )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)
        if i == 0:
            ax.legend(fontsize=6, markerscale=3, frameon=False, loc="best")

    ax = fig.add_subplot(gs[3])
    lls = metrics.set_index("likelihood")["marginal_ll"]
    colors = [DIST_COLORS.get(ll, "#999") for ll in likelihoods]
    ax.bar([DIST_LABELS.get(ll, ll) for ll in likelihoods],
           [lls.get(ll, 0) for ll in likelihoods],
           color=colors, edgecolor="white")
    ax.set_ylabel("Marginal log-likelihood")
    ax.set_title("(d) scVI marginal LL")
    ax.set_xticks(range(len(likelihoods)))
    ax.set_xticklabels([DIST_LABELS.get(ll, ll) for ll in likelihoods],
                       rotation=30, ha="right", fontsize=8)

    save_fig(fig, "fig6_scvi_comparison")
    plt.close(fig)
    print("  Saved fig6_scvi_comparison")


# ============================================================
#  Figure 7 — Posterior predictive checks (existing)
# ============================================================
def figure7_ppc():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    likelihoods_to_plot = ["nb", "zinb"]
    colors_map = {"nb": "#DD8452", "zinb": "#C44E52"}

    ax = axes[0]
    for ll in likelihoods_to_plot:
        ppc_path = os.path.join(SCVI_DIR, f"ppc_{ll}.csv")
        if not os.path.exists(ppc_path):
            continue
        ppc = pd.read_csv(ppc_path)
        ax.scatter(ppc["real_zero_frac"], ppc["pp_zero_frac"],
                   s=3, alpha=0.3, color=colors_map[ll],
                   label=DIST_LABELS.get(ll, ll), edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Observed Zero Fraction")
    ax.set_ylabel("Posterior Predictive Zero Fraction")
    ax.set_title("(a) scVI Zero-Fraction PPC")
    ax.legend(fontsize=8, frameon=False, markerscale=3)

    ax = axes[1]
    for ll in likelihoods_to_plot:
        ppc_path = os.path.join(SCVI_DIR, f"ppc_{ll}.csv")
        if not os.path.exists(ppc_path):
            continue
        ppc = pd.read_csv(ppc_path)
        mask = (ppc["pp_mean"] > 0) & (ppc["pp_var"] > 0)
        m = ppc[mask]
        ax.scatter(m["pp_mean"], m["pp_var"],
                   s=3, alpha=0.3, color=colors_map[ll],
                   label=f"{DIST_LABELS.get(ll, ll)} (PPC)",
                   edgecolors="none")

    ppc_any = None
    for ll in likelihoods_to_plot:
        p = os.path.join(SCVI_DIR, f"ppc_{ll}.csv")
        if os.path.exists(p):
            ppc_any = pd.read_csv(p)
            break
    if ppc_any is not None:
        mask = (ppc_any["real_mean"] > 0) & (ppc_any["real_var"] > 0)
        ax.scatter(ppc_any.loc[mask, "real_mean"], ppc_any.loc[mask, "real_var"],
                   s=3, alpha=0.15, color="#333", label="Real",
                   edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gene mean")
    ax.set_ylabel("Gene variance")
    ax.set_title("(b) scVI mean-variance PPC")
    ax.legend(fontsize=7, frameon=False, markerscale=3)

    save_fig(fig, "fig7_ppc")
    plt.close(fig)
    print("  Saved fig7_ppc")


# ============================================================
#  Figure 8 — GLM-PCA multinomial / Poisson / NB
# ============================================================
def figure8_glmpca():
    metrics_path = os.path.join(GLMPCA_DIR, "metrics_summary.csv")
    obs_path = os.path.join(GLMPCA_DIR, "obs_metadata.csv")
    if not os.path.exists(metrics_path) or not os.path.exists(obs_path):
        print("  SKIPPED fig8 (GLM-PCA results not found)")
        return
    metrics = pd.read_csv(metrics_path)
    obs = pd.read_csv(obs_path, index_col=0)

    fams = ["mult", "poi", "nb"]
    labels = {"mult": "(a) Multinomial",
              "poi": "(b) Poisson",
              "nb": "(c) Negative Binomial"}

    cell_types = sorted(obs["cell_type"].unique())
    palette = sns.color_palette("tab10", len(cell_types))
    ct_colors = dict(zip(cell_types, palette))

    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8])

    for i, fam in enumerate(fams):
        ax = fig.add_subplot(gs[i])
        umap_path = os.path.join(GLMPCA_DIR, f"umap_{fam}.npy")
        if not os.path.exists(umap_path):
            ax.set_title(f"{labels[fam]} (no data)")
            continue
        umap = np.load(umap_path)
        for ct in cell_types:
            mask = obs["cell_type"].values == ct
            ax.scatter(umap[mask, 0], umap[mask, 1], s=2, alpha=0.55,
                       color=ct_colors[ct],
                       label=ct if i == 0 else None, edgecolors="none")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(labels[fam])
        if i == 0:
            ax.legend(fontsize=6, markerscale=3, frameon=False, loc="best")

    ax = fig.add_subplot(gs[3])
    ax.bar(metrics["family"].map({"mult": "Mult", "poi": "Poi", "nb": "NB"}),
           metrics["leiden_ari"],
           color=["#7A5195", "#4C72B0", "#DD8452"], edgecolor="white")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("(d) Cell-type ARI")
    ax.set_ylim(0, 1)

    save_fig(fig, "fig8_glmpca")
    plt.close(fig)
    print("  Saved fig8_glmpca")


# ============================================================
def main():
    print("=" * 60)
    print("Step 6: Figure Generation")
    print("=" * 60)

    print("\nLoading data...")
    adata = sc.read_h5ad(PROCESSED_H5AD)
    sel_noff = _load_selection("noff")
    sel_offset = _load_selection("offset")
    sel_clu_offset = _load_selection_cluster("offset")
    fit_df_offset = None
    fit_path = FIT_ALL_CELLS_FMT.format(mode="offset")
    if os.path.exists(fit_path):
        fit_df_offset = pd.read_csv(fit_path)

    print("\nGenerating figures...")
    figure1_dataset_overview(adata)
    if sel_noff is not None and sel_offset is not None:
        figure2_offset_ablation(sel_noff, sel_offset)
    if sel_offset is not None:
        figure3_by_expression(sel_offset)
    if sel_offset is not None and fit_df_offset is not None:
        figure4_diagnostics(sel_offset, fit_df_offset)
    if sel_offset is not None and sel_clu_offset is not None:
        figure5_heterogeneity(sel_offset, sel_clu_offset)
    figure6_scvi_comparison()
    figure7_ppc()
    figure8_glmpca()

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
