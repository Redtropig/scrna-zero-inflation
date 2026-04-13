"""
05_scvi.py — scVI deep generative model comparison.

Trains scVI with three likelihoods (ZINB, NB, Poisson), extracts latent
spaces, computes marginal log-likelihoods, and runs posterior predictive checks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scvi
from scipy.sparse import issparse

anndata.settings.allow_write_nullable_strings = True

from config import (
    SEED, PROCESSED_H5AD, SCVI_DIR,
    SCVI_MAX_EPOCHS, SCVI_TRAIN_SIZE, SCVI_N_LATENT,
    SCVI_N_HIDDEN, SCVI_N_LAYERS, SCVI_LIKELIHOODS,
)

np.random.seed(SEED)
scvi.settings.seed = SEED


def main():
    print("=" * 60)
    print("Step 5: scVI Deep Generative Model Comparison")
    print("=" * 60)

    # Load data
    print("\nLoading processed data...")
    adata = sc.read_h5ad(PROCESSED_H5AD)

    # Use HVGs for scVI
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print(f"  {adata_hvg.n_obs} cells, {adata_hvg.n_vars} HVGs")

    metrics = {}

    for likelihood in SCVI_LIKELIHOODS:
        print(f"\n{'='*40}")
        print(f"Training scVI with gene_likelihood='{likelihood}'")
        print(f"{'='*40}")

        # Setup anndata for scVI
        scvi.model.SCVI.setup_anndata(adata_hvg, layer="counts")

        # Create model
        model = scvi.model.SCVI(
            adata_hvg,
            gene_likelihood=likelihood,
            n_latent=SCVI_N_LATENT,
            n_hidden=SCVI_N_HIDDEN,
            n_layers=SCVI_N_LAYERS,
        )

        # Train
        model.train(
            max_epochs=SCVI_MAX_EPOCHS,
            train_size=SCVI_TRAIN_SIZE,
            early_stopping=True,
            early_stopping_patience=10,
        )

        # Extract latent space
        latent = model.get_latent_representation()
        adata_hvg.obsm[f"X_scvi_{likelihood}"] = latent

        # Compute UMAP on latent space
        adata_tmp = sc.AnnData(X=latent, obs=adata_hvg.obs.copy())
        sc.pp.neighbors(adata_tmp, use_rep="X", random_state=SEED)
        sc.tl.umap(adata_tmp, random_state=SEED)
        adata_hvg.obsm[f"X_umap_scvi_{likelihood}"] = adata_tmp.obsm["X_umap"]

        # Marginal log-likelihood (evaluated on full training set)
        try:
            ll = model.get_marginal_ll(adata_hvg)
            print(f"  Marginal LL: {ll:.2f}")
        except Exception as e:
            print(f"  Could not compute marginal LL: {e}")
            ll = np.nan

        # Training history
        history = model.history
        train_loss_raw = history["train_loss"].values[-1] if "train_loss" in history else np.nan
        train_loss = float(np.asarray(train_loss_raw).flat[0]) if not np.isnan(np.asarray(train_loss_raw).flat[0]) else np.nan

        # Posterior predictive check: sample from model
        try:
            pp_samples = model.posterior_predictive_sample(adata_hvg, n_samples=1)
            # Handle different return types (sparse GCXS, AnnData, dict, ndarray)
            if hasattr(pp_samples, 'todense'):
                pp_counts = np.asarray(pp_samples.todense())
            elif isinstance(pp_samples, dict):
                pp_counts = pp_samples["X"]
                if hasattr(pp_counts, 'toarray'):
                    pp_counts = pp_counts.toarray()
            elif hasattr(pp_samples, 'X'):
                pp_counts = pp_samples.X
                if hasattr(pp_counts, 'toarray'):
                    pp_counts = pp_counts.toarray()
            else:
                pp_counts = np.asarray(pp_samples)
            if issparse(pp_counts):
                pp_counts = pp_counts.toarray()

            real_counts = adata_hvg.layers["counts"]
            if issparse(real_counts):
                real_counts = real_counts.toarray()

            # Per-gene zero fractions
            real_zero_frac = (real_counts == 0).mean(axis=0)
            pp_zero_frac = (pp_counts == 0).mean(axis=0)
            zero_frac_corr = np.corrcoef(real_zero_frac, pp_zero_frac)[0, 1]

            # Per-gene means
            real_means = real_counts.mean(axis=0)
            pp_means = pp_counts.mean(axis=0)
            mean_corr = np.corrcoef(real_means, pp_means)[0, 1]

            # Per-gene variances
            real_vars = real_counts.var(axis=0)
            pp_vars = pp_counts.var(axis=0)
            var_corr = np.corrcoef(real_vars, pp_vars)[0, 1]

            print(f"  PPC zero-fraction correlation: {zero_frac_corr:.4f}")
            print(f"  PPC mean correlation:          {mean_corr:.4f}")
            print(f"  PPC variance correlation:      {var_corr:.4f}")

        except Exception as e:
            print(f"  Posterior predictive check failed: {e}")
            zero_frac_corr = mean_corr = var_corr = np.nan
            real_zero_frac = pp_zero_frac = None
            real_means = pp_means = None
            real_vars = pp_vars = None

        metrics[likelihood] = {
            "likelihood": likelihood,
            "marginal_ll": float(ll) if np.isfinite(ll) else None,
            "train_loss_final": train_loss,
            "ppc_zero_frac_corr": float(zero_frac_corr) if np.isfinite(zero_frac_corr) else None,
            "ppc_mean_corr": float(mean_corr) if np.isfinite(mean_corr) else None,
            "ppc_var_corr": float(var_corr) if np.isfinite(var_corr) else None,
        }

        # Save per-gene PPC data
        if real_zero_frac is not None:
            ppc_df = pd.DataFrame({
                "gene": adata_hvg.var_names,
                "real_zero_frac": real_zero_frac,
                "pp_zero_frac": pp_zero_frac,
                "real_mean": real_means,
                "pp_mean": pp_means,
                "real_var": real_vars,
                "pp_var": pp_vars,
            })
            ppc_df.to_csv(os.path.join(SCVI_DIR, f"ppc_{likelihood}.csv"), index=False)

        # Save model
        model_dir = os.path.join(SCVI_DIR, f"model_{likelihood}")
        model.save(model_dir, overwrite=True)
        print(f"  Model saved to {model_dir}")

    # Save UMAP embeddings in the adata
    adata_hvg.write(os.path.join(SCVI_DIR, "adata_with_scvi.h5ad"))

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics.values())
    metrics_df.to_csv(os.path.join(SCVI_DIR, "metrics_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("scVI Comparison Summary")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\nAll results saved to {SCVI_DIR}")


if __name__ == "__main__":
    main()
