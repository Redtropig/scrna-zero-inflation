# Modelling Single-cell RNA-Seq Data Distributions

CPSC 445 (Algorithms in Bioinformatics) course project, University of British Columbia

## Versions

- `main` (September 2026): revised report and README. The analysis, code, and figures are unchanged.
  - Wording: notes about earlier drafts were replaced with a direct description of the study design, and an unsupported novelty claim was removed.
  - Credit: the feedback on our proposal that pointed out the circularity of HVG-only fitting is now acknowledged.
  - Corrections, checked against the archived pipeline outputs: the 99.1% figure is now stated over the 1,719 of the top 2,000 HVGs that pass the mean filter; the descriptions of the expression-decile pattern, of which cell types have the highest UMI totals, of the GLM-PCA clustering comparison, and of zero-inflation after cell-type conditioning now match the data; the 0.4% bound on zero-inflation is limited to the analyses reported; and one rounding error in Table 1 is fixed (per-cell-type, no-offset ZINB: from 0.1% to 0.2%).
- [`submitted-2026-04-13`](https://github.com/Redtropig/scrna-zero-inflation/tree/submitted-2026-04-13): the version submitted for grading on April 13, 2026.

## Overview

This project tests which statistical distributions best model single-cell RNA-seq UMI count data, addressing the "zero-inflation" debate in the field. We compare Poisson, Negative Binomial (NB), Zero-Inflated Poisson (ZIP), and Zero-Inflated Negative Binomial (ZINB) distributions on the PBMC 3k dataset, fitting **all 9,107 post-QC genes** (not just HVGs) under both no-offset and per-cell library-size offset modes. We complement the univariate analysis with scVI and GLM-PCA latent-variable models under multiple observation likelihoods, including an explicit multinomial family.

### Key Findings

- **HVG-only fitting is circularly biased**: Seurat v3 HVG selection picks genes whose variance exceeds the empirical mean-variance trend, which tilts a Poisson-vs-NB BIC contest toward NB (99.1% NB on HVGs vs. 43.6% on all genes)
- **Library-size offset shifts the balance further**: adding a per-cell log-offset moves ~4 pp from NB to Poisson (56.2% Poisson, 39.8% NB)
- **Cell-type conditioning makes Poisson dominant**: within Leiden clusters, Poisson wins 86.5% of 58,614 gene–cluster BIC contests; NB 11.2%
- **Zero-inflation is negligible**: 0.1–0.3% of genes/gene-cluster pairs are flagged at ΔBIC > 10, depending on offset mode
- **scVI confirms overdispersion > zero-inflation**: ZINB–NB gap is ~10 nats/cell vs. NB–Poisson gap of ~50 nats/cell
- **GLM-PCA confirms Poisson–multinomial equivalence**: the two families differ by 0.2% in deviance and give nearly identical clustering ARI (0.501 vs. 0.500)

## Project Structure

```text
├── report.tex              # LaTeX report (~10 pages)
├── report.pdf              # Compiled report
├── ref.bib                 # Bibliography (11 references)
├── requirements.txt        # Pinned Python dependencies
├── README.md               # This file
├── src/
│   ├── config.py           # Centralized constants, paths, size-factor computation
│   ├── utils.py            # Offset-aware distribution fitters, AIC/BIC, sanity tests
│   ├── 01_data.py          # Data acquisition, QC, clustering, cell type annotation
│   ├── 02_fit.py           # Per-gene fitting (all 9,107 genes × {noff, offset} modes)
│   ├── 03_select.py        # Model selection via BIC, zero-fraction analysis, decile stratification
│   ├── 04_heterogeneity.py # Per-cluster re-fitting under both offset modes
│   ├── 05_scvi.py          # scVI comparison (ZINB, NB, Poisson likelihoods)
│   ├── 05b_glmpca.py       # GLM-PCA comparison (multinomial, Poisson, NB families)
│   └── 06_figures.py       # Figure generation (8 figures)
└── data/
    ├── pbmc3k_processed.h5ad              # Processed AnnData object
    ├── fit_results_all_cells_{noff,offset}.csv   # Distribution fitting results
    ├── model_selection_all_cells_{noff,offset}.csv  # Model selection summaries
    ├── fit_results_per_cluster_{noff,offset}.csv    # Per-cluster fitting results
    ├── model_selection_per_cluster_{noff,offset}.csv # Per-cluster model selection
    ├── figures/                            # All generated figures (PDF + PNG)
    ├── scvi_results/                       # scVI trained models and metrics
    └── glmpca_results/                     # GLM-PCA factors, UMAPs, and metrics
```

## Setup and Reproduction

### Requirements

- Python 3.12 (scanpy ≥ 1.12 requires Python ≥ 3.12)
- **Python 3.13 is not supported**: scVI crashes due to a multiprocessing/OMP incompatibility in PyTorch Lightning. Use Python 3.12.x.
- Internet connection (Step 1 downloads the PBMC 3k dataset from 10x Genomics via Scanpy)
- ~2 GB disk space for data and models
- GPU is **not required**; all scripts run on CPU. scVI training (Step 5a) benefits from a CUDA GPU but completes in ~5 min on CPU as well.
- **Total pipeline runtime**: ~60 minutes on a modern multi-core CPU (Steps 2 and 4 dominate).

### Installation

```bash
# Create virtual environment (using uv, or pip below)
uv venv .venv --python 3.12
source .venv/bin/activate

# Install pinned dependencies
uv pip install -r requirements.txt
```

**Alternative (without `uv`):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Alternative (conda):**

```bash
conda create -y -p .venv python=3.12
conda activate ./.venv
pip install -r requirements.txt
```

### Running the Pipeline

Scripts are numbered to indicate execution order. **Run all scripts from the `src/` directory:**

```bash
cd src/

# Step 1: Download and preprocess PBMC 3k data (~1 min; requires internet)
#   -> data/pbmc3k_processed.h5ad
python 01_data.py

# Step 2: Fit 4 distributions to all 9,107 post-QC genes (~12 min, both offset modes)
#   -> data/fit_results_all_cells_{noff,offset}.csv
python 02_fit.py

# Step 3: Model selection, zero-fraction analysis, decile stratification
#   -> data/model_selection_all_cells_{noff,offset}.csv
python 03_select.py

# Step 4: Per-cluster heterogeneity analysis (~30 min, both offset modes)
#   -> data/fit_results_per_cluster_{noff,offset}.csv
#   -> data/model_selection_per_cluster_{noff,offset}.csv
python 04_heterogeneity.py

# Step 5a: scVI comparison with 3 likelihoods (~5 min on CPU)
#   -> data/scvi_results/
python 05_scvi.py

# Step 5b: GLM-PCA multinomial/Poisson/NB comparison (~12 min)
#   -> data/glmpca_results/
python 05b_glmpca.py

# Step 6: Generate all figures
#   -> data/figures/fig{1..8}_*.{pdf,png}
python 06_figures.py
```

### Compiling the Report

Run from the **project root** (not `src/`):

```bash
cd ..   # return to project root if still in src/
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
```

## Methods Summary

All fits are performed both without offset (`noff`) and with a per-cell log-library-size offset (`offset`), where size factor $s_i = T_i / \text{median}(T)$.

| Distribution | Parameters | Estimation Method |
| --- | --- | --- |
| Poisson | μ | Closed-form MLE: μ̂ = ΣX / Σs |
| Negative Binomial | μ, φ | Profile likelihood (Brent on log φ) |
| Zero-Inflated Poisson | π₀, μ | statsmodels (BFGS/NM) with log-offset |
| Zero-Inflated NB | π₀, μ, φ | statsmodels (BFGS/NM) with log-offset |

Model selection uses BIC with ΔBIC > 10 threshold for zero-inflation evidence (Kass–Raftery "very strong" scale).

## References

1. Svensson (2020). Droplet scRNA-seq is not zero-inflated. *Nature Biotechnology*.
2. Choi et al. (2020). Bayesian model selection reveals biological origins of zero inflation. *Genome Biology*.
3. Kim et al. (2020). Demystifying "drop-outs" in single-cell UMI data. *Genome Biology*.
4. Pan et al. (2023). The Poisson distribution model fits UMI-based scRNA-seq data. *BMC Bioinformatics*.
5. Pierson & Yau (2015). ZIFA: Dimensionality reduction for zero-inflated scRNA-seq. *Genome Biology*.
6. Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*.
7. Risso et al. (2018). A general and flexible method for signal extraction from single-cell RNA-seq data. *Nature Communications*.
8. Jiang et al. (2022). Statistics or biology: the zero-inflation controversy about scRNA-seq data. *Genome Biology*.
9. Townes et al. (2019). Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model. *Genome Biology*.
10. Wolf et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*.
11. Kass & Raftery (1995). Bayes Factors. *Journal of the American Statistical Association*.
