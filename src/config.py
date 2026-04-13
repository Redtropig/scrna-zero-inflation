"""
Centralized configuration for the scRNA-seq distribution modelling project.
All tunable parameters live here for reproducibility.
"""
import os

# ---------- Paths ----------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(DATA_DIR, "figures")
SRC_DIR = os.path.join(PROJECT_DIR, "src")

PROCESSED_H5AD = os.path.join(DATA_DIR, "pbmc3k_processed.h5ad")

# Mode-aware output paths. The {mode} slot is either "noff" (no offset,
# legacy behaviour used in the first draft) or "offset" (with a per-cell
# log(library-size) offset).  We keep both so the report can show explicitly
# how much of the overdispersion in the pooled analysis is explained by
# sequencing-depth differences alone.
FIT_ALL_CELLS_FMT = os.path.join(DATA_DIR, "fit_results_all_cells_{mode}.csv")
MODEL_SEL_ALL_FMT = os.path.join(DATA_DIR, "model_selection_all_cells_{mode}.csv")
FIT_PER_CLUSTER_FMT = os.path.join(DATA_DIR, "fit_results_per_cluster_{mode}.csv")
MODEL_SEL_CLUSTER_FMT = os.path.join(DATA_DIR, "model_selection_per_cluster_{mode}.csv")

# Backwards-compatible names (default = with offset, the main result)
FIT_ALL_CELLS = FIT_ALL_CELLS_FMT.format(mode="offset")
MODEL_SEL_ALL = MODEL_SEL_ALL_FMT.format(mode="offset")
FIT_PER_CLUSTER = FIT_PER_CLUSTER_FMT.format(mode="offset")
MODEL_SEL_CLUSTER = MODEL_SEL_CLUSTER_FMT.format(mode="offset")

SCVI_DIR = os.path.join(DATA_DIR, "scvi_results")
GLMPCA_DIR = os.path.join(DATA_DIR, "glmpca_results")

# ---------- Random seed ----------
SEED = 42

# ---------- QC thresholds ----------
MIN_GENES_PER_CELL = 200
MAX_GENES_PER_CELL = 5000
MAX_PCT_MITO = 10.0          # percent
MIN_CELLS_PER_GENE = 3

# ---------- Analysis parameters ----------
N_HIGHLY_VARIABLE = 2000
LEIDEN_RESOLUTION = 1.0
MIN_CELLS_PER_CLUSTER = 50   # skip clusters smaller than this

# Gene-filter threshold applied at fit time.  HVG-based selection
# over-samples overdispersed genes and biases the model-selection contest
# toward NB.  We instead fit every post-QC gene that exceeds this mean-count
# floor — set to 0.01 so near-silent genes (which have degenerate
# likelihoods under ZI models) are skipped but everything else is kept.
MIN_MEAN_COUNT_FOR_FIT = 0.01

# Size factor convention: s_i = total_counts_i / median(total_counts).
# Used as a per-cell log offset in the GLM fits.
SIZE_FACTOR_MODES = ["offset", "noff"]

# ---------- Distribution fitting ----------
DISTRIBUTIONS = ["poisson", "nb", "zip", "zinb"]
N_PARAMS = {"poisson": 1, "nb": 2, "zip": 2, "zinb": 3}
BIC_DELTA_THRESHOLD = 10     # strong evidence on Kass-Raftery scale

# ---------- scVI training ----------
SCVI_MAX_EPOCHS = 200
SCVI_TRAIN_SIZE = 0.9
SCVI_N_LATENT = 10
SCVI_N_HIDDEN = 128
SCVI_N_LAYERS = 1
SCVI_LIKELIHOODS = ["zinb", "nb", "poisson"]

# ---------- Cell type markers (for PBMC annotation) ----------
CELL_TYPE_MARKERS = {
    "CD4 T":      ["IL7R", "CD4"],
    "CD8 T":      ["CD8A", "CD8B"],
    "NK":         ["GNLY", "NKG7"],
    "B":          ["MS4A1", "CD79A"],
    "CD14 Mono":  ["CD14", "LYZ"],
    "FCGR3A Mono":["FCGR3A", "MS4A7"],
    "DC":         ["FCER1A", "CST3"],
    "Platelet":   ["PPBP"],
}

# ---------- Ensure output directories exist ----------
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SCVI_DIR, exist_ok=True)
os.makedirs(GLMPCA_DIR, exist_ok=True)


# ============================================================
#  Helpers
# ============================================================
def compute_size_factors(total_counts):
    """Return s_i = total_counts_i / median(total_counts)."""
    import numpy as np
    tc = np.asarray(total_counts, dtype=float)
    med = float(np.median(tc))
    if med <= 0:
        return np.ones_like(tc)
    return tc / med
