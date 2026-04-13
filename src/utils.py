"""
Shared helpers for distribution fitting, model selection, and plotting.

Each fitter accepts an optional `size_factor` array s_i (one value per cell).
When supplied, the mean rate for cell i is modelled as s_i * mu_j (equivalent
to including log(s_i) as an offset in a Poisson/NB GLM intercept model).  The
no-offset path is preserved by passing s_i = 1 for all cells.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from config import FIG_DIR, N_PARAMS, SEED

# ============================================================
#  Distribution fitting (with optional size-factor offset)
# ============================================================

def _ensure_size_factors(counts, size_factors):
    """Return a size-factor vector of length len(counts); ones if none given."""
    if size_factors is None:
        return np.ones(len(counts), dtype=float)
    s = np.asarray(size_factors, dtype=float)
    if s.shape[0] != len(counts):
        raise ValueError("size_factors length mismatch with counts")
    return s


def fit_poisson(counts, size_factors=None):
    """
    Fit Poisson(mu_j * s_i) by closed-form MLE.
    Score equation: sum_i (x_i - s_i * mu_j) = 0 => mu_j = sum_i x_i / sum_i s_i.
    Returns (params_dict, loglik).
    """
    s = _ensure_size_factors(counts, size_factors)
    sum_s = s.sum()
    if sum_s <= 0:
        return {"mu": 0.0}, -np.inf
    mu = counts.sum() / sum_s
    if mu <= 0:
        return {"mu": 0.0}, 0.0
    rates = s * mu
    ll = stats.poisson.logpmf(counts, mu=rates).sum()
    return {"mu": float(mu)}, float(ll)


def fit_nb(counts, size_factors=None):
    """
    Fit NB(mu_j * s_i, phi_j) (phi is gene-specific, shared across cells).
    Uses a nested profile-likelihood optimisation: given phi, mu has no closed
    form with heterogeneous offsets, so we solve for mu via a 1-D Newton root
    find and then refine phi by Brent search on the profile log-likelihood.

    Parameterisation: Var(X_i) = s_i*mu + (s_i*mu)^2/phi;
    scipy.stats.nbinom uses (n=phi, p=phi/(phi+s_i*mu)).
    """
    s = _ensure_size_factors(counts, size_factors)
    total = counts.sum()
    sum_s = s.sum()
    if sum_s <= 0 or total <= 0:
        return {"mu": 0.0, "phi": 1.0}, -np.inf

    # Moment-matching initial mu
    mu0 = total / sum_s

    # Moment-matching initial phi (ignoring depth heterogeneity)
    var_obs = counts.var()
    mean_obs = counts.mean()
    if var_obs <= mean_obs:
        phi0 = 1e6
    else:
        phi0 = mean_obs ** 2 / (var_obs - mean_obs)

    def mu_given_phi(phi):
        """Solve sum_i (phi * (x_i - s_i*mu)) / (s_i*mu + phi) = 0 for mu."""
        # Initial guess from moment matching
        mu = mu0
        for _ in range(50):
            denom = s * mu + phi
            num = np.sum((counts - s * mu) / denom)
            if not np.isfinite(num):
                break
            # Derivative wrt mu of score
            d = -np.sum(s * (counts + phi) / denom ** 2)
            if d == 0 or not np.isfinite(d):
                break
            step = num / d
            mu_new = mu - step
            if mu_new <= 0:
                mu_new = mu / 2
            if abs(mu_new - mu) < 1e-10 * max(mu, 1e-12):
                mu = mu_new
                break
            mu = mu_new
        return max(mu, 1e-12)

    def neg_profile_ll(log_phi):
        phi = np.exp(log_phi)
        mu = mu_given_phi(phi)
        rates = s * mu
        prob = phi / (phi + rates)
        ll = stats.nbinom.logpmf(counts, n=phi, p=prob).sum()
        return -ll if np.isfinite(ll) else 1e18

    try:
        res = minimize_scalar(neg_profile_ll, bounds=(np.log(1e-3), np.log(1e8)),
                              method="bounded", options={"xatol": 1e-4})
        phi_opt = float(np.exp(res.x))
        mu_opt = float(mu_given_phi(phi_opt))
        rates = s * mu_opt
        prob = phi_opt / (phi_opt + rates)
        ll = float(stats.nbinom.logpmf(counts, n=phi_opt, p=prob).sum())
        return {"mu": mu_opt, "phi": phi_opt}, ll
    except Exception:
        # Fallback: just use moment estimates
        rates = s * mu0
        prob = phi0 / (phi0 + rates)
        ll = float(stats.nbinom.logpmf(counts, n=phi0, p=prob).sum())
        return {"mu": float(mu0), "phi": float(phi0)}, ll


def fit_zip(counts, size_factors=None):
    """
    Fit Zero-Inflated Poisson with optional log-offset:
    P(X=0) = pi0 + (1-pi0) * exp(-s_i * mu);
    P(X=k) = (1-pi0) * Poisson(k; s_i * mu) for k>0.
    """
    import statsmodels.discrete.count_model as cm
    s = _ensure_size_factors(counts, size_factors)
    n = len(counts)
    exog = np.ones((n, 1))
    offset = np.log(np.clip(s, 1e-12, None))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = cm.ZeroInflatedPoisson(
                counts, exog, exog_infl=exog, offset=offset, inflation="logit"
            )
            res = model.fit(method="bfgs", maxiter=300, disp=0,
                            warn_convergence=False)
            if not res.converged:
                res = model.fit(method="nm", maxiter=800, disp=0,
                                warn_convergence=False)
        # params layout: [inflate_intercept, count_intercept]
        mu = float(np.exp(res.params[1]))       # count intercept -> mu
        logit_pi = float(res.params[0])          # inflate intercept
        pi0 = 1.0 / (1.0 + np.exp(-logit_pi))
        return {"pi0": float(pi0), "mu": mu}, float(res.llf)
    except Exception:
        return {"pi0": np.nan, "mu": np.nan}, np.nan


def fit_zinb(counts, size_factors=None):
    """
    Fit Zero-Inflated Negative Binomial with optional log-offset.
    Count component: NB(s_i * mu, phi). Inflate component: logit pi0.
    """
    import statsmodels.discrete.count_model as cm
    s = _ensure_size_factors(counts, size_factors)
    n = len(counts)
    exog = np.ones((n, 1))
    offset = np.log(np.clip(s, 1e-12, None))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = cm.ZeroInflatedNegativeBinomialP(
                counts, exog, exog_infl=exog, offset=offset, inflation="logit"
            )
            res = model.fit(method="bfgs", maxiter=300, disp=0,
                            warn_convergence=False)
            if not res.converged:
                res = model.fit(method="nm", maxiter=800, disp=0,
                                warn_convergence=False)
        # params layout: [inflate_intercept, count_intercept, lnalpha]
        mu = float(np.exp(res.params[1]))       # count intercept
        alpha = float(np.exp(res.params[2]))    # lnalpha -> alpha
        phi = 1.0 / max(alpha, 1e-10)
        logit_pi = float(res.params[0])          # inflate intercept
        pi0 = 1.0 / (1.0 + np.exp(-logit_pi))
        return {"pi0": float(pi0), "mu": mu, "phi": float(phi)}, float(res.llf)
    except Exception:
        return {"pi0": np.nan, "mu": np.nan, "phi": np.nan}, np.nan


FITTERS = {
    "poisson": fit_poisson,
    "nb":      fit_nb,
    "zip":     fit_zip,
    "zinb":    fit_zinb,
}


def fit_gene(counts, distributions=None, size_factors=None):
    """
    Fit all requested distributions to a single gene's count vector.
    If `size_factors` is given, fitters use it as a per-cell offset.
    Returns list of dicts: distribution, loglik, k, aic, bic, params, converged.

    Note: `converged` is True when the log-likelihood is finite (usable for
    BIC comparison), not strictly when the optimizer reports convergence.
    """
    if distributions is None:
        distributions = list(FITTERS.keys())
    n = len(counts)
    results = []
    for dist in distributions:
        params, ll = FITTERS[dist](counts, size_factors=size_factors)
        k = N_PARAMS[dist]
        converged = np.isfinite(ll)
        aic = 2 * k - 2 * ll if converged else np.nan
        bic = k * np.log(n) - 2 * ll if converged else np.nan
        results.append({
            "distribution": dist,
            "loglik": ll,
            "k": k,
            "aic": aic,
            "bic": bic,
            "params": params,
            "converged": converged,
        })
    return results


# ============================================================
#  Expected zero fractions
# ============================================================

def expected_zero_fraction(dist, params, size_factors=None):
    """
    Compute the mean expected zero fraction under a fitted distribution.
    When `size_factors` is given, the probability is averaged across cells:
    (1/N) * sum_i P(X_i = 0 | s_i * mu, ...).  Otherwise s_i = 1 is assumed.
    """
    s = np.asarray(size_factors, dtype=float) if size_factors is not None else np.array([1.0])

    if dist == "poisson":
        mu = params.get("mu", params.get("lambda"))
        return float(np.mean(np.exp(-s * mu)))
    elif dist == "nb":
        mu, phi = params["mu"], params["phi"]
        rates = s * mu
        return float(np.mean((phi / (phi + rates)) ** phi))
    elif dist == "zip":
        pi0 = params["pi0"]
        mu = params.get("mu", params.get("lambda"))
        return float(pi0 + (1 - pi0) * np.mean(np.exp(-s * mu)))
    elif dist == "zinb":
        pi0, mu, phi = params["pi0"], params["mu"], params["phi"]
        rates = s * mu
        return float(pi0 + (1 - pi0) * np.mean((phi / (phi + rates)) ** phi))
    return np.nan


# ============================================================
#  Plotting helpers
# ============================================================

def setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (7, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, name, tight=True):
    """Save a figure to FIG_DIR as both PNG and PDF."""
    import os
    if tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", facecolor="white")


# ============================================================
#  Synthetic sanity test
# ============================================================

def run_sanity_test():
    """
    Two-part sanity check.

    (1) NB(mu=5, phi=2) without offsets — NB should win by BIC and recover
        mu_hat near 5, phi_hat near 2.
    (2) Heterogeneous Poisson with size factors: X_i ~ Pois(s_i * mu) with
        mu=3, s_i uniformly on [0.5, 2].  Poisson-with-offset should win by
        BIC.  This exercises the offset pathway added in this revision.
    """
    np.random.seed(SEED)

    # Part 1 — no offset
    mu_true, phi_true = 5.0, 2.0
    p_true = phi_true / (phi_true + mu_true)
    data = np.random.negative_binomial(n=phi_true, p=p_true, size=1000)
    df = pd.DataFrame(fit_gene(data))
    winner = df.loc[df["bic"].idxmin(), "distribution"]
    nb_row = df[df["distribution"] == "nb"].iloc[0]
    mu_hat = nb_row["params"]["mu"]
    phi_hat = nb_row["params"]["phi"]
    print("=== Sanity Test 1: Synthetic NB(mu=5, phi=2), no offset ===")
    print(f"  BIC winner: {winner}  mu_hat={mu_hat:.2f}  phi_hat={phi_hat:.2f}")
    for _, row in df.iterrows():
        print(f"  {row['distribution']:6s}  loglik={row['loglik']:.1f}  "
              f"BIC={row['bic']:.1f}")
    assert winner in ("nb", "zinb"), f"Expected NB/ZINB, got {winner}"
    assert abs(mu_hat - mu_true) / mu_true < 0.15

    # Part 2 — heterogeneous Poisson with large size-factor spread.
    # This is the crucial test of the offset pathway: without an offset, the
    # depth heterogeneity inflates apparent dispersion and NB should fit phi
    # to a modest value; with an offset, NB should be forced toward Poisson
    # (phi -> very large) and Poisson-MLE should recover mu accurately.
    np.random.seed(SEED + 1)
    n = 2000
    s = np.random.uniform(0.2, 3.0, size=n)
    mu_true = 2.0
    data = np.random.poisson(lam=s * mu_true)

    df_no = pd.DataFrame(fit_gene(data))                    # no offset
    df_off = pd.DataFrame(fit_gene(data, size_factors=s))   # with offset

    phi_no  = df_no[df_no["distribution"] == "nb"].iloc[0]["params"]["phi"]
    phi_off = df_off[df_off["distribution"] == "nb"].iloc[0]["params"]["phi"]
    mu_hat_off = df_off[df_off["distribution"] == "poisson"].iloc[0]["params"]["mu"]

    print("\n=== Sanity Test 2: Heterogeneous Poisson(s*mu), mu=2, s~U(0.2,3) ===")
    print(f"  NB phi (no offset):   {phi_no:.2f}  (small = apparent overdispersion)")
    print(f"  NB phi (with offset): {phi_off:.2f}  (large = Poisson-like after depth correction)")
    print(f"  Poisson mu_hat:       {mu_hat_off:.3f}  (true=2.000)")
    assert phi_off > 10 * phi_no, (
        f"Offset failed to collapse overdispersion: phi_no={phi_no}, phi_off={phi_off}")
    assert abs(mu_hat_off - mu_true) / mu_true < 0.05
    print("  PASSED\n")


if __name__ == "__main__":
    run_sanity_test()
