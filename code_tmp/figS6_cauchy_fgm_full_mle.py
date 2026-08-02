r"""Full three-parameter MLE of the shared-buffer Cauchy-FGM ancestral DFEs.

This is the Cauchy-mutation counterpart of ``figS6_fgm_exper.py``.  It uses the same
Couce/Limdi data loaders and replicate pooling.  The fit is restricted to the prespecified
range ``s >= -0.4`` where the empirical heavy-tail law applies, and uses the likelihood
conditional on that truncation.  Couce remains a secondary analysis because its missing
essential disruptions are an ascertainment process, not a literal sharp cutoff in s.

Model
-----

    delta = sigma Z / |Z0|,  Z ~ N_n(0,I), Z0 ~ N(0,1)
    s = -r . delta - |delta|^2/2.

The exact DFE density is in ``cmn/cmn_cauchy_fgm.py``.  A Gaussian measurement error with
the same fixed s.d. used by the old figure (0.005) is convolved numerically with the exact
density.  The reported fit maximizes the approximate unbinned log likelihood of every
observed effect jointly over continuous (n, r, sigma); no moments are locked.

Outputs
-------
    data/cauchy_fgm_full_mle.json
    data/cauchy_fgm_full_mle_params.txt
    figs_paper/figS6_cauchy_fgm_full_mle.pdf
"""

import json
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cmn.cmn_cauchy_fgm import (  # noqa: E402
    cauchy_fgm_dfe_pdf,
    cauchy_fgm_survival,
    cauchy_fgm_large_n_pdf,
    cauchy_fgm_large_n_survival,
)
from cmn.cmn_fgm_exper import MEAS_ERR, load_couce, load_limdi  # noqa: E402

DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figs_paper")
JSON_PATH = os.path.join(DATA_DIR, "cauchy_fgm_full_mle.json")
TXT_PATH = os.path.join(DATA_DIR, "cauchy_fgm_full_mle_params.txt")
FIG_PATH = os.path.join(FIG_DIR, "figS6_cauchy_fgm_full_mle.pdf")
PROFILE_FIG_PATH = os.path.join(FIG_DIR, "figS6_cauchy_fgm_n_profiles.pdf")

# The optimizer uses the combinations that remain finite on the large-n ridge:
# C = n sigma^2 and A = r sigma.  Reported values are transformed back to n,r,sigma.
# These are broad numerical bounds, not biological priors.
OPT_BOUNDS = {
    # n is a physical phenotype dimension.  We use the continuous effective-dimension
    # extension for inference, but restrict it to n >= 2; below 2 the true density has an
    # integrable singularity at the hard support edge that a fixed FFT grid cannot resolve.
    "n": (2.0, 500.0),
    "C": (1e-6, 0.2),
    "A": (1e-5, 0.2),
}
DX = 2.0e-4
ERROR_PAD_SD = 10.0
DE_SEED = 1729
DE_MAXITER = 45
DE_POPSIZE = 10
PROFILE_N = np.geomspace(2.0, 300.0, 61)
LOWER_CUT = -0.4

DISPLAY_NAMES = {
    "Couce 0K": "REL607 (Couce 0K)",
    "Couce 2K": "ARA+2 (Couce 2K)",
    "REL606": "REL606 (Limdi)",
    "REL607": "REL607 (Limdi)",
}
DISPLAY_ORDER = ["Couce 0K", "REL607", "Couce 2K", "REL606"]


def ancestor_dfes(trim=(0.0, 0.0), lower_cut=LOWER_CUT):
    """Use the old figure's loaders, retaining the prespecified fitted range."""
    couce = dict(load_couce(trim=trim, labels=("0K", "2K")))
    limdi = load_limdi(populations=["REL606", "REL607"], trim=trim)
    raw = [
        ("Couce 0K", couce["0K"]),
        ("Couce 2K", couce["2K"]),
        ("REL606", limdi["REL606"]),
        ("REL607", limdi["REL607"]),
    ]
    return [(name, effects[effects >= lower_cut]) for name, effects in raw]


class UnbinnedNoisyLikelihood:
    """Numerically convolved, interpolated unbinned likelihood for one fixed dataset."""

    def __init__(self, effects, eps=MEAS_ERR, dx=DX, lower_cut=LOWER_CUT):
        self.effects = np.asarray(effects, dtype=float)
        self.eps = float(eps)
        self.dx = float(dx)
        self.lower_cut = float(lower_cut)
        pad = ERROR_PAD_SD * self.eps if self.eps > 0.0 else 2.0 * self.dx
        lo = float(self.effects.min()) - pad
        hi = float(self.effects.max()) + pad
        number = int(np.ceil((hi - lo) / self.dx)) + 1
        self.x = lo + self.dx * np.arange(number)
        if self.eps > 0.0:
            half = int(np.ceil(ERROR_PAD_SD * self.eps / self.dx))
            offset = self.dx * np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * (offset / self.eps) ** 2)
            self.kernel = kernel / kernel.sum()
        else:
            self.kernel = None

    def observed_pdf_grid(self, n, r, sigma):
        true_pdf = cauchy_fgm_dfe_pdf(self.x, n=n, sigma=sigma, r=r)
        true_pdf = np.where(np.isfinite(true_pdf), true_pdf, 0.0)
        if self.kernel is None:
            return true_pdf
        noisy = fftconvolve(true_pdf, self.kernel, mode="same")
        return np.maximum(noisy, 0.0)

    def loglik(self, n, r, sigma):
        pdf = self.observed_pdf_grid(n=n, r=r, sigma=sigma)
        at_data = np.interp(self.effects, self.x, pdf)
        if np.any(~np.isfinite(at_data)) or np.any(at_data <= 0.0):
            return -np.inf
        keep_probability = cauchy_fgm_survival(
            self.lower_cut, n=n, sigma=sigma, r=r, eps=self.eps
        )
        if not np.isfinite(keep_probability) or keep_probability <= 0.0:
            return -np.inf
        return float(np.log(at_data).sum() - self.effects.size * np.log(keep_probability))

    @staticmethod
    def unpack(theta):
        n, C, A = np.exp(np.asarray(theta, dtype=float))
        sigma = np.sqrt(C / n)
        r = A / sigma
        return n, r, sigma, C, A

    def objective(self, theta):
        n, r, sigma, _, _ = self.unpack(theta)
        ll = self.loglik(n=n, r=r, sigma=sigma)
        return -ll if np.isfinite(ll) else 1e300


class LargeNNoisyLikelihood(UnbinnedNoisyLikelihood):
    """Two-parameter likelihood for the exact n -> infinity limiting DFE."""

    def observed_pdf_grid(self, C, A):
        true_pdf = cauchy_fgm_large_n_pdf(self.x, C=C, A=A)
        true_pdf = np.where(np.isfinite(true_pdf), true_pdf, 0.0)
        if self.kernel is None:
            return true_pdf
        noisy = fftconvolve(true_pdf, self.kernel, mode="same")
        return np.maximum(noisy, 0.0)

    def loglik(self, C, A):
        pdf = self.observed_pdf_grid(C=C, A=A)
        at_data = np.interp(self.effects, self.x, pdf)
        if np.any(~np.isfinite(at_data)) or np.any(at_data <= 0.0):
            return -np.inf
        keep_probability = cauchy_fgm_large_n_survival(
            self.lower_cut, C=C, A=A, eps=self.eps
        )
        if not np.isfinite(keep_probability) or keep_probability <= 0.0:
            return -np.inf
        return float(np.log(at_data).sum() - self.effects.size * np.log(keep_probability))

    def objective(self, theta):
        C, A = np.exp(np.asarray(theta, dtype=float))
        ll = self.loglik(C=C, A=A)
        return -ll if np.isfinite(ll) else 1e300


def fit_one(effects, eps=MEAS_ERR, dx=DX):
    """Global differential-evolution search followed by local L-BFGS polishing."""
    likelihood = UnbinnedNoisyLikelihood(effects, eps=eps, dx=dx)
    log_bounds = [
        (np.log(OPT_BOUNDS["n"][0]), np.log(OPT_BOUNDS["n"][1])),
        (np.log(OPT_BOUNDS["C"][0]), np.log(OPT_BOUNDS["C"][1])),
        (np.log(OPT_BOUNDS["A"][0]), np.log(OPT_BOUNDS["A"][1])),
    ]
    global_fit = differential_evolution(
        likelihood.objective,
        bounds=log_bounds,
        seed=DE_SEED,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-7,
    )
    local_fit = minimize(
        likelihood.objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=log_bounds,
        options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 1000},
    )
    best = local_fit if local_fit.fun <= global_fit.fun else global_fit
    n, r, sigma, C, A = likelihood.unpack(best.x)
    bounds_hit = {
        key: bool(
            value <= lower * 1.001 or value >= upper / 1.001
        )
        for key, value, (lower, upper) in zip(
            ("n", "C", "A"), (n, C, A), OPT_BOUNDS.values()
        )
    }
    return {
        "n": float(n),
        "r": float(r),
        "sigma": float(sigma),
        "C_n_sigma2": float(C),
        "A_r_sigma": float(A),
        "loglik": float(-best.fun),
        "success": bool(best.success),
        "message": str(best.message),
        "bounds_hit": bounds_hit,
        "_theta": np.asarray(best.x, float),
        "_likelihood": likelihood,
    }


def grid_resolution_check(effects, fit, eps=MEAS_ERR):
    """Re-evaluate the fitted parameters at half the numerical grid spacing."""
    fine = UnbinnedNoisyLikelihood(effects, eps=eps, dx=0.5 * DX)
    ll = fine.loglik(fit["n"], fit["r"], fit["sigma"])
    return {
        "dx": 0.5 * DX,
        "loglik_at_fit": float(ll),
        "delta_loglik": float(ll - fit["loglik"]),
    }


def fit_large_n(effects, eps=MEAS_ERR, dx=DX):
    """Fit the exact n -> infinity DFE over its two surviving scale parameters."""
    likelihood = LargeNNoisyLikelihood(effects, eps=eps, dx=dx)
    bounds = [
        (np.log(OPT_BOUNDS["C"][0]), np.log(OPT_BOUNDS["C"][1])),
        (np.log(OPT_BOUNDS["A"][0]), np.log(OPT_BOUNDS["A"][1])),
    ]
    global_fit = differential_evolution(
        likelihood.objective,
        bounds=bounds,
        seed=DE_SEED,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        polish=False,
        tol=1e-8,
    )
    local_fit = minimize(
        likelihood.objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 1000},
    )
    best = local_fit if local_fit.fun <= global_fit.fun else global_fit
    C, A = np.exp(best.x)
    return {
        "C_n_sigma2": float(C),
        "A_r_sigma": float(A),
        "loglik": float(-best.fun),
        "success": bool(best.success),
        "message": str(best.message),
    }


def observed_hessian(fit, step=2e-3):
    """Finite-difference observed-information matrix in log(n,C,A) coordinates."""
    objective = fit["_likelihood"].objective
    theta = fit["_theta"]
    size = theta.size
    H = np.empty((size, size), dtype=float)
    f0 = objective(theta)
    for i in range(size):
        ei = np.zeros(size)
        ei[i] = step
        H[i, i] = (objective(theta + ei) - 2.0 * f0 + objective(theta - ei)) / step**2
        for j in range(i):
            ej = np.zeros(size)
            ej[j] = step
            H[i, j] = H[j, i] = (
                objective(theta + ei + ej)
                - objective(theta + ei - ej)
                - objective(theta - ei + ej)
                + objective(theta - ei - ej)
            ) / (4.0 * step**2)
    try:
        covariance = np.linalg.inv(H)
        eig = np.linalg.eigvalsh(H)
        valid = bool(np.all(eig > 0.0) and np.all(np.diag(covariance) > 0.0))
    except np.linalg.LinAlgError:
        covariance = np.full_like(H, np.nan)
        eig = np.full(size, np.nan)
        valid = False

    # log(n), log(r), log(sigma) as linear transforms of log(n),log(C),log(A).
    transform = np.array([
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 1.0],
        [-0.5, 0.5, 0.0],
    ])
    cov_report = transform @ covariance @ transform.T
    estimates = np.array([fit["n"], fit["r"], fit["sigma"]])
    if valid and np.all(np.diag(cov_report) > 0.0):
        se_log = np.sqrt(np.diag(cov_report))
        intervals = np.column_stack([
            estimates * np.exp(-1.96 * se_log),
            estimates * np.exp(1.96 * se_log),
        ])
    else:
        se_log = np.full(3, np.nan)
        intervals = np.full((3, 2), np.nan)
    return {
        "valid": valid,
        "eigenvalues": eig.tolist(),
        "se_log": dict(zip(("n", "r", "sigma"), se_log.tolist())),
        "wald_95": {
            key: interval.tolist()
            for key, interval in zip(("n", "r", "sigma"), intervals)
        },
    }


def profile_n(fit, large_n_fit):
    """Profile n while optimizing the two stable scale combinations C and A."""
    likelihood = fit["_likelihood"]
    ca_bounds = [
        (np.log(OPT_BOUNDS["C"][0]), np.log(OPT_BOUNDS["C"][1])),
        (np.log(OPT_BOUNDS["A"][0]), np.log(OPT_BOUNDS["A"][1])),
    ]
    start = fit["_theta"][1:].copy()
    loglik = []
    C_values = []
    A_values = []
    r_values = []
    sigma_values = []
    for n in PROFILE_N:
        logn = np.log(n)

        def objective_ca(log_ca):
            return likelihood.objective(np.r_[logn, log_ca])

        candidates = []
        for initial in (
            start,
            fit["_theta"][1:],
            np.log([large_n_fit["C_n_sigma2"], large_n_fit["A_r_sigma"]]),
        ):
            candidates.append(minimize(
                objective_ca,
                initial,
                method="L-BFGS-B",
                bounds=ca_bounds,
                options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 500},
            ))
        best = min(candidates, key=lambda result: result.fun)
        # Never return an optimizer result worse than simply evaluating a supplied
        # starting point; this guards against finite-difference failures.
        starts = [
            start,
            fit["_theta"][1:],
            np.log([large_n_fit["C_n_sigma2"], large_n_fit["A_r_sigma"]]),
        ]
        start_values = [objective_ca(candidate) for candidate in starts]
        if min(start_values) < best.fun:
            index = int(np.argmin(start_values))
            best.x = np.asarray(starts[index])
            best.fun = float(start_values[index])
        start = best.x
        C, A = np.exp(best.x)
        sigma = np.sqrt(C / n)
        r = A / sigma
        loglik.append(float(-best.fun))
        C_values.append(float(C))
        A_values.append(float(A))
        r_values.append(float(r))
        sigma_values.append(float(sigma))

    loglik = np.asarray(loglik)
    reference = float(max(fit["loglik"], np.max(loglik)))
    delta = reference - loglik
    large_n_twice_delta = 2.0 * (reference - large_n_fit["loglik"])
    inside = 2.0 * delta <= 3.841458820694124
    if np.any(inside):
        selected = PROFILE_N[inside]
        ci = [float(selected.min()), float(selected.max())]
        ci_open_low = bool(inside[0])
        ci_open_high = bool(
            inside[-1] or large_n_twice_delta <= 3.841458820694124
        )
    else:
        ci = [float("nan"), float("nan")]
        ci_open_low = ci_open_high = True
    return {
        "n": PROFILE_N.tolist(),
        "loglik": loglik.tolist(),
        "delta_loglik": delta.tolist(),
        "C_n_sigma2": C_values,
        "A_r_sigma": A_values,
        "r": r_values,
        "sigma": sigma_values,
        "profile_95_n": ci,
        "profile_95_open_low": ci_open_low,
        "profile_95_open_high": ci_open_high,
        "large_n_limit": large_n_fit,
        "large_n_twice_delta_loglik": float(large_n_twice_delta),
    }


def plot_results(results, data_map):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    for ax, name in zip(axes.ravel(), DISPLAY_ORDER):
        effects = data_map[name]
        fit = results[name]
        like = fit["_likelihood"]
        pdf = like.observed_pdf_grid(fit["n"], fit["r"], fit["sigma"])
        keep_probability = cauchy_fgm_survival(
            LOWER_CUT,
            n=fit["n"],
            sigma=fit["sigma"],
            r=fit["r"],
            eps=MEAS_ERR,
        )
        pdf = np.where(like.x >= LOWER_CUT, pdf / keep_probability, np.nan)
        ax.hist(effects, bins=75, density=True, color="0.65", alpha=0.45,
                edgecolor="none", label="data")
        ax.plot(like.x, pdf, color="#6a00a8", lw=2.0, label="Cauchy-FGM MLE")
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.set_xlim(float(effects.min()), float(effects.max()))
        ax.set_title(DISPLAY_NAMES[name])
        ax.set_xlabel(r"Fitness effect $s$")
        ax.set_yticks([])
        ax.text(
            0.03, 0.97,
            rf"$n={fit['n']:.3g}$" "\n"
            rf"$r={fit['r']:.3g}$" "\n"
            rf"$\sigma={fit['sigma']:.3g}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.75", "alpha": 0.9},
        )
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def serializable(results):
    out = {}
    for name, fit in results.items():
        out[name] = {
            key: value for key, value in fit.items()
            if key not in {"_likelihood", "_theta"}
        }
    return out


def write_text(results):
    lines = [
        "Shared-buffer Cauchy FGM: full 3-D maximum-likelihood fits",
        "Model: delta = sigma Z / |Z0|; s = -r.delta - |delta|^2/2",
        f"Fitted observations: cleaned effects with s >= {LOWER_CUT:g}",
        f"Likelihood is conditional on observed s >= {LOWER_CUT:g}",
        "Interpretation: Limdi is primary; Couce library ascertainment is not a sharp-s cutoff.",
        f"Measurement error: Gaussian s.d. = {MEAS_ERR:g}",
        f"Numerical convolution grid: dx = {DX:g}",
        "",
        f"{'dataset':<13}{'N':>8}{'skew':>10}{'n':>12}{'r':>12}{'sigma':>12}"
        f"{'n*sigma2':>12}{'r*sigma':>12}{'logLik':>15}{'boundary':>12}",
    ]
    for name, fit in results.items():
        boundary = ",".join(key for key, hit in fit["bounds_hit"].items() if hit) or "none"
        lines.append(
            f"{name:<13}{fit['data']['N']:>8d}{fit['data']['skew']:>10.3f}"
            f"{fit['n']:>12.5g}{fit['r']:>12.5g}{fit['sigma']:>12.5g}"
            f"{fit['C_n_sigma2']:>12.5g}{fit['A_r_sigma']:>12.5g}"
            f"{fit['loglik']:>15.3f}{boundary:>12}"
        )
        prof = fit["n_profile"]
        upper = "inf" if prof["profile_95_open_high"] else f"{prof['profile_95_n'][1]:.5g}"
        lines.append(
            f"  n profile 95% interval: [{prof['profile_95_n'][0]:.5g}, {upper}]"
        )
        lines.append(
            f"  n->infinity: 2*DeltaLogLik={prof['large_n_twice_delta_loglik']:.3f}, "
            f"C={prof['large_n_limit']['C_n_sigma2']:.5g}, "
            f"A={prof['large_n_limit']['A_r_sigma']:.5g}"
        )
    with open(TXT_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_n_profiles(results):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    for ax, name in zip(axes.ravel(), DISPLAY_ORDER):
        profile = results[name]["n_profile"]
        n = np.asarray(profile["n"])
        twice_delta = 2.0 * np.asarray(profile["delta_loglik"])
        ax.plot(n, twice_delta, color="#6a00a8", lw=2.0)
        ax.axhline(3.841458820694124, color="0.35", lw=1.0, ls="--",
                   label="95% cutoff")
        ax.axvline(results[name]["n"], color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0.0, min(25.0, max(5.0, float(np.nanmax(twice_delta)))))
        ax.set_title(DISPLAY_NAMES[name])
        ax.set_xlabel(r"Fixed $n$")
        ax.set_ylabel(r"$2[\ell_{\max}-\ell_p(n)]$")
        ax.text(
            0.97,
            0.96,
            rf"$n\to\infty$: {profile['large_n_twice_delta_loglik']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(PROFILE_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    specs = ancestor_dfes()
    data_map = dict(specs)
    results = {}
    for name, effects in specs:
        started = time.time()
        print(f"Fitting {name}: N={effects.size}")
        fit = fit_one(effects)
        fit["data"] = {
            "N": int(effects.size),
            "min": float(effects.min()),
            "max": float(effects.max()),
            "mean": float(effects.mean()),
            "sd": float(effects.std()),
            "skew": float(skew(effects)),
        }
        fit["grid_check"] = grid_resolution_check(effects, fit)
        fit["hessian"] = observed_hessian(fit)
        large_n_fit = fit_large_n(effects)
        fit["n_profile"] = profile_n(fit, large_n_fit)
        fit["elapsed_seconds"] = float(time.time() - started)
        results[name] = fit
        print(
            f"  n={fit['n']:.6g}, r={fit['r']:.6g}, sigma={fit['sigma']:.6g}, "
            f"C={fit['C_n_sigma2']:.6g}, A={fit['A_r_sigma']:.6g}, "
            f"logLik={fit['loglik']:.3f}, boundary={fit['bounds_hit']}, "
            f"time={fit['elapsed_seconds']:.1f}s"
        )

    payload = {
        "model": "shared_buffer_multivariate_cauchy_fgm",
        "method": "full_3d_unbinned_mle_numeric_gaussian_convolution",
        "config": {
            "measurement_error_sd": MEAS_ERR,
            "dx": DX,
            "optimization_bounds": {
                key: list(value) for key, value in OPT_BOUNDS.items()
            },
            "tail_trim": [0.0, 0.0],
            "conditional_lower_cut": LOWER_CUT,
            "de_seed": DE_SEED,
            "de_maxiter": DE_MAXITER,
            "de_popsize": DE_POPSIZE,
        },
        "per_dfe": serializable(results),
    }
    with open(JSON_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    write_text(results)
    plot_results(results, data_map)
    plot_n_profiles(results)
    print(f"Saved {JSON_PATH}")
    print(f"Saved {TXT_PATH}")
    print(f"Saved {FIG_PATH}")
    print(f"Saved {PROFILE_FIG_PATH}")


if __name__ == "__main__":
    main()
