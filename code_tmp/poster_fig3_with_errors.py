r"""With-error backend for Poster Figure 3.

The two panels show the same REL607 ancestor DFE and two fitted observed-density
curves, with separate views of the deleterious tail and the central bulk:

1. moment-constrained canonical FGM with gene-specific measurement errors;
2. heavy-tailed log-fitness FGM with the same gene-specific errors.

The displayed data contain every REL607 gene with a finite effect and a
positive reported uncertainty after averaging the green and red libraries per
gene.  The canonical fit excludes the most deleterious 5% of displayed
effects; the heavy-tailed fit uses all displayed effects.  The heavy-tailed
model identifies the measured effect directly with the log-fitness effect.
Both likelihoods convolve the latent DFE separately with every gene's reported
Gaussian error.  Fits are conditional on s >= -0.5; more negative measurements
are excluded from both the fit and displayed histogram.

Canonical FGM
-------------
The Gaussian-mutation model remains moment constrained.  For every candidate
sigma, n and r are solved numerically so its exact absolute-fitness moments
match the de-noised sample moments.  Only sigma is optimized, with
2 <= n <= 500.

Heavy-tailed FGM
----------------

    delta = sigma Z / sqrt(2 G)
    Z ~ Normal_n(0, I), G ~ Gamma(mu, 1)

so |delta|^2 / sigma^2 ~ BetaPrime(n/2, mu).  The full unbinned
likelihood is optimized over n, r, sigma, and mu, using the stable coordinates
C = n sigma^2 and A = r sigma.  Mu is inferred rather than fixed.

For the heavy-tail likelihood, we report the highest converged interior local
maximum found by multistart optimization, then polish it without parameter
bounds.  This avoids selecting a point solely because it lies on an arbitrary
numerical ceiling.

Run from any directory:

    python code_tmp/poster_fig3.py --errors with

Outputs:

    figs_paper/poster_fig3_with_errors.pdf
    data/poster_fig3_fit.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import (
    least_squares,
    minimize,
    minimize_scalar,
)
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Paths and shared model/data helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cmn import cmn_fgm  # noqa: E402
from cmn.cmn_cauchy_fgm import (  # noqa: E402
    cauchy_fgm_dfe_pdf,
    cauchy_fgm_survival_many_eps,
)
from cmn.cmn_exper import limdi_gene_series  # noqa: E402

FIG_DIR = os.path.join(REPO_ROOT, "figs_paper")
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_PDF = os.path.join(FIG_DIR, "poster_fig3_with_errors.pdf")
OUT_JSON = os.path.join(DATA_DIR, "poster_fig3_fit.json")


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------

FIT_DX = 2.0e-4
CHECK_DX = 1.0e-4
FIT_ERROR_NODES = 28
FINE_ERROR_NODES = 56
CHECK_ERROR_NODES = 96
CI_ERROR_NODES = 20
CANONICAL_LOWER_TRIM = 0.05
CANONICAL_BOOTSTRAPS = 100
CANONICAL_BOOTSTRAP_SEED = 260729
GAUSSIAN_PAD_SD = 8.0
LOWER_FIT_CUT = -0.5

N_BOUNDS = (2.0, 500.0)
R_BOUNDS = (1.0e-5, 5.0)
CANONICAL_SIGMA_BOUNDS = (1.0e-3, 1.0)
C_BOUNDS = (1.0e-6, 0.2)
A_BOUNDS = (1.0e-5, 0.2)
MU_BOUNDS = (0.10, 5.0)

TAIL_BIN_WIDTH = 0.020
BULK_BIN_WIDTH = 0.0025
TAIL_XLIM = (-0.505, -0.065)
BULK_XLIM = (-0.065, 0.060)


# ---------------------------------------------------------------------------
# Poster styling
# ---------------------------------------------------------------------------

for font_path in (
    "/Library/Fonts/AGaramondPro-Regular.otf",
    "/Library/Fonts/AGaramondPro-Italic.otf",
    "/Library/Fonts/AGaramondPro-Bold.otf",
    "/Library/Fonts/AGaramondPro-BoldItalic.otf",
):
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)

mpl.rcParams.update({
    "font.family": "Adobe Garamond Pro",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Adobe Garamond Pro",
    "mathtext.it": "Adobe Garamond Pro:italic",
    "mathtext.bf": "Adobe Garamond Pro:bold",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATA_COLOR = "#666666"
DATA_EDGE = "#4A4A4A"
# Exact CMRmap colors used in poster_fig1.py:
# DFE_COLOR (red) for canonical FGM and EVO_COLOR (purple) for HT FGM.
CANONICAL_COLOR = "#FF4126"
HEAVY_COLOR = "#802F95"


# ---------------------------------------------------------------------------
# Common unbinned noisy likelihood
# ---------------------------------------------------------------------------

@dataclass
class ModelParameters:
    n: float
    r: float
    sigma: float
    loglik: float
    mu: float | None = None

    def serializable(self) -> dict[str, float]:
        output = {
            "n": float(self.n),
            "r": float(self.r),
            "sigma": float(self.sigma),
            "loglik": float(self.loglik),
        }
        if self.mu is not None:
            output["mu"] = float(self.mu)
            output["C_n_sigma2"] = float(self.n * self.sigma * self.sigma)
            output["A_r_sigma"] = float(self.r * self.sigma)
        return output


class GeneErrorLikelihood:
    """Grid likelihood with a separate reported Gaussian error per gene."""

    def __init__(
        self,
        effects,
        errors,
        dx=FIT_DX,
        number_error_nodes=FIT_ERROR_NODES,
        lower_cut=LOWER_FIT_CUT,
    ):
        self.effects = np.asarray(effects, dtype=float)
        self.errors = np.asarray(errors, dtype=float)
        self.dx = float(dx)
        self.number_error_nodes = int(number_error_nodes)
        self.lower_cut = None if lower_cut is None else float(lower_cut)
        if (
            self.effects.ndim != 1
            or self.effects.size == 0
            or self.effects.shape != self.errors.shape
            or np.any(~np.isfinite(self.effects))
            or np.any(~np.isfinite(self.errors))
            or np.any(self.errors <= 0.0)
            or self.dx <= 0.0
            or self.number_error_nodes < 2
            or (
                self.lower_cut is not None
                and np.any(self.effects < self.lower_cut)
            )
        ):
            raise ValueError(
                "Effects and errors must be aligned finite vectors with "
                "strictly positive errors."
            )

        self.log_error_nodes = np.linspace(
            float(np.log(self.errors.min())),
            float(np.log(self.errors.max())),
            self.number_error_nodes,
        )
        self.error_nodes = np.exp(self.log_error_nodes)
        position = np.interp(
            np.log(self.errors),
            self.log_error_nodes,
            np.arange(self.number_error_nodes, dtype=float),
        )
        self.error_lower = np.minimum(
            np.floor(position).astype(int),
            self.number_error_nodes - 2,
        )
        self.error_weight = position - self.error_lower

        pad = GAUSSIAN_PAD_SD * float(self.errors.max())
        lo = float(self.effects.min()) - pad
        hi = float(self.effects.max()) + pad
        number = int(np.ceil((hi - lo) / self.dx)) + 1
        self.x = lo + self.dx * np.arange(number)
        self.edges = np.r_[
            self.x - 0.5 * self.dx,
            self.x[-1] + 0.5 * self.dx,
        ]
        data_position = (self.effects - lo) / self.dx
        self.data_lower = np.minimum(
            np.floor(data_position).astype(int),
            number - 2,
        )
        self.data_weight = data_position - self.data_lower

        self.kernels = []
        for error in self.error_nodes:
            half = int(np.ceil(GAUSSIAN_PAD_SD * error / self.dx))
            offsets = self.dx * np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * np.square(offsets / error))
            self.kernels.append(kernel / kernel.sum())

        self.mixture_weights = np.zeros(self.number_error_nodes)
        np.add.at(
            self.mixture_weights,
            self.error_lower,
            1.0 - self.error_weight,
        )
        np.add.at(
            self.mixture_weights,
            self.error_lower + 1,
            self.error_weight,
        )
        self.mixture_weights /= self.effects.size

    def convolved_grids(self, latent_pdf):
        latent_pdf = np.asarray(latent_pdf, dtype=float)
        latent_pdf = np.where(np.isfinite(latent_pdf), latent_pdf, 0.0)
        latent_pdf = np.maximum(latent_pdf, 0.0)
        return np.asarray([
            np.maximum(fftconvolve(latent_pdf, kernel, mode="same"), 0.0)
            for kernel in self.kernels
        ])

    def loglik_from_grids(self, grids, survival_nodes):
        at_data = (
            grids[:, self.data_lower]
            * (1.0 - self.data_weight[None, :])
            + grids[:, self.data_lower + 1]
            * self.data_weight[None, :]
        )
        if np.any(~np.isfinite(at_data)):
            return -np.inf
        log_density_nodes = np.log(
            np.maximum(at_data, np.finfo(float).tiny)
        )
        gene_index = np.arange(self.effects.size)
        lower = self.error_lower
        weight = self.error_weight
        log_density = (
            (1.0 - weight) * log_density_nodes[lower, gene_index]
            + weight * log_density_nodes[lower + 1, gene_index]
        )
        if self.lower_cut is not None:
            if (
                np.any(~np.isfinite(survival_nodes))
                or np.any(survival_nodes <= 0.0)
            ):
                return -np.inf
            log_survival_nodes = np.log(survival_nodes)
            log_survival = (
                (1.0 - weight) * log_survival_nodes[lower]
                + weight * log_survival_nodes[lower + 1]
            )
            log_density = log_density - log_survival
        return float(np.sum(log_density))

    def conditional_grids(self, grids, survival_nodes):
        if self.lower_cut is None:
            return grids
        return grids / survival_nodes[:, None]


class CanonicalGeneErrorLikelihood(GeneErrorLikelihood):
    """Canonical absolute-fitness DFE with gene-specific errors."""

    def grids(self, n, r, sigma):
        latent = cmn_fgm.fgm_fitness_bin_probs(
            self.edges,
            n=float(n),
            sigma=float(sigma),
            r=float(r),
        ) / self.dx
        return self.convolved_grids(latent)

    def survival_nodes(self, n, r, sigma):
        if self.lower_cut is None:
            return np.ones(self.number_error_nodes)
        return cmn_fgm.fgm_fitness_survival_many_eps(
            self.lower_cut,
            n=float(n),
            sigma=float(sigma),
            r=float(r),
            eps=self.error_nodes,
        )

    def canonical_loglik(self, n, r, sigma):
        return self.loglik_from_grids(
            self.grids(n=n, r=r, sigma=sigma),
            self.survival_nodes(n=n, r=r, sigma=sigma),
        )

    def canonical_predictive_pdf(self, n, r, sigma):
        return self.mixture_weights @ self.conditional_grids(
            self.grids(n=n, r=r, sigma=sigma),
            self.survival_nodes(n=n, r=r, sigma=sigma),
        )


class HeavyLogGeneErrorLikelihood(GeneErrorLikelihood):
    """Heavy-tailed log-fitness DFE with gene-specific errors."""

    def grids(self, n, r, sigma, mu):
        latent = cauchy_fgm_dfe_pdf(
            self.x,
            n=float(n),
            sigma=float(sigma),
            r=float(r),
            mu=float(mu),
        )
        return self.convolved_grids(latent)

    def survival_nodes(self, n, r, sigma, mu):
        if self.lower_cut is None:
            return np.ones(self.number_error_nodes)
        return cauchy_fgm_survival_many_eps(
            self.lower_cut,
            n=float(n),
            sigma=float(sigma),
            r=float(r),
            eps=self.error_nodes,
            mu=float(mu),
        )

    def heavy_loglik(self, n, r, sigma, mu):
        return self.loglik_from_grids(
            self.grids(n=n, r=r, sigma=sigma, mu=mu),
            self.survival_nodes(n=n, r=r, sigma=sigma, mu=mu),
        )

    def heavy_predictive_pdf(self, n, r, sigma, mu):
        return self.mixture_weights @ self.conditional_grids(
            self.grids(n=n, r=r, sigma=sigma, mu=mu),
            self.survival_nodes(n=n, r=r, sigma=sigma, mu=mu),
        )


# ---------------------------------------------------------------------------
# Canonical moment-constrained MLE
# ---------------------------------------------------------------------------

def solve_moment_parameter_branches(
    sigma,
    observed_mean,
    true_variance,
    starts=(),
):
    """Solve the two exact raw-fitness moment constraints for n and r."""
    abs_mean = abs(float(observed_mean))
    approximate_n = np.clip(
        2.0 * abs_mean / (sigma * sigma),
        *N_BOUNDS,
    )
    approximate_s0 = 0.5 * (
        true_variance / (sigma * sigma) - abs_mean
    )
    approximate_r = np.clip(
        np.sqrt(max(2.0 * approximate_s0, R_BOUNDS[0] ** 2)),
        *R_BOUNDS,
    )
    candidates = [
        *starts,
        (approximate_n, approximate_r),
        (10.0, 0.25),
        (100.0, 0.25),
        (3.0, 1.0),
        (30.0, 1.5),
    ]
    lower = np.log([N_BOUNDS[0], R_BOUNDS[0]])
    upper = np.log([N_BOUNDS[1], R_BOUNDS[1]])

    def residual(log_parameters):
        n, r = np.exp(log_parameters)
        model_mean, model_variance = cmn_fgm.fgm_fitness_moments(
            n=n,
            sigma=sigma,
            r=r,
        )
        return np.array([
            (model_mean - observed_mean) / abs_mean,
            (model_variance - true_variance) / true_variance,
        ])

    results = []
    for start in candidates:
        start = np.clip(np.asarray(start, dtype=float),
                        np.exp(lower), np.exp(upper))
        result = least_squares(
            residual,
            np.log(start),
            bounds=(lower, upper),
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
            max_nfev=500,
        )
        results.append(result)
    branches = []
    for result in results:
        maximum_relative_residual = float(np.max(np.abs(result.fun)))
        if maximum_relative_residual > 1.0e-5:
            continue
        n, r = np.exp(result.x)
        if any(
            abs(np.log(n / old_n)) < 1.0e-5
            and abs(np.log(r / old_r)) < 1.0e-5
            for old_n, old_r, _ in branches
        ):
            continue
        branches.append(
            (float(n), float(r), maximum_relative_residual)
        )
    return branches


def fit_canonical(effects, errors, likelihood, scan_points=350):
    observed_mean = float(np.mean(effects))
    abs_mean = abs(observed_mean)
    true_variance = max(
        float(np.var(effects)) - float(np.mean(np.square(errors))),
        np.finfo(float).eps,
    )
    if abs_mean <= 0.0:
        raise ValueError("Moment-constrained FGM requires a nonzero mean.")

    sigma_lo, sigma_hi = CANONICAL_SIGMA_BOUNDS
    solution_cache = {}

    def constrained_parameters(sigma):
        key = float(sigma)
        if key in solution_cache:
            return solution_cache[key]
        starts = []
        valid = [
            (other_sigma, values)
            for other_sigma, values in solution_cache.items()
            if np.isfinite(values[0]) and np.isfinite(values[1])
        ]
        if valid:
            _, nearest = min(
                valid,
                key=lambda item: abs(np.log(item[0] / sigma)),
            )
            starts.append(nearest[:2])
        branches = solve_moment_parameter_branches(
            sigma,
            observed_mean,
            true_variance,
            starts=starts,
        )
        scored_branches = []
        for n, r, residual in branches:
            loglik = likelihood.canonical_loglik(
                n=n,
                r=r,
                sigma=sigma,
            )
            if np.isfinite(loglik):
                scored_branches.append((loglik, n, r, residual))
        if scored_branches:
            loglik, n, r, residual = max(
                scored_branches,
                key=lambda values: values[0],
            )
            solution_cache[key] = (n, r, residual, loglik)
        else:
            solution_cache[key] = (np.nan, np.nan, np.inf, -np.inf)
        return solution_cache[key]

    def objective(log_sigma):
        sigma = float(np.exp(log_sigma))
        n, r, _, value = constrained_parameters(sigma)
        if (
            not np.isfinite(r)
            or not np.isfinite(n)
            or n < N_BOUNDS[0]
            or n > N_BOUNDS[1]
        ):
            return 1.0e100
        return -value if np.isfinite(value) else 1.0e100

    # A dense deterministic scan makes boundary optima and secondary peaks clear.
    log_grid = np.linspace(
        np.log(sigma_lo),
        np.log(sigma_hi),
        int(scan_points),
    )
    objectives = np.asarray([objective(value) for value in log_grid])
    best_index = int(np.argmin(objectives))
    left = log_grid[max(0, best_index - 2)]
    right = log_grid[min(log_grid.size - 1, best_index + 2)]
    if left == right:
        left, right = log_grid[0], log_grid[-1]
    polished = minimize_scalar(
        objective,
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-11, "maxiter": 500},
    )
    candidates = [
        (float(objectives[best_index]), float(log_grid[best_index])),
        (float(polished.fun), float(polished.x)),
    ]
    best_value, best_log_sigma = min(candidates, key=lambda item: item[0])
    sigma = float(np.exp(best_log_sigma))
    n, r, moment_residual, _ = constrained_parameters(sigma)
    if not np.isfinite(n) or not np.isfinite(r):
        raise RuntimeError("Canonical optimum has no valid exact-moment solution.")
    fit = ModelParameters(
        n=float(n),
        r=float(r),
        sigma=sigma,
        loglik=float(-best_value),
    )
    diagnostics = {
        "sample_mean": observed_mean,
        "sample_variance": float(np.var(effects)),
        "mean_measurement_error_variance": float(
            np.mean(np.square(errors))
        ),
        "deconvolved_variance": true_variance,
        "sigma_bounds": [float(sigma_lo), float(sigma_hi)],
        "n_bounds": list(N_BOUNDS),
        "r_bounds": list(R_BOUNDS),
        "exact_fitness_moment_constraint": True,
        "maximum_relative_moment_residual": moment_residual,
        "at_n_lower_bound": bool(n <= N_BOUNDS[0] * 1.001),
        "scan": {
            "sigma": np.exp(log_grid).tolist(),
            "loglik": (-objectives).tolist(),
        },
    }
    return fit, diagnostics


# ---------------------------------------------------------------------------
# Heavy-tailed free-mu MLE
# ---------------------------------------------------------------------------

def unpack_heavy(theta):
    n, C, A, mu = np.exp(np.asarray(theta, dtype=float))
    sigma = np.sqrt(C / n)
    r = A / sigma
    return float(n), float(r), float(sigma), float(mu), float(C), float(A)


def heavy_bounds():
    return [
        (np.log(N_BOUNDS[0]), np.log(N_BOUNDS[1])),
        (np.log(C_BOUNDS[0]), np.log(C_BOUNDS[1])),
        (np.log(A_BOUNDS[0]), np.log(A_BOUNDS[1])),
        (np.log(MU_BOUNDS[0]), np.log(MU_BOUNDS[1])),
    ]


def fit_heavy_interior(likelihood, canonical_fit):
    """Find the highest finite interior local MLE by multistart optimization.

    This is used for the log-fitness heavy-tail likelihood, whose global
    high-n ridge otherwise selects the arbitrary numerical n ceiling.  Bounds
    are used only to discover basins.  Each interior candidate is then
    polished in unconstrained log coordinates, and the best finite stationary
    candidate is retained.
    """
    bounds = heavy_bounds()
    lower = np.asarray(bounds)[:, 0]
    upper = np.asarray(bounds)[:, 1]

    def objective(theta):
        n, r, sigma, mu, _, _ = unpack_heavy(theta)
        if r < R_BOUNDS[0] or r > R_BOUNDS[1]:
            return 1.0e100
        value = likelihood.heavy_loglik(
            n=n,
            r=r,
            sigma=sigma,
            mu=mu,
        )
        return -value if np.isfinite(value) else 1.0e100

    starts = [
        np.log([
            canonical_fit.n,
            canonical_fit.n * canonical_fit.sigma**2,
            max(canonical_fit.r * canonical_fit.sigma, A_BOUNDS[0]),
            0.25,
        ]),
        np.log([4.0, 1.0e-4, 4.0e-3, 0.13]),
        np.log([4.0, 1.0e-4, 4.0e-3, 0.25]),
        np.log([6.0, 1.2e-4, 2.0e-3, 0.13]),
        np.log([7.0, 1.2e-4, 2.0e-3, 0.21]),
        np.log([10.0, 2.5e-4, 4.0e-3, 0.20]),
        np.log([12.0, 2.0e-4, 2.5e-3, 0.35]),
        np.log([30.0, 2.0e-4, 2.0e-3, 0.20]),
    ]

    bounded_results = []
    for start in starts:
        bounded_results.append(minimize(
            objective,
            np.clip(start, lower, upper),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1.0e-12,
                "gtol": 1.0e-6,
                "maxiter": 1500,
                "maxls": 40,
            },
        ))

    def is_interior(result):
        theta = np.asarray(result.x, dtype=float)
        n, r, _, _, _, _ = unpack_heavy(theta)
        log_margin = 2.0e-3
        return bool(
            np.isfinite(result.fun)
            and np.all(theta > lower + log_margin)
            and np.all(theta < upper - log_margin)
            and n > N_BOUNDS[0] * 1.002
            and n < N_BOUNDS[1] / 1.002
            and r > R_BOUNDS[0] * 1.002
            and r < R_BOUNDS[1] / 1.002
        )

    interior_bounded = [
        result for result in bounded_results if is_interior(result)
    ]
    if not interior_bounded:
        raise RuntimeError(
            "No finite interior heavy-tail likelihood maximum was found."
        )

    # Polish without bounds so the reported point is not defined by them.
    unbounded_results = []
    for result in interior_bounded:
        polished = minimize(
            objective,
            result.x,
            method="BFGS",
            options={
                "gtol": 2.0e-6,
                "maxiter": 1500,
            },
        )
        if is_interior(polished):
            unbounded_results.append(polished)
    eligible = unbounded_results or interior_bounded
    best = min(eligible, key=lambda result: result.fun)
    n, r, sigma, mu, C, A = unpack_heavy(best.x)
    fit = ModelParameters(
        n=n,
        r=r,
        sigma=sigma,
        mu=mu,
        loglik=float(-best.fun),
    )
    diagnostics = {
        "success": bool(best.success),
        "message": str(best.message),
        "selection_rule": (
            "highest converged interior local maximum from multistart; "
            "final polish has no parameter bounds"
        ),
        "number_local_starts": len(starts),
        "number_interior_bounded_candidates": len(interior_bounded),
        "number_interior_unbounded_candidates": len(unbounded_results),
        "bounds_hit": {
            "n": False,
            "C": False,
            "A": False,
            "mu": False,
        },
        "C_n_sigma2": C,
        "A_r_sigma": A,
        "_theta": np.asarray(best.x, dtype=float),
        "_objective": objective,
    }
    return fit, diagnostics


def polish_heavy(likelihood, initial_fit):
    """Polish an interior heavy-tail fit without parameter bounds."""
    initial_theta = np.log([
        initial_fit.n,
        initial_fit.n * initial_fit.sigma**2,
        initial_fit.r * initial_fit.sigma,
        initial_fit.mu,
    ])

    def objective(theta):
        n, r, sigma, mu, _, _ = unpack_heavy(theta)
        if r < R_BOUNDS[0] or r > R_BOUNDS[1]:
            return 1.0e100
        value = likelihood.heavy_loglik(
            n=n,
            r=r,
            sigma=sigma,
            mu=mu,
        )
        return -value if np.isfinite(value) else 1.0e100

    result = minimize(
        objective,
        initial_theta,
        method="BFGS",
        options={
            "gtol": 2.0e-6,
            "maxiter": 1500,
        },
    )
    n, r, sigma, mu, C, A = unpack_heavy(result.x)
    fit = ModelParameters(
        n=n,
        r=r,
        sigma=sigma,
        mu=mu,
        loglik=float(-result.fun),
    )
    raw_bounds = (N_BOUNDS, C_BOUNDS, A_BOUNDS, MU_BOUNDS)
    bounds_hit = {
        key: bool(
            value <= lower * 1.002 or value >= upper / 1.002
        )
        for key, value, (lower, upper) in zip(
            ("n", "C", "A", "mu"),
            (n, C, A, mu),
            raw_bounds,
        )
    }
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "bounds_hit": bounds_hit,
        "final_polish_used_bounds": False,
        "initial_parameters": initial_fit.serializable(),
        "_theta": np.asarray(result.x, dtype=float),
        "_objective": objective,
    }
    return fit, diagnostics


def finite_difference_hessian(objective, theta, step=2.0e-3):
    """Observed Hessian of an objective in unconstrained log coordinates."""
    theta = np.asarray(theta, dtype=float)
    dimension = theta.size
    hessian = np.empty((dimension, dimension), dtype=float)
    center = float(objective(theta))
    for i in range(dimension):
        ei = np.zeros(dimension)
        ei[i] = step
        hessian[i, i] = (
            objective(theta + ei)
            - 2.0 * center
            + objective(theta - ei)
        ) / step**2
        for j in range(i):
            ej = np.zeros(dimension)
            ej[j] = step
            value = (
                objective(theta + ei + ej)
                - objective(theta + ei - ej)
                - objective(theta - ei + ej)
                + objective(theta - ei - ej)
            ) / (4.0 * step**2)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def heavy_wald_intervals(heavy_fit, heavy_diagnostics):
    """Log-scale Wald intervals transformed to n, r, sigma, and mu."""
    theta = heavy_diagnostics["_theta"]
    objective = heavy_diagnostics["_objective"]
    # The exact endpoint is represented by finite bins, which can make an
    # excessively small finite-difference stencil follow grid-scale roughness.
    # Use the smallest stencil whose observed Hessian is positive definite.
    hessian = None
    eigenvalues = None
    selected_step = None
    attempted_steps = (2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2)
    for step in attempted_steps:
        candidate = finite_difference_hessian(
            objective,
            theta,
            step=step,
        )
        candidate_eigenvalues = np.linalg.eigvalsh(candidate)
        if np.all(candidate_eigenvalues > 0.0):
            hessian = candidate
            eigenvalues = candidate_eigenvalues
            selected_step = step
            break
    if hessian is None:
        raise RuntimeError(
            "Heavy-tail observed-information Hessian is not positive definite "
            f"for finite-difference steps {attempted_steps}."
        )
    covariance_theta = np.linalg.inv(hessian)

    # theta = log(n, C=n*sigma^2, A=r*sigma, mu).
    # Rows map theta to log(n, r, sigma, mu).
    transform = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, 1.0, 0.0],
        [-0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    covariance_log_parameters = (
        transform @ covariance_theta @ transform.T
    )
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance_log_parameters), 0.0)
    )
    estimates = {
        "n": heavy_fit.n,
        "r": heavy_fit.r,
        "sigma": heavy_fit.sigma,
        "mu": heavy_fit.mu,
    }
    intervals = {}
    for index, (name, estimate) in enumerate(estimates.items()):
        half_width = 1.959964 * standard_errors[index]
        intervals[name] = [
            float(np.exp(np.log(estimate) - half_width)),
            float(np.exp(np.log(estimate) + half_width)),
        ]
    return intervals, {
        "method": "observed_information_log_scale_wald",
        "hessian_eigenvalues": eigenvalues.tolist(),
        "valid_positive_definite": True,
        "finite_difference_step": selected_step,
        "attempted_steps": list(attempted_steps),
    }


def trim_canonical_tail(effects, errors):
    """Remove only the most deleterious canonical-fit fraction."""
    cutoff = float(np.quantile(effects, CANONICAL_LOWER_TRIM))
    keep = effects >= cutoff
    return effects[keep], errors[keep]


def canonical_bootstrap_intervals(effects, errors):
    """Gene bootstrap CIs for the moment-constrained canonical estimator."""
    rng = np.random.default_rng(CANONICAL_BOOTSTRAP_SEED)
    estimates = {"n": [], "r": [], "sigma": []}
    failures = 0
    for _ in range(CANONICAL_BOOTSTRAPS):
        index = rng.integers(0, effects.size, effects.size)
        sampled_effects, sampled_errors = trim_canonical_tail(
            effects[index],
            errors[index],
        )
        try:
            likelihood = CanonicalGeneErrorLikelihood(
                sampled_effects,
                sampled_errors,
                dx=3.0e-4,
                number_error_nodes=CI_ERROR_NODES,
                lower_cut=LOWER_FIT_CUT,
            )
            fit, _ = fit_canonical(
                sampled_effects,
                sampled_errors,
                likelihood,
                scan_points=45,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
            continue
        estimates["n"].append(fit.n)
        estimates["r"].append(fit.r)
        estimates["sigma"].append(fit.sigma)

    intervals = {
        name: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for name, values in estimates.items()
        if values
    }
    return intervals, {
        "method": "gene_bootstrap_percentile",
        "replicates_requested": CANONICAL_BOOTSTRAPS,
        "replicates_retained": int(
            CANONICAL_BOOTSTRAPS - failures
        ),
        "failures": int(failures),
        "seed": CANONICAL_BOOTSTRAP_SEED,
        "error_nodes": CI_ERROR_NODES,
    }


# ---------------------------------------------------------------------------
# Diagnostics and plotting
# ---------------------------------------------------------------------------

def grid_check(
    canonical_effects,
    canonical_errors,
    heavy_effects,
    heavy_errors,
    canonical_fit,
    heavy_error_fit,
):
    fine_canonical = CanonicalGeneErrorLikelihood(
        canonical_effects,
        canonical_errors,
        dx=CHECK_DX,
        number_error_nodes=CHECK_ERROR_NODES,
    )
    fine_heavy_error = HeavyLogGeneErrorLikelihood(
        heavy_effects,
        heavy_errors,
        dx=CHECK_DX,
        number_error_nodes=CHECK_ERROR_NODES,
    )
    canonical_ll = fine_canonical.canonical_loglik(
        canonical_fit.n,
        canonical_fit.r,
        canonical_fit.sigma,
    )
    heavy_error_ll = fine_heavy_error.heavy_loglik(
        heavy_error_fit.n,
        heavy_error_fit.r,
        heavy_error_fit.sigma,
        heavy_error_fit.mu,
    )
    return {
        "dx": CHECK_DX,
        "canonical_loglik": float(canonical_ll),
        "canonical_delta": float(
            canonical_ll - canonical_fit.loglik
        ),
        "heavy_with_gene_errors_loglik": float(heavy_error_ll),
        "heavy_with_gene_errors_delta": float(
            heavy_error_ll - heavy_error_fit.loglik
        ),
    }


def histogram(effects, bin_width):
    left = bin_width * np.floor(float(np.min(effects)) / bin_width)
    right = bin_width * np.ceil(float(np.max(effects)) / bin_width)
    edges = np.arange(left, right + 1.0001 * bin_width, bin_width)
    counts, edges = np.histogram(effects, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (effects.size * bin_width)
    error = np.sqrt(counts) / (effects.size * bin_width)
    return centers, density, error, counts


def plot_figure(
    effects,
    canonical_likelihood,
    heavy_error_likelihood,
    canonical_fit,
    heavy_error_fit,
    canonical_intervals,
    heavy_error_intervals,
    path,
    heavy_parameter_text=None,
):
    canonical_curve = canonical_likelihood.canonical_predictive_pdf(
        canonical_fit.n,
        canonical_fit.r,
        canonical_fit.sigma,
    )
    heavy_error_curve = heavy_error_likelihood.heavy_predictive_pdf(
        heavy_error_fit.n,
        heavy_error_fit.r,
        heavy_error_fit.sigma,
        heavy_error_fit.mu,
    )
    tail_histogram = histogram(effects, TAIL_BIN_WIDTH)
    bulk_histogram = histogram(effects, BULK_BIN_WIDTH)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.5),
        gridspec_kw={"wspace": 0.22},
    )

    def draw_common(ax, empirical_histogram, xlim):
        if xlim == TAIL_XLIM:
            ax.set_yscale("log")
        centers, density, density_error, counts = empirical_histogram
        nonzero = (
            (counts > 0)
            & (centers >= xlim[0])
            & (centers <= xlim[1])
        )
        canonical_visible = (
            (canonical_likelihood.x >= xlim[0])
            & (canonical_likelihood.x <= xlim[1])
        )
        heavy_visible = (
            (heavy_error_likelihood.x >= xlim[0])
            & (heavy_error_likelihood.x <= xlim[1])
        )
        ax.errorbar(
            centers[nonzero],
            density[nonzero],
            yerr=density_error[nonzero],
            fmt="o",
            ms=4.8,
            mfc=DATA_COLOR,
            mec=DATA_EDGE,
            mew=0.45,
            ecolor=DATA_COLOR,
            elinewidth=0.75,
            capsize=0,
            alpha=0.88,
            label="REL607 data",
            zorder=4,
        )
        ax.plot(
            heavy_error_likelihood.x[heavy_visible],
            heavy_error_curve[heavy_visible],
            color=HEAVY_COLOR,
            lw=3.0,
            ls="-",
            label="Heavy-tailed",
            zorder=3,
        )
        ax.autoscale_view(scalex=False, scaley=True)
        ax.set_autoscaley_on(False)
        # Let the observed data and the full-data HT fit determine the automatic
        # y-scale.  The canonical tail is an extrapolation beyond its trimmed
        # fit range and can otherwise expand the log axis by many empty decades.
        ax.plot(
            canonical_likelihood.x[canonical_visible],
            canonical_curve[canonical_visible],
            color=CANONICAL_COLOR,
            lw=2.8,
            ls="-",
            label="Canonical",
            zorder=2,
            scaley=False,
        )
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Absolute fitness effect $(s)$")
        ax.set_ylabel("Probability density")
        ax.tick_params(direction="out", length=4.5, width=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    draw_common(axes[0], tail_histogram, TAIL_XLIM)
    draw_common(axes[1], bulk_histogram, BULK_XLIM)

    axes[0].set_ylim(bottom=1.0e-2)

    axes[1].set_ylabel("")

    def interval_line(symbol, estimate, interval, decimals):
        return (
            rf"${symbol}={estimate:.{decimals}f}$"
            "\n"
            rf"$[{interval[0]:.{decimals}f},"
            rf"{interval[1]:.{decimals}f}]$"
        )

    canonical_lines = "\n".join([
        interval_line(
            "n",
            canonical_fit.n,
            canonical_intervals["n"],
            2,
        ),
        interval_line(
            "r",
            canonical_fit.r,
            canonical_intervals["r"],
            3,
        ),
        interval_line(
            r"\sigma",
            canonical_fit.sigma,
            canonical_intervals["sigma"],
            3,
        ),
    ])
    heavy_error_lines = heavy_parameter_text
    if heavy_error_lines is None:
        heavy_error_lines = "\n".join([
            interval_line(
                "n",
                heavy_error_fit.n,
                heavy_error_intervals["n"],
                2,
            ),
            interval_line(
                "r",
                heavy_error_fit.r,
                heavy_error_intervals["r"],
                3,
            ),
            interval_line(
                r"\sigma",
                heavy_error_fit.sigma,
                heavy_error_intervals["sigma"],
                4,
            ),
            interval_line(
                r"\mu",
                heavy_error_fit.mu,
                heavy_error_intervals["mu"],
                2,
            ),
        ])
    # Color-matched parameter values; keep them behind every plotted curve and
    # data point in panel B so they do not obscure the fit comparison.
    axes[1].text(
        0.025,
        0.965,
        canonical_lines,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=17.3,
        linespacing=1.00,
        color=CANONICAL_COLOR,
        zorder=1,
    )

    axes[1].text(
        0.975,
        0.965,
        heavy_error_lines,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=17.3,
        linespacing=1.00,
        color=HEAVY_COLOR,
        zorder=1,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    semantic_handles = [
        Line2D([], [], color=CANONICAL_COLOR, lw=2.8, ls="-"),
        Line2D([], [], color=HEAVY_COLOR, lw=3.0, ls="-"),
        handles_by_label["REL607 data"],
    ]
    semantic_labels = [
        "Gaussian (canonical)",
        "Heavy-tailed",
        "REL607 data",
    ]
    axes[0].legend(
        semantic_handles,
        semantic_labels,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        handlelength=2.5,
        borderaxespad=0.3,
        fontsize=17.5,
    )

    for label, ax in zip(("A", "B"), axes):
        ax.text(
            -0.14,
            1.08,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=25,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.89)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def clean_heavy_diagnostics(diagnostics):
    return {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("_")
    }


def rel607_effects_and_errors():
    """Aligned REL607 mean effects and reported per-gene 1-sigma errors."""
    effect_series, error_series = limdi_gene_series("REL607", errors=True)
    errors = error_series.reindex(effect_series.index).to_numpy(float)
    effects = effect_series.to_numpy(float)
    keep = (
        np.isfinite(effects)
        & np.isfinite(errors)
        & (errors > 0.0)
    )
    return effects[keep], errors[keep]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    all_effects, all_errors = rel607_effects_and_errors()
    above_cut = all_effects >= LOWER_FIT_CUT
    effects = all_effects[above_cut]
    errors = all_errors[above_cut]
    excluded_below_cut = int(np.sum(~above_cut))
    canonical_effects, canonical_errors = trim_canonical_tail(
        effects,
        errors,
    )

    print(
        f"REL607: N={effects.size}, range=({effects.min():.6f}, "
        f"{effects.max():.6f}), mean={effects.mean():.6f}, "
        f"sd={effects.std():.6f}, error range=({errors.min():.6f}, "
        f"{errors.max():.6f})"
    )
    coarse_canonical_likelihood = CanonicalGeneErrorLikelihood(
        canonical_effects,
        canonical_errors,
        dx=FIT_DX,
        number_error_nodes=FIT_ERROR_NODES,
        lower_cut=LOWER_FIT_CUT,
    )
    coarse_heavy_error_likelihood = HeavyLogGeneErrorLikelihood(
        effects,
        errors,
        dx=FIT_DX,
        number_error_nodes=FIT_ERROR_NODES,
        lower_cut=LOWER_FIT_CUT,
    )
    started = time.perf_counter()
    coarse_canonical_fit, _ = fit_canonical(
        canonical_effects,
        canonical_errors,
        coarse_canonical_likelihood,
    )
    coarse_heavy_error_fit, coarse_heavy_error_diagnostics = (
        fit_heavy_interior(
            coarse_heavy_error_likelihood,
            coarse_canonical_fit,
        )
    )
    # Final error-aware fits and displayed curves use twice as many nodes.
    canonical_likelihood = CanonicalGeneErrorLikelihood(
        canonical_effects,
        canonical_errors,
        dx=FIT_DX,
        number_error_nodes=FINE_ERROR_NODES,
        lower_cut=LOWER_FIT_CUT,
    )
    heavy_error_likelihood = HeavyLogGeneErrorLikelihood(
        effects,
        errors,
        dx=FIT_DX,
        number_error_nodes=FINE_ERROR_NODES,
        lower_cut=LOWER_FIT_CUT,
    )
    canonical_fit, canonical_diagnostics = fit_canonical(
        canonical_effects,
        canonical_errors,
        canonical_likelihood,
    )
    heavy_error_fit, heavy_error_diagnostics = polish_heavy(
        heavy_error_likelihood,
        coarse_heavy_error_fit,
    )
    canonical_intervals, canonical_ci_diagnostics = (
        canonical_bootstrap_intervals(effects, errors)
    )
    heavy_error_intervals, heavy_error_ci_diagnostics = heavy_wald_intervals(
        heavy_error_fit,
        heavy_error_diagnostics,
    )

    print(
        "Canonical fit with gene errors: "
        f"n={canonical_fit.n:.6g}, r={canonical_fit.r:.6g}, "
        f"sigma={canonical_fit.sigma:.6g}, "
        f"logL={canonical_fit.loglik:.6f}"
    )
    print(
        "Heavy-tailed log-fitness fit with gene errors: "
        f"n={heavy_error_fit.n:.6g}, r={heavy_error_fit.r:.6g}, "
        f"sigma={heavy_error_fit.sigma:.6g}, "
        f"mu={heavy_error_fit.mu:.6g}, "
        f"logL={heavy_error_fit.loglik:.6f}"
    )
    check = grid_check(
        canonical_effects,
        canonical_errors,
        effects,
        errors,
        canonical_fit,
        heavy_error_fit,
    )
    elapsed = time.perf_counter() - started

    output = {
        "dataset": {
            "name": "REL607",
            "source": "Limdi",
            "replicate_pooling": "mean_per_gene",
            "displayed_tail_trim": [0.0, 0.0],
            "observed_lower_cut": LOWER_FIT_CUT,
            "excluded_below_lower_cut": excluded_below_cut,
            "N": int(effects.size),
            "minimum": float(effects.min()),
            "maximum": float(effects.max()),
            "mean": float(effects.mean()),
            "sd": float(effects.std()),
            "error_minimum": float(errors.min()),
            "error_median": float(np.median(errors)),
            "error_maximum": float(errors.max()),
        },
        "likelihood": {
            "canonical": (
                "moment-constrained absolute-fitness DFE with "
                "gene-specific Gaussian errors"
            ),
            "heavy_tail_effect_convention": (
                "measured s is identified directly with log fitness"
            ),
            "heavy_tail_error_treatment": (
                "gene-specific Gaussian error convolution"
            ),
            "heavy_tail_support": "(-infinity, r^2/2]",
            "error_source": "errors_genes_inv.npy via limdi_gene_series",
            "fit_dx": FIT_DX,
            "initial_fit_error_nodes": FIT_ERROR_NODES,
            "final_fit_error_nodes": FINE_ERROR_NODES,
            "check_error_nodes": CHECK_ERROR_NODES,
            "conditional_on_observed_effect_above_cut": True,
            "observed_lower_cut": LOWER_FIT_CUT,
            "canonical_lower_tail_trim": CANONICAL_LOWER_TRIM,
        },
        "plot": {
            "tail_bin_width": TAIL_BIN_WIDTH,
            "bulk_bin_width": BULK_BIN_WIDTH,
        },
        "canonical_moment_constrained_mle": {
            "shared_fit_data": {
                "tail_trim": [CANONICAL_LOWER_TRIM, 0.0],
                "N": int(canonical_effects.size),
                "minimum": float(canonical_effects.min()),
                "maximum": float(canonical_effects.max()),
                "error_minimum": float(canonical_errors.min()),
                "error_median": float(np.median(canonical_errors)),
                "error_maximum": float(canonical_errors.max()),
                "tail_curve_is_extrapolation": True,
            },
            "fit": canonical_fit.serializable(),
            "confidence_intervals_95": canonical_intervals,
            "confidence_interval_diagnostics": (
                canonical_ci_diagnostics
            ),
            "diagnostics": canonical_diagnostics,
            "coarse_fit": coarse_canonical_fit.serializable(),
        },
        "heavy_tailed_log_fitness_free_mu_mle": {
            "shared_fit_data": {
                "tail_trim": [0.0, 0.0],
                "N": int(effects.size),
                "minimum": float(effects.min()),
                "maximum": float(effects.max()),
                "error_minimum": float(errors.min()),
                "error_median": float(np.median(errors)),
                "error_maximum": float(errors.max()),
            },
            "with_gene_specific_errors": {
                "fit": heavy_error_fit.serializable(),
                "confidence_intervals_95": heavy_error_intervals,
                "confidence_interval_diagnostics": (
                    heavy_error_ci_diagnostics
                ),
                "diagnostics": clean_heavy_diagnostics(
                    heavy_error_diagnostics
                ),
                "coarse_interior_fit": (
                    coarse_heavy_error_fit.serializable()
                ),
                "coarse_interior_diagnostics": clean_heavy_diagnostics(
                    coarse_heavy_error_diagnostics
                ),
                "selection_note": (
                    "highest finite interior local MLE; final polish "
                    "performed without parameter bounds"
                ),
            },
        },
        "grid_check": check,
        "elapsed_seconds": float(elapsed),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    plot_figure(
        effects,
        canonical_likelihood,
        heavy_error_likelihood,
        canonical_fit,
        heavy_error_fit,
        canonical_intervals,
        heavy_error_intervals,
        OUT_PDF,
    )

    print(f"Grid check: {check}")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
