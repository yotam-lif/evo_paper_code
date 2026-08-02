r"""Heteroskedastic Cauchy-FGM MLEs for ten evolved Limdi backgrounds.

Each gene i has its own reported selection-coefficient uncertainty eps_i:

    y_i | s_i ~ Normal(s_i, eps_i^2)
    s_i ~ Cauchy-FGM(n, r, sigma).

The likelihood is conditional on each observed y_i >= -0.4, so both the
convolved density and the truncation normalization depend on eps_i.  During
optimization, log density and log survival are interpolated over a logarithmic
grid of error standard deviations.  A finer grid and half-sized convolution
spacing are used for final local polishing.

The effective dimension is constrained to n >= 2, as requested.  Ara-2 and
Ara+4 are excluded following the published Limdi analysis.

Outputs
-------
    data/cauchy_fgm_limdi_evolved_heteroskedastic_mle.json
    data/cauchy_fgm_limdi_evolved_heteroskedastic_mle_params.csv
    data/cauchy_fgm_limdi_evolved_heteroskedastic_mle_plot.json
    figs_paper/figS6_cauchy_fgm_limdi_evolved_heteroskedastic_mle_linear.png
    figs_paper/figS6_cauchy_fgm_limdi_evolved_heteroskedastic_mle_log.png
"""

import csv
import json
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
from scipy.special import betaln, roots_jacobi
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cmn.cmn_cauchy_fgm import (  # noqa: E402
    cauchy_fgm_dfe_pdf,
    cauchy_fgm_survival,
    cauchy_fgm_survival_many_eps,
)
from cmn.cmn_exper import limdi_gene_series  # noqa: E402
from code_tmp.figS6_cauchy_fgm_full_mle import (  # noqa: E402
    LOWER_CUT,
    OPT_BOUNDS,
    UnbinnedNoisyLikelihood,
    observed_hessian,
)
from code_tmp.figS6_cauchy_fgm_limdi_evolved_mle import (  # noqa: E402
    BIN_WIDTH,
    EVOLVED_ORDER,
    LOG_Y_LIMITS,
    UPPER_PLOT_LIMIT,
)

DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figs_paper")
FLAT_MLE_PATH = os.path.join(
    DATA_DIR, "cauchy_fgm_limdi_evolved_mle.json"
)
JSON_PATH = os.path.join(
    DATA_DIR, "cauchy_fgm_limdi_evolved_heteroskedastic_mle.json"
)
CSV_PATH = os.path.join(
    DATA_DIR, "cauchy_fgm_limdi_evolved_heteroskedastic_mle_params.csv"
)
PLOT_DATA_PATH = os.path.join(
    DATA_DIR, "cauchy_fgm_limdi_evolved_heteroskedastic_mle_plot.json"
)
LINEAR_FIG_PATH = os.path.join(
    FIG_DIR,
    "figS6_cauchy_fgm_limdi_evolved_heteroskedastic_mle_linear.png",
)
LOG_FIG_PATH = os.path.join(
    FIG_DIR,
    "figS6_cauchy_fgm_limdi_evolved_heteroskedastic_mle_log.png",
)

COARSE_DX = 2.0e-4
COARSE_ERROR_NODES = 24
FINE_DX = 1.0e-4
FINE_ERROR_NODES = 64
CHECK_ERROR_NODES = 96
GAUSSIAN_PAD_SD = 8.0
COARSE_MAGNITUDE_NODES = 160
COARSE_DIRECTION_NODES = 40
FINE_MAGNITUDE_NODES = 768
FINE_DIRECTION_NODES = 160
CHECK_MAGNITUDE_NODES = 1024
CHECK_DIRECTION_NODES = 192
DE_SEED = 6427
DE_MAXITER = 32
DE_POPSIZE = 9


def evolved_effects_and_errors():
    """Load aligned mean effects and reported errors for the ten backgrounds."""
    output = []
    for name in EVOLVED_ORDER:
        effects, errors = limdi_gene_series(name, errors=True)
        values = effects.to_numpy(float)
        uncertainty = errors.reindex(effects.index).to_numpy(float)
        keep = (
            np.isfinite(values)
            & np.isfinite(uncertainty)
            & (uncertainty > 0.0)
            & (values >= LOWER_CUT)
        )
        output.append((name, values[keep], uncertainty[keep]))
    return output


class HeteroskedasticLikelihood:
    """Interpolated gene-specific Gaussian-convolution likelihood."""

    def __init__(
        self,
        effects,
        errors,
        dx,
        number_error_nodes,
        magnitude_nodes,
        direction_nodes,
        mu=0.5,
    ):
        self.effects = np.asarray(effects, dtype=float)
        self.errors = np.asarray(errors, dtype=float)
        self.dx = float(dx)
        self.number_error_nodes = int(number_error_nodes)
        self.number_magnitude_nodes = int(magnitude_nodes)
        self.number_direction_nodes = int(direction_nodes)
        self.mu = float(mu)
        if (
            self.effects.shape != self.errors.shape
            or np.any(~np.isfinite(self.effects))
            or np.any(~np.isfinite(self.errors))
            or np.any(self.errors <= 0.0)
            or not np.isfinite(self.mu)
            or self.mu <= 0.0
        ):
            raise ValueError(
                "effects/errors and mu must be aligned finite positive values"
            )

        log_min = float(np.log(self.errors.min()))
        log_max = float(np.log(self.errors.max()))
        self.log_error_nodes = np.linspace(
            log_min,
            log_max,
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
        data_position = (self.effects - lo) / self.dx
        self.data_lower = np.minimum(
            np.floor(data_position).astype(int),
            number - 2,
        )
        self.data_weight = data_position - self.data_lower

        self.kernels = []
        for eps in self.error_nodes:
            half = int(np.ceil(GAUSSIAN_PAD_SD * eps / self.dx))
            offsets = self.dx * np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * np.square(offsets / eps))
            self.kernels.append(kernel / kernel.sum())

    @staticmethod
    def unpack(theta):
        n, C, A = np.exp(np.asarray(theta, dtype=float))
        sigma = np.sqrt(C / n)
        r = A / sigma
        return n, r, sigma, C, A

    def latent_density_grid(self, n, r, sigma):
        """Deposit a normalized latent quadrature measure onto the FFT grid.

        Using probability masses rather than point evaluations of the analytic
        density remains accurate when the true DFE becomes much narrower than
        the grid but measurement error is still resolved.
        """
        magnitude_nodes, magnitude_weights = leggauss(
            self.number_magnitude_nodes
        )
        z_lo, z_hi = -20.0, 30.0
        z = (
            0.5 * (z_hi - z_lo) * magnitude_nodes
            + 0.5 * (z_hi + z_lo)
        )
        y = np.log(n) + z
        a_q = 0.5 * n
        b_q = self.mu
        log_density = (
            a_q * y
            - (a_q + b_q) * np.logaddexp(0.0, y)
            - betaln(a_q, b_q)
        )
        q_weights = (
            0.5
            * (z_hi - z_lo)
            * magnitude_weights
            * np.exp(log_density)
        )
        q_weights /= q_weights.sum()
        q = sigma * sigma * np.exp(y)

        sphere_alpha = 0.5 * (n - 3.0)
        direction, direction_weights = roots_jacobi(
            self.number_direction_nodes,
            sphere_alpha,
            sphere_alpha,
        )
        direction_weights /= direction_weights.sum()
        latent_effects = (
            -r * np.sqrt(q)[:, None] * direction[None, :]
            - 0.5 * q[:, None]
        ).ravel()
        latent_weights = (
            q_weights[:, None] * direction_weights[None, :]
        ).ravel()

        position = (latent_effects - self.x[0]) / self.dx
        lower = np.floor(position).astype(int)
        keep = (lower >= 0) & (lower < self.x.size - 1)
        lower = lower[keep]
        fraction = position[keep] - lower
        mass = np.zeros(self.x.size, dtype=float)
        np.add.at(
            mass,
            lower,
            latent_weights[keep] * (1.0 - fraction),
        )
        np.add.at(
            mass,
            lower + 1,
            latent_weights[keep] * fraction,
        )
        return mass / self.dx

    def convolved_grids(self, n, r, sigma):
        true_pdf = self.latent_density_grid(n=n, r=r, sigma=sigma)
        return np.asarray([
            np.maximum(
                fftconvolve(true_pdf, kernel, mode="same"),
                0.0,
            )
            for kernel in self.kernels
        ])

    def loglik(self, n, r, sigma):
        grids = self.convolved_grids(n=n, r=r, sigma=sigma)
        at_data = (
            grids[:, self.data_lower]
            * (1.0 - self.data_weight[None, :])
            + grids[:, self.data_lower + 1]
            * self.data_weight[None, :]
        )
        if np.any(~np.isfinite(at_data)):
            return -np.inf

        # Some error nodes are irrelevant to a given gene and can legitimately
        # underflow at that gene's observed effect.  Clamp before selecting the
        # two neighboring error nodes instead of rejecting the whole parameter
        # vector because of unused node-by-gene combinations.
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

        survival = cauchy_fgm_survival_many_eps(
            LOWER_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=self.error_nodes,
            mu=self.mu,
        )
        if np.any(~np.isfinite(survival)) or np.any(survival <= 0.0):
            return -np.inf
        log_survival_nodes = np.log(survival)
        log_survival = (
            (1.0 - weight) * log_survival_nodes[lower]
            + weight * log_survival_nodes[lower + 1]
        )
        return float(np.sum(log_density - log_survival))

    def objective(self, theta):
        n, r, sigma, _, _ = self.unpack(theta)
        value = self.loglik(n=n, r=r, sigma=sigma)
        return -value if np.isfinite(value) else 1.0e300

    def analytic_loglik(self, n, r, sigma):
        """Likelihood check using pointwise evaluation of the exact latent PDF."""
        true_pdf = cauchy_fgm_dfe_pdf(
            self.x,
            n=n,
            sigma=sigma,
            r=r,
            mu=self.mu,
        )
        true_pdf = np.where(np.isfinite(true_pdf), true_pdf, 0.0)
        grids = np.asarray([
            np.maximum(
                fftconvolve(true_pdf, kernel, mode="same"),
                0.0,
            )
            for kernel in self.kernels
        ])
        at_data = (
            grids[:, self.data_lower]
            * (1.0 - self.data_weight[None, :])
            + grids[:, self.data_lower + 1]
            * self.data_weight[None, :]
        )
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
        survival = cauchy_fgm_survival_many_eps(
            LOWER_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=self.error_nodes,
            mu=self.mu,
        )
        log_survival_nodes = np.log(survival)
        log_survival = (
            (1.0 - weight) * log_survival_nodes[lower]
            + weight * log_survival_nodes[lower + 1]
        )
        return float(np.sum(log_density - log_survival))

    def predictive_mixture(self, n, r, sigma):
        """Average conditional observed density over the sample's error values."""
        grids = self.convolved_grids(n=n, r=r, sigma=sigma)
        survival = cauchy_fgm_survival_many_eps(
            LOWER_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=self.error_nodes,
            mu=self.mu,
        )
        conditional = grids / survival[:, None]
        node_weights = np.zeros(self.number_error_nodes)
        np.add.at(
            node_weights,
            self.error_lower,
            1.0 - self.error_weight,
        )
        np.add.at(
            node_weights,
            self.error_lower + 1,
            self.error_weight,
        )
        node_weights /= self.effects.size
        return self.x, node_weights @ conditional

    def analytic_predictive_mixture(self, n, r, sigma):
        """Smooth plotting curve from the exact analytic latent density."""
        true_pdf = cauchy_fgm_dfe_pdf(
            self.x,
            n=n,
            sigma=sigma,
            r=r,
            mu=self.mu,
        )
        true_pdf = np.where(np.isfinite(true_pdf), true_pdf, 0.0)
        grids = np.asarray([
            np.maximum(
                fftconvolve(true_pdf, kernel, mode="same"),
                0.0,
            )
            for kernel in self.kernels
        ])
        survival = cauchy_fgm_survival_many_eps(
            LOWER_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=self.error_nodes,
            mu=self.mu,
        )
        conditional = grids / survival[:, None]
        node_weights = np.zeros(self.number_error_nodes)
        np.add.at(
            node_weights,
            self.error_lower,
            1.0 - self.error_weight,
        )
        np.add.at(
            node_weights,
            self.error_lower + 1,
            self.error_weight,
        )
        node_weights /= self.effects.size
        return self.x, node_weights @ conditional

    def probability_integral_transforms(self, n, r, sigma):
        """Conditional model CDF evaluated at each gene's own observation/error."""
        grids = self.convolved_grids(n=n, r=r, sigma=sigma)
        survival = cauchy_fgm_survival_many_eps(
            LOWER_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=self.error_nodes,
            mu=self.mu,
        )
        cdfs = cumulative_trapezoid(
            grids,
            self.x,
            axis=1,
            initial=0.0,
        )
        cut_value = np.asarray([
            np.interp(LOWER_CUT, self.x, row)
            for row in cdfs
        ])
        cdfs = np.clip(
            (cdfs - cut_value[:, None]) / survival[:, None],
            0.0,
            1.0,
        )
        at_data = (
            cdfs[:, self.data_lower]
            * (1.0 - self.data_weight[None, :])
            + cdfs[:, self.data_lower + 1]
            * self.data_weight[None, :]
        )
        gene_index = np.arange(self.effects.size)
        lower = self.error_lower
        weight = self.error_weight
        return np.clip(
            (1.0 - weight) * at_data[lower, gene_index]
            + weight * at_data[lower + 1, gene_index],
            0.0,
            1.0,
        )


def fit_one(effects, errors, flat_fit, mu=0.5):
    coarse = HeteroskedasticLikelihood(
        effects,
        errors,
        dx=COARSE_DX,
        number_error_nodes=COARSE_ERROR_NODES,
        magnitude_nodes=COARSE_MAGNITUDE_NODES,
        direction_nodes=COARSE_DIRECTION_NODES,
        mu=mu,
    )
    bounds = [
        (np.log(OPT_BOUNDS["n"][0]), np.log(OPT_BOUNDS["n"][1])),
        (np.log(OPT_BOUNDS["C"][0]), np.log(OPT_BOUNDS["C"][1])),
        (np.log(OPT_BOUNDS["A"][0]), np.log(OPT_BOUNDS["A"][1])),
    ]
    flat_theta = np.log([
        flat_fit["n"],
        flat_fit["C_n_sigma2"],
        flat_fit["A_r_sigma"],
    ])
    global_fit = differential_evolution(
        coarse.objective,
        bounds=bounds,
        seed=DE_SEED,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1.0e-8,
        x0=flat_theta,
    )
    coarse_candidates = [global_fit]
    for start in (
        global_fit.x,
        flat_theta,
        np.clip(
            flat_theta + np.array([np.log(2.0), 0.0, 0.0]),
            np.asarray(bounds)[:, 0],
            np.asarray(bounds)[:, 1],
        ),
        np.clip(
            flat_theta - np.array([np.log(2.0), 0.0, 0.0]),
            np.asarray(bounds)[:, 0],
            np.asarray(bounds)[:, 1],
        ),
    ):
        coarse_candidates.append(minimize(
            coarse.objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-11, "gtol": 1.0e-6, "maxiter": 1000},
        ))
    coarse_best = min(coarse_candidates, key=lambda result: result.fun)

    fine = HeteroskedasticLikelihood(
        effects,
        errors,
        dx=FINE_DX,
        number_error_nodes=FINE_ERROR_NODES,
        magnitude_nodes=FINE_MAGNITUDE_NODES,
        direction_nodes=FINE_DIRECTION_NODES,
        mu=mu,
    )
    fine_candidates = []
    for start in (coarse_best.x, flat_theta):
        fine_candidates.append(minimize(
            fine.objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-12, "gtol": 1.0e-6, "maxiter": 1000},
        ))
    best = min(fine_candidates, key=lambda result: result.fun)
    n, r, sigma, C, A = fine.unpack(best.x)
    bounds_hit = {
        key: bool(value <= lower * 1.001 or value >= upper / 1.001)
        for key, value, (lower, upper) in zip(
            ("n", "C", "A"),
            (n, C, A),
            OPT_BOUNDS.values(),
        )
    }

    check = HeteroskedasticLikelihood(
        effects,
        errors,
        dx=FINE_DX,
        number_error_nodes=CHECK_ERROR_NODES,
        magnitude_nodes=CHECK_MAGNITUDE_NODES,
        direction_nodes=CHECK_DIRECTION_NODES,
        mu=mu,
    )
    check_ll = check.loglik(n=n, r=r, sigma=sigma)
    flat_ll = fine.loglik(
        n=flat_fit["n"],
        r=flat_fit["r"],
        sigma=flat_fit["sigma"],
    )
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
        "heteroskedastic_loglik_at_flat_error_fit": float(flat_ll),
        "reoptimization_delta_loglik": float(-best.fun - flat_ll),
        "error_node_check": {
            "number_error_nodes": CHECK_ERROR_NODES,
            "loglik_at_fit": float(check_ll),
            "delta_loglik": float(check_ll + best.fun),
        },
        "_theta": np.asarray(best.x, dtype=float),
        "_likelihood": fine,
    }


def exact_error_positive_fraction(errors, fit, mu=0.5, chunk=128):
    conditional = []
    for start in range(0, errors.size, chunk):
        eps = errors[start:start + chunk]
        above_zero = cauchy_fgm_survival_many_eps(
            0.0,
            n=fit["n"],
            sigma=fit["sigma"],
            r=fit["r"],
            eps=eps,
            mu=mu,
        )
        retained = cauchy_fgm_survival_many_eps(
            LOWER_CUT,
            n=fit["n"],
            sigma=fit["sigma"],
            r=fit["r"],
            eps=eps,
            mu=mu,
        )
        conditional.append(above_zero / retained)
    return float(np.mean(np.concatenate(conditional)))


def diagnostics(effects, errors, fit, mu=0.5):
    pit = np.sort(fit["_likelihood"].probability_integral_transforms(
        n=fit["n"],
        r=fit["r"],
        sigma=fit["sigma"],
    ))
    number = pit.size
    uniform_right = np.arange(1, number + 1) / number
    uniform_left = np.arange(number) / number
    pit_ks = max(
        float(np.max(np.abs(uniform_right - pit))),
        float(np.max(np.abs(uniform_left - pit))),
    )
    observed_positive = float(np.mean(effects > 0.0))
    predicted_positive = exact_error_positive_fraction(errors, fit, mu=mu)
    return {
        "conditional_pit_ks": pit_ks,
        "observed_positive_fraction": observed_positive,
        "predicted_positive_fraction": predicted_positive,
        "positive_fraction_residual": observed_positive - predicted_positive,
    }


def histogram(effects):
    edges = np.arange(
        LOWER_CUT,
        UPPER_PLOT_LIMIT + BIN_WIDTH * 1.0001,
        BIN_WIDTH,
    )
    counts, edges = np.histogram(effects, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (effects.size * np.diff(edges))
    return centers, density, counts


def flat_curve(flat_fit):
    support = np.array([LOWER_CUT - 0.06, UPPER_PLOT_LIMIT + 0.06])
    likelihood = UnbinnedNoisyLikelihood(
        support,
        eps=0.005,
        dx=5.0e-4,
        lower_cut=LOWER_CUT,
    )
    pdf = likelihood.observed_pdf_grid(
        flat_fit["n"],
        flat_fit["r"],
        flat_fit["sigma"],
    )
    keep_probability = cauchy_fgm_survival(
        LOWER_CUT,
        n=flat_fit["n"],
        sigma=flat_fit["sigma"],
        r=flat_fit["r"],
        eps=0.005,
    )
    keep = (
        (likelihood.x >= LOWER_CUT)
        & (likelihood.x <= UPPER_PLOT_LIMIT)
    )
    return likelihood.x[keep], pdf[keep] / keep_probability


def plot_results(results, data_map, flat_fits, logarithmic):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(9.2, 13.0),
        sharex=True,
        sharey=True,
    )
    plot_values = {}
    linear_max = 0.0
    for name in EVOLVED_ORDER:
        effects, errors = data_map[name]
        fit = results[name]
        centers, density, counts = histogram(effects)
        plot_likelihood = HeteroskedasticLikelihood(
            effects,
            errors,
            dx=FINE_DX,
            number_error_nodes=CHECK_ERROR_NODES,
            magnitude_nodes=CHECK_MAGNITUDE_NODES,
            direction_nodes=CHECK_DIRECTION_NODES,
        )
        model_x, model_y = plot_likelihood.analytic_predictive_mixture(
            n=fit["n"],
            r=fit["r"],
            sigma=fit["sigma"],
        )
        keep = (
            (model_x >= LOWER_CUT)
            & (model_x <= UPPER_PLOT_LIMIT)
        )
        old_x, old_y = flat_curve(flat_fits[name])
        plot_values[name] = (
            centers,
            density,
            counts,
            model_x[keep],
            model_y[keep],
            old_x,
            old_y,
        )
        linear_max = max(
            linear_max,
            float(np.max(density)),
            float(np.max(model_y[keep])),
            float(np.max(old_y)),
        )

    for ax, name in zip(axes.ravel(), EVOLVED_ORDER):
        fit = results[name]
        (
            centers,
            density,
            counts,
            model_x,
            model_y,
            old_x,
            old_y,
        ) = plot_values[name]
        nonempty = counts > 0
        ax.plot(
            old_x,
            old_y,
            color="#0072B2",
            lw=1.1,
            ls="--",
            label=r"flat $\epsilon=0.005$",
            zorder=1,
        )
        ax.plot(
            model_x,
            model_y,
            color="#6a00a8",
            lw=1.8,
            label="per-gene error",
            zorder=2,
        )
        ax.scatter(
            centers[nonempty],
            density[nonempty],
            s=9,
            facecolor="0.45",
            edgecolor="none",
            label="data",
            zorder=3,
        )
        ax.axvline(0.0, color="k", lw=0.6, ls=":", zorder=1)
        ax.set_xlim(LOWER_CUT, UPPER_PLOT_LIMIT)
        if logarithmic:
            ax.set_yscale("log")
            ax.set_ylim(*LOG_Y_LIMITS)
        else:
            ax.set_ylim(-0.5, 1.08 * linear_max)
        ax.set_title(name)
        ax.text(
            0.025,
            0.055,
            rf"$n={fit['n']:.3g},\ r={fit['r']:.3g},\ "
            rf"\sigma={fit['sigma']:.3g}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fitness effect $s$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")
    fig.tight_layout()
    path = LOG_FIG_PATH if logarithmic else LINEAR_FIG_PATH
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def serializable(results):
    return {
        name: {
            key: value
            for key, value in fit.items()
            if key not in {"_likelihood", "_theta"}
        }
        for name, fit in results.items()
    }


def write_csv(results, flat_fits):
    columns = [
        "dataset",
        "N",
        "error_min",
        "error_median",
        "error_mean",
        "error_max",
        "flat_n",
        "flat_r",
        "flat_sigma",
        "n",
        "r",
        "sigma",
        "n_sigma2",
        "r_sigma",
        "loglik",
        "reoptimization_delta_loglik",
        "conditional_pit_ks",
        "observed_positive_fraction",
        "predicted_positive_fraction",
        "positive_fraction_residual",
        "boundary_hits",
        "error_node_check_delta_loglik",
    ]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in EVOLVED_ORDER:
            fit = results[name]
            flat = flat_fits[name]
            summary = fit["error_summary"]
            diag = fit["diagnostics"]
            writer.writerow({
                "dataset": name,
                "N": fit["data"]["N"],
                "error_min": summary["min"],
                "error_median": summary["median"],
                "error_mean": summary["mean"],
                "error_max": summary["max"],
                "flat_n": flat["n"],
                "flat_r": flat["r"],
                "flat_sigma": flat["sigma"],
                "n": fit["n"],
                "r": fit["r"],
                "sigma": fit["sigma"],
                "n_sigma2": fit["C_n_sigma2"],
                "r_sigma": fit["A_r_sigma"],
                "loglik": fit["loglik"],
                "reoptimization_delta_loglik": fit[
                    "reoptimization_delta_loglik"
                ],
                "conditional_pit_ks": diag["conditional_pit_ks"],
                "observed_positive_fraction": diag[
                    "observed_positive_fraction"
                ],
                "predicted_positive_fraction": diag[
                    "predicted_positive_fraction"
                ],
                "positive_fraction_residual": diag[
                    "positive_fraction_residual"
                ],
                "boundary_hits": ";".join(
                    key for key, hit in fit["bounds_hit"].items() if hit
                ),
                "error_node_check_delta_loglik": fit[
                    "error_node_check"
                ]["delta_loglik"],
            })


def write_plot_data(results, data_map, flat_fits):
    payload = {
        "lower_cut": LOWER_CUT,
        "upper_plot_limit": UPPER_PLOT_LIMIT,
        "bin_width": BIN_WIDTH,
        "panels": [],
    }
    for name in EVOLVED_ORDER:
        effects, errors = data_map[name]
        fit = results[name]
        centers, density, counts = histogram(effects)
        plot_likelihood = HeteroskedasticLikelihood(
            effects,
            errors,
            dx=FINE_DX,
            number_error_nodes=CHECK_ERROR_NODES,
            magnitude_nodes=CHECK_MAGNITUDE_NODES,
            direction_nodes=CHECK_DIRECTION_NODES,
        )
        x, model = plot_likelihood.analytic_predictive_mixture(
            n=fit["n"],
            r=fit["r"],
            sigma=fit["sigma"],
        )
        keep = (x >= LOWER_CUT) & (x <= UPPER_PLOT_LIMIT)
        old_x, old_y = flat_curve(flat_fits[name])
        payload["panels"].append({
            "key": name,
            "N": int(effects.size),
            "n": fit["n"],
            "r": fit["r"],
            "sigma": fit["sigma"],
            "histogram": [
                [
                    round(float(center), 6),
                    round(float(value), 8),
                    int(count),
                ]
                for center, value, count in zip(centers, density, counts)
            ],
            "model": [
                [round(float(xi), 6), round(float(yi), 8)]
                for xi, yi in zip(x[keep][::5], model[keep][::5])
            ],
            "flat_model": [
                [round(float(xi), 6), round(float(yi), 8)]
                for xi, yi in zip(old_x, old_y)
            ],
        })
    with open(PLOT_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    with open(FLAT_MLE_PATH, encoding="utf-8") as handle:
        flat_fits = json.load(handle)["per_dfe"]
    specs = evolved_effects_and_errors()
    data_map = {
        name: (effects, errors)
        for name, effects, errors in specs
    }
    results = {}
    for name, effects, errors in specs:
        started = time.time()
        print(
            f"Fitting {name}: N={effects.size}, "
            f"median error={np.median(errors):.5g}",
            flush=True,
        )
        fit = fit_one(effects, errors, flat_fits[name])
        fit["data"] = {
            "N": int(effects.size),
            "min": float(effects.min()),
            "max": float(effects.max()),
            "mean": float(effects.mean()),
            "sd": float(effects.std()),
            "skew": float(skew(effects)),
        }
        fit["error_summary"] = {
            "min": float(errors.min()),
            "median": float(np.median(errors)),
            "mean": float(errors.mean()),
            "max": float(errors.max()),
        }
        fit["hessian"] = observed_hessian(fit)
        fit["diagnostics"] = diagnostics(effects, errors, fit)
        fit["elapsed_seconds"] = float(time.time() - started)
        results[name] = fit
        print(
            f"  n={fit['n']:.6g}, r={fit['r']:.6g}, "
            f"sigma={fit['sigma']:.6g}, "
            f"PIT-KS={fit['diagnostics']['conditional_pit_ks']:.4f}, "
            f"node-check dLL={fit['error_node_check']['delta_loglik']:.4g}, "
            f"time={fit['elapsed_seconds']:.1f}s",
            flush=True,
        )

    payload = {
        "model": "shared_buffer_multivariate_cauchy_fgm",
        "method": "full_3d_unbinned_conditional_mle_per_gene_gaussian_error",
        "config": {
            "conditional_lower_cut": LOWER_CUT,
            "n_lower_bound": OPT_BOUNDS["n"][0],
            "replicate_pooling": "mean_per_gene",
            "per_gene_error_source": "errors_genes_inv.npy via limdi_gene_series",
            "coarse_dx": COARSE_DX,
            "coarse_error_nodes": COARSE_ERROR_NODES,
            "fine_dx": FINE_DX,
            "fine_error_nodes": FINE_ERROR_NODES,
            "check_error_nodes": CHECK_ERROR_NODES,
            "coarse_magnitude_nodes": COARSE_MAGNITUDE_NODES,
            "coarse_direction_nodes": COARSE_DIRECTION_NODES,
            "fine_magnitude_nodes": FINE_MAGNITUDE_NODES,
            "fine_direction_nodes": FINE_DIRECTION_NODES,
            "check_magnitude_nodes": CHECK_MAGNITUDE_NODES,
            "check_direction_nodes": CHECK_DIRECTION_NODES,
            "excluded_published_anomalies": ["Ara-2", "Ara+4"],
            "backgrounds": list(EVOLVED_ORDER),
        },
        "per_dfe": serializable(results),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    write_csv(results, flat_fits)
    write_plot_data(results, data_map, flat_fits)
    plot_results(results, data_map, flat_fits, logarithmic=False)
    plot_results(results, data_map, flat_fits, logarithmic=True)
    print(JSON_PATH)
    print(CSV_PATH)
    print(PLOT_DATA_PATH)
    print(LINEAR_FIG_PATH)
    print(LOG_FIG_PATH)


if __name__ == "__main__":
    main()
