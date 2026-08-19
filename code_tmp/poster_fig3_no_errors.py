r"""Poster Figure 3 variant: full MLEs without measurement-error convolution.

Both curves are fit independently to the individual REL607 and REL606 effects:

* Gaussian FGM: exact absolute-fitness DFE, free n, r, and sigma;
* heavy-tailed FGM: log-fitness DFE, free n, r, sigma, and mu.

As in the poster figure, observations are restricted to s >= -0.5 and each
Gaussian fit excludes the most deleterious 5% of its retained measurements.
Each heavy-tailed fit uses all retained measurements.  Neither likelihood nor
any displayed predictive curve is convolved with measurement error.  Figure
rows correspond to REL607 and REL606; columns show the deleterious tail and
central bulk, respectively.

Outputs:

    data/poster_fig3_no_errors_fit.json
    figs_paper/poster_fig3_no_errors.pdf
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import poster_fig3_with_errors as base
from cmn import cmn_fgm
from cmn.cmn_cauchy_fgm import (
    cauchy_fgm_dfe_logpdf,
    cauchy_fgm_dfe_pdf,
    cauchy_fgm_survival,
)


OUT_PDF = os.path.join(base.FIG_DIR, "poster_fig3_no_errors.pdf")
OUT_JSON = os.path.join(base.DATA_DIR, "poster_fig3_no_errors_fit.json")
PLOT_DX = 2.0e-4
CANONICAL_BOOTSTRAPS = 100
CANONICAL_BOOTSTRAP_SEED = 260731
SHARED_Y_LIMITS = ((1.0e-2, 1.45), (-2.0, 47.0))
POPULATIONS = ("REL607", "REL606")


class CanonicalNoErrorLikelihood:
    """Exact conditional absolute-fitness likelihood without convolution."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)
        self.x = np.arange(-0.56, 0.1001, PLOT_DX)

    @staticmethod
    def survival(n, r, sigma):
        return float(cmn_fgm.fgm_fitness_survival_many_eps(
            base.LOWER_FIT_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=np.array([0.0]),
        )[0])

    def canonical_loglik(self, n, r, sigma):
        log_density = cmn_fgm.fgm_fitness_dfe_logpdf(
            self.effects,
            n=n,
            sigma=sigma,
            r=r,
        )
        survival = self.survival(n=n, r=r, sigma=sigma)
        if (
            np.any(~np.isfinite(log_density))
            or not np.isfinite(survival)
            or survival <= 0.0
        ):
            return -np.inf
        return float(
            np.sum(log_density) - self.effects.size * np.log(survival)
        )

    def canonical_predictive_pdf(self, n, r, sigma):
        survival = self.survival(n=n, r=r, sigma=sigma)
        return (
            cmn_fgm.fgm_fitness_dfe_pdf(
                self.x,
                n=n,
                sigma=sigma,
                r=r,
            )
            / survival
        )


class HeavyNoErrorLikelihood:
    """Exact conditional heavy-tailed likelihood without convolution."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)
        self.x = np.arange(-0.56, 0.1001, PLOT_DX)

    @staticmethod
    def survival(n, r, sigma, mu):
        return cauchy_fgm_survival(
            base.LOWER_FIT_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=0.0,
            mu=mu,
        )

    def heavy_loglik(self, n, r, sigma, mu):
        log_density = cauchy_fgm_dfe_logpdf(
            self.effects,
            n=n,
            sigma=sigma,
            r=r,
            mu=mu,
        )
        survival = self.survival(
            n=n,
            r=r,
            sigma=sigma,
            mu=mu,
        )
        if (
            np.any(~np.isfinite(log_density))
            or not np.isfinite(survival)
            or survival <= 0.0
        ):
            return -np.inf
        return float(
            np.sum(log_density) - self.effects.size * np.log(survival)
        )

    def heavy_predictive_pdf(self, n, r, sigma, mu):
        survival = self.survival(
            n=n,
            r=r,
            sigma=sigma,
            mu=mu,
        )
        return (
            cauchy_fgm_dfe_pdf(
                self.x,
                n=n,
                sigma=sigma,
                r=r,
                mu=mu,
            )
            / survival
        )


def unpack_canonical(theta):
    """Transform log(n, C=n sigma^2, A=r sigma) to model parameters."""
    n, c_scale, a_scale = np.exp(np.asarray(theta, dtype=float))
    sigma = np.sqrt(c_scale / n)
    r = a_scale / sigma
    return (
        float(n),
        float(r),
        float(sigma),
        float(c_scale),
        float(a_scale),
    )


def fit_canonical_full_mle(likelihood, saved_fit):
    """Multistart full MLE, with bounds used only to discover local basins."""
    bounds = [
        (np.log(base.N_BOUNDS[0]), np.log(base.N_BOUNDS[1])),
        (np.log(base.C_BOUNDS[0]), np.log(base.C_BOUNDS[1])),
        (np.log(base.A_BOUNDS[0]), np.log(base.A_BOUNDS[1])),
    ]
    lower = np.asarray(bounds)[:, 0]
    upper = np.asarray(bounds)[:, 1]

    def objective(theta):
        n, r, sigma, _, _ = unpack_canonical(theta)
        if r < base.R_BOUNDS[0] or r > base.R_BOUNDS[1]:
            return 1.0e100
        value = likelihood.canonical_loglik(n=n, r=r, sigma=sigma)
        return -value if np.isfinite(value) else 1.0e100

    saved_start = np.array([
        saved_fit["n"],
        saved_fit["n"] * saved_fit["sigma"] ** 2,
        saved_fit["r"] * saved_fit["sigma"],
    ])
    starts = [
        np.log(saved_start),
        np.log([2.2, 3.0e-2, 4.0e-2]),
        np.log([3.0, 2.0e-2, 3.0e-2]),
        np.log([5.0, 1.0e-2, 2.0e-2]),
        np.log([10.0, 5.0e-3, 1.0e-2]),
        np.log([30.0, 5.0e-3, 1.0e-2]),
        np.log([100.0, 5.0e-3, 1.0e-2]),
    ]
    bounded = [
        minimize(
            objective,
            np.clip(start, lower, upper),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1.0e-12,
                "gtol": 1.0e-7,
                "maxiter": 2000,
                "maxls": 60,
            },
        )
        for start in starts
    ]

    def interior(result):
        theta = np.asarray(result.x)
        n, r, _, _, _ = unpack_canonical(theta)
        margin = 2.0e-3
        return bool(
            np.isfinite(result.fun)
            and np.all(theta > lower + margin)
            and np.all(theta < upper - margin)
            and n > base.N_BOUNDS[0] * 1.002
            and n < base.N_BOUNDS[1] / 1.002
            and r > base.R_BOUNDS[0] * 1.002
            and r < base.R_BOUNDS[1] / 1.002
        )

    interior_bounded = [result for result in bounded if interior(result)]
    if not interior_bounded:
        finite = [result for result in bounded if np.isfinite(result.fun)]
        if not finite:
            raise RuntimeError("No finite canonical no-error MLE was found.")
        candidates = finite
        final_polish = "bounded because no interior basin was found"
    else:
        polished = [
            minimize(
                objective,
                result.x,
                method="BFGS",
                options={"gtol": 2.0e-6, "maxiter": 2000},
            )
            for result in interior_bounded
        ]
        candidates = [
            result for result in polished if interior(result)
        ] or interior_bounded
        final_polish = (
            "unconstrained log-coordinate polish of an interior basin"
        )

    best = min(candidates, key=lambda result: result.fun)
    n, r, sigma, c_scale, a_scale = unpack_canonical(best.x)
    fit = base.ModelParameters(
        n=n,
        r=r,
        sigma=sigma,
        loglik=float(-best.fun),
    )
    diagnostics = {
        "success": bool(best.success),
        "message": str(best.message),
        "selection_rule": "highest finite local maximum from multistart",
        "number_starts": len(starts),
        "number_interior_bounded_candidates": len(interior_bounded),
        "final_polish": final_polish,
        "C_n_sigma2": c_scale,
        "A_r_sigma": a_scale,
        "_theta": np.asarray(best.x, dtype=float),
        "_objective": objective,
    }
    return fit, diagnostics


def wald_intervals(fit, diagnostics, heavy=False):
    """Observed-information Wald intervals in transformed log coordinates."""
    theta = diagnostics["_theta"]
    objective = diagnostics["_objective"]
    attempted_steps = (
        2.0e-4,
        5.0e-4,
        1.0e-3,
        2.0e-3,
        5.0e-3,
        1.0e-2,
        2.0e-2,
    )
    hessian = None
    eigenvalues = None
    selected_step = None
    for step in attempted_steps:
        candidate = base.finite_difference_hessian(
            objective, theta, step=step
        )
        candidate_eigenvalues = np.linalg.eigvalsh(candidate)
        if np.all(candidate_eigenvalues > 0.0):
            hessian = candidate
            eigenvalues = candidate_eigenvalues
            selected_step = step
            break
    if hessian is None:
        raise RuntimeError(
            "No positive-definite observed-information Hessian was found."
        )

    covariance_theta = np.linalg.inv(hessian)
    transform = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, 1.0, 0.0],
        [-0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    if not heavy:
        transform = transform[:3, :3]
    covariance_log_parameters = (
        transform @ covariance_theta @ transform.T
    )
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance_log_parameters), 0.0)
    )
    estimates = {
        "n": fit.n,
        "r": fit.r,
        "sigma": fit.sigma,
    }
    if heavy:
        estimates["mu"] = fit.mu

    intervals = {}
    for index, (name, estimate) in enumerate(estimates.items()):
        half_width = 1.959964 * standard_errors[index]
        intervals[name] = [
            float(np.exp(np.log(estimate) - half_width)),
            float(np.exp(np.log(estimate) + half_width)),
        ]
    return intervals, {
        "method": "observed_information_log_scale_wald",
        "finite_difference_step": selected_step,
        "hessian_eigenvalues": eigenvalues.tolist(),
    }


def canonical_bootstrap_intervals(
    effects,
    canonical_diagnostics,
    seed=CANONICAL_BOOTSTRAP_SEED,
):
    """Percentile intervals for the boundary-sensitive full Gaussian MLE."""
    bounds = [
        (np.log(base.N_BOUNDS[0]), np.log(base.N_BOUNDS[1])),
        (np.log(base.C_BOUNDS[0]), np.log(base.C_BOUNDS[1])),
        (np.log(base.A_BOUNDS[0]), np.log(base.A_BOUNDS[1])),
    ]
    start = np.asarray(canonical_diagnostics["_theta"], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = {"n": [], "r": [], "sigma": []}
    failures = 0

    for bootstrap in range(CANONICAL_BOOTSTRAPS):
        index = rng.integers(0, effects.size, effects.size)
        sample = effects[index]
        cutoff = float(np.quantile(sample, base.CANONICAL_LOWER_TRIM))
        sample = sample[sample >= cutoff]
        likelihood = CanonicalNoErrorLikelihood(sample)

        def objective(theta):
            n, r, sigma, _, _ = unpack_canonical(theta)
            if r < base.R_BOUNDS[0] or r > base.R_BOUNDS[1]:
                return 1.0e100
            value = likelihood.canonical_loglik(
                n=n, r=r, sigma=sigma
            )
            return -value if np.isfinite(value) else 1.0e100

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1.0e-11,
                "gtol": 1.0e-6,
                "maxiter": 1000,
                "maxls": 60,
            },
        )
        if not np.isfinite(result.fun) or result.fun >= 1.0e90:
            failures += 1
            continue
        n, r, sigma, _, _ = unpack_canonical(result.x)
        estimates["n"].append(n)
        estimates["r"].append(r)
        estimates["sigma"].append(sigma)
        if (bootstrap + 1) % 20 == 0:
            print(
                f"Canonical bootstrap: {bootstrap + 1}/"
                f"{CANONICAL_BOOTSTRAPS}",
                flush=True,
            )

    retained = CANONICAL_BOOTSTRAPS - failures
    if retained < 0.8 * CANONICAL_BOOTSTRAPS:
        raise RuntimeError(
            f"Only {retained}/{CANONICAL_BOOTSTRAPS} canonical "
            "bootstrap fits were finite."
        )
    intervals = {
        name: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for name, values in estimates.items()
    }
    return intervals, {
        "method": "gene_bootstrap_percentile",
        "replicates_requested": CANONICAL_BOOTSTRAPS,
        "replicates_retained": retained,
        "failures": failures,
        "seed": seed,
        "canonical_tail_retrimmed_within_each_bootstrap": True,
    }


def clean_diagnostics(diagnostics):
    return {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("_")
    }


def population_effects_and_errors(population):
    """Aligned finite mean effects and positive reported errors."""
    effect_series, error_series = base.limdi_gene_series(
        population, errors=True
    )
    errors = error_series.reindex(effect_series.index).to_numpy(float)
    effects = effect_series.to_numpy(float)
    keep = (
        np.isfinite(effects)
        & np.isfinite(errors)
        & (errors > 0.0)
    )
    return effects[keep], errors[keep]


def interval_line(symbol, estimate, interval, decimals):
    return (
        rf"${symbol}={estimate:.{decimals}f}$"
        "\n"
        rf"$[{interval[0]:.{decimals}f},"
        rf"{interval[1]:.{decimals}f}]$"
    )


def parameter_text(fit, intervals, heavy=False):
    specifications = [
        ("n", "n", 2),
        ("r", "r", 3),
        ("sigma", r"\sigma", 3 if not heavy else 4),
    ]
    if heavy:
        specifications.append(("mu", r"\mu", 2))
    return "\n".join(
        interval_line(
            symbol,
            getattr(fit, name),
            intervals[name],
            decimals,
        )
        for name, symbol, decimals in specifications
    )


def plot_two_row_figure(results, path):
    """Plot tail/bulk columns for one independently fitted population per row."""
    fig, axes = base.plt.subplots(
        len(POPULATIONS),
        2,
        figsize=(11.4, 9.6),
        sharex="col",
        gridspec_kw={"wspace": 0.22, "hspace": 0.18},
    )

    for row, population in enumerate(POPULATIONS):
        result = results[population]
        effects = result["effects"]
        canonical_likelihood = result["canonical_likelihood"]
        heavy_likelihood = result["heavy_likelihood"]
        canonical_fit = result["canonical_fit"]
        heavy_fit = result["heavy_fit"]
        canonical_curve = canonical_likelihood.canonical_predictive_pdf(
            canonical_fit.n,
            canonical_fit.r,
            canonical_fit.sigma,
        )
        heavy_curve = heavy_likelihood.heavy_predictive_pdf(
            heavy_fit.n,
            heavy_fit.r,
            heavy_fit.sigma,
            heavy_fit.mu,
        )
        histograms = (
            base.histogram(effects, base.TAIL_BIN_WIDTH),
            base.histogram(effects, base.BULK_BIN_WIDTH),
        )

        for col, (empirical_histogram, xlim) in enumerate(zip(
            histograms, (base.TAIL_XLIM, base.BULK_XLIM)
        )):
            ax = axes[row, col]
            if col == 0:
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
                (heavy_likelihood.x >= xlim[0])
                & (heavy_likelihood.x <= xlim[1])
            )
            ax.errorbar(
                centers[nonzero],
                density[nonzero],
                yerr=density_error[nonzero],
                fmt="o",
                ms=4.8,
                mfc=base.DATA_COLOR,
                mec=base.DATA_EDGE,
                mew=0.45,
                ecolor=base.DATA_COLOR,
                elinewidth=0.75,
                capsize=0,
                alpha=0.88,
                label="Data",
                zorder=4,
            )
            ax.plot(
                heavy_likelihood.x[heavy_visible],
                heavy_curve[heavy_visible],
                color=base.HEAVY_COLOR,
                lw=3.0,
                label="Heavy-tailed",
                zorder=3,
            )
            ax.autoscale_view(scalex=False, scaley=True)
            ax.set_autoscaley_on(False)
            # The Gaussian tail is extrapolated beyond the trimmed fit range,
            # so it must not determine the tail-panel scale.
            ax.plot(
                canonical_likelihood.x[canonical_visible],
                canonical_curve[canonical_visible],
                color=base.CANONICAL_COLOR,
                lw=2.8,
                label="Gaussian (canonical)",
                zorder=2,
                scaley=False,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*SHARED_Y_LIMITS[col])
            ax.tick_params(direction="out", length=4.5, width=0.9)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        axes[row, 0].set_ylabel("Probability density")
        axes[row, 1].set_ylabel("")
        axes[row, 0].text(
            -0.205,
            0.5,
            population,
            transform=axes[row, 0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=20,
            fontweight="bold",
        )
        axes[row, 1].text(
            0.025,
            0.965,
            parameter_text(
                canonical_fit,
                result["canonical_intervals"],
            ),
            transform=axes[row, 1].transAxes,
            ha="left",
            va="top",
            fontsize=17.3,
            linespacing=1.00,
            color=base.CANONICAL_COLOR,
            zorder=1,
        )
        axes[row, 1].text(
            0.975,
            0.965,
            parameter_text(
                heavy_fit,
                result["heavy_intervals"],
                heavy=True,
            ),
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=17.3,
            linespacing=1.00,
            color=base.HEAVY_COLOR,
            zorder=1,
        )

    axes[0, 0].set_title("Deleterious tail", pad=9)
    axes[0, 1].set_title("Central bulk", pad=9)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fitness effect $(s)$")

    handles = [
        base.Line2D([], [], color=base.HEAVY_COLOR, lw=3.0),
        base.Line2D([], [], color=base.CANONICAL_COLOR, lw=2.8),
        base.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            ms=4.8,
            mfc=base.DATA_COLOR,
            mec=base.DATA_EDGE,
        ),
    ]
    axes[0, 0].legend(
        handles,
        ("Heavy-tailed", "Gaussian (canonical)", "Data"),
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

    for label, ax in zip(("A", "B", "C", "D"), axes.ravel()):
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

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.10,
        top=0.93,
    )
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    base.plt.close(fig)


def fit_population(population, saved_canonical, bootstrap_seed):
    all_effects, all_errors = population_effects_and_errors(population)
    above_cut = all_effects >= base.LOWER_FIT_CUT
    effects = all_effects[above_cut]
    errors = all_errors[above_cut]
    canonical_effects, _ = base.trim_canonical_tail(effects, errors)

    print(
        f"{population}: N={effects.size}, "
        f"range=({effects.min():.6f}, {effects.max():.6f})",
        flush=True,
    )
    canonical_likelihood = CanonicalNoErrorLikelihood(canonical_effects)
    heavy_likelihood = HeavyNoErrorLikelihood(effects)
    canonical_fit, canonical_diagnostics = fit_canonical_full_mle(
        canonical_likelihood, saved_canonical
    )
    heavy_fit, heavy_diagnostics = base.fit_heavy_interior(
        heavy_likelihood, canonical_fit
    )
    canonical_intervals, canonical_ci_diagnostics = (
        canonical_bootstrap_intervals(
            effects,
            canonical_diagnostics,
            seed=bootstrap_seed,
        )
    )
    heavy_intervals, heavy_ci_diagnostics = wald_intervals(
        heavy_fit, heavy_diagnostics, heavy=True
    )
    return {
        "effects": effects,
        "canonical_effects": canonical_effects,
        "canonical_likelihood": canonical_likelihood,
        "heavy_likelihood": heavy_likelihood,
        "canonical_fit": canonical_fit,
        "heavy_fit": heavy_fit,
        "canonical_intervals": canonical_intervals,
        "heavy_intervals": heavy_intervals,
        "canonical_diagnostics": canonical_diagnostics,
        "heavy_diagnostics": heavy_diagnostics,
        "canonical_ci_diagnostics": canonical_ci_diagnostics,
        "heavy_ci_diagnostics": heavy_ci_diagnostics,
    }


def serializable_population_result(population, result):
    return {
        "dataset": {
            "name": population,
            "observed_lower_cut": base.LOWER_FIT_CUT,
            "N_heavy": int(result["effects"].size),
            "N_canonical_after_5_percent_trim": int(
                result["canonical_effects"].size
            ),
        },
        "canonical_full_mle": {
            "fit": result["canonical_fit"].serializable(),
            "confidence_intervals_95": result["canonical_intervals"],
            "confidence_interval_diagnostics": result[
                "canonical_ci_diagnostics"
            ],
            "diagnostics": clean_diagnostics(
                result["canonical_diagnostics"]
            ),
        },
        "heavy_tailed_full_mle": {
            "fit": result["heavy_fit"].serializable(),
            "confidence_intervals_95": result["heavy_intervals"],
            "confidence_interval_diagnostics": result[
                "heavy_ci_diagnostics"
            ],
            "diagnostics": clean_diagnostics(
                result["heavy_diagnostics"]
            ),
        },
    }


def main():
    started = time.perf_counter()
    os.makedirs(base.FIG_DIR, exist_ok=True)
    os.makedirs(base.DATA_DIR, exist_ok=True)

    with open(base.OUT_JSON, encoding="utf-8") as handle:
        saved = json.load(handle)
    saved_canonical = saved["canonical_moment_constrained_mle"]["fit"]

    results = {
        population: fit_population(
            population,
            saved_canonical,
            bootstrap_seed=CANONICAL_BOOTSTRAP_SEED + index,
        )
        for index, population in enumerate(POPULATIONS)
    }
    plot_two_row_figure(results, OUT_PDF)

    elapsed = time.perf_counter() - started
    populations_output = {
        population: serializable_population_result(
            population, results[population]
        )
        for population in POPULATIONS
    }
    # Keep the original REL607 top-level entries as compatibility aliases for
    # consumers such as poster_fig5.py, while exposing both fits explicitly.
    rel607_output = populations_output["REL607"]
    output = {
        "dataset": rel607_output["dataset"],
        "likelihood": {
            "measurement_error_convolution": False,
            "canonical": (
                "full unbinned MLE of exact absolute-fitness DFE"
            ),
            "heavy_tailed": (
                "full unbinned MLE of exact log-fitness DFE"
            ),
            "conditional_on_s_at_least": base.LOWER_FIT_CUT,
            "canonical_lower_tail_trim": base.CANONICAL_LOWER_TRIM,
        },
        "canonical_full_mle": rel607_output["canonical_full_mle"],
        "heavy_tailed_full_mle": rel607_output[
            "heavy_tailed_full_mle"
        ],
        "populations": populations_output,
        "elapsed_seconds": elapsed,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    for population in POPULATIONS:
        canonical_fit = results[population]["canonical_fit"]
        heavy_fit = results[population]["heavy_fit"]
        print(
            f"{population} canonical no-error full MLE: "
            f"n={canonical_fit.n:.8g}, r={canonical_fit.r:.8g}, "
            f"sigma={canonical_fit.sigma:.8g}, "
            f"logL={canonical_fit.loglik:.8f}"
        )
        print(
            f"{population} heavy-tailed no-error full MLE: "
            f"n={heavy_fit.n:.8g}, r={heavy_fit.r:.8g}, "
            f"sigma={heavy_fit.sigma:.8g}, mu={heavy_fit.mu:.8g}, "
            f"logL={heavy_fit.loglik:.8f}"
        )
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
