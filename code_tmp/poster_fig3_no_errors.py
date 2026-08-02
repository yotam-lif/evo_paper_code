r"""Poster Figure 3 variant: full MLEs without measurement-error convolution.

Both curves are fit directly to the individual REL607 effects:

* Gaussian FGM: exact absolute-fitness DFE, free n, r, and sigma;
* heavy-tailed FGM: log-fitness DFE, free n, r, sigma, and mu.

As in the poster figure, observations are restricted to s >= -0.5 and the
Gaussian fit excludes the most deleterious 5% of the retained measurements.
The heavy-tailed fit uses all retained measurements.  Neither likelihood nor
either displayed predictive curve is convolved with measurement error.

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


def canonical_bootstrap_intervals(effects, canonical_diagnostics):
    """Percentile intervals for the boundary-sensitive full Gaussian MLE."""
    bounds = [
        (np.log(base.N_BOUNDS[0]), np.log(base.N_BOUNDS[1])),
        (np.log(base.C_BOUNDS[0]), np.log(base.C_BOUNDS[1])),
        (np.log(base.A_BOUNDS[0]), np.log(base.A_BOUNDS[1])),
    ]
    start = np.asarray(canonical_diagnostics["_theta"], dtype=float)
    rng = np.random.default_rng(CANONICAL_BOOTSTRAP_SEED)
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
        "seed": CANONICAL_BOOTSTRAP_SEED,
        "canonical_tail_retrimmed_within_each_bootstrap": True,
    }


def clean_diagnostics(diagnostics):
    return {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("_")
    }


def main():
    started = time.perf_counter()
    os.makedirs(base.FIG_DIR, exist_ok=True)
    os.makedirs(base.DATA_DIR, exist_ok=True)

    all_effects, all_errors = base.rel607_effects_and_errors()
    above_cut = all_effects >= base.LOWER_FIT_CUT
    effects = all_effects[above_cut]
    errors = all_errors[above_cut]
    canonical_effects, _ = base.trim_canonical_tail(effects, errors)

    with open(base.OUT_JSON, encoding="utf-8") as handle:
        saved = json.load(handle)
    saved_canonical = saved["canonical_moment_constrained_mle"]["fit"]

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
            effects, canonical_diagnostics
        )
    )
    heavy_intervals, heavy_ci_diagnostics = wald_intervals(
        heavy_fit, heavy_diagnostics, heavy=True
    )

    base.plot_figure(
        effects,
        canonical_likelihood,
        heavy_likelihood,
        canonical_fit,
        heavy_fit,
        canonical_intervals,
        heavy_intervals,
        OUT_PDF,
    )

    elapsed = time.perf_counter() - started
    output = {
        "dataset": {
            "name": "REL607",
            "observed_lower_cut": base.LOWER_FIT_CUT,
            "N_heavy": int(effects.size),
            "N_canonical_after_5_percent_trim": int(
                canonical_effects.size
            ),
        },
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
        "canonical_full_mle": {
            "fit": canonical_fit.serializable(),
            "confidence_intervals_95": canonical_intervals,
            "confidence_interval_diagnostics": canonical_ci_diagnostics,
            "diagnostics": clean_diagnostics(canonical_diagnostics),
        },
        "heavy_tailed_full_mle": {
            "fit": heavy_fit.serializable(),
            "confidence_intervals_95": heavy_intervals,
            "confidence_interval_diagnostics": heavy_ci_diagnostics,
            "diagnostics": clean_diagnostics(heavy_diagnostics),
        },
        "elapsed_seconds": elapsed,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print(
        "Canonical no-error full MLE: "
        f"n={canonical_fit.n:.8g}, r={canonical_fit.r:.8g}, "
        f"sigma={canonical_fit.sigma:.8g}, "
        f"logL={canonical_fit.loglik:.8f}"
    )
    print(
        "Heavy-tailed no-error full MLE: "
        f"n={heavy_fit.n:.8g}, r={heavy_fit.r:.8g}, "
        f"sigma={heavy_fit.sigma:.8g}, mu={heavy_fit.mu:.8g}, "
        f"logL={heavy_fit.loglik:.8f}"
    )
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
