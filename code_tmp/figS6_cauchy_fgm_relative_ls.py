r"""Fit truncated Cauchy-FGM DFEs by mean squared relative density residuals.

For equal-width histogram bins, minimize

    mean_j [ (model_density_j - empirical_density_j) / empirical_density_j ]^2

over bins with at least one observation.  Empty bins are omitted because their
relative residual is undefined.  The model density is averaged over each bin,
includes the same Gaussian measurement error as the MLE analysis, and is
conditioned on the prespecified fitted range s >= -0.4.

This criterion intentionally gives low-density tail bins much more influence
than maximum likelihood.  Because one-count bins receive the greatest relative
weight and zero-count bins are ignored, the result is a descriptive
histogram-dependent fit rather than a statistically calibrated estimator.

Outputs
-------
    data/cauchy_fgm_relative_ls.json
    data/cauchy_fgm_relative_ls_params.txt
    data/cauchy_fgm_relative_ls_plot.json
    figs_paper/figS6_cauchy_fgm_relative_ls_linear.png
    figs_paper/figS6_cauchy_fgm_relative_ls_log.png
"""

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cmn.cmn_cauchy_fgm import cauchy_fgm_survival  # noqa: E402
from cmn.cmn_fgm_exper import MEAS_ERR  # noqa: E402
from code_tmp.figS6_cauchy_fgm_full_mle import (  # noqa: E402
    DISPLAY_NAMES,
    DISPLAY_ORDER,
    LOWER_CUT,
    OPT_BOUNDS,
    UnbinnedNoisyLikelihood,
    ancestor_dfes,
)

DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figs_paper")
MLE_PATH = os.path.join(DATA_DIR, "cauchy_fgm_full_mle.json")
JSON_PATH = os.path.join(DATA_DIR, "cauchy_fgm_relative_ls.json")
TXT_PATH = os.path.join(DATA_DIR, "cauchy_fgm_relative_ls_params.txt")
PLOT_DATA_PATH = os.path.join(DATA_DIR, "cauchy_fgm_relative_ls_plot.json")
LINEAR_FIG_PATH = os.path.join(
    FIG_DIR, "figS6_cauchy_fgm_relative_ls_linear.png"
)
LOG_FIG_PATH = os.path.join(FIG_DIR, "figS6_cauchy_fgm_relative_ls_log.png")

UPPER_LIMIT = 0.12
BIN_WIDTH = 0.005
MODEL_DX = 2.0e-4
DE_SEEDS = (1729, 2718)
DE_MAXITER = 70
DE_POPSIZE = 12
LOG_Y_LIMITS = (1.0e-2, 1.0e2)
LINEAR_Y_LIMITS = (-0.6, 68.0)


class RelativeDensityLeastSquares:
    """Bin-integrated mean squared relative-residual objective."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)
        self.edges = np.arange(
            LOWER_CUT,
            UPPER_LIMIT + BIN_WIDTH * 1.0001,
            BIN_WIDTH,
        )
        self.counts, self.edges = np.histogram(self.effects, bins=self.edges)
        self.widths = np.diff(self.edges)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.empirical_density = (
            self.counts / (self.effects.size * self.widths)
        )
        self.used = self.counts > 0

        # The endpoint padding makes the Gaussian convolution accurate throughout
        # every fitted bin.
        support = np.array([LOWER_CUT - 0.06, UPPER_LIMIT + 0.06])
        self.likelihood = UnbinnedNoisyLikelihood(
            support,
            eps=MEAS_ERR,
            dx=MODEL_DX,
            lower_cut=LOWER_CUT,
        )

    @staticmethod
    def unpack(theta):
        n, C, A = np.exp(np.asarray(theta, dtype=float))
        sigma = np.sqrt(C / n)
        r = A / sigma
        return n, r, sigma, C, A

    def model_bin_density(self, n, r, sigma):
        pdf = self.likelihood.observed_pdf_grid(n=n, r=r, sigma=sigma)
        if np.any(~np.isfinite(pdf)):
            return None
        cdf_grid = cumulative_trapezoid(
            pdf,
            self.likelihood.x,
            initial=0.0,
        )
        cdf_edges = np.interp(self.edges, self.likelihood.x, cdf_grid)
        keep_probability = cauchy_fgm_survival(
            LOWER_CUT,
            n=n,
            sigma=sigma,
            r=r,
            eps=MEAS_ERR,
        )
        if not np.isfinite(keep_probability) or keep_probability <= 0.0:
            return None
        return np.diff(cdf_edges) / (self.widths * keep_probability)

    def metrics(self, n, r, sigma):
        model = self.model_bin_density(n=n, r=r, sigma=sigma)
        if model is None or np.any(~np.isfinite(model)):
            return None
        residual = model[self.used] - self.empirical_density[self.used]
        relative = residual / self.empirical_density[self.used]
        return {
            "mean_relative_residual_sq": float(np.mean(relative**2)),
            "root_mean_relative_residual_sq": float(
                np.sqrt(np.mean(relative**2))
            ),
            "mean_density_residual_sq": float(np.mean(residual**2)),
            "model_bin_density": model,
        }

    def objective(self, theta):
        n, r, sigma, _, _ = self.unpack(theta)
        metrics = self.metrics(n=n, r=r, sigma=sigma)
        if metrics is None:
            return 1.0e300
        return metrics["mean_relative_residual_sq"]


def fit_one(effects):
    objective = RelativeDensityLeastSquares(effects)
    log_bounds = [
        (np.log(OPT_BOUNDS["n"][0]), np.log(OPT_BOUNDS["n"][1])),
        (np.log(OPT_BOUNDS["C"][0]), np.log(OPT_BOUNDS["C"][1])),
        (np.log(OPT_BOUNDS["A"][0]), np.log(OPT_BOUNDS["A"][1])),
    ]
    candidates = []
    for seed in DE_SEEDS:
        global_fit = differential_evolution(
            objective.objective,
            bounds=log_bounds,
            seed=seed,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            polish=False,
            updating="immediate",
            workers=1,
            tol=1.0e-9,
        )
        local_fit = minimize(
            objective.objective,
            global_fit.x,
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"ftol": 1.0e-13, "gtol": 1.0e-8, "maxiter": 1500},
        )
        candidates.extend((global_fit, local_fit))

    best = min(candidates, key=lambda result: result.fun)
    n, r, sigma, C, A = objective.unpack(best.x)
    metrics = objective.metrics(n=n, r=r, sigma=sigma)
    bounds_hit = {
        key: bool(value <= lower * 1.001 or value >= upper / 1.001)
        for key, value, (lower, upper) in zip(
            ("n", "C", "A"),
            (n, C, A),
            OPT_BOUNDS.values(),
        )
    }
    return {
        "n": float(n),
        "r": float(r),
        "sigma": float(sigma),
        "C_n_sigma2": float(C),
        "A_r_sigma": float(A),
        "mean_relative_residual_sq": metrics[
            "mean_relative_residual_sq"
        ],
        "root_mean_relative_residual_sq": metrics[
            "root_mean_relative_residual_sq"
        ],
        "mean_density_residual_sq": metrics["mean_density_residual_sq"],
        "success": bool(best.success),
        "message": str(best.message),
        "bounds_hit": bounds_hit,
        "number_nonempty_bins": int(objective.used.sum()),
        "number_empty_bins": int((~objective.used).sum()),
        "_objective": objective,
        "_model_bin_density": metrics["model_bin_density"],
    }


def model_curve(fit):
    objective = fit["_objective"]
    pdf = objective.likelihood.observed_pdf_grid(
        n=fit["n"],
        r=fit["r"],
        sigma=fit["sigma"],
    )
    keep_probability = cauchy_fgm_survival(
        LOWER_CUT,
        n=fit["n"],
        sigma=fit["sigma"],
        r=fit["r"],
        eps=MEAS_ERR,
    )
    keep = (
        (objective.likelihood.x >= LOWER_CUT)
        & (objective.likelihood.x <= UPPER_LIMIT)
    )
    return (
        objective.likelihood.x[keep],
        pdf[keep] / keep_probability,
    )


def evaluate_old_mle(fit, old_fit):
    metrics = fit["_objective"].metrics(
        n=old_fit["n"],
        r=old_fit["r"],
        sigma=old_fit["sigma"],
    )
    return {
        "n": old_fit["n"],
        "r": old_fit["r"],
        "sigma": old_fit["sigma"],
        "mean_relative_residual_sq": metrics[
            "mean_relative_residual_sq"
        ],
        "root_mean_relative_residual_sq": metrics[
            "root_mean_relative_residual_sq"
        ],
        "mean_density_residual_sq": metrics["mean_density_residual_sq"],
    }


def plot_results(results, logarithmic):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.4, 6.5),
        sharex=True,
        sharey=True,
    )
    for ax, name in zip(axes.ravel(), DISPLAY_ORDER):
        fit = results[name]
        objective = fit["_objective"]
        model_x, model_y = model_curve(fit)
        used = objective.used

        ax.plot(
            model_x,
            model_y,
            color="#0072B2",
            lw=2.0,
            label="relative-residual fit",
            zorder=2,
        )
        ax.scatter(
            objective.centers[used],
            objective.empirical_density[used],
            s=12,
            facecolor="0.45",
            edgecolor="none",
            label="data (nonempty bins)",
            zorder=3,
        )
        if not logarithmic:
            empty = ~used
            ax.scatter(
                objective.centers[empty],
                np.zeros(empty.sum()),
                s=9,
                facecolor="none",
                edgecolor="0.65",
                linewidth=0.6,
                label="empty bins",
                zorder=3,
            )
        ax.axvline(0.0, color="k", lw=0.7, ls=":", zorder=1)
        ax.set_xlim(LOWER_CUT, UPPER_LIMIT)
        if logarithmic:
            ax.set_yscale("log")
            ax.set_ylim(*LOG_Y_LIMITS)
        else:
            ax.set_ylim(*LINEAR_Y_LIMITS)
        ax.set_title(DISPLAY_NAMES[name])
        ax.text(
            0.03,
            0.05,
            rf"$n={fit['n']:.3g},\ r={fit['r']:.3g},\ "
            rf"\sigma={fit['sigma']:.3g}$" "\n"
            rf"$\langle e_{{rel}}^2\rangle="
            rf"{fit['mean_relative_residual_sq']:.3g}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fitness effect $s$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")
    fig.tight_layout()
    path = LOG_FIG_PATH if logarithmic else LINEAR_FIG_PATH
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def serializable(results):
    output = {}
    for name, fit in results.items():
        output[name] = {
            key: value
            for key, value in fit.items()
            if not key.startswith("_")
        }
    return output


def write_text(results):
    header = (
        "dataset\tN\tn\tr\tsigma\tn*sigma^2\tr*sigma\t"
        "mean_rel_resid_sq\trms_rel_resid\t"
        "MLE_mean_rel_resid_sq\tbounds_hit"
    )
    lines = [
        "Shared-buffer Cauchy FGM: mean squared relative-density-residual fits",
        (
            f"Bins: width={BIN_WIDTH:g}, range=[{LOWER_CUT:g},"
            f"{UPPER_LIMIT:g}], nonempty bins only"
        ),
        header,
    ]
    for name in ("Couce 0K", "Couce 2K", "REL606", "REL607"):
        fit = results[name]
        hit = ",".join(
            key for key, value in fit["bounds_hit"].items() if value
        ) or "none"
        lines.append(
            f"{name}\t{fit['N']}\t{fit['n']:.8g}\t{fit['r']:.8g}\t"
            f"{fit['sigma']:.8g}\t{fit['C_n_sigma2']:.8g}\t"
            f"{fit['A_r_sigma']:.8g}\t"
            f"{fit['mean_relative_residual_sq']:.8g}\t"
            f"{fit['root_mean_relative_residual_sq']:.8g}\t"
            f"{fit['old_mle']['mean_relative_residual_sq']:.8g}\t{hit}"
        )
    with open(TXT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    with open(MLE_PATH, encoding="utf-8") as handle:
        old_mle = json.load(handle)["per_dfe"]
    effects_by_name = dict(ancestor_dfes(lower_cut=LOWER_CUT))

    results = {}
    for name, effects in effects_by_name.items():
        print(f"Fitting {name} ({effects.size} observations)", flush=True)
        fit = fit_one(effects)
        fit["N"] = int(effects.size)
        fit["old_mle"] = evaluate_old_mle(fit, old_mle[name])
        results[name] = fit
        print(
            f"  n={fit['n']:.6g}, r={fit['r']:.6g}, "
            f"sigma={fit['sigma']:.6g}, "
            f"mean relative residual^2="
            f"{fit['mean_relative_residual_sq']:.6g}",
            flush=True,
        )

    output = {
        "model": "shared_buffer_multivariate_cauchy_fgm",
        "method": "binned_mean_squared_relative_density_residuals",
        "config": {
            "measurement_error_sd": MEAS_ERR,
            "conditional_lower_cut": LOWER_CUT,
            "upper_plot_and_bin_limit": UPPER_LIMIT,
            "bin_width": BIN_WIDTH,
            "empty_bins_excluded": True,
            "model_grid_dx": MODEL_DX,
            "optimization_bounds": OPT_BOUNDS,
            "de_seeds": list(DE_SEEDS),
            "de_maxiter": DE_MAXITER,
            "de_popsize": DE_POPSIZE,
        },
        "per_dfe": serializable(results),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    plot_data = {
        "lower_cut": LOWER_CUT,
        "upper_limit": UPPER_LIMIT,
        "bin_width": BIN_WIDTH,
        "log_y_limits": list(LOG_Y_LIMITS),
        "panels": [],
    }
    for name in DISPLAY_ORDER:
        fit = results[name]
        objective = fit["_objective"]
        model_x, model_y = model_curve(fit)
        plot_data["panels"].append({
            "key": name,
            "title": DISPLAY_NAMES[name],
            "N": fit["N"],
            "n": fit["n"],
            "r": fit["r"],
            "sigma": fit["sigma"],
            "mean_relative_residual_sq": fit[
                "mean_relative_residual_sq"
            ],
            "histogram": [
                [
                    round(float(x), 6),
                    round(float(y), 8),
                    int(count),
                ]
                for x, y, count in zip(
                    objective.centers,
                    objective.empirical_density,
                    objective.counts,
                )
            ],
            "model": [
                [round(float(x), 6), round(float(y), 8)]
                for x, y in zip(model_x[::10], model_y[::10])
            ],
        })
    with open(PLOT_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(plot_data, handle, indent=2)
        handle.write("\n")

    write_text(results)
    plot_results(results, logarithmic=False)
    plot_results(results, logarithmic=True)
    print(JSON_PATH)
    print(TXT_PATH)
    print(PLOT_DATA_PATH)
    print(LINEAR_FIG_PATH)
    print(LOG_FIG_PATH)


if __name__ == "__main__":
    main()
