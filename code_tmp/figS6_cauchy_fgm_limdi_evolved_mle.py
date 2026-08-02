r"""Full Cauchy-FGM MLEs for the ten published Limdi evolved backgrounds.

This is the evolved-background companion to
``figS6_cauchy_fgm_full_mle.py``.  It uses the same Limdi loader, averages the
Green/Red libraries per gene, excludes the published anomalous Ara-2 and Ara+4
backgrounds, retains every cleaned effect with s >= -0.4, and fits each DFE
independently by the conditional unbinned likelihood

    sum_i log p_obs(s_i | n,r,sigma)
        - N log P(S_obs >= -0.4 | n,r,sigma).

All three biological parameters are free.  The optimizer uses the invertible
coordinates (n, C=n*sigma^2, A=r*sigma) only for numerical stability.  The
Gaussian measurement-error s.d. is 0.005, as in the ancestor analysis.

Outputs
-------
    data/cauchy_fgm_limdi_evolved_mle.json
    data/cauchy_fgm_limdi_evolved_mle_params.csv
    data/cauchy_fgm_limdi_evolved_mle_plot.json
    figs_paper/figS6_cauchy_fgm_limdi_evolved_mle_linear.png
    figs_paper/figS6_cauchy_fgm_limdi_evolved_mle_log.png
    figs_paper/figS6_cauchy_fgm_limdi_evolved_n_profiles.png
"""

import csv
import json
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cmn.cmn_cauchy_fgm import cauchy_fgm_survival  # noqa: E402
from cmn.cmn_fgm_exper import MEAS_ERR, ORDER, load_limdi  # noqa: E402
from code_tmp.figS6_cauchy_fgm_full_mle import (  # noqa: E402
    DX,
    LOWER_CUT,
    UnbinnedNoisyLikelihood,
    fit_large_n,
    fit_one,
    grid_resolution_check,
    observed_hessian,
    profile_n,
)

DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figs_paper")
JSON_PATH = os.path.join(DATA_DIR, "cauchy_fgm_limdi_evolved_mle.json")
CSV_PATH = os.path.join(DATA_DIR, "cauchy_fgm_limdi_evolved_mle_params.csv")
PLOT_DATA_PATH = os.path.join(
    DATA_DIR, "cauchy_fgm_limdi_evolved_mle_plot.json"
)
LINEAR_FIG_PATH = os.path.join(
    FIG_DIR, "figS6_cauchy_fgm_limdi_evolved_mle_linear.png"
)
LOG_FIG_PATH = os.path.join(
    FIG_DIR, "figS6_cauchy_fgm_limdi_evolved_mle_log.png"
)
PROFILE_FIG_PATH = os.path.join(
    FIG_DIR, "figS6_cauchy_fgm_limdi_evolved_n_profiles.png"
)

EVOLVED_ORDER = tuple(
    population for population in ORDER if population not in {"REL606", "REL607"}
)
UPPER_PLOT_LIMIT = 0.12
BIN_WIDTH = 0.005
PLOT_DX = 5.0e-4
LOG_Y_LIMITS = (1.0e-2, 1.0e2)
PROFILE_95_THRESHOLD = 3.841458820694124


def evolved_dfes():
    """Return the ten main-analysis 50K Limdi DFEs with only the s cutoff."""
    loaded = load_limdi(populations=EVOLVED_ORDER, trim=(0.0, 0.0))
    return [
        (name, loaded[name][loaded[name] >= LOWER_CUT])
        for name in EVOLVED_ORDER
    ]


def conditional_curve_and_cdf(fit, upper=0.45):
    """Dense conditional noisy density and CDF for plotting/diagnostics."""
    support = np.array([LOWER_CUT - 0.06, upper])
    likelihood = UnbinnedNoisyLikelihood(
        support,
        eps=MEAS_ERR,
        dx=PLOT_DX,
        lower_cut=LOWER_CUT,
    )
    pdf = likelihood.observed_pdf_grid(
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
    conditional_pdf = pdf / keep_probability
    cdf = cumulative_trapezoid(
        conditional_pdf,
        likelihood.x,
        initial=0.0,
    )
    at_cut = float(np.interp(LOWER_CUT, likelihood.x, cdf))
    cdf = np.clip(cdf - at_cut, 0.0, 1.0)
    return likelihood.x, conditional_pdf, cdf


def descriptive_diagnostics(effects, fit):
    """Approximate conditional KS distance and beneficial fraction mismatch."""
    x, _, cdf = conditional_curve_and_cdf(fit)
    ordered = np.sort(effects)
    model_cdf = np.interp(ordered, x, cdf)
    number = ordered.size
    empirical_right = np.arange(1, number + 1) / number
    empirical_left = np.arange(number) / number
    ks = max(
        float(np.max(np.abs(empirical_right - model_cdf))),
        float(np.max(np.abs(empirical_left - model_cdf))),
    )
    observed_positive = float(np.mean(effects > 0.0))
    predicted_positive = float(
        cauchy_fgm_survival(
            0.0,
            n=fit["n"],
            sigma=fit["sigma"],
            r=fit["r"],
            eps=MEAS_ERR,
        )
        / cauchy_fgm_survival(
            LOWER_CUT,
            n=fit["n"],
            sigma=fit["sigma"],
            r=fit["r"],
            eps=MEAS_ERR,
        )
    )
    return {
        "approx_conditional_ks": ks,
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


def plot_fits(results, data_map, logarithmic):
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
    linear_max = 0.0
    panel_values = {}
    for name in EVOLVED_ORDER:
        effects = data_map[name]
        fit = results[name]
        centers, density, counts = histogram(effects)
        x, model, _ = conditional_curve_and_cdf(fit)
        keep = (x >= LOWER_CUT) & (x <= UPPER_PLOT_LIMIT)
        panel_values[name] = (
            centers,
            density,
            counts,
            x[keep],
            model[keep],
        )
        linear_max = max(
            linear_max,
            float(np.max(density)),
            float(np.max(model[keep])),
        )

    for ax, name in zip(axes.ravel(), EVOLVED_ORDER):
        fit = results[name]
        centers, density, counts, model_x, model_y = panel_values[name]
        nonempty = counts > 0
        ax.plot(
            model_x,
            model_y,
            color="#6a00a8",
            lw=1.7,
            label="conditional MLE",
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
        if not logarithmic:
            empty = ~nonempty
            ax.scatter(
                centers[empty],
                np.zeros(empty.sum()),
                s=7,
                facecolor="none",
                edgecolor="0.7",
                linewidth=0.5,
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

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fitness effect $s$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")
    fig.tight_layout()
    path = LOG_FIG_PATH if logarithmic else LINEAR_FIG_PATH
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(results):
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
        sharey=False,
    )
    for ax, name in zip(axes.ravel(), EVOLVED_ORDER):
        profile = results[name]["n_profile"]
        n_grid = np.asarray(profile["n"])
        twice_delta = 2.0 * np.asarray(profile["delta_loglik"])
        ax.plot(n_grid, twice_delta, color="#6a00a8", lw=1.7)
        ax.axhline(
            PROFILE_95_THRESHOLD,
            color="0.4",
            lw=0.8,
            ls="--",
            label="95% cutoff",
        )
        ax.axvline(results[name]["n"], color="k", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(
            0.0,
            min(25.0, max(5.0, float(np.nanmax(twice_delta)))),
        )
        ax.set_title(name)
        ax.text(
            0.97,
            0.94,
            rf"$n\to\infty$: "
            rf"{profile['large_n_twice_delta_loglik']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fixed $n$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$2[\ell_{\max}-\ell_p(n)]$")
    fig.tight_layout()
    fig.savefig(PROFILE_FIG_PATH, dpi=220, bbox_inches="tight")
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


def write_csv(results):
    columns = [
        "dataset",
        "N",
        "min_s",
        "max_s",
        "mean_s",
        "sd_s",
        "n",
        "r",
        "sigma",
        "n_sigma2",
        "r_sigma",
        "loglik",
        "n_profile_95_low",
        "n_profile_95_high",
        "n_profile_open_high",
        "large_n_twice_delta_loglik",
        "approx_conditional_ks",
        "observed_positive_fraction",
        "predicted_positive_fraction",
        "positive_fraction_residual",
        "boundary_hits",
        "grid_delta_loglik",
    ]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in EVOLVED_ORDER:
            fit = results[name]
            profile = fit["n_profile"]
            diagnostics = fit["diagnostics"]
            writer.writerow({
                "dataset": name,
                "N": fit["data"]["N"],
                "min_s": fit["data"]["min"],
                "max_s": fit["data"]["max"],
                "mean_s": fit["data"]["mean"],
                "sd_s": fit["data"]["sd"],
                "n": fit["n"],
                "r": fit["r"],
                "sigma": fit["sigma"],
                "n_sigma2": fit["C_n_sigma2"],
                "r_sigma": fit["A_r_sigma"],
                "loglik": fit["loglik"],
                "n_profile_95_low": profile["profile_95_n"][0],
                "n_profile_95_high": profile["profile_95_n"][1],
                "n_profile_open_high": profile["profile_95_open_high"],
                "large_n_twice_delta_loglik": profile[
                    "large_n_twice_delta_loglik"
                ],
                "approx_conditional_ks": diagnostics[
                    "approx_conditional_ks"
                ],
                "observed_positive_fraction": diagnostics[
                    "observed_positive_fraction"
                ],
                "predicted_positive_fraction": diagnostics[
                    "predicted_positive_fraction"
                ],
                "positive_fraction_residual": diagnostics[
                    "positive_fraction_residual"
                ],
                "boundary_hits": ";".join(
                    key for key, hit in fit["bounds_hit"].items() if hit
                ),
                "grid_delta_loglik": fit["grid_check"]["delta_loglik"],
            })


def write_plot_data(results, data_map):
    payload = {
        "lower_cut": LOWER_CUT,
        "upper_plot_limit": UPPER_PLOT_LIMIT,
        "bin_width": BIN_WIDTH,
        "log_y_limits": list(LOG_Y_LIMITS),
        "panels": [],
    }
    for name in EVOLVED_ORDER:
        effects = data_map[name]
        fit = results[name]
        centers, density, counts = histogram(effects)
        x, model, _ = conditional_curve_and_cdf(fit)
        keep = (x >= LOWER_CUT) & (x <= UPPER_PLOT_LIMIT)
        payload["panels"].append({
            "key": name,
            "title": name,
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
                for xi, yi in zip(x[keep][::4], model[keep][::4])
            ],
        })
    with open(PLOT_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    specs = evolved_dfes()
    data_map = dict(specs)
    results = {}
    for name, effects in specs:
        started = time.time()
        print(f"Fitting {name}: N={effects.size}", flush=True)
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
        fit["diagnostics"] = descriptive_diagnostics(effects, fit)
        fit["elapsed_seconds"] = float(time.time() - started)
        results[name] = fit
        print(
            f"  n={fit['n']:.6g}, r={fit['r']:.6g}, "
            f"sigma={fit['sigma']:.6g}, "
            f"n95={fit['n_profile']['profile_95_n']}, "
            f"KS={fit['diagnostics']['approx_conditional_ks']:.4f}, "
            f"time={fit['elapsed_seconds']:.1f}s",
            flush=True,
        )

    payload = {
        "model": "shared_buffer_multivariate_cauchy_fgm",
        "method": "full_3d_unbinned_conditional_mle",
        "config": {
            "measurement_error_sd": MEAS_ERR,
            "dx": DX,
            "replicate_pooling": "mean_per_gene",
            "tail_trim": [0.0, 0.0],
            "conditional_lower_cut": LOWER_CUT,
            "excluded_published_anomalies": ["Ara-2", "Ara+4"],
            "backgrounds": list(EVOLVED_ORDER),
        },
        "per_dfe": serializable(results),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    write_csv(results)
    write_plot_data(results, data_map)
    plot_fits(results, data_map, logarithmic=False)
    plot_fits(results, data_map, logarithmic=True)
    plot_profiles(results)
    print(JSON_PATH)
    print(CSV_PATH)
    print(PLOT_DATA_PATH)
    print(LINEAR_FIG_PATH)
    print(LOG_FIG_PATH)
    print(PROFILE_FIG_PATH)


if __name__ == "__main__":
    main()
