r"""Fit the ten evolved Limdi DFEs with a fixed magnitude-tail exponent mu=0.45.

The generalized shared-buffer mutation model is

    delta = sigma Z / sqrt(2 G),
    Z ~ N_n(0,I), G ~ Gamma(mu,1),

so q=|delta|^2 obeys q/sigma^2 ~ BetaPrime(n/2,mu).  Thus the q density
has tail q^(-1-mu), and mu=0.5 recovers the multivariate Cauchy model.

The data handling and likelihood are identical to the heteroskedastic Cauchy
fit: per-gene Gaussian errors, conditioning on observed s >= -0.4, n >= 2,
and exclusion of the published anomalous Ara-2 and Ara+4 backgrounds.
Only mu is changed, and it is held fixed rather than estimated.
"""

import csv
import json
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from code_tmp.figS6_cauchy_fgm_full_mle import (  # noqa: E402
    LOWER_CUT,
    OPT_BOUNDS,
)
from code_tmp.figS6_cauchy_fgm_limdi_evolved_mle import (  # noqa: E402
    BIN_WIDTH,
    EVOLVED_ORDER,
    LOG_Y_LIMITS,
    UPPER_PLOT_LIMIT,
)
from code_tmp.figS6_cauchy_fgm_limdi_evolved_heteroskedastic_mle import (  # noqa: E402
    CHECK_DIRECTION_NODES,
    CHECK_ERROR_NODES,
    CHECK_MAGNITUDE_NODES,
    FINE_DX,
    HeteroskedasticLikelihood,
    diagnostics,
    evolved_effects_and_errors,
    fit_one,
    histogram,
    serializable,
)

MU = 0.45
BASELINE_MU = 0.50
DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figs_paper")
BASELINE_PATH = os.path.join(
    DATA_DIR, "cauchy_fgm_limdi_evolved_heteroskedastic_mle.json"
)
JSON_PATH = os.path.join(
    DATA_DIR, "student_fgm_mu045_limdi_evolved_heteroskedastic_mle.json"
)
CSV_PATH = os.path.join(
    DATA_DIR, "student_fgm_mu045_limdi_evolved_heteroskedastic_mle_params.csv"
)
LINEAR_FIG_PATH = os.path.join(
    FIG_DIR,
    "figS6_student_fgm_mu045_limdi_evolved_heteroskedastic_mle_linear.png",
)
LOG_FIG_PATH = os.path.join(
    FIG_DIR,
    "figS6_student_fgm_mu045_limdi_evolved_heteroskedastic_mle_log.png",
)


def plot_likelihood(effects, errors, mu):
    return HeteroskedasticLikelihood(
        effects,
        errors,
        dx=FINE_DX,
        number_error_nodes=CHECK_ERROR_NODES,
        magnitude_nodes=CHECK_MAGNITUDE_NODES,
        direction_nodes=CHECK_DIRECTION_NODES,
        mu=mu,
    )


def analytic_polish(effects, errors, initial_fit, mu):
    """Polish a global latent-quadrature fit with the exact analytic DFE."""
    likelihood = plot_likelihood(effects, errors, mu)
    bounds = [
        (
            np.log(OPT_BOUNDS[key][0]),
            np.log(OPT_BOUNDS[key][1]),
        )
        for key in ("n", "C", "A")
    ]
    start = np.log([
        initial_fit["n"],
        initial_fit["C_n_sigma2"],
        initial_fit["A_r_sigma"],
    ])

    def objective(theta):
        n, r, sigma, _, _ = likelihood.unpack(theta)
        value = likelihood.analytic_loglik(n=n, r=r, sigma=sigma)
        return -value if np.isfinite(value) else 1.0e300

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1.0e-12, "gtol": 1.0e-6, "maxiter": 1000},
    )
    n, r, sigma, C, A = likelihood.unpack(result.x)
    latent_check = likelihood.loglik(n=n, r=r, sigma=sigma)
    polished = dict(initial_fit)
    polished["latent_quadrature_start"] = {
        "n": initial_fit["n"],
        "r": initial_fit["r"],
        "sigma": initial_fit["sigma"],
        "loglik": initial_fit["loglik"],
    }
    polished.update({
        "n": float(n),
        "r": float(r),
        "sigma": float(sigma),
        "C_n_sigma2": float(C),
        "A_r_sigma": float(A),
        "loglik": float(-result.fun),
        "comparison_loglik_mu045": (
            float(-result.fun) if mu == MU else None
        ),
        "analytic_polish_success": bool(result.success),
        "analytic_polish_message": str(result.message),
        "latent_check_loglik_at_analytic_mle": float(latent_check),
        "error_node_check": {
            "number_error_nodes": CHECK_ERROR_NODES,
            "loglik_at_fit": float(latent_check),
            "delta_loglik": float(latent_check + result.fun),
        },
        "bounds_hit": {
            key: bool(value <= lower * 1.001 or value >= upper / 1.001)
            for key, value, (lower, upper) in zip(
                ("n", "C", "A"),
                (n, C, A),
                OPT_BOUNDS.values(),
            )
        },
        "_theta": np.asarray(result.x, dtype=float),
        "_likelihood": likelihood,
    })
    return polished


def conditional_curve(effects, errors, fit, mu):
    likelihood = plot_likelihood(effects, errors, mu)
    x, density = likelihood.analytic_predictive_mixture(
        n=fit["n"],
        r=fit["r"],
        sigma=fit["sigma"],
    )
    keep = (x >= LOWER_CUT) & (x <= UPPER_PLOT_LIMIT)
    return x[keep], density[keep]


def plot_results(results, data_map, baseline, logarithmic):
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
    values = {}
    linear_max = 0.0
    for name in EVOLVED_ORDER:
        effects, errors = data_map[name]
        centers, density, counts = histogram(effects)
        new_x, new_y = conditional_curve(
            effects, errors, results[name], MU
        )
        old_x, old_y = conditional_curve(
            effects, errors, baseline[name], BASELINE_MU
        )
        values[name] = (
            centers, density, counts, new_x, new_y, old_x, old_y
        )
        linear_max = max(
            linear_max,
            float(np.max(density)),
            float(np.max(new_y)),
            float(np.max(old_y)),
        )

    for ax, name in zip(axes.ravel(), EVOLVED_ORDER):
        fit = results[name]
        (
            centers,
            density,
            counts,
            new_x,
            new_y,
            old_x,
            old_y,
        ) = values[name]
        nonempty = counts > 0
        ax.plot(
            old_x,
            old_y,
            color="#0072B2",
            lw=1.2,
            ls="--",
            label=r"$\mu=0.50$ MLE",
            zorder=1,
        )
        ax.plot(
            new_x,
            new_y,
            color="#6a00a8",
            lw=1.8,
            label=r"$\mu=0.45$ MLE",
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
            rf"\sigma={fit['sigma']:.3g}$"
            "\n"
            rf"$\Delta\ell={fit['delta_loglik_vs_mu050']:+.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
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


def write_csv(results, baseline):
    columns = [
        "dataset",
        "N",
        "mu",
        "n",
        "r",
        "sigma",
        "n_sigma2",
        "r_sigma",
        "loglik",
        "comparison_loglik_mu045",
        "mu050_n",
        "mu050_r",
        "mu050_sigma",
        "mu050_recomputed_loglik",
        "delta_loglik_vs_mu050",
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
            old = baseline[name]
            diag = fit["diagnostics"]
            writer.writerow({
                "dataset": name,
                "N": fit["data"]["N"],
                "mu": MU,
                "n": fit["n"],
                "r": fit["r"],
                "sigma": fit["sigma"],
                "n_sigma2": fit["C_n_sigma2"],
                "r_sigma": fit["A_r_sigma"],
                "loglik": fit["loglik"],
                "comparison_loglik_mu045": fit[
                    "comparison_loglik_mu045"
                ],
                "mu050_n": old["n"],
                "mu050_r": old["r"],
                "mu050_sigma": old["sigma"],
                "mu050_recomputed_loglik": fit[
                    "mu050_recomputed_loglik"
                ],
                "delta_loglik_vs_mu050": fit[
                    "delta_loglik_vs_mu050"
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


def main():
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    baseline = baseline_payload["per_dfe"]
    specs = evolved_effects_and_errors()
    data_map = {
        name: (effects, errors)
        for name, effects, errors in specs
    }
    results = {}
    for name, effects, errors in specs:
        started = time.time()
        print(
            f"Fitting {name} at fixed mu={MU}: N={effects.size}",
            flush=True,
        )
        latent_fit = fit_one(effects, errors, baseline[name], mu=MU)
        fit = analytic_polish(
            effects, errors, latent_fit, mu=MU
        )
        old_fit = analytic_polish(
            effects, errors, baseline[name], mu=BASELINE_MU
        )
        baseline[name] = old_fit
        old_ll = old_fit["loglik"]
        fit["mu"] = MU
        fit["mu050_recomputed_loglik"] = float(old_ll)
        fit["delta_loglik_vs_mu050"] = float(
            fit["comparison_loglik_mu045"] - old_ll
        )
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
        fit["diagnostics"] = diagnostics(
            effects, errors, fit, mu=MU
        )
        fit["elapsed_seconds"] = float(time.time() - started)
        results[name] = fit
        print(
            f"  n={fit['n']:.6g}, r={fit['r']:.6g}, "
            f"sigma={fit['sigma']:.6g}, "
            f"dLL={fit['delta_loglik_vs_mu050']:+.4f}, "
            f"PIT-KS={fit['diagnostics']['conditional_pit_ks']:.4f}, "
            f"time={fit['elapsed_seconds']:.1f}s",
            flush=True,
        )

    clean_results = serializable(results)
    payload = {
        "model": "shared_buffer_isotropic_student_fgm",
        "method": "full_3d_unbinned_conditional_mle_per_gene_gaussian_error",
        "config": {
            "fixed_magnitude_tail_mu": MU,
            "magnitude_density_tail": "q^(-1-mu)",
            "baseline_mu": BASELINE_MU,
            "conditional_lower_cut": LOWER_CUT,
            "n_lower_bound": 2.0,
            "replicate_pooling": "mean_per_gene",
            "per_gene_error_source": "errors_genes_inv.npy via limdi_gene_series",
            "excluded_published_anomalies": ["Ara-2", "Ara+4"],
            "backgrounds": list(EVOLVED_ORDER),
        },
        "per_dfe": clean_results,
        "comparison": {
            "sum_comparison_loglik_mu045": float(sum(
                fit["loglik"] for fit in clean_results.values()
            )),
            "sum_recomputed_loglik_mu050": float(sum(
                fit["mu050_recomputed_loglik"]
                for fit in clean_results.values()
            )),
            "sum_delta_loglik_mu045_minus_mu050": float(sum(
                fit["delta_loglik_vs_mu050"]
                for fit in clean_results.values()
            )),
        },
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    write_csv(results, baseline)
    plot_results(results, data_map, baseline, logarithmic=False)
    plot_results(results, data_map, baseline, logarithmic=True)
    print(JSON_PATH)
    print(CSV_PATH)
    print(LINEAR_FIG_PATH)
    print(LOG_FIG_PATH)


if __name__ == "__main__":
    main()
