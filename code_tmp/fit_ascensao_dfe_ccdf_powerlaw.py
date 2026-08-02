r"""Fit Ascensao deleterious DFE tails by log-log CCDF regression.

For x = -s > 0 and p(s) proportional to |s|^(-(1 + mu)), the raw
deleterious survival count obeys

    N(S <= -x) = B x^(-mu).

The lower magnitude is fixed at x = 0.01.  The upper magnitude is selected
separately for each DFE from 0.15, 0.20, and 0.30: use the largest candidate
with at least three observations beyond it, falling back to 0.15 when none
qualifies.  This preserves the requested candidate ranges without pretending
that a one- or two-gene terminal tail is well resolved.  Zero-count thresholds
cannot be logged and are excluded; this matters only for MNO:L at x = 0.15,
whose most deleterious observed effect has x = 0.1483.

The regression is descriptive OLS on correlated empirical-CCDF points, not an
unbinned power-law MLE.  Bootstrap intervals are therefore descriptive.

Outputs
-------
    data/ascensao_dfe_ccdf_powerlaw_results.csv
    data/ascensao_dfe_ccdf_powerlaw_cutoff_sensitivity.csv
    data/ascensao_dfe_ccdf_powerlaw_curves.csv
    figs_paper/ascensao_dfe_ccdf_powerlaw.png
"""

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from cmn.cmn_exper import load_asencao_array  # noqa: E402


EXPERIMENTS = ("GHI", "MNO", "PQT", "SLR")
BACKGROUNDS = ("R", "S", "L")
BACKGROUND_LABELS = {
    "R": "R (ancestor)",
    "S": "S (evolved)",
    "L": "L (evolved)",
}

X_MIN = 0.01
X_MAX_CANDIDATES = (0.15, 0.20, 0.30)
MIN_ENDPOINT_COUNT = 3
N_THRESHOLDS = 100
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 260726

RESULT_PATH = REPO_DIR / "data" / "ascensao_dfe_ccdf_powerlaw_results.csv"
SENSITIVITY_PATH = (
    REPO_DIR / "data" / "ascensao_dfe_ccdf_powerlaw_cutoff_sensitivity.csv"
)
CURVE_PATH = REPO_DIR / "data" / "ascensao_dfe_ccdf_powerlaw_curves.csv"
FIG_PATH = REPO_DIR / "figs_paper" / "ascensao_dfe_ccdf_powerlaw.png"


def survival_counts(magnitudes, thresholds):
    ordered = np.sort(np.asarray(magnitudes, float))
    return ordered.size - np.searchsorted(ordered, thresholds, side="left")


def regress_log_ccdf(thresholds, counts):
    """OLS log count = intercept - mu log threshold, dropping zero counts."""
    thresholds = np.asarray(thresholds, float)
    counts = np.asarray(counts, float)
    keep = counts > 0
    log_x = np.log(thresholds[keep])
    log_y = np.log(counts[keep])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted_log = intercept + slope * log_x
    residual = log_y - fitted_log
    sse = float(np.sum(residual**2))
    sst = float(np.sum((log_y - log_y.mean()) ** 2))
    return {
        "mu": float(-slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - sse / sst),
        "thresholds": thresholds[keep],
        "counts": counts[keep].astype(int),
        "fitted": np.exp(fitted_log),
        "sse": sse,
        "sst": sst,
    }


def choose_upper_cutoff(magnitudes):
    counts = {
        candidate: int(np.sum(magnitudes >= candidate))
        for candidate in X_MAX_CANDIDATES
    }
    eligible = [
        candidate
        for candidate in X_MAX_CANDIDATES
        if counts[candidate] >= MIN_ENDPOINT_COUNT
    ]
    return (max(eligible) if eligible else X_MAX_CANDIDATES[0]), counts


def fit_at_cutoff(magnitudes, x_max):
    thresholds = np.geomspace(X_MIN, x_max, N_THRESHOLDS)
    counts = survival_counts(magnitudes, thresholds)
    fit = regress_log_ccdf(thresholds, counts)
    fit.update(
        {
            "requested_x_max": float(x_max),
            "endpoint_count": int(np.sum(magnitudes >= x_max)),
            "last_fitted_x": float(fit["thresholds"][-1]),
            "last_fitted_count": int(fit["counts"][-1]),
        }
    )
    return fit


def bootstrap_mu(magnitudes, x_max, rng):
    magnitudes = np.asarray(magnitudes, float)
    estimates = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.choice(magnitudes, size=magnitudes.size, replace=True)
        estimates[index] = fit_at_cutoff(sample, x_max)["mu"]
    return estimates


def common_fixed_effect_slope(fits, keys):
    """Joint log-CCDF slope with a free intercept for each DFE."""
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        fit = fits[key]
        log_x = np.log(fit["thresholds"])
        log_y = np.log(fit["counts"])
        numerator += float(
            np.sum((log_x - log_x.mean()) * (log_y - log_y.mean()))
        )
        denominator += float(np.sum((log_x - log_x.mean()) ** 2))
    return float(-numerator / denominator)


def load_and_fit():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fits = {}
    sensitivity_rows = []
    curve_rows = []

    for experiment in EXPERIMENTS:
        for background in BACKGROUNDS:
            key = (experiment, background)
            effects = np.asarray(
                load_asencao_array(experiment, background), float
            )
            effects = effects[np.isfinite(effects)]
            magnitudes = -effects[effects < 0.0]
            chosen_x_max, endpoint_counts = choose_upper_cutoff(magnitudes)

            for candidate in X_MAX_CANDIDATES:
                candidate_fit = fit_at_cutoff(magnitudes, candidate)
                sensitivity_rows.append(
                    {
                        "experiment": experiment,
                        "background": background,
                        "x_min": X_MIN,
                        "requested_x_max": candidate,
                        "endpoint_count": endpoint_counts[candidate],
                        "last_fitted_x": candidate_fit["last_fitted_x"],
                        "last_fitted_count": candidate_fit[
                            "last_fitted_count"
                        ],
                        "mu": candidate_fit["mu"],
                        "loglog_r_squared": candidate_fit["r_squared"],
                        "selected": candidate == chosen_x_max,
                    }
                )

            fit = fit_at_cutoff(magnitudes, chosen_x_max)
            bootstrap = bootstrap_mu(magnitudes, chosen_x_max, rng)
            fit.update(
                {
                    "experiment": experiment,
                    "background": background,
                    "n_measured": int(effects.size),
                    "n_negative": int(magnitudes.size),
                    "count_at_x_min": int(np.sum(magnitudes >= X_MIN)),
                    "maximum_magnitude": float(magnitudes.max()),
                    "mu_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "mu_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "bootstrap_mu": bootstrap,
                }
            )
            fits[key] = fit

            curve_rows.extend(
                {
                    "experiment": experiment,
                    "background": background,
                    "x": float(x),
                    "s": float(-x),
                    "tail_count": int(count),
                    "fitted_tail_count": float(fitted),
                    "requested_x_max": chosen_x_max,
                }
                for x, count, fitted in zip(
                    fit["thresholds"], fit["counts"], fit["fitted"]
                )
            )

    keys = tuple(fits)
    common_mu = common_fixed_effect_slope(fits, keys)
    mean_mu = float(np.mean([fits[key]["mu"] for key in keys]))

    # Bootstrap the equal-background mean of the 12 individual slopes.  The
    # joint fixed-effect slope is also reported as the primary common slope.
    mean_bootstrap = np.mean(
        np.vstack([fits[key]["bootstrap_mu"] for key in keys]), axis=0
    )
    summary = {
        "common_fixed_effect_mu": common_mu,
        "equal_background_mean_mu": mean_mu,
        "mean_mu_ci_low": float(np.quantile(mean_bootstrap, 0.025)),
        "mean_mu_ci_high": float(np.quantile(mean_bootstrap, 0.975)),
    }

    return (
        fits,
        summary,
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(curve_rows),
    )


def write_results(fits, summary):
    rows = []
    for experiment in EXPERIMENTS:
        for background in BACKGROUNDS:
            fit = fits[(experiment, background)]
            rows.append(
                {
                    "experiment": experiment,
                    "background": background,
                    "x_min": X_MIN,
                    "requested_x_max": fit["requested_x_max"],
                    "endpoint_count": fit["endpoint_count"],
                    "last_fitted_x": fit["last_fitted_x"],
                    "last_fitted_count": fit["last_fitted_count"],
                    "mu": fit["mu"],
                    "mu_ci_low": fit["mu_ci_low"],
                    "mu_ci_high": fit["mu_ci_high"],
                    "loglog_r_squared": fit["r_squared"],
                    "n_measured": fit["n_measured"],
                    "n_negative": fit["n_negative"],
                    "tail_count_at_x_0.01": fit["count_at_x_min"],
                    "maximum_deleterious_magnitude": fit[
                        "maximum_magnitude"
                    ],
                }
            )

    rows.append(
        {
            "experiment": "ALL_COMMON_FIXED_INTERCEPTS",
            "background": "ALL",
            "x_min": X_MIN,
            "requested_x_max": np.nan,
            "endpoint_count": np.nan,
            "last_fitted_x": np.nan,
            "last_fitted_count": np.nan,
            "mu": summary["common_fixed_effect_mu"],
            "mu_ci_low": np.nan,
            "mu_ci_high": np.nan,
            "loglog_r_squared": np.nan,
            "n_measured": sum(fit["n_measured"] for fit in fits.values()),
            "n_negative": sum(fit["n_negative"] for fit in fits.values()),
            "tail_count_at_x_0.01": sum(
                fit["count_at_x_min"] for fit in fits.values()
            ),
            "maximum_deleterious_magnitude": np.nan,
        }
    )
    rows.append(
        {
            "experiment": "ALL_EQUAL_BACKGROUND_MEAN",
            "background": "ALL",
            "x_min": X_MIN,
            "requested_x_max": np.nan,
            "endpoint_count": np.nan,
            "last_fitted_x": np.nan,
            "last_fitted_count": np.nan,
            "mu": summary["equal_background_mean_mu"],
            "mu_ci_low": summary["mean_mu_ci_low"],
            "mu_ci_high": summary["mean_mu_ci_high"],
            "loglog_r_squared": np.nan,
            "n_measured": sum(fit["n_measured"] for fit in fits.values()),
            "n_negative": sum(fit["n_negative"] for fit in fits.values()),
            "tail_count_at_x_0.01": sum(
                fit["count_at_x_min"] for fit in fits.values()
            ),
            "maximum_deleterious_magnitude": np.nan,
        }
    )
    pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)


def plot(fits, summary):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 10,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        len(EXPERIMENTS),
        len(BACKGROUNDS),
        figsize=(10.2, 9.8),
        constrained_layout=True,
    )

    for row_index, experiment in enumerate(EXPERIMENTS):
        for column_index, background in enumerate(BACKGROUNDS):
            ax = axes[row_index, column_index]
            fit = fits[(experiment, background)]
            ax.scatter(
                fit["thresholds"],
                fit["counts"],
                s=11,
                color="#31688e",
                edgecolors="none",
                label="empirical cumulative count",
            )
            ax.plot(
                fit["thresholds"],
                fit["fitted"],
                color="#b12a90",
                linewidth=1.3,
                linestyle="--",
                label="power-law fit",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="major", color="0.88", linewidth=0.6)
            ax.set_xlim(X_MIN, fit["requested_x_max"])
            ax.set_title(
                (
                    f"{experiment}:{background}   "
                    rf"$\mu={fit['mu']:.2f}$, "
                    rf"$x_{{\max}}={fit['requested_x_max']:.2f}$"
                ),
                loc="left",
            )
            ax.text(
                0.98,
                0.06,
                (
                    f"N({fit['requested_x_max']:.2f})="
                    f"{fit['endpoint_count']}, "
                    rf"$R^2={fit['r_squared']:.3f}$"
                ),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="0.35",
            )

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Deleterious magnitude $x=-s$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Raw count with $S\leq-x$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        (
            r"Ascensao deleterious-tail CCDF fits: "
            r"$N(S\leq-x)\propto x^{-\mu}$"
            "\n"
            rf"common fixed-effect slope $\mu="
            rf"{summary['common_fixed_effect_mu']:.3f}$"
        ),
        fontsize=13,
    )
    fig.savefig(FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULT_PATH)
    parser.add_argument("--sensitivity", type=Path, default=SENSITIVITY_PATH)
    parser.add_argument("--curves", type=Path, default=CURVE_PATH)
    parser.add_argument("--figure", type=Path, default=FIG_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    global RESULT_PATH, SENSITIVITY_PATH, CURVE_PATH, FIG_PATH
    args = parse_args(sys.argv[1:] if argv is None else argv)
    RESULT_PATH = args.results
    SENSITIVITY_PATH = args.sensitivity
    CURVE_PATH = args.curves
    FIG_PATH = args.figure
    for path in (RESULT_PATH, SENSITIVITY_PATH, CURVE_PATH, FIG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)

    fits, summary, sensitivity, curves = load_and_fit()
    write_results(fits, summary)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    curves.to_csv(CURVE_PATH, index=False)
    plot(fits, summary)

    print(
        "Ascensao cumulative-tail fits: "
        "N(S <= -x) = B x^(-mu), x >= 0.01."
    )
    print(
        "Upper-cutoff rule: largest of 0.15, 0.20, 0.30 with at "
        f"least {MIN_ENDPOINT_COUNT} endpoint genes; otherwise 0.15."
    )
    print(
        f"{'DFE':<8}{'xmax':>7}{'N(.01)':>8}{'N(xmax)':>9}"
        f"{'mu':>9}{'95% boot':>20}{'R2':>9}"
    )
    print("-" * 70)
    for experiment in EXPERIMENTS:
        for background in BACKGROUNDS:
            fit = fits[(experiment, background)]
            print(
                f"{experiment + ':' + background:<8}"
                f"{fit['requested_x_max']:>7.2f}"
                f"{fit['count_at_x_min']:>8d}"
                f"{fit['endpoint_count']:>9d}"
                f"{fit['mu']:>9.3f}"
                f"{'[' + format(fit['mu_ci_low'], '.3f') + ', ' + format(fit['mu_ci_high'], '.3f') + ']':>20}"
                f"{fit['r_squared']:>9.4f}"
            )
    print(
        "\nCommon fixed-effect mu = "
        f"{summary['common_fixed_effect_mu']:.6f}"
    )
    print(
        "Equal-background mean mu = "
        f"{summary['equal_background_mean_mu']:.6f} "
        f"[{summary['mean_mu_ci_low']:.6f}, "
        f"{summary['mean_mu_ci_high']:.6f}] descriptive bootstrap"
    )
    print(f"\nSaved {RESULT_PATH}")
    print(f"Saved {SENSITIVITY_PATH}")
    print(f"Saved {CURVE_PATH}")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
