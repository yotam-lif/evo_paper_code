r"""Log-density view of the truncated shared-buffer Cauchy-FGM DFE fits.

This plotting-only companion to ``figS6_cauchy_fgm_full_mle.py`` reads the saved
maximum-likelihood estimates; it does not refit the model.  All panels use common
histogram bins and axes over the fitted range s >= -0.4, so the experimentally
missing Couce tail remains visible instead of being hidden by panel-specific limits.

Outputs
-------
    figs_paper/figS6_cauchy_fgm_full_mle_log.png
    data/cauchy_fgm_full_mle_log_plot.json
"""

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    UnbinnedNoisyLikelihood,
    ancestor_dfes,
)

DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figs_paper")
FIT_PATH = os.path.join(DATA_DIR, "cauchy_fgm_full_mle.json")
FIG_PATH = os.path.join(FIG_DIR, "figS6_cauchy_fgm_full_mle_log.png")
PLOT_DATA_PATH = os.path.join(DATA_DIR, "cauchy_fgm_full_mle_log_plot.json")

UPPER_LIMIT = 0.12
BIN_WIDTH = 0.005
MODEL_DX = 0.0005
Y_LIMITS = (1.0e-2, 1.0e2)


def conditional_model_curve(fit):
    """Return the noisy fitted density conditional on the same lower cutoff."""
    # Pad the requested plotting interval so the Gaussian convolution is accurate
    # at both plot boundaries.
    support = np.array([LOWER_CUT - 0.06, UPPER_LIMIT + 0.06])
    likelihood = UnbinnedNoisyLikelihood(support, eps=MEAS_ERR, dx=MODEL_DX)
    observed_pdf = likelihood.observed_pdf_grid(
        fit["n"], fit["r"], fit["sigma"]
    )
    keep_probability = cauchy_fgm_survival(
        LOWER_CUT,
        n=fit["n"],
        sigma=fit["sigma"],
        r=fit["r"],
        eps=MEAS_ERR,
    )
    keep = (likelihood.x >= LOWER_CUT) & (likelihood.x <= UPPER_LIMIT)
    return likelihood.x[keep], observed_pdf[keep] / keep_probability


def histogram_density(effects):
    """Equal-width empirical density; omit empty bins on the logarithmic axis."""
    edges = np.arange(
        LOWER_CUT,
        UPPER_LIMIT + BIN_WIDTH * 1.0001,
        BIN_WIDTH,
    )
    counts, edges = np.histogram(effects, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (effects.size * np.diff(edges))
    nonzero = counts > 0
    return centers[nonzero], density[nonzero], counts[nonzero]


def main():
    with open(FIT_PATH, encoding="utf-8") as handle:
        fits = json.load(handle)["per_dfe"]
    effects_by_name = dict(ancestor_dfes(lower_cut=LOWER_CUT))

    plot_data = {
        "lower_cut": LOWER_CUT,
        "upper_limit": UPPER_LIMIT,
        "bin_width": BIN_WIDTH,
        "y_limits": list(Y_LIMITS),
        "panels": [],
    }

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5), sharex=True, sharey=True)

    for ax, name in zip(axes.ravel(), DISPLAY_ORDER):
        effects = effects_by_name[name]
        fit = fits[name]
        hist_x, hist_y, counts = histogram_density(effects)
        model_x, model_y = conditional_model_curve(fit)

        ax.plot(
            model_x,
            model_y,
            color="#6a00a8",
            lw=2.0,
            label="Cauchy-FGM MLE",
            zorder=2,
        )
        ax.scatter(
            hist_x,
            hist_y,
            s=12,
            facecolor="0.45",
            edgecolor="none",
            label="data (nonempty bins)",
            zorder=3,
        )
        ax.axvline(0.0, color="k", lw=0.7, ls=":", zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(LOWER_CUT, UPPER_LIMIT)
        ax.set_ylim(*Y_LIMITS)
        ax.set_title(DISPLAY_NAMES[name])
        ax.text(
            0.03,
            0.05,
            rf"$n={fit['n']:.3g},\ r={fit['r']:.3g},\ \sigma={fit['sigma']:.3g}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        plot_data["panels"].append({
            "key": name,
            "title": DISPLAY_NAMES[name],
            "N": int(effects.size),
            "n": fit["n"],
            "r": fit["r"],
            "sigma": fit["sigma"],
            "histogram": [
                [round(float(x), 6), round(float(y), 8), int(c)]
                for x, y, c in zip(hist_x, hist_y, counts)
            ],
            # 0.002-wide sampling is more than sufficient for the inline display.
            "model": [
                [round(float(x), 6), round(float(y), 8)]
                for x, y in zip(model_x[::4], model_y[::4])
            ],
        })

    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fitness effect $s$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)

    with open(PLOT_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(plot_data, handle, indent=2)
        handle.write("\n")

    print(FIG_PATH)
    print(PLOT_DATA_PATH)


if __name__ == "__main__":
    main()
