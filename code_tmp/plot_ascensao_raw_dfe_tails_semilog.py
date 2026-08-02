r"""Plot the raw deleterious tails of all Ascensao DFEs.

This is deliberately descriptive: no model fit, smoothing, measurement-error
weighting, or normalization is applied.  Finite per-gene selection
coefficients at s <= 0.01 are placed in equal-width bins and the nonzero raw
counts are shown as points on a logarithmic y axis.  The left edge is not a
chosen cutoff: it is the smallest observed finite effect across the 12 DFEs,
rounded down to the bin grid.

Outputs
-------
    data/ascensao_raw_dfe_tails_histogram.csv
    figs_paper/ascensao_raw_dfe_tails_semilog.png
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

UPPER = 0.01
BIN_WIDTH = 0.002

CSV_PATH = REPO_DIR / "data" / "ascensao_raw_dfe_tails_histogram.csv"
FIG_PATH = REPO_DIR / "figs_paper" / "ascensao_raw_dfe_tails_semilog.png"


def load_histograms():
    arrays = {}
    for experiment in EXPERIMENTS:
        for background in BACKGROUNDS:
            values = load_asencao_array(experiment, background)
            values = np.asarray(values, float)
            arrays[(experiment, background)] = values[np.isfinite(values)]

    observed_minimum = min(values.min() for values in arrays.values())
    lower_edge = np.floor(observed_minimum / BIN_WIDTH) * BIN_WIDTH
    n_bins = int(np.ceil((UPPER - lower_edge) / BIN_WIDTH))
    edges = lower_edge + np.arange(n_bins + 1) * BIN_WIDTH
    edges[-1] = UPPER
    centers = 0.5 * (edges[:-1] + edges[1:])

    histograms = {}
    rows = []
    for experiment in EXPERIMENTS:
        for background in BACKGROUNDS:
            values = arrays[(experiment, background)]
            window = values[values <= UPPER]
            counts, _ = np.histogram(window, bins=edges)
            histograms[(experiment, background)] = {
                "counts": counts,
                "n_measured": int(values.size),
                "n_window": int(window.size),
                "minimum": float(values.min()),
            }
            rows.extend(
                {
                    "experiment": experiment,
                    "background": background,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_center": float(center),
                    "count": int(count),
                    "n_measured": int(values.size),
                    "n_at_or_below_0.01": int(window.size),
                    "minimum_observed_s": float(values.min()),
                }
                for left, right, center, count in zip(
                    edges[:-1], edges[1:], centers, counts
                )
            )

    return edges, histograms, pd.DataFrame(rows), float(observed_minimum)


def plot(edges, histograms, observed_minimum):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        len(EXPERIMENTS),
        len(BACKGROUNDS),
        figsize=(10.2, 9.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    all_positive = np.concatenate(
        [
            spec["counts"][spec["counts"] > 0]
            for spec in histograms.values()
        ]
    )
    y_max = 10 ** np.ceil(np.log10(all_positive.max()))
    centers = 0.5 * (edges[:-1] + edges[1:])

    for row_index, experiment in enumerate(EXPERIMENTS):
        for column_index, background in enumerate(BACKGROUNDS):
            ax = axes[row_index, column_index]
            spec = histograms[(experiment, background)]
            positive = spec["counts"] > 0
            ax.scatter(
                centers[positive],
                spec["counts"][positive],
                s=11,
                color="#355f8d",
                edgecolors="none",
            )
            ax.axvline(0.0, color="0.55", linewidth=0.8)
            ax.set_yscale("log")
            ax.set_xlim(edges[0], UPPER)
            ax.set_ylim(0.8, y_max)
            ax.grid(axis="y", which="major", color="0.88", linewidth=0.6)
            ax.set_title(
                f"{experiment}: {BACKGROUND_LABELS[background]}",
                loc="left",
                fontweight="bold",
            )
            ax.text(
                0.98,
                0.94,
                (
                    f"{spec['n_window']:,} / {spec['n_measured']:,} genes\n"
                    rf"$s_{{\min}}={spec['minimum']:.3f}$"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="0.35",
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Selection coefficient, s")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Raw genes per {BIN_WIDTH:g}-wide bin")

    fig.suptitle(
        (
            "Raw Ascensao DFE tails: equal-width counts, no normalization\n"
            rf"all observed effects with $s\leq {UPPER:g}$; "
            rf"global observed minimum $s={observed_minimum:.3f}$"
        ),
        fontsize=13,
    )
    fig.savefig(FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--figure", type=Path, default=FIG_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    global CSV_PATH, FIG_PATH
    args = parse_args(sys.argv[1:] if argv is None else argv)
    CSV_PATH = args.csv
    FIG_PATH = args.figure
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    edges, histograms, table, observed_minimum = load_histograms()
    table.to_csv(CSV_PATH, index=False)
    plot(edges, histograms, observed_minimum)

    print(
        "Raw Ascensao DFE tails; no fit, normalization, or lower selection "
        "coefficient cutoff."
    )
    print(
        f"Displayed s <= {UPPER:g} in {BIN_WIDTH:g}-wide bins; "
        f"minimum observed s = {observed_minimum:.6g}."
    )
    for experiment in EXPERIMENTS:
        for background in BACKGROUNDS:
            spec = histograms[(experiment, background)]
            print(
                f"{experiment}:{background}  "
                f"{spec['n_window']:4d}/{spec['n_measured']:4d} genes  "
                f"min={spec['minimum']:.6g}"
            )
    print(f"Saved {CSV_PATH}")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
