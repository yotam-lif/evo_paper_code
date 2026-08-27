"""Supplement to fig. 4 (peak-DFE floor inference). Two panels:

  A. Pseudogap exponent theta vs the per-model control parameter (K / p / n),
     for all three models overlaid. theta is the near-zero rise exponent in
     p(u;N) = c N^{-alpha} + d u^theta; alpha itself is the inset of fig. 4.
     Every point is a free-theta posterior median with a 95% credible
     interval, one colour per model (CMRmap).
  B. FGM radial decay exponent gamma vs phenotype dimension n. The endpoint
     radius r = |z*| shrinks with the number of loci m as r ~ m^{-gamma}; gamma
     is fit per n and shown with its 95% bootstrap credible interval. r->0
     with system size is the mechanism that closes the FGM off-optimum floor
     (see the "FGM caveat" in cmn/cmn_bayes.py); gamma is that rate.

Values are precomputed by cmn/cmn_bayes.py:
  python cmn/cmn_bayes.py theta   -> data/floor_theta_by_param.json
  python cmn/cmn_bayes.py radius  -> data/fgm_radius_by_n.json
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, NullFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(REPO_DIR, "data")
os.chdir(SCRIPT_DIR)

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

THETA_PATH = os.path.join(DATA, "floor_theta_by_param.json")
RADIUS_PATH = os.path.join(DATA, "fgm_radius_by_n.json")

# Per-model styling for the overlaid theta panel: all dots, one CMRmap colour
# each (mid-tone samples; the near-black/near-white ends are skipped).
_CMR = sns.color_palette("CMRmap", 3)
MODEL_STYLE = {
    "NK":    dict(label="NK ($K$)",       color=_CMR[0], marker="o", dx=1.06),
    "PSPIN": dict(label="$p$-spin ($p$)", color=_CMR[1], marker="o", dx=1.00),
    "FGM":   dict(label="FGM ($n$)",      color=_CMR[2], marker="o", dx=0.94),
}
# Panel B (FGM-only) uses a single colour, matching FGM in panel A.
GAMMA_COLOR = "grey"


def style_axis(ax, label):
    ax.text(-0.12, 1.04, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")


# ── Panel A: theta vs control parameter, three models overlaid ─────────────────
def theta_panel(ax):
    data = json.load(open(THETA_PATH))

    for model in ("NK", "PSPIN", "FGM"):
        st = MODEL_STYLE[model]
        params = sorted(int(p) for p in data.get(model, {}))
        xs, meds, los, his = [], [], [], []
        for p in params:
            med, lo, hi = data[model][str(p)]
            xs.append(p * st["dx"]); meds.append(med); los.append(lo); his.append(hi)
        if not xs:
            continue
        xs = np.asarray(xs); meds = np.asarray(meds)
        yerr = np.vstack([meds - np.asarray(los), np.asarray(his) - meds])
        ax.plot(xs, meds, "-", color=st["color"], lw=1.0, alpha=0.5, zorder=1)
        ax.errorbar(xs, meds, yerr=yerr, fmt=st["marker"], ms=8, color=st["color"],
                    mec="k", mew=0.6, ecolor=st["color"], elinewidth=1.6,
                    capsize=3.5, label=st["label"], zorder=3)

    ax.set_xscale("log", base=2)
    ticks = [2, 3, 4, 8, 16, 32]
    ax.set_xlim(1.7, 38)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Control parameter ($K/p/n$)")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(r"Small-$\Delta$ exponent")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)


# ── Panel B: FGM radial decay exponent gamma vs phenotype dimension n ──────────
def gamma_panel(ax):
    data = json.load(open(RADIUS_PATH))

    ns = sorted(int(n) for n in data)
    meds = np.array([data[str(n)]["gamma"][0] for n in ns])
    los = np.array([data[str(n)]["gamma"][1] for n in ns])
    his = np.array([data[str(n)]["gamma"][2] for n in ns])
    yerr = np.vstack([meds - los, his - meds])

    # Thin connector, then all points in a single colour (matches FGM in panel A).
    ax.plot(ns, meds, "-", color=GAMMA_COLOR, lw=1.0, alpha=0.5, zorder=1)
    ax.errorbar(ns, meds, yerr=yerr, fmt="o", ms=9, color=GAMMA_COLOR,
                mec="k", mew=0.7, ecolor=GAMMA_COLOR, elinewidth=1.8,
                capsize=4, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xlim(ns[0] / 1.4, ns[-1] * 1.4)
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\gamma_n$")
    ax.set_title(r"FGM: $r_{\text{final}} \sim m^{-\gamma_n}$", fontsize=16, pad=8)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.subplots_adjust(wspace=0.28)

    theta_panel(axA)
    style_axis(axA, "A")

    gamma_panel(axB)
    style_axis(axB, "B")

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS12_peak_dfe_bayes.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
