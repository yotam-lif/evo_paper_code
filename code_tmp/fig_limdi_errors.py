r"""The measurement error on the Limdi knockout fitness effects, plotted two ways.

TableS1_autocorr.py disattenuates every autocorrelation with the per-gene 1-sigma errors
(errors_genes_inv.npy).  This figure just shows those errors, so the size of the correction is
something you can see rather than take on trust.

  Left   Measurement error sigma against the fitness effect s, pooled over the ten usable evolved
         clones.  Faint points are individual genes; the line is the median sigma in effect bins,
         the band its interquartile range.  sigma is smallest (~0.005) in the dense near-neutral
         bulk and grows to ~0.02 in the deep deleterious tail.  The dashed line is the overall
         median error.
  Right  One representative clone (REL606 -> Ara-5), ancestor vs evolved, with the actual error
         bars drawn on the points -- all deleterious-tail genes plus a random sample of the bulk,
         over the faint full cloud.  This is the same scatter whose tightness is the reported r.

Read together they explain why the autocorrelation lives in the tail (see
fig_limdi_autocorr_vs_cutoff.py).  What matters is sigma *relative to the signal*, not sigma
itself.  In the tail the error bars (~0.02) are tiny next to how far the gene sits from the
origin (~0.4), so those points are placed precisely and pin the correlation.  In the bulk the
error bars (~0.005) are a large fraction of the gene's own effect (~0.02) and of the cloud's
spread, so the bulk points scatter about the diagonal mostly from noise and contribute little
real correlation.  Absolutely larger errors in the tail, but far smaller *relative* to signal.

Conventions match TableS1_autocorr.py: genes matched on metadata row index, sigma is the
inverse-variance SEM the source paper reports, and effects are the Green/Red average.  The
right panel keeps the pair only where both sides exceed NONLETHAL_CUT = -0.3, as the table does.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper  # noqa: E402

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
color = sns.color_palette("CMRmap", 5)
BULK_COLOR = color[1]
TAIL_COLOR = color[3]
MEDIAN_COLOR = color[2]

# ─────────────────────────────────── Parameters ──────────────────────────────────
EXCLUDED = ("Ara-2", "Ara+4")
NONLETHAL_CUT = -0.3
TAIL_CUT = -0.1                 # genes below this (either side) are "the deleterious tail"
BULK_SAMPLE = 180              # random bulk genes to draw error bars for, in the right panel
REP_ANCESTOR, REP_EVOLVED = "REL606", "Ara-5"   # representative clone for the right panel
AXIS_LIM = (-0.32, 0.14)
RNG = np.random.default_rng(0)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")
OUT_PATH = os.path.join(OUT_DIR, "fig_limdi_errors.pdf")


def clone_effects_errors(pop):
    """Effects and 1-sigma errors for one Limdi population, indexed by metadata gene row."""
    return cmn_exper.limdi_gene_series(pop, errors=True)


def matched_pair(early, late):
    """Matched (a, b, sig_a, sig_b) over the kept range, exactly as TableS1_autocorr.py does."""
    a_eff, a_sig = clone_effects_errors(early)
    b_eff, b_sig = clone_effects_errors(late)
    idx = a_eff.index.intersection(b_eff.index)
    a, b = a_eff[idx].to_numpy(float), b_eff[idx].to_numpy(float)
    sa, sb = a_sig[idx].to_numpy(float), b_sig[idx].to_numpy(float)
    m = (a > NONLETHAL_CUT) & (b > NONLETHAL_CUT)
    return a[m], b[m], sa[m], sb[m]


def pooled_effect_sigma():
    """Every (effect, sigma) from the ten usable evolved clones, pooled, over the kept range."""
    eff, sig = [], []
    for anc in cmn_exper.LIMDI_ANCESTORS:
        for evo in cmn_exper.LIMDI_EVOLVED[anc]:
            if evo in EXCLUDED:
                continue
            e, s = clone_effects_errors(evo)
            keep = e.to_numpy(float) > NONLETHAL_CUT
            eff.append(e.to_numpy(float)[keep])
            sig.append(s.to_numpy(float)[keep])
    return np.concatenate(eff), np.concatenate(sig)


def panel_sigma_vs_effect(ax):
    """Left panel: measurement error against effect size, pooled, with a binned-median band."""
    eff, sig = pooled_effect_sigma()
    ax.scatter(eff, sig, s=2.0, color=BULK_COLOR, alpha=0.05, linewidths=0,
               rasterized=True, zorder=2)

    edges = np.linspace(NONLETHAL_CUT, 0.12, 25)
    centers, med, q1, q3 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (eff >= lo) & (eff < hi)
        if m.sum() >= 15:
            centers.append(0.5 * (lo + hi))
            med.append(np.median(sig[m]))
            q1.append(np.percentile(sig[m], 25))
            q3.append(np.percentile(sig[m], 75))
    centers, med, q1, q3 = map(np.asarray, (centers, med, q1, q3))
    ax.fill_between(centers, q1, q3, color=MEDIAN_COLOR, alpha=0.25, zorder=3,
                    label="interquartile range")
    ax.plot(centers, med, color=MEDIAN_COLOR, lw=2.6, zorder=4, label="median $\\sigma$ in bin")

    overall = np.median(sig)
    ax.axhline(overall, color="black", lw=1.2, ls="--", zorder=5)
    ax.text(0.10, overall, f"  overall median $\\sigma = {overall:.3f}$",
            va="bottom", ha="right", fontsize=12)

    ax.set_xlim(NONLETHAL_CUT - 0.01, 0.13)
    ax.set_ylim(0.0, 0.035)
    ax.set_xlabel(r"fitness effect  $s$")
    ax.set_ylabel(r"measurement error  $\sigma$")
    ax.set_title("Error grows toward the deleterious tail\n(pooled over the ten usable clones)",
                 pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    _spines(ax)


def panel_error_bars(ax):
    """Right panel: one clone's paired effects with the real error bars drawn on the points."""
    a, b, sa, sb = matched_pair(REP_ANCESTOR, REP_EVOLVED)
    lo, hi = AXIS_LIM

    ax.axhline(0.0, color="grey", lw=0.7, ls="--", zorder=1)
    ax.axvline(0.0, color="grey", lw=0.7, ls="--", zorder=1)
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, zorder=2)
    ax.scatter(a, b, s=2.0, color="0.7", alpha=0.35, linewidths=0, rasterized=True, zorder=3)

    tail = (a < TAIL_CUT) | (b < TAIL_CUT)
    bulk = ~tail
    bulk_idx = np.where(bulk)[0]
    sample = RNG.choice(bulk_idx, size=min(BULK_SAMPLE, bulk_idx.size), replace=False)

    for sel, col, z in ((sample, BULK_COLOR, 4), (np.where(tail)[0], TAIL_COLOR, 5)):
        ax.errorbar(a[sel], b[sel], xerr=sa[sel], yerr=sb[sel], fmt="o", ms=3.0,
                    color=col, ecolor=col, elinewidth=0.9, capsize=0, alpha=0.85,
                    markeredgewidth=0, zorder=z)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"effect in {REP_ANCESTOR}  $s$")
    ax.set_ylabel(rf"effect in {REP_EVOLVED}  $s$")
    ax.set_title(f"Error bars on the paired effects\n({REP_ANCESTOR} "
                 rf"$\rightarrow$ {REP_EVOLVED}, representative)", pad=8)
    handles = [
        Line2D([0], [0], marker="o", ls="none", color=TAIL_COLOR,
               label=rf"deleterious tail ($s < {TAIL_CUT}$): {int(tail.sum())} genes"),
        Line2D([0], [0], marker="o", ls="none", color=BULK_COLOR,
               label=f"near-neutral bulk: {BULK_SAMPLE} of {int(bulk.sum())} shown"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11)
    _spines(ax)


def _spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=6, width=1.5)


def main():
    fig = plt.figure(figsize=(15, 6.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28, width_ratios=[1.15, 1.0])
    panel_sigma_vs_effect(fig.add_subplot(gs[0]))
    panel_error_bars(fig.add_subplot(gs[1]))

    fig.suptitle("Measurement error on the knockout fitness effects", fontsize=17, y=1.02)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
