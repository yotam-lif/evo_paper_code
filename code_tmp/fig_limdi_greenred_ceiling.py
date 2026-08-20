r"""The Limdi assay's noise ceiling, from the Green/Red reference replicates.

Each Limdi library is assayed against two fluorescent references, a green- and a red-marked
competitor, giving two independent fitness estimates per gene from the *same* library -- the
last axis of ``fitness_corrected_genes.npy``.  The main analysis (TableS1_couce_autocorr.py,
fig1_limdi_clones.py) averages them.  Correlating them against each other instead measures the
pure technical noise of the assay: there is no genetic difference between a clone's green and
red measurement, so their correlation is the highest this assay can report for two things that
are genuinely identical.

This figure exists to answer a specific worry: the REL606 -> REL607 "control" correlates at
only r = 0.872, which looks low for two backgrounds that differ by just the Ara marker.  The
top row shows why that is fine.  Green vs red within one clone -- zero genetic difference at
all -- already lands at r ~ 0.88-0.91, so 0.872 for the marker control is the noise floor, not
scrambling; consistently, it disattenuates back to 0.942.  The bottom row shows the 50K
evolved clones, whose correlation drops well below that floor: that decorrelation is real.

  Top row     REL606 green-vs-red, REL607 green-vs-red, and the REL606 -> REL607 marker control
              -- three comparisons of genotypes that are identical or nearly so.  The first two
              are the raw single-channel replicate; the third uses the two-channel average, as
              the analysis does, and so carries its disattenuated value too.
  Bottom row  The lowest, median and highest of the ten usable evolved clones (Ara+6, Ara-4,
              Ara-5), on the same axes.

The green-vs-red correlation is also a model-free check on the disattenuation the table relies
on.  Averaging two channels each correlating at 0.88 gives a reliability of 2*0.88/1.88 = 0.94,
which matches the 0.92 the reported errors imply -- so the classical correction rests on a
measurement, not just on trusting the error bars.

The 0/1 order of the last array axis is taken as green/red; the correlation is symmetric in the
two, so the panel r does not depend on which is which.  Genes are matched on metadata row index
and a pair is kept only where both sides exceed NONLETHAL_CUT = -0.3, exactly as in
TableS1_couce_autocorr.py, so the evolved-panel r values reproduce that table.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper  # noqa: E402

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
color = sns.color_palette('CMRmap', 5)
CEIL_COLOR = color[1]       # same-genotype comparisons (the ceiling)
EVO_COLOR = color[2]        # evolved comparisons
DIAG_COLOR = "black"

# ─────────────────────────────────── Parameters ──────────────────────────────────
NONLETHAL_CUT = -0.3
AXIS_LIM = (-0.32, 0.14)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")
OUT_PATH = os.path.join(OUT_DIR, "fig_limdi_greenred_ceiling.pdf")

_FIT, _ERR, _NAMES = cmn_exper.load_limdi_arrays()


def channel_pair(pop):
    """Green vs red single-channel effects for one clone, over the kept range.

    The two independent reference assays of the *same* library -- their correlation is the pure
    technical-replicate ceiling, with no genetic difference between the two sides.
    """
    k = cmn_exper.LIMDI_LIBRARIES.index(pop)
    g, r = _FIT[:, k, 0], _FIT[:, k, 1]
    keep = (g > cmn_exper.LIMDI_MISSING) & (r > cmn_exper.LIMDI_MISSING)
    g, r = g[keep], r[keep]
    m = (g > NONLETHAL_CUT) & (r > NONLETHAL_CUT)
    return g[m], r[m]


def clone_pair(early, late, errors=False):
    """Two-channel-averaged effects for two clones, matched on gene row index, over the range.

    This is what TableS1_couce_autocorr.py and fig1_limdi_clones.py use, so its r reproduces the
    table.  With ``errors`` it also returns the paired 1-sigma errors for disattenuation.
    """
    a_eff, a_sig = cmn_exper.limdi_gene_series(early, errors=True)
    b_eff, b_sig = cmn_exper.limdi_gene_series(late, errors=True)
    idx = a_eff.index.intersection(b_eff.index)
    a, b = a_eff[idx].to_numpy(float), b_eff[idx].to_numpy(float)
    m = (a > NONLETHAL_CUT) & (b > NONLETHAL_CUT)
    if not errors:
        return a[m], b[m]
    return a[m], b[m], a_sig[idx].to_numpy(float)[m], b_sig[idx].to_numpy(float)[m]


def disattenuate(r, a, b, sig_a, sig_b):
    """Classical correction: r / sqrt(rel_a * rel_b), reliability (V - mean sig^2)/V."""
    rel = [(v.var() - np.mean(s ** 2)) / v.var() for v, s in ((a, sig_a), (b, sig_b))]
    return r / np.sqrt(rel[0] * rel[1]) if min(rel) > 0.0 else np.nan


def scatter(ax, x, y, pt_color, title, stat_lines, xlabel, ylabel):
    lo, hi = AXIS_LIM
    ax.axhline(0.0, color="grey", lw=0.7, ls="--", zorder=1)
    ax.axvline(0.0, color="grey", lw=0.7, ls="--", zorder=1)
    ax.plot([lo, hi], [lo, hi], color=DIAG_COLOR, lw=1.0, ls="-", zorder=2)
    ax.scatter(x, y, s=2.5, color=pt_color, alpha=0.16, linewidths=0,
               rasterized=True, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, pad=6)
    ax.text(0.04, 0.96, "\n".join(stat_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=6, width=1.5)


def evolved_by_rank():
    """The lowest / median / highest of the ten usable evolved clones, by Pearson r."""
    excluded = ("Ara-2", "Ara+4")
    rs = []
    for anc in cmn_exper.LIMDI_ANCESTORS:
        for evo in cmn_exper.LIMDI_EVOLVED[anc]:
            if evo in excluded:
                continue
            a, b = clone_pair(anc, evo)
            rs.append((pearsonr(a, b)[0], anc, evo))
    rs.sort()
    return [rs[0], rs[len(rs) // 2], rs[-1]]   # lowest, median, highest


def main():
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.42, top=0.9)

    # ── Top row: same-genotype comparisons -- the noise ceiling ──────────────────
    for col, founder in enumerate(("REL606", "REL607")):
        g, r = channel_pair(founder)
        rr = pearsonr(g, r)[0]
        scatter(fig.add_subplot(gs[0, col]), g, r, CEIL_COLOR,
                f"{founder}: green vs red",
                [f"$r = {rr:.3f}$", f"$n = {g.size}$"],
                "Green-reference effect $(s)$", "Red-reference effect $(s)$")
        print(f"{founder} green/red   r={rr:.3f}  n={g.size}")

    a, b, sa, sb = clone_pair("REL606", "REL607", errors=True)
    r_marker = pearsonr(a, b)[0]
    r_marker_c = disattenuate(r_marker, a, b, sa, sb)
    scatter(fig.add_subplot(gs[0, 2]), a, b, CEIL_COLOR,
            "REL606 $\\rightarrow$ REL607 (marker)",
            [f"$r = {r_marker:.3f}$", f"$r_{{corr}} = {r_marker_c:.3f}$", f"$n = {a.size}$"],
            "Effect in REL606 $(s)$", "Effect in REL607 $(s)$")
    print(f"REL606->REL607 marker r={r_marker:.3f}  r_corr={r_marker_c:.3f}  n={a.size}")

    # ── Bottom row: evolved clones spanning the range, on the same axes ───────────
    labels = ("lowest", "median", "highest")
    for col, ((r_evo, anc, evo), lab) in enumerate(zip(evolved_by_rank(), labels)):
        a, b, sa, sb = clone_pair(anc, evo, errors=True)
        r_evo_c = disattenuate(r_evo, a, b, sa, sb)
        scatter(fig.add_subplot(gs[1, col]), a, b, EVO_COLOR,
                f"{anc} $\\rightarrow$ {evo}  ({lab} evolved)",
                [f"$r = {r_evo:.3f}$", f"$r_{{corr}} = {r_evo_c:.3f}$", f"$n = {a.size}$"],
                f"Effect in {anc} $(s)$", f"Effect in {evo} $(s)$")
        print(f"{anc}->{evo:6s} ({lab:7s}) r={r_evo:.3f}  r_corr={r_evo_c:.3f}  n={a.size}")

    fig.suptitle(
        "Same genotype sits at the assay's noise ceiling; 50K evolution falls below it\n"
        "top: identical or marker-only genotypes (green/red replicate + the control)   "
        "bottom: real 50K divergence",
        fontsize=15, y=0.99)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
