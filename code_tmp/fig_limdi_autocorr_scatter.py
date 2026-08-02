r"""Ancestor-vs-evolved scatter of knockout fitness effects for the 10 usable Limdi clones.

Two 2x5 panel grids showing the raw material behind the Limdi block of TableS1_autocorr.py.
The top row of each is the five REL606 (Ara-) descendants, the bottom row the five REL607
(Ara+) descendants -- dropping the two flagged populations leaves exactly five of each.

  fig_limdi_autocorr_scatter.pdf    Each point is one gene at (effect in the founder, effect
                                    in the 50K clone), so the Pearson r reported in the table
                                    is literally the tightness of each cloud about the diagonal.
  fig_limdi_autocorr_spearman.pdf   The same genes plotted by percentile rank instead of by
                                    effect, so the cloud's tightness about the diagonal is
                                    Spearman rho.

The two say different things, and the difference is large:  rho is 0.26-0.39 below r in every
population.  Pearson is dominated by the ~200 strongly deleterious genes, which are both the
best measured and the furthest from the origin; rank-transforming removes their leverage and
what remains is mostly the near-neutral bulk, whose reliability is only 0.3-0.6.  Neither is
wrong -- they weight the DFE differently -- but a claim about "the DFE autocorrelation" that
holds for r and not for rho is a claim about the deleterious tail, and should say so.

Note there is no disattenuated rho.  The classical correction assumes the noise is additive on
the measured scale, which the rank transform destroys, so applying r / sqrt(rel_a * rel_b) to
a rank correlation would not mean anything.  The isogenic REL606 -> REL607 control is the
honest noise reference for the Spearman figure: it is the rho this assay returns when nothing
has evolved, and it is only 0.554.

Ara-2 and Ara+4 are excluded, for the reasons recorded in TableS1_autocorr.py: sweeping
mutants bias the Ara-2 assay, and Ara+4's technical replicates are poor (its per-gene error is
0.023, about 2.5x every other population).  They are the two lowest autocorrelations in the
table, and both are measurement artefacts rather than biology.

Conventions follow TableS1_autocorr.py exactly, so the r printed in each panel reproduces the
table: genes matched on metadata row index, a pair kept only if both sides are above
NONLETHAL_CUT = -0.3, Pearson r over that set, and r_corr the same r disattenuated for
measurement error by the classical formula r / sqrt(rel_anc * rel_evo), with the reliability
of a side being (V - mean(sigma^2)) / V.

The isogenic REL606 -> REL607 control is not one of the ten panels -- no evolution separates
those two backgrounds -- but its correlation is quoted in each figure title, since it is the
empirical ceiling this assay can reach and the right thing to read the ten panels against.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, rankdata, spearmanr

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
color = sns.color_palette('CMRmap', 5)
PT_COLOR = color[1]
DIAG_COLOR = color[2]

# ─────────────────────────────────── Parameters ──────────────────────────────────
NONLETHAL_CUT = -0.3        # same cut as TableS1_autocorr.py
EXCLUDED = ("Ara-2", "Ara+4")
NROWS, NCOLS = 2, 5
AXIS_LIM = (-0.32, 0.14)    # covers every kept pair in the ten panels

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")
OUT_PEARSON = os.path.join(OUT_DIR, "fig_limdi_autocorr_scatter.pdf")
OUT_SPEARMAN = os.path.join(OUT_DIR, "fig_limdi_autocorr_spearman.pdf")


def limdi_pair(early, late):
    """Matched (effects, effects, sigma, sigma) for two Limdi populations, over the kept range.

    Genes are matched on metadata row index -- the shared gene identity across the Limdi
    matrices; see the block comment in cmn/cmn_exper.py for why the labelled CSV must not be
    used for this.
    """
    a_eff, a_sig = cmn_exper.limdi_gene_series(early, errors=True)
    b_eff, b_sig = cmn_exper.limdi_gene_series(late, errors=True)
    idx = a_eff.index.intersection(b_eff.index)
    a, b = a_eff[idx].to_numpy(float), b_eff[idx].to_numpy(float)
    sa, sb = a_sig[idx].to_numpy(float), b_sig[idx].to_numpy(float)
    m = (a > NONLETHAL_CUT) & (b > NONLETHAL_CUT)
    return a[m], b[m], sa[m], sb[m]


def autocorr(a, b, sig_a, sig_b):
    """``(r, r_disattenuated, n)`` -- the two numbers TableS1_autocorr.py reports.

    The reliability of a side is the fraction of its observed variance that is real signal,
    ``(V - mean(sigma^2)) / V``, and ``r_true = r_obs / sqrt(rel_a * rel_b)``.
    """
    r, _ = pearsonr(a, b)
    rel = [(v.var() - np.mean(s ** 2)) / v.var() for v, s in ((a, sig_a), (b, sig_b))]
    r_corr = r / np.sqrt(rel[0] * rel[1]) if min(rel) > 0.0 else np.nan
    return float(r), float(r_corr), a.size


def panel(ax, ancestor, evolved, rank=False):
    """One ancestor-vs-evolved scatter -- by effect, or by percentile rank. Returns the count.

    With ``rank=True`` both axes carry the gene's percentile within its own background, so the
    cloud's tightness about the diagonal is Spearman rho rather than Pearson r.  Percentiles
    rather than raw ranks, so the axes do not depend on how many genes the population happens
    to have measured.
    """
    a, b, sa, sb = limdi_pair(ancestor, evolved)
    r, r_corr, n = autocorr(a, b, sa, sb)
    rho = float(spearmanr(a, b).statistic)

    if rank:
        x, y = rankdata(a) / n, rankdata(b) / n
        lo, hi = -0.02, 1.02
        zero_lines = ()
    else:
        x, y = a, b
        lo, hi = AXIS_LIM
        zero_lines = (0.0,)

    for z in zero_lines:
        ax.axhline(z, color="grey", lw=0.8, ls="--", zorder=1)
        ax.axvline(z, color="grey", lw=0.8, ls="--", zorder=1)
    ax.plot([lo, hi], [lo, hi], color=DIAG_COLOR, lw=1.2, ls="-", zorder=2, label="1:1")
    # Small, faint markers: ~3200 points, and in the effect panels most of them are piled
    # within +-0.02 of the origin, so anything heavier saturates the bulk into a solid disc.
    # Ranks are uniform by construction and cannot saturate, so those panels take more ink.
    ax.scatter(x, y, s=2.5, color=PT_COLOR, alpha=0.24 if rank else 0.16, linewidths=0,
               rasterized=True, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(f"{ancestor} $\\rightarrow$ {evolved}", pad=6)
    # Each figure leads with its own statistic and carries the other for comparison.  There is
    # deliberately no disattenuated rho -- see the module docstring.
    stats = (f"$\\rho = {rho:.3f}$\n$r = {r:.3f}$\n$n = {n}$" if rank else
             f"$r = {r:.3f}$\n$r_{{corr}} = {r_corr:.3f}$\n$n = {n}$")
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, va="top", ha="left", fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=7, width=1.5)
    return n


def build(grid, rank, out_path):
    """Render one 2x5 grid -- by effect (``rank=False``) or by percentile rank."""
    fig = plt.figure(figsize=(23, 10))
    gs = GridSpec(NROWS, NCOLS, figure=fig, wspace=0.25, hspace=0.3, top=0.99)

    for row, (ancestor, pops) in enumerate(zip(cmn_exper.LIMDI_ANCESTORS, grid)):
        for col, evolved in enumerate(pops):
            ax = fig.add_subplot(gs[row, col])
            panel(ax, ancestor, evolved, rank=rank)
            if row == NROWS - 1:
                ax.set_xlabel('Rank in ancestor' if rank else r'Effect in ancestor $(s)$')
            if col == 0:
                ax.set_ylabel('Rank in evolved clone' if rank
                              else r'Effect in evolved clone $(s)$')

    # The isogenic control is the ceiling the ten panels should be read against.
    a, b, sa, sb = limdi_pair(*cmn_exper.LIMDI_ANCESTORS)
    r, r_corr, _ = autocorr(a, b, sa, sb)
    ctrl = (f"$\\rho = {spearmanr(a, b).statistic:.3f}$" if rank else
            f"$r = {r:.3f}$, $r_{{corr}} = {r_corr:.3f}$")
    fig.suptitle(
        f"Knockout fitness effects by {'rank' if rank else 'effect'}, founder vs 50K clone\n"
        f"isogenic control REL606 $\\rightarrow$ REL607: {ctrl}",
        fontsize=17, y=1.06)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    # Five REL606 descendants across the top row, five REL607 descendants across the bottom.
    grid = [[p for p in cmn_exper.LIMDI_EVOLVED[anc] if p not in EXCLUDED]
            for anc in cmn_exper.LIMDI_ANCESTORS]
    assert all(len(g) == NCOLS for g in grid), "expected 5 usable populations per ancestor"

    print(f'{"pair":22s} {"n":>6s} {"pearson":>8s} {"spearman":>9s}')
    for ancestor, pops in zip(cmn_exper.LIMDI_ANCESTORS, grid):
        for evolved in pops:
            a, b, sa, sb = limdi_pair(ancestor, evolved)
            r, _, n = autocorr(a, b, sa, sb)
            print(f'{ancestor + " -> " + evolved:22s} {n:6d} {r:8.3f} '
                  f'{spearmanr(a, b).statistic:9.3f}')
    print()

    build(grid, rank=False, out_path=OUT_PEARSON)
    build(grid, rank=True, out_path=OUT_SPEARMAN)


if __name__ == "__main__":
    main()
