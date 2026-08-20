r"""How the Limdi DFE autocorrelation depends on where the deleterious tail is cut.

TableS1_couce_autocorr.py keeps a gene pair only where both sides exceed NONLETHAL_CUT = -0.3, then
reports the Pearson r of the surviving cloud.  That -0.3 is a single choice, and this figure
asks how much the reported autocorrelation depends on it: instead of one cut we sweep the lower
cutoff continuously, from "keep everything measured" (c ~ -0.75, nothing removed) up to c = -0.05
(the whole deleterious side removed), and recompute r at each step.

  Top     Pearson r of the ancestor-vs-evolved cloud as the lower cutoff is raised.  The ten
          usable evolved clones (five REL606/Ara- descendants, five REL607/Ara+ descendants) and
          the isogenic REL606 -> REL607 control.  The dashed vertical line marks the -0.3 used in
          the table, so each curve passes through its TableS1 value there.
  Bottom  The fraction of matched genes still kept at that cutoff -- how much data the cut costs.

Autocorrelation r measures how much the DFE is *preserved*; scrambling is the decorrelation,
1 - r.  Read that way the figure says the opposite of what a high r looks like.  The high r is
carried by the deleterious tail -- consistently-lethal knockouts sit far from the origin in both
backgrounds and dominate a Pearson correlation -- and that tail is *preserved*, not scrambled:
essential genes stay essential.  The scrambling lives in the near-neutral bulk, which is exactly
what the sweep exposes.  Strip the preserved tail (only the ~15-18% of genes on the deleterious
side) and the evolved clones fall to r ~ 0.0-0.35, while the isogenic control -- same stripping,
same noise and range-restriction, no evolution -- stays at ~0.48 and would disattenuate above 1.
That right-hand gap between the control and the evolved clones is the real bulk scrambling; the
full-range r (0.5-0.7 at -0.3) hides it because the preserved tail dominates the number.  (The
disattenuated r is not to be trusted in the bulk: reliability there is only ~0.4-0.6 and the
control's correction runs past 1.  Use the control as the empirical bulk floor instead.)

The sweep stops at -0.05 on purpose: past there the cut eats into the dense near-zero bulk, the
kept count crashes (from ~85% to ~30% of genes between -0.05 and 0), and what remains is the
winner's-curse regime of genes beneficial in both backgrounds -- a different question from tail
autocorrelation.  Conventions otherwise match TableS1_couce_autocorr.py exactly (genes matched on
metadata row index, both sides above the cutoff, Pearson r over the survivors).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

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
ARA_MINUS_COLOR = sns.color_palette("crest", 5)      # REL606 (Ara-) descendants
ARA_PLUS_COLOR = sns.color_palette("flare", 5)       # REL607 (Ara+) descendants
CTRL_COLOR = "black"

# ─────────────────────────────────── Parameters ──────────────────────────────────
EXCLUDED = ("Ara-2", "Ara+4")
ANALYSIS_CUT = -0.3                                  # the cut TableS1_couce_autocorr.py uses
CUTS = np.round(np.arange(-0.75, -0.049, 0.01), 3)  # sweep: keep-everything -> tail-removed
MIN_N = 200                                          # do not report r on fewer pairs than this

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")
OUT_PATH = os.path.join(OUT_DIR, "fig_limdi_autocorr_vs_cutoff.pdf")


def matched_effects(early, late):
    """Matched fitness-effect arrays for two Limdi populations, genes keyed on metadata row.

    No value cut is applied here -- the cut is what the sweep varies -- so the pair covers every
    gene measured in both backgrounds, down to the deepest lethal knockouts near -0.75.
    """
    a = cmn_exper.limdi_gene_series(early)
    b = cmn_exper.limdi_gene_series(late)
    idx = a.index.intersection(b.index)
    return a[idx].to_numpy(float), b[idx].to_numpy(float)


def r_vs_cut(a, b):
    """Pearson r and kept-fraction at each cutoff in CUTS (both sides above the cutoff)."""
    n_total = a.size
    r = np.full(CUTS.size, np.nan)
    frac = np.full(CUTS.size, np.nan)
    for i, c in enumerate(CUTS):
        m = (a > c) & (b > c)
        frac[i] = m.sum() / n_total
        if m.sum() >= MIN_N:
            r[i] = pearsonr(a[m], b[m])[0]
    return r, frac


def main():
    fig = plt.figure(figsize=(11, 10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.4, 1.0], hspace=0.12)
    ax_r = fig.add_subplot(gs[0])
    ax_n = fig.add_subplot(gs[1], sharex=ax_r)

    print(f'{"pair":22s} {"r(no cut)":>10s} {"r(-0.3)":>9s} {"r(-0.05)":>9s}')
    last_fracs = []

    def draw(early, late, colr, lw, ls, z):
        a, b = matched_effects(early, late)
        r, frac = r_vs_cut(a, b)
        ax_r.plot(CUTS, r, color=colr, lw=lw, ls=ls, zorder=z)
        ax_n.plot(CUTS, frac, color=colr, lw=lw, ls=ls, zorder=z)
        last_fracs.append(frac[-1])
        i03 = int(np.argmin(np.abs(CUTS - ANALYSIS_CUT)))
        print(f'{early + " -> " + late:22s} {r[0]:10.3f} {r[i03]:9.3f} {r[-1]:9.3f}')

    # Evolved clones, five per ancestor, shaded within each family.
    for anc, palette in ((cmn_exper.LIMDI_ANCESTORS[0], ARA_MINUS_COLOR),
                         (cmn_exper.LIMDI_ANCESTORS[1], ARA_PLUS_COLOR)):
        pops = [p for p in cmn_exper.LIMDI_EVOLVED[anc] if p not in EXCLUDED]
        for shade, evo in enumerate(pops):
            draw(anc, evo, palette[shade], 1.8, "-", 3)

    # Isogenic control on top, as the noise ceiling.
    draw(*cmn_exper.LIMDI_ANCESTORS, CTRL_COLOR, 2.6, "-", 5)
    frac_last = float(np.median(last_fracs))

    # ── cosmetics ────────────────────────────────────────────────────────────────
    for ax in (ax_r, ax_n):
        ax.axvline(ANALYSIS_CUT, color="grey", lw=1.2, ls="--", zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', length=6, width=1.5)

    ax_r.set_ylabel("DFE autocorrelation  (Pearson $r$)")
    ax_r.set_ylim(-0.05, 1.0)
    ax_r.axhline(0.0, color="grey", lw=0.7, ls=":", zorder=1)
    ax_r.text(ANALYSIS_CUT - 0.006, 0.98, "cut used in analysis ($-0.3$)  ",
              rotation=90, va="top", ha="right", fontsize=11, color="dimgrey")
    ax_r.annotate("", xy=(-0.06, 0.9), xytext=(-0.5, 0.9),
                  arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1.3))
    ax_r.text(-0.28, 0.915, "cutting away more of the deleterious tail",
              fontsize=11, color="dimgrey", va="bottom", ha="center")
    plt.setp(ax_r.get_xticklabels(), visible=False)

    ax_n.set_ylabel("fraction of\ngenes kept")
    ax_n.set_xlabel(r"lower cutoff on fitness effect  $s$  (keep both sides $> c$)")
    ax_n.set_ylim(0.0, 1.02)
    ax_n.set_xlim(CUTS[0] - 0.01, CUTS[-1] + 0.01)
    ax_n.text(CUTS[-1], 0.72, f"  {frac_last:.0%} kept at $-0.05$", va="top", ha="right",
              fontsize=11, color="dimgrey")

    legend = [
        Line2D([0], [0], color=CTRL_COLOR, lw=2.6,
               label=r"REL606 $\rightarrow$ REL607 (isogenic control)"),
        Line2D([0], [0], color=ARA_MINUS_COLOR[2], lw=1.8,
               label=r"REL606 (Ara$-$) descendants"),
        Line2D([0], [0], color=ARA_PLUS_COLOR[2], lw=1.8,
               label=r"REL607 (Ara$+$) descendants"),
    ]
    ax_r.legend(handles=legend, loc="lower left", frameon=False, fontsize=12)
    ax_r.set_title("The deleterious tail is preserved; the near-neutral bulk is scrambled\n"
                   "stripping the preserved tail drops the evolved clones far below the control",
                   pad=10)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
