r"""Figure S1: ancestor-to-evolved Limdi scatters -- four replicates of fig1 E.

Fig1 E pairs each knockout's effect in the REL607 ancestor against its effect in the 50K
ARA+2 clone.  This figure repeats that comparison for four further LTEE populations, two from
each ancestor, so the collapse of the correlation is not read off a single lineage:

    A  REL607 -> ARA+5      B  REL607 -> ARA+6
    C  REL606 -> ARA-1      D  REL606 -> ARA-4

Titles name the evolved population only, matching fig1 E.  The ancestor on the x axis is
therefore NOT the same in both rows -- REL607 founds the Ara+ populations and REL606 the Ara- --
so the panel key above is the record of which, and the caption should say so.  Read against fig
S2, whose panels are the same libraries measured twice with no evolution in between, the drop
from r ~ 0.95-0.98 there to what these panels show is epistasis rather than assay noise.

Panels are drawn by the same code as fig1 row 2 (``cmn/cmn_scatter.py``): the raw cloud split
into the near-neutral bulk and the excluded large-effect points, the identity line, Pearson r
over every pair and again after dropping the largest 10% of |ancestral effect|, and a hexbin
inset on the dense core.  The exclusion is defined only from the x (ancestral) measurement and
never from y, so the retained subset is not conditioned on the outcome whose correlation is
reported, and ranking on |s| keeps it symmetric about zero.

Genes are matched on metadata row index, the shared gene identity across the Limdi matrices;
see the block comment in ``cmn/cmn_exper.py`` for why the labelled CSV must not be used.  Gene
sets differ between libraries, so the two indices are intersected per panel.

Nothing is clipped: each panel is on the envelope of its own data, with x and y sharing that
envelope so the identity line is the panel diagonal.  The four therefore differ slightly --
ARA+6 alone carries a knockout at s = +0.125.

Run from anywhere:  python code_figs/figS1_limdi_evolved.py
Output:             figs_paper/figS1_limdi_evolved.pdf
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_scatter import envelope_limits  # noqa: E402

cmn_scatter.apply_style()

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# (ancestor, evolved population).  REL607 founds Ara+, REL606 founds Ara-.
TRANSITIONS = (
    ("REL607", "Ara+5"),
    ("REL607", "Ara+6"),
    ("REL606", "Ara-1"),
    ("REL606", "Ara-4"),
)


def transition_pair(ancestor, evolved):
    """Matched effects for one ancestor -> evolved pair, intersected on gene identity."""
    anc = cmn_exper.limdi_gene_series(ancestor)
    evo = cmn_exper.limdi_gene_series(evolved)
    shared = anc.index.intersection(evo.index)
    x = anc.loc[shared].to_numpy(float)
    y = evo.loc[shared].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def main():
    panels = []
    for ancestor, evolved in TRANSITIONS:
        x, y = transition_pair(ancestor, evolved)
        panels.append({
            "name": f"{ancestor} -> {evolved}", "x": x, "y": y,
            "title": f"{evolved.upper()} (LB), 50K",
            "xlabel": r"Ancestral effect $(s)$",
            "ylabel": r"Evolved effect $(s)$",
            "limits": envelope_limits(x, y),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5))
    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    ladders = cmn_scatter.draw_panel_grid(axes, panels)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "figS1_limdi_evolved.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for name, results in ladders:
        cmn_scatter.print_correlations(name, results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
