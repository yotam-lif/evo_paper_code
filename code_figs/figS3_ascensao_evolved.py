r"""Figure S3: ancestor-to-evolved Ascensao scatters -- fig1 E in a second dataset.

Each panel pairs a knockout's effect in the REL606 ancestor against its effect in one of the
two diversified Ara-2 ecotypes, both measured in the same condition, so the only difference
between the two axes is 6.5K generations of evolution:

    A  REL606 -> S, acetate (GHI)     B  REL606 -> S, DM25 (SLR)
    C  REL606 -> L, DM27.8 (MNO)      D  REL606 -> L, DM25 exp. (PQT)

Both evolved ecotypes appear, each in two conditions, so the loss of correlation is not a
property of one lineage or one medium.  Read against fig S4, whose panels are single strains
fit twice in the same experiment with no evolution in between, the drop from r ~ 0.90-0.95
there to what these panels show is epistasis rather than assay noise.  The experiment code in
each title identifies the growth regime; the five monoculture regimes are documented in
``cmn/cmn_exper.py`` under ``ASENCAO_MONO``.

Panels are drawn by the same code as fig1 row 2 (``cmn/cmn_scatter.py``).  The partition drops the largest 2% of
|ancestral effect| rather than fig1 E's 10%: the Ascensao DFEs are an order of magnitude more
compact than Limdi's -- their 5th percentile already sits inside the bulk -- so anything deeper
would reach past the large-effect points.  The exclusion is defined only from the x (ancestral) measurement and never from y.

Rows are keyed on ``gene_ID``, never on row position -- gene sets differ substantially between
strains, and L is covered in ~700 fewer genes than R -- so the two indices are intersected per
panel.  Nothing is clipped: each panel's limits are the envelope of its own data, which is why
the four differ.  D is the widest because REL606 carries a single, tightly measured knockout at
s = +0.50 (ECB_01645, sigma = 0.0014) that reads +0.004 in L.

Run from anywhere:  python code_figs/figS3_ascensao_evolved.py
Output:             figs_paper/figS3_ascensao_evolved.pdf
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_scatter import SHALLOW_MAGNITUDE_EXCLUSIONS, envelope_limits  # noqa: E402

cmn_scatter.apply_style()

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# Short condition tag per experiment folder; the full regime lives in cmn_exper.ASENCAO_MONO.
REGIME = {"SLR": "DM25", "GHI": "acetate", "MNO": "DM27.8", "PQT": "DM25 exp."}

# (ancestor letter, evolved letter) per panel.  Letters are unique across the release and each
# pair is one experiment, so both members share a condition: I/G = GHI, R/S = SLR, O/N = MNO,
# T/Q = PQT.
TRANSITIONS = (("I", "G"), ("R", "S"), ("O", "N"), ("T", "Q"))

# The Ascensao core is an order of magnitude tighter than Limdi's, so the inset zooms harder.
ASENCAO_INSET_LIMITS = (-0.03, 0.03)


def transition_pair(ancestor_letter, evolved_letter):
    """Matched combined fits for one ancestor -> evolved pair, intersected on gene_ID."""
    anc = cmn_exper.asencao_mono_series(ancestor_letter)
    evo = cmn_exper.asencao_mono_series(evolved_letter)
    shared = anc.index.intersection(evo.index)
    x = anc.loc[shared].to_numpy(float)
    y = evo.loc[shared].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def main():
    panels = []
    for ancestor_letter, evolved_letter in TRANSITIONS:
        folder, ancestor = cmn_exper.ASENCAO_MONO[ancestor_letter][:2]
        evolved = cmn_exper.ASENCAO_MONO[evolved_letter][1]
        x, y = transition_pair(ancestor_letter, evolved_letter)
        panels.append({
            "name": f"{ancestor} -> {evolved} ({folder})", "x": x, "y": y,
            "title": (f"{ancestor} " + r"$\rightarrow$ "
                      + f"{evolved}, {REGIME[folder]} ({folder})"),
            "xlabel": r"Ancestral effect $(s)$",
            "ylabel": r"Evolved effect $(s)$",
            "limits": envelope_limits(x, y),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5))
    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    ladders = cmn_scatter.draw_panel_grid(
        axes, panels, exclusions=SHALLOW_MAGNITUDE_EXCLUSIONS,
        inset_limits=ASENCAO_INSET_LIMITS)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "figS3_ascensao_evolved.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for (name, results), panel in zip(ladders, panels):
        cmn_scatter.print_correlations(name, results)
        print(f"{'':30s} limits {panel['limits']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
