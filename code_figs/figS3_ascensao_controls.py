r"""Figure S3: within-experiment Ascensao controls -- fig1 D in a second dataset.

Fig1 D is an isogenic control built from the Limdi assay's two reference channels.  The
Ascensao release supports the same control by a different route: each strain in each
experiment was fit twice, once per biological replicate, and replicate 1 against replicate 2
is one genotype, one library and one condition with nothing between the two numbers but assay
noise.  No evolution, so whatever decorrelation these panels show is measurement error, and it
is what calibrates the ancestor-to-evolved panels of fig S4.

    A  REL606, acetate (GHI)     B  REL606, DM27.8 (MNO)
    C  S, DM25 exp. (PQT)        D  S, DM27.8 (MNO)

Two genotypes in two conditions each, so a high correlation cannot be an artefact of one
strain or one medium.  The experiment code in each title identifies the growth regime; the
five monoculture regimes -- all of them serial dilution, differing in dilution factor and
transfer interval -- are documented in ``cmn/cmn_exper.py`` under ``ASENCAO_MONO``.

Panels are drawn by the same code as fig1 row 2 (``cmn/cmn_scatter.py``).  The partition drops the largest 2% of
|ancestral effect| rather than fig1 D's 10%: the Ascensao DFEs are an order of magnitude more
compact than Limdi's -- their 5th percentile already sits inside the bulk -- so anything deeper
would reach past the large-effect points.  The exclusion is defined only from the replicate-1 measurement and never from
replicate 2.

Rows are keyed on ``gene_ID``, never on row position -- gene sets differ substantially between
strains -- so the two indices are intersected per panel.  Nothing is clipped: each panel's
limits are the envelope of its own data, which is why the four differ.

Run from anywhere:  python code_figs/figS3_ascensao_controls.py
Output:             figs_paper/figS3_ascensao_controls.pdf
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

# Ascensao monoculture strain letters, in panel order: REL606 in GHI and MNO, S in PQT and MNO.
CONTROLS = ("I", "O", "P", "M")

# The Ascensao core is an order of magnitude tighter than Limdi's, so the inset zooms harder.
ASENCAO_INSET_LIMITS = (-0.03, 0.03)


def replicate_pair(letter):
    """The two biological-replicate fits of one Ascensao monoculture, matched on gene_ID."""
    first = cmn_exper.asencao_mono_series(letter, rep=1)
    second = cmn_exper.asencao_mono_series(letter, rep=2)
    shared = first.index.intersection(second.index)
    x = first.loc[shared].to_numpy(float)
    y = second.loc[shared].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def mono_title(letter):
    folder, ecotype, _media, _description, _n_rep = cmn_exper.ASENCAO_MONO[letter]
    return f"{ecotype}, {REGIME[folder]} ({folder})"


def main():
    panels = []
    for letter in CONTROLS:
        x, y = replicate_pair(letter)
        folder, ecotype = cmn_exper.ASENCAO_MONO[letter][:2]
        panels.append({
            "name": f"{ecotype} {folder} rep1 vs rep2", "x": x, "y": y,
            "title": mono_title(letter),
            "xlabel": r"Fitness effect $(s)$, replicate 1",
            "ylabel": r"Fitness effect $(s)$, replicate 2",
            "limits": envelope_limits(x, y),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5))
    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    ladders = cmn_scatter.draw_panel_grid(
        axes, panels, exclusions=SHALLOW_MAGNITUDE_EXCLUSIONS,
        inset_limits=ASENCAO_INSET_LIMITS)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "figS3_ascensao_controls.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for (name, results), panel in zip(ladders, panels):
        cmn_scatter.print_correlations(name, results)
        print(f"{'':30s} limits {panel['limits']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
