r"""Figure S2: isogenic Limdi controls -- four replicates of fig1 D.

Fig1 D shows one paired-effect scatter of a Limdi library measured twice, green- against
red-reference, in a single background with zero evolution between the two numbers.  Whatever
decorrelation it shows is therefore assay noise, not epistasis, and it is what calibrates the
evolved panels beside it.  This figure repeats that control in four independent backgrounds --
the REL606 ancestor and three 50K LTEE clones -- so the noise floor behind fig S1 is not read
off a single library.

    A  REL606      B  ARA+2      C  ARA-3      D  ARA-6

Panels are drawn by the same code as fig1 row 2 (``cmn/cmn_scatter.py``): the raw cloud split
into the near-neutral bulk and the excluded large-effect points, the identity line, Pearson r
over every pair and again after dropping the largest 10% of |first measurement|, and a hexbin
inset on the dense core.  The exclusion is defined only from the x measurement and never from
y, so the retained subset is not conditioned on the outcome whose correlation is reported, and
ranking on |s| keeps it symmetric about zero.

Genes are matched on metadata row index, the shared gene identity across the Limdi matrices;
see the block comment in ``cmn/cmn_exper.py`` for why the labelled CSV must not be used.  The
two channels of one library cover exactly the same genes -- the missing-value sentinel marks a
gene absent from both at once, never from just one -- so no intersection is needed here.

Nothing is clipped: each panel is on the envelope of its own data, with x and y sharing that
envelope so the identity line is the panel diagonal.

Run from anywhere:  python code_figs/figS2_limdi_controls.py
Output:             figs_paper/figS2_limdi_controls.pdf
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

# (library, panel title).  REL606 is the Ara- founder; the other three are 50K clones.
CONTROLS = (
    ("REL606", "REL606 (LB)"),
    ("Ara+2",  "ARA+2 (LB), 50K"),
    ("Ara-3",  "ARA-3 (LB), 50K"),
    ("Ara-6",  "ARA-6 (LB), 50K"),
)


def control_pair(pop):
    """The two unaveraged technical replicates of one Limdi library, as finite arrays."""
    green, red = cmn_exper.limdi_channel_series(pop)
    x = green.to_numpy(float)
    y = red.to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def main():
    panels = []
    for pop, title in CONTROLS:
        x, y = control_pair(pop)
        panels.append({
            "name": f"{pop} green vs red", "x": x, "y": y, "title": title,
            "xlabel": r"Fitness effect $(s)$, measurement 1",
            "ylabel": r"Fitness effect $(s)$, measurement 2",
            "limits": envelope_limits(x, y),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5))
    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    ladders = cmn_scatter.draw_panel_grid(axes, panels)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "figS2_limdi_controls.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for name, results in ladders:
        cmn_scatter.print_correlations(name, results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
