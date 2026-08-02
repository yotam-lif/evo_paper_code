r"""Per-clone version of fig1 for the 12 Limdi LTEE populations -- one figure per clone.

Same three panels as ``code_figs/fig1_scrambling_exper_res.py``, with the Couce Ara+2
timepoint pair (0K -> 2K, 2K -> 15K) replaced by an ancestor -> evolved pair from the Limdi
TnSeq panel.  Each population's founder is its own ancestor: REL606 for the six Ara- lines
and REL607 for the six Ara+ lines.  Twelve figures are written to ``out_tmp/``.

  A  Beneficial knockouts at one timepoint, with an arrow to the same knockout's effect at
     the other -- forward (grey, anchored on the ancestor) and backward (orange, anchored on
     the evolved clone).
  B  Forward: the ancestor's beneficial DFE (grey) and where those same knockouts land in the
     evolved background (orange), against the evolved clone's full DFE (line).
  C  Backward: the evolved clone's beneficial DFE (orange) and where those same knockouts sat
     in the ancestor (grey), against the ancestor's full DFE (line).

Genes are matched on metadata row index, which is the shared gene identity across the Limdi
matrices -- see the block comment in ``cmn/cmn_exper.py`` for why the labelled CSV must not
be used for this.

Everything is drawn on fig1's settings, including its +-0.06 window, with two changes forced
by the Limdi DFE having a real deleterious tail where the Couce one has none:

  NONLETHAL_CUT   The Couce library has no lethal tail (its deepest segment is -0.23); Limdi
                  measures knockouts down to -0.75.  Pairs where either side is below -0.3 are
                  dropped, the same cut TableS1_autocorr.py uses, so these figures show the
                  same range of effects the reported autocorrelations are computed on.
  dfe_histogram   B and C are clipped to the plotted window before binning, and fig1's
                  sparse-bin threshold is rescaled to sample size -- see that function.

Coverage of the +-0.06 window is therefore lower here than in fig1: 93% of Limdi pairs against
98.3% of Couce's.  What falls outside is almost entirely deleterious -- 99.5% of the
beneficial knockouts these panels are about are inside it -- and panel A is not clipped at
all, so the full paired range including that tail is visible there.

Two populations carry the quality flags recorded in TableS1_autocorr.py and are labelled as
such in the figure: Ara-2 (sweeping mutants bias the assay) and Ara+4 (poor technical
replicates).  They are plotted anyway, since the point of a per-clone panel is to see them.

"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper  # noqa: E402

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14
color = sns.color_palette('CMRmap', 5)
EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.5)
ANC_FILL = (0.5, 0.5, 0.5, 0.15)
DFE_FILL = color[2]

# ─────────────────────────────────── Parameters ──────────────────────────────────
XLIM = 0.06                 # half-width of the plotted fitness-effect window (fig1's)
SHIFT_FRAC = 0.025          # sideways offset between the paired histograms
UPPER_BEN_LIMIT = 0.3       # arrows are drawn for LOWER < s < UPPER ...
LOWER_BEN_LIMIT = 0.005     # ... same cut as fig1 (median measurement error is 0.008)
NONLETHAL_CUT = -0.3        # drop a pair if either side is below this

# Quality flags from code_figs/TableS1_autocorr.py.
FLAGGED = {"Ara-2": "sweeping mutants bias assay", "Ara+4": "poor technical replicates"}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")


def load_pair(ancestor, evolved):
    """Matched (ancestor, evolved) knockout-effect arrays for one population.

    Genes are matched on metadata row index -- the shared gene identity across the Limdi
    matrices -- and a pair is kept only if both sides are above NONLETHAL_CUT.
    """
    anc = cmn_exper.limdi_gene_series(ancestor)
    evo = cmn_exper.limdi_gene_series(evolved)
    shared = anc.index.intersection(evo.index)
    a = anc.loc[shared].to_numpy(float)
    e = evo.loc[shared].to_numpy(float)
    keep = (a > NONLETHAL_CUT) & (e > NONLETHAL_CUT)
    return a[keep], e[keep]


def dfe_histogram(data, final_bins):
    """fig1's thresholded histogram, restricted to the plotted window.

    ``thresholded_histogram`` drops bins holding fewer than ``threshold`` points and then
    re-bins what survives, so the threshold sets both the trimming and the visible resolution:
    the final bins are spread over the surviving range, not over the frame.  On the Couce data
    those two jobs coincide -- fig1's hand-set thresholds trim its histograms to [-0.07, +0.03],
    which is its +-0.06 frame, so the bins fill it.  They do not coincide here.  Limdi's
    deleterious tail is real and densely populated all the way down to -0.3, so no count
    threshold trims it; the bins would be spread over five times the frame and only the central
    fifth would be visible.  Raising the threshold until the range fits the frame is circular,
    since the frame would then be setting its own contents.

    So the two jobs are separated: clip to the window first (what is plotted decides what is
    plotted), then apply fig1's sparse-bin cut inside it (removing under-populated bins).  The
    threshold is rescaled to sample size as n/1500, which reproduces every one of fig1's
    hand-set values -- 3, 3, 6 and 2, 3, 8 -- to within 2, and follows Limdi's ~3200 pairs down
    from Couce's ~9000.

    Clipping then costs one thing, which the renormalisation below undoes.  ``density=True``
    divides each curve by the points it kept, and curves in the same panel keep different
    fractions: in panel B the ancestor's beneficial histogram loses 0.1% to the window while
    the full evolved DFE loses 6%, so the two would be scaled differently and their relative
    heights would be an artefact of the frame.  Rescaling by (kept / full sample) makes every
    curve a density over its whole sample instead, so heights stay comparable and the clipped
    mass simply shows up as the curve not integrating to 1 inside the frame.  This is the one
    place the figures depart from fig1, where the same effect is present but unlabelled.
    """
    inside = data[np.abs(data) <= XLIM]
    counts, edges, kept = thresholded_histogram(
        inside, max(2, int(round(inside.size / 1500.0))), final_bins)
    return counts * (kept.size / float(data.size)), edges, kept


def thresholded_histogram(data, threshold, final_bins):
    # Step 1: Use many initial bins to capture fine structure
    init_bins = 10 * final_bins
    counts, bin_edges = np.histogram(data, bins=init_bins)

    # Step 2: Mask bins below threshold
    valid_indices = counts >= threshold
    valid_data = []
    for i, keep in enumerate(valid_indices):
        if keep:
            # Get data in that bin
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
            valid_data.append(data[bin_mask])
    if not valid_data:
        raise ValueError("No bins passed the threshold.")

    # Concatenate all valid data
    cleaned_data = np.concatenate(valid_data)

    # Step 3: Create final histogram with desired number of bins
    final_counts, final_edges = np.histogram(cleaned_data, bins=final_bins, density=True)

    return final_counts, final_edges, cleaned_data


def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo):
    # Vertical shift for the "evolved" histograms
    z_frac = 0.1
    lw_main = 1.0
    valid_indices = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid_indices]
    dfe_evo = dfe_evo[valid_indices]

    def draw_custom_segments(ax, _xlim, _ylim):
        z = _ylim * z_frac * 1.1
        ax.plot([-_xlim * 0.9, _xlim * 0.9], [z, z],
                linestyle="--", color="grey", lw=lw_main)
        segs = [
            ((-_xlim, -0.75), (-_xlim * 0.9, z)),
            ((_xlim, -0.75), (_xlim * 0.9, z)),
            ((-_xlim / 2, -0.75), (-_xlim / 2 * 0.9, z)),
            ((_xlim / 2, -0.75), (_xlim / 2 * 0.9, z)),
            ((0, -0.75), (0, z))
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]

    bdfe_anc_inds = np.where(dfe_anc > 0)
    bdfe_evo_inds = np.where(dfe_evo > 0)

    prop_bdfe_anc = dfe_evo[bdfe_anc_inds]
    prop_bdfe_evo = dfe_anc[bdfe_evo_inds]

    # Left Panel - Forward propagate
    counts, bin_edges, _ = dfe_histogram(prop_bdfe_anc, 25)
    anc_counts, anc_bin_edges, _ = dfe_histogram(bdfe_anc, 20)
    dfe_counts, dfe_bin_edges, _ = dfe_histogram(dfe_evo, 30)
    bin_edges = bin_edges - XLIM * SHIFT_FRAC
    dfe_bin_edges = dfe_bin_edges - XLIM * SHIFT_FRAC
    anc_bin_edges = anc_bin_edges + XLIM * SHIFT_FRAC
    ymax = max(np.max(counts), np.max(anc_counts), np.max(dfe_counts))
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts_shifted = counts + z
    dfe_counts_shifted = dfe_counts + z
    ax_left.set_xlim(-XLIM, XLIM)
    ax_left.tick_params(labelsize=16)
    ax_left.set_ylim(0, ylim * 1.15)

    ax_left.stairs(
        values=counts_shifted,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evo."
    )

    ax_left.stairs(
        values=dfe_counts_shifted,
        edges=dfe_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=DFE_FILL,
        lw=1.1,
        label="DFE Evo."
    )

    rect = Rectangle((-XLIM, 0), 2 * XLIM, z, facecolor="white", edgecolor="none")
    ax_left.add_patch(rect)
    draw_custom_segments(ax_left, XLIM, ylim)

    ax_left.stairs(
        values=anc_counts,
        edges=anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Anc."
    )
    ax_left.legend(frameon=False)
    ax_left.set_xlabel(r'Fitness effect $(s)$')

    # Right Panel
    counts2, bin_edges2, _ = dfe_histogram(bdfe_evo, 12)
    anc2_counts, anc2_bin_edges, _ = dfe_histogram(prop_bdfe_evo, 22)
    dfe2_counts, dfe2_bin_edges, _ = dfe_histogram(dfe_anc, 24)
    bin_edges2 = bin_edges2 + XLIM * SHIFT_FRAC
    dfe2_bin_edges = dfe2_bin_edges - XLIM * SHIFT_FRAC
    anc2_bin_edges = anc2_bin_edges - XLIM * SHIFT_FRAC
    ymax = max(np.max(counts2), np.max(anc2_counts), np.max(dfe2_counts))
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts2_shifted = counts2 + z
    ax_right.set_xlim(-XLIM, XLIM)
    ax_right.tick_params(labelsize=16)
    ax_right.set_ylim(0, ylim * 1.15)

    ax_right.stairs(
        values=counts2_shifted,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evo."
    )
    rect2 = Rectangle((-XLIM, 0), 2 * XLIM, z, facecolor="white", edgecolor="none")
    ax_right.add_patch(rect2)
    draw_custom_segments(ax_right, XLIM, ylim)

    ax_right.stairs(
        values=anc2_counts,
        edges=anc2_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Anc."
    )

    ax_right.stairs(
        values=dfe2_counts,
        edges=dfe2_bin_edges,
        baseline=0,
        edgecolor=DFE_FILL,
        lw=1.1,
        label="DFE Anc."
    )

    ax_right.legend(frameon=False)
    ax_right.set_xlabel(r'Fitness effect $(s)$')

    # Adjust spines and tick positions for a cleaner look
    for ax in [ax_left, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def create_segben(ax, dfe_anc, dfe_evo, labels=(r'$t_1$', r'$t_2$')):
    valid_indices = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid_indices]
    dfe_evo = dfe_evo[valid_indices]

    anc_mask = (dfe_anc > LOWER_BEN_LIMIT) & (dfe_anc < UPPER_BEN_LIMIT)
    evo_mask = (dfe_evo > LOWER_BEN_LIMIT) & (dfe_evo < UPPER_BEN_LIMIT)

    # positions
    x0, x1 = 1.0, 2.0

    # fetch the paired values
    anc_vals = dfe_anc[anc_mask]
    evo_from_anc = dfe_evo[anc_mask]

    evo_vals = dfe_evo[evo_mask]
    anc_from_evo = dfe_anc[evo_mask]

    # scatter evo→anc (reverse)
    ax.scatter(np.full_like(evo_vals, x1), evo_vals,
               color=EVO_FILL, label="Backwards")
    ax.scatter(np.full_like(evo_vals, x0), anc_from_evo,
               facecolors='none', edgecolors=EVO_FILL)

    # arrows from evo→anc
    for y1, y0 in zip(evo_vals, anc_from_evo):
        ax.add_patch(FancyArrowPatch((x1, y1), (x0, y0),
                                     arrowstyle='-|>', mutation_scale=8,
                                     color=EVO_FILL, linewidth=0.7))

    # scatter ancestor→evo
    ax.scatter(np.full_like(anc_vals, x0), anc_vals,
               color=ANC_FILL, label="Forward")
    ax.scatter(np.full_like(anc_vals, x1), evo_from_anc,
               facecolors='none', edgecolors=ANC_FILL)

    # arrows from anc→evo
    for y0, y1 in zip(anc_vals, evo_from_anc):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle='-|>', mutation_scale=8,
                                     color=ANC_FILL, linewidth=0.7))

    # styling
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(labels)
    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylabel(r'Fitness effect $(s)$')
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.tick_params(labelsize=16)


def make_figure(ancestor, evolved):
    """One three-panel figure for a single ancestor -> evolved pair.

    Returns ``(path, n_pairs, off_frame)``, where ``off_frame`` is the largest fraction of
    either side falling outside the +-XLIM window of panels B and C.  It is 6-8% for every
    population except Ara+4, whose whole DFE is shifted negative and which loses 25%.
    """
    dfe_anc, dfe_evo = load_pair(ancestor, evolved)
    off_frame = max(np.mean(np.abs(dfe_anc) > XLIM), np.mean(np.abs(dfe_evo) > XLIM))

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_middle = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    create_segben(ax_left, dfe_anc, dfe_evo, labels=(ancestor, evolved))
    create_overlapping_dfes(ax_middle, ax_right, dfe_anc, dfe_evo)

    for ax, label in zip((ax_left, ax_middle, ax_right), "ABC"):
        ax.text(-0.01, 1.1, label, transform=ax.transAxes, fontsize=18,
                fontweight='heavy', va='top', ha='left')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', width=1.5)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.tick_params(axis='both', which='minor', length=5, width=1.6)

    title = f"{ancestor} $\\rightarrow$ {evolved}   ({dfe_anc.size} matched genes)"
    if evolved in FLAGGED:
        title += f"   [{FLAGGED[evolved]}]"
    fig.suptitle(title, fontsize=18, y=1.02)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"fig1_limdi_{evolved}.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.close(fig)
    return out_path, dfe_anc.size, off_frame


def main():
    # The isogenic REL606 -> REL607 control first: zero evolution, so whatever collapse of the
    # beneficial set it shows is regression to the mean from measurement error, not epistasis.
    # Inside the +-0.06 window the reliability of a Limdi effect is only 0.3-0.6 (rms error
    # 0.010 against an in-window spread of 0.014), and selecting on s > 0 selects partly on
    # positive noise, so a sizeable collapse is expected with no background change at all --
    # the control's beneficial mean falls 53%.  B and C are only interpretable against it.
    pairs = [("REL606", "REL607")]
    pairs += [(a, e) for a in cmn_exper.LIMDI_ANCESTORS for e in cmn_exper.LIMDI_EVOLVED[a]]
    for ancestor, evolved in pairs:
        out_path, n, off_frame = make_figure(ancestor, evolved)
        print(f"{ancestor:>7s} -> {evolved:<6s}  n={n:5d}  off-frame {off_frame:5.1%}  "
              f"{os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
