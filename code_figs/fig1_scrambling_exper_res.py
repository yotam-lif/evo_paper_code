r"""Figure 1: experimental scrambling in two independent LTEE knockout panels.

Two rows, three columns.

Row 1 (A-C)  Couce Ara+2, generation 0K -> 2K.

    A  Beneficial knockouts at one timepoint, with an arrow to the same knockout's
       effect at the other -- forward (grey, anchored on the ancestor) and backward
       (orange, anchored on the evolved clone).
    B  Forward: the ancestor's beneficial DFE (grey) and where those same knockouts
       land in the evolved background (orange), against the evolved full DFE (line).
    C  Backward: the evolved clone's beneficial DFE (orange) and where those same
       knockouts sat in the ancestor (grey), against the ancestor's full DFE (line).

Row 2 (D-F)  Paired-effect scatters with nested ancestor-defined tail exclusions.

    D  Limdi REL607 green- versus red-reference estimates.  Zero evolution, so whatever
       decorrelation it shows is measurement error, not epistasis; D calibrates E and F.
    E  Limdi REL607 -> Ara+2.
    F  Couce 0K -> 2K.

    Each reports Pearson r over every pair and again over the pairs whose x effect is
    smallest in ABSOLUTE value -- one partition per panel, dropping the largest 10% of |s| in
    D and E and the largest 2% in F, since the Couce effects are compact enough that 10%
    would reach well inside the bulk.  The exclusion is defined only from the x
    (ancestor/control) measurement and never from y, so the retained subset is not
    conditioned on the outcome whose correlation is reported.  Ranking on |s| rather than on
    s keeps the retained set symmetric about zero: a knockout is dropped for being large, not
    for being deleterious.

Data conventions
----------------
Couce genes are matched on allele name after dropping duplicate fitted values and keeping
abundance > 1, as in the published panel.  Limdi genes are matched on metadata row index --
the shared gene identity across the Limdi matrices; see the block comment in
``cmn/cmn_exper.py`` for why the labelled CSV must not be used for this.

Nothing here is clipped or cut -- both rows show the full measured range, and each row-2
panel is on the envelope of its own data rather than a shared limit, with x and y sharing
that envelope so the identity line is the panel diagonal.  The Limdi
counterpart of row 1, which needs a nonlethal cut and window-clipped histograms because
its deleterious tail is real where Couce's is not, lives in
``code_tmp/fig1_limdi_clones.py``.

Run from anywhere:  python code_figs/fig1_scrambling_exper_res.py
Output:             figs_paper/fig1_scrambling_exper_res.pdf
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_scatter import (  # noqa: E402  (row 2 is shared with figs S1-S4)
    SHALLOW_MAGNITUDE_EXCLUSIONS, envelope_limits, print_correlations, scatter_panel,
    share_density_norm,
)

# ───────────────────────────────────── Style ─────────────────────────────────────
cmn_scatter.apply_style()

color = sns.color_palette('CMRmap', 5)
EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.5)
ANC_FILL = (0.5, 0.5, 0.5, 0.15)
DFE_FILL = color[2]

# ─────────────────────────────────── Parameters ──────────────────────────────────
XLIM = 0.06                 # half-width of the plotted fitness-effect window
SHIFT_FRAC = 0.025          # sideways offset between the paired histograms
UPPER_BEN_LIMIT = 0.3       # arrows are drawn for LOWER < s < UPPER ...
LOWER_BEN_LIMIT = 0.005     # ... median Limdi measurement error is 0.008

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# Row 1 spans +-0.06, so every tick label would read "-0.02", "0.00", ... .  Pulling a
# fixed 10^-2 out into the axis offset text turns those into "-2", "0", ... , which is
# both shorter and easier to read.
EFFECT_AXIS_ORDER = -2


class FixedOrderFormatter(mticker.ScalarFormatter):
    """ScalarFormatter with the power-of-ten offset pinned rather than auto-chosen."""

    def __init__(self, order):
        super().__init__(useOffset=False, useMathText=True)
        self._fixed_order = order
        self.set_scientific(True)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order


def scale_effect_axis(ax):
    ax.xaxis.set_major_formatter(FixedOrderFormatter(EFFECT_AXIS_ORDER))
    ax.xaxis.get_offset_text().set_fontsize(14)


# ─────────────────────────────────── Data loading ─────────────────────────────────
def load_couce_pair(ancestor="0K", evolved="2K"):
    """Matched Couce segment effects for one transition.

    Segments are keyed on ``alle`` ("<ORF>-<segment>"), the sub-genic unit the authors'
    own scripts match on and the only key comparable across independently mutagenised
    libraries -- see the block comment in ``cmn/cmn_exper.py``.  The cleaning (abundance
    above one, duplicate and failed fits dropped) lives in that loader so every figure and
    table in the repo starts from the same rows.
    """
    early = cmn_exper.load_couce_segment_series(ancestor)
    late = cmn_exper.load_couce_segment_series(evolved)
    shared = early.index.intersection(late.index)
    return (early.loc[shared].to_numpy(float),
            late.loc[shared].to_numpy(float))


# ───────────────────────────────────  Histograms  ─────────────────────────────────
def thresholded_histogram(data, threshold, final_bins):
    """Drop bins holding fewer than ``threshold`` points, then re-bin what survives."""
    counts, bin_edges = np.histogram(data, bins=10 * final_bins)
    valid_data = []
    for i, keep in enumerate(counts >= threshold):
        if keep:
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
            valid_data.append(data[bin_mask])
    if not valid_data:
        raise ValueError("No bins passed the threshold.")
    cleaned_data = np.concatenate(valid_data)
    final_counts, final_edges = np.histogram(
        cleaned_data, bins=final_bins, density=True)
    return final_counts, final_edges


def couce_histogram(threshold):
    """Row 1: fig1's hand-set count thresholds, applied to the unclipped data."""
    def histogram(data, final_bins):
        return thresholded_histogram(data, threshold, final_bins)
    return histogram


# ─────────────────────────────────  Row 1 panels  ─────────────────────────────────
def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo, histograms, headroom):
    """Forward (left) and backward (right) subset-DFE panels for one transition.

    ``histograms`` maps a curve role to the histogram callable for that curve, so row 1
    keeps fig1's per-curve hand-set thresholds while row 2 uses the rescaled clipped one.
    ``headroom`` turns a panel's y-limit into the extra space left above it.
    """
    z_frac = 0.1
    lw_main = 1.0
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc, dfe_evo = dfe_anc[valid], dfe_evo[valid]

    def draw_custom_segments(ax, _xlim, _ylim):
        z = _ylim * z_frac * 1.1
        ax.plot([-_xlim * 0.9, _xlim * 0.9], [z, z],
                linestyle="--", color="grey", lw=lw_main)
        for (x0, y0), (x1, y1) in [
            ((-_xlim, -0.75), (-_xlim * 0.9, z)),
            ((_xlim, -0.75), (_xlim * 0.9, z)),
            ((-_xlim / 2, -0.75), (-_xlim / 2 * 0.9, z)),
            ((_xlim / 2, -0.75), (_xlim / 2 * 0.9, z)),
            ((0, -0.75), (0, z)),
        ]:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]
    prop_bdfe_anc = dfe_evo[dfe_anc > 0]     # ancestor-beneficial, read on the evolved
    prop_bdfe_evo = dfe_anc[dfe_evo > 0]     # evolved-beneficial, read on the ancestor

    def frame(ax, ylim):
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(0, ylim + headroom(ylim))
        ax.tick_params(labelsize=16)
        scale_effect_axis(ax)

    # ── Left panel: forward propagation ──
    counts, bin_edges = histograms["forward_subset"](prop_bdfe_anc, 25)
    anc_counts, anc_bin_edges = histograms["forward_anchor"](bdfe_anc, 20)
    dfe_counts, dfe_bin_edges = histograms["forward_backdrop"](dfe_evo, 30)
    bin_edges = bin_edges - XLIM * SHIFT_FRAC
    dfe_bin_edges = dfe_bin_edges - XLIM * SHIFT_FRAC
    anc_bin_edges = anc_bin_edges + XLIM * SHIFT_FRAC

    ylim = max(counts.max(), anc_counts.max(), dfe_counts.max()) * (1 + z_frac)
    z = ylim * z_frac
    frame(ax_left, ylim)
    ax_left.stairs(values=counts + z, edges=bin_edges, baseline=0, fill=True,
                   facecolor=EVO_FILL, edgecolor="black", lw=1.1, label="Evo.")
    ax_left.stairs(values=dfe_counts + z, edges=dfe_bin_edges, baseline=0, fill=False,
                   edgecolor=DFE_FILL, lw=1.1, label="DFE Evo.")
    ax_left.add_patch(Rectangle((-XLIM, 0), 2 * XLIM, z,
                                facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_left, XLIM, ylim)
    ax_left.stairs(values=anc_counts, edges=anc_bin_edges, baseline=0, fill=True,
                   facecolor=ANC_FILL, edgecolor="black", lw=1.1, label="Anc.")
    ax_left.legend(frameon=False)
    ax_left.set_xlabel(r'Fitness effect $(s)$')
    ax_left.set_ylabel('Density')

    # ── Right panel: backward propagation ──
    counts2, bin_edges2 = histograms["backward_subset"](bdfe_evo, 12)
    anc2_counts, anc2_bin_edges = histograms["backward_anchor"](prop_bdfe_evo, 22)
    dfe2_counts, dfe2_bin_edges = histograms["backward_backdrop"](dfe_anc, 24)
    bin_edges2 = bin_edges2 + XLIM * SHIFT_FRAC
    dfe2_bin_edges = dfe2_bin_edges - XLIM * SHIFT_FRAC
    anc2_bin_edges = anc2_bin_edges - XLIM * SHIFT_FRAC

    ylim = max(counts2.max(), anc2_counts.max(), dfe2_counts.max()) * (1 + z_frac)
    z = ylim * z_frac
    frame(ax_right, ylim)
    ax_right.stairs(values=counts2 + z, edges=bin_edges2, baseline=0, fill=True,
                    facecolor=EVO_FILL, edgecolor="black", lw=1.1, label="Evo.")
    ax_right.add_patch(Rectangle((-XLIM, 0), 2 * XLIM, z,
                                 facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_right, XLIM, ylim)
    ax_right.stairs(values=anc2_counts, edges=anc2_bin_edges, baseline=0, fill=True,
                    facecolor=ANC_FILL, edgecolor="black", lw=1.1, label="Anc.")
    ax_right.stairs(values=dfe2_counts, edges=dfe2_bin_edges, baseline=0,
                    edgecolor=DFE_FILL, lw=1.1, label="DFE Anc.")
    ax_right.legend(frameon=False)
    ax_right.set_xlabel(r'Fitness effect $(s)$')
    ax_right.set_ylabel('Density')

    for ax in (ax_left, ax_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def create_segben(ax, dfe_anc, dfe_evo, labels):
    """Paired beneficial effects at both timepoints, joined by arrows."""
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc, dfe_evo = dfe_anc[valid], dfe_evo[valid]

    anc_mask = (dfe_anc > LOWER_BEN_LIMIT) & (dfe_anc < UPPER_BEN_LIMIT)
    evo_mask = (dfe_evo > LOWER_BEN_LIMIT) & (dfe_evo < UPPER_BEN_LIMIT)
    x0, x1 = 1.0, 2.0

    anc_vals, evo_from_anc = dfe_anc[anc_mask], dfe_evo[anc_mask]
    evo_vals, anc_from_evo = dfe_evo[evo_mask], dfe_anc[evo_mask]

    ax.scatter(np.full_like(evo_vals, x1), evo_vals, color=EVO_FILL, label="Backwards")
    ax.scatter(np.full_like(evo_vals, x0), anc_from_evo,
               facecolors='none', edgecolors=EVO_FILL)
    for y1, y0 in zip(evo_vals, anc_from_evo):
        ax.add_patch(FancyArrowPatch((x1, y1), (x0, y0), arrowstyle='-|>',
                                     mutation_scale=8, color=EVO_FILL, linewidth=0.7))

    ax.scatter(np.full_like(anc_vals, x0), anc_vals, color=ANC_FILL, label="Forward")
    ax.scatter(np.full_like(anc_vals, x1), evo_from_anc,
               facecolors='none', edgecolors=ANC_FILL)
    for y0, y1 in zip(anc_vals, evo_from_anc):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                     mutation_scale=8, color=ANC_FILL, linewidth=0.7))

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(labels)
    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylabel(r'Fitness effect $(s)$')
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.tick_params(labelsize=16)


# ─────────────────────────────────────  Figure  ───────────────────────────────────
def main():
    couce_anc, couce_evo = load_couce_pair()
    control_green, control_red = cmn_exper.limdi_channel_series("REL607")
    control_green = np.asarray(control_green, dtype=float)
    control_red = np.asarray(control_red, dtype=float)

    scatter_anc = cmn_exper.limdi_gene_series("REL607")
    scatter_evo = cmn_exper.limdi_gene_series("Ara+2")
    shared = scatter_anc.index.intersection(scatter_evo.index)
    scatter_anc = scatter_anc.loc[shared].to_numpy(float)
    scatter_evo = scatter_evo.loc[shared].to_numpy(float)

    fig = plt.figure(figsize=(18, 12.5))
    # An explicit spacer row rather than hspace, so the gap can be set independently of
    # the row heights -- row 2 is the taller of the two.
    gs = GridSpec(3, 3, figure=fig, wspace=0.36, hspace=0.0,
                  height_ratios=(1.0, 0.28, 1.32))
    panel_rows = (0, 2)
    axes = np.array([[fig.add_subplot(gs[panel_rows[row], col]) for col in range(3)]
                     for row in range(2)])

    # Row 1: Couce, fig1's hand-set per-curve histogram thresholds.
    couce_histograms = {
        "forward_subset": couce_histogram(3), "forward_anchor": couce_histogram(3),
        "forward_backdrop": couce_histogram(6), "backward_subset": couce_histogram(2),
        "backward_anchor": couce_histogram(3), "backward_backdrop": couce_histogram(8),
    }
    create_segben(axes[0, 0], couce_anc, couce_evo, labels=('0', '2K'))
    create_overlapping_dfes(axes[0, 1], axes[0, 2], couce_anc, couce_evo,
                            couce_histograms, headroom=lambda ylim: 10.0)

    # Row 2: paired-effect scatters, each on the envelope of its own data.
    control_results, control_density = scatter_panel(
        axes[1, 0], control_green, control_red, "Isogenic control (REL607, LB)",
        r"Fitness effect $(s)$, measurement 1",
        r"Fitness effect $(s)$, measurement 2",
        envelope_limits(control_green, control_red))
    ara2_results, ara2_density = scatter_panel(
        axes[1, 1], scatter_anc, scatter_evo, "ARA+2 (LB), 50K",
        r"Ancestral effect $(s)$", r"Evolved effect $(s)$",
        envelope_limits(scatter_anc, scatter_evo))
    couce_results, couce_density = scatter_panel(
        axes[1, 2], couce_anc, couce_evo, "ARA+2 (DM25), 2K",
        r"Ancestral effect $(s)$", r"Evolved effect $(s)$",
        envelope_limits(couce_anc, couce_evo),
        marker_size=6.0, exclusions=SHALLOW_MAGNITUDE_EXCLUSIONS)

    share_density_norm((control_density, ara2_density, couce_density))

    for index, (ax, label) in enumerate(zip(axes.ravel(), "ABCDEF")):
        cmn_scatter.panel_label(ax, label)
        if index >= 3:      # row 2: open, detached frame, matching B and C
            cmn_scatter.style_scatter_axes(ax)
        else:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', length=10, width=1.5)
            ax.tick_params(axis='both', which='minor', length=5, width=1.6)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fig1_scrambling_exper_res.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

    print_correlations("REL607 green vs red", control_results)
    print_correlations("REL607 -> Ara+2", ara2_results)
    print_correlations("Couce 0K -> 2K", couce_results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
