r"""Poster Figure 1: experimental scrambling across short and long timescales.

The top row reproduces panels A-B of ``fig1_scrambling_exper_res.py`` for the
Couce Ara+2 0K -> 2K transition.  The plotting code is copied here deliberately
so this poster figure does not import another figure-generating script.

The bottom row shows two Limdi comparisons:

  C. REL607 green-reference versus red-reference estimates (technical control).
  D. REL607 versus Ara+2 at 50K.

Each lower panel reports Pearson autocorrelation after removing exactly 0%, 5%,
or 10% of the most deleterious effects on the x-axis.  The exclusion is defined
only from the ancestor/control x measurement; it never uses the y measurement.
This produces nested, ancestor-defined subsets and avoids conditioning on the
outcome whose correlation is being calculated.

Run from any directory:

    python code_tmp/poster_fig1.py

Output:

    figs_paper/poster_fig1.pdf
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
import cmasher as cmr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Paths and data layout
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
COUCE_DIR = os.path.join(REPO_ROOT, "data", "alex_code")
LIMDI_DIR = os.path.join(
    REPO_ROOT,
    "data",
    "anurag_data",
    "Analysis",
    "Part_3_TnSeq_analysis",
    "Processed_data_for_plotting",
)
LIMDI_FIT_PATH = os.path.join(LIMDI_DIR, "fitness_corrected_genes.npy")
OUT_DIR = os.path.join(REPO_ROOT, "figs_paper")
OUT_PDF = os.path.join(OUT_DIR, "poster_fig1.pdf")

LIMDI_LIBRARIES = (
    "REL606",
    "REL607",
    "Ara-1",
    "Ara-2",
    "Ara-3",
    "Ara-4",
    "Ara-5",
    "Ara-6",
    "Ara+1",
    "Ara+2",
    "Ara+3",
    "Ara+4",
    "Ara+5",
    "Ara+6",
)
LIMDI_MISSING = -1.0


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

for font_path in (
    "/Library/Fonts/AGaramondPro-Regular.otf",
    "/Library/Fonts/AGaramondPro-Italic.otf",
    "/Library/Fonts/AGaramondPro-Bold.otf",
    "/Library/Fonts/AGaramondPro-BoldItalic.otf",
):
    font_manager.fontManager.addfont(font_path)

plt.rcParams["font.family"] = "Adobe Garamond Pro"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Adobe Garamond Pro"
plt.rcParams["mathtext.it"] = "Adobe Garamond Pro:italic"
plt.rcParams["mathtext.bf"] = "Adobe Garamond Pro:bold"
plt.rcParams["font.size"] = 18
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["legend.fontsize"] = 15

PALETTE = sns.color_palette("CMRmap", 5)
EVO_COLOR = PALETTE[1]
EVO_FILL = (*EVO_COLOR, 0.50)
ANC_FILL = (0.5, 0.5, 0.5, 0.18)
DFE_COLOR = PALETTE[2]
IDENTITY_COLOR = (0.18, 0.18, 0.18)
CUTOFF_COLOR = "#18C63E"
LOWEST_5_COLOR, NEXT_5_COLOR, RETAINED_90_COLOR = cmr.take_cmap_colors(
    "cmr.freeze", 3, cmap_range=(0.15, 0.85), return_fmt="hex"
)
DENSITY_MAX_COLOR = mpl.colors.to_hex(mpl.colormaps["cmr.freeze"](0.75))
DENSITY_MIN_COLOR = "#E2F2F2"
DENSITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "moderate_density", (DENSITY_MIN_COLOR, DENSITY_MAX_COLOR)
)

# Couce top-row settings copied from the current paper figure.
TOP_XLIM = 0.06
SHIFT_FRAC = 0.025
UPPER_BEN_LIMIT = 0.3
LOWER_BEN_LIMIT = 0.005
MIN_ABUNDANCE = 1

# Shared settings for both Limdi panels.
SCATTER_LIMITS = (-0.65, 0.11)
INSET_LIMITS = (-0.05, 0.05)
TAIL_EXCLUSIONS = (0.00, 0.05, 0.10)


# ---------------------------------------------------------------------------
# Couce data and top-row plotting code
# ---------------------------------------------------------------------------

def load_couce_pair():
    """Return matched 0K and 2K Couce segment effects."""
    ancestor = (
        pd.read_csv(os.path.join(COUCE_DIR, "Rfitted_fil.txt"), sep="\t")
        .dropna(subset=["fitted1"])
        .drop_duplicates(subset=["fitted1"])
    )
    evolved = (
        pd.read_csv(os.path.join(COUCE_DIR, "2Kfitted_fil.txt"), sep="\t")
        .dropna(subset=["fitted1"])
        .drop_duplicates(subset=["fitted1"])
    )
    ancestor = ancestor[ancestor["abn"] > MIN_ABUNDANCE]
    evolved = evolved[evolved["abn"] > MIN_ABUNDANCE]
    ancestor = ancestor.set_index("alle")["fitted1"]
    evolved = evolved.set_index("alle")["fitted1"]
    shared = ancestor.index.intersection(evolved.index)
    return (
        ancestor.loc[shared].to_numpy(float),
        evolved.loc[shared].to_numpy(float),
    )


def thresholded_histogram(data, threshold, final_bins):
    """Copy of the sparse-bin cleaning used by the current Figure 1."""
    init_bins = 10 * final_bins
    counts, bin_edges = np.histogram(data, bins=init_bins)
    valid_indices = counts >= threshold
    valid_data = []
    for i, keep in enumerate(valid_indices):
        if keep:
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
            valid_data.append(data[bin_mask])
    if not valid_data:
        raise ValueError("No histogram bins passed the count threshold.")
    cleaned_data = np.concatenate(valid_data)
    final_counts, final_edges = np.histogram(
        cleaned_data, bins=final_bins, density=True
    )
    return final_counts, final_edges


def create_segben(ax, dfe_anc, dfe_evo, labels=("0", "2K")):
    """Connected beneficial-mutation effects, copied from the paper figure."""
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid]
    dfe_evo = dfe_evo[valid]

    anc_mask = (dfe_anc > LOWER_BEN_LIMIT) & (dfe_anc < UPPER_BEN_LIMIT)
    evo_mask = (dfe_evo > LOWER_BEN_LIMIT) & (dfe_evo < UPPER_BEN_LIMIT)
    anc_vals = dfe_anc[anc_mask]
    evo_from_anc = dfe_evo[anc_mask]
    evo_vals = dfe_evo[evo_mask]
    anc_from_evo = dfe_anc[evo_mask]
    x0, x1 = 1.0, 2.0

    ax.scatter(
        np.full_like(evo_vals, x1),
        evo_vals,
        s=15,
        color=EVO_FILL,
        zorder=3,
    )
    ax.scatter(
        np.full_like(evo_vals, x0),
        anc_from_evo,
        s=15,
        facecolors="none",
        edgecolors=EVO_FILL,
        linewidths=0.7,
        zorder=3,
    )
    for y1, y0 in zip(evo_vals, anc_from_evo):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x0, y0),
                arrowstyle="-|>",
                mutation_scale=7,
                color=EVO_FILL,
                linewidth=0.55,
                zorder=2,
            )
        )

    ax.scatter(
        np.full_like(anc_vals, x0),
        anc_vals,
        s=15,
        color=ANC_FILL,
        zorder=3,
    )
    ax.scatter(
        np.full_like(anc_vals, x1),
        evo_from_anc,
        s=15,
        facecolors="none",
        edgecolors=ANC_FILL,
        linewidths=0.7,
        zorder=3,
    )
    for y0, y1 in zip(anc_vals, evo_from_anc):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=7,
                color=ANC_FILL,
                linewidth=0.55,
                zorder=2,
            )
        )

    ax.set_xticks([x0, x1], labels)
    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylabel(r"Fitness effect $(s)$")
    ax.axhline(0, linestyle="--", color="black", linewidth=0.8)


def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo):
    """Subset DFEs copied from the paper figure.

    Pass ``ax_right=None`` when only the ancestor-beneficial comparison is
    needed, as in the reduced four-panel poster figure.
    """
    z_frac = 0.1
    lw_main = 1.0
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid]
    dfe_evo = dfe_evo[valid]

    def draw_custom_segments(ax, xlim, ylim):
        z = ylim * z_frac * 1.1
        ax.plot(
            [-xlim * 0.9, xlim * 0.9],
            [z, z],
            linestyle="--",
            color="grey",
            lw=lw_main,
        )
        segments = [
            ((-xlim, -0.75), (-xlim * 0.9, z)),
            ((xlim, -0.75), (xlim * 0.9, z)),
            ((-xlim / 2, -0.75), (-xlim / 2 * 0.9, z)),
            ((xlim / 2, -0.75), (xlim / 2 * 0.9, z)),
            ((0, -0.75), (0, z)),
        ]
        for (x0, y0), (x1, y1) in segments:
            ax.plot(
                [x0, x1],
                [y0, y1],
                linestyle="--",
                color="grey",
                lw=lw_main,
            )

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]
    prop_bdfe_anc = dfe_evo[dfe_anc > 0]
    prop_bdfe_evo = dfe_anc[dfe_evo > 0]

    # Forward: ancestor-beneficial subset propagated to the evolved background.
    counts, bin_edges = thresholded_histogram(prop_bdfe_anc, 3, 25)
    anc_counts, anc_bin_edges = thresholded_histogram(bdfe_anc, 3, 20)
    dfe_counts, dfe_bin_edges = thresholded_histogram(dfe_evo, 6, 30)
    bin_edges = bin_edges - TOP_XLIM * SHIFT_FRAC
    dfe_bin_edges = dfe_bin_edges - TOP_XLIM * SHIFT_FRAC
    anc_bin_edges = anc_bin_edges + TOP_XLIM * SHIFT_FRAC
    ylim = max(np.max(counts), np.max(anc_counts), np.max(dfe_counts)) * (
        1 + z_frac
    )
    z = ylim * z_frac
    ax_left.set_xlim(-TOP_XLIM, TOP_XLIM)
    ax_left.set_ylim(0, ylim + 10)
    ax_left.stairs(
        counts + z,
        bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved",
    )
    ax_left.stairs(
        dfe_counts + z,
        dfe_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=DFE_COLOR,
        lw=1.2,
        label="Evolved DFE",
    )
    ax_left.add_patch(
        Rectangle(
            (-TOP_XLIM, 0),
            2 * TOP_XLIM,
            z,
            facecolor="white",
            edgecolor="none",
        )
    )
    draw_custom_segments(ax_left, TOP_XLIM, ylim)
    ax_left.stairs(
        anc_counts,
        anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor",
    )
    ax_left.legend(frameon=False, loc="upper left")
    ax_left.set_xlabel(r"Fitness effect $(s)$")
    ax_left.set_ylabel("Density")
    ax_left.set_title("Ancestor-beneficial subset")

    if ax_right is None:
        return

    # Reverse: evolved-beneficial subset propagated to the ancestor.
    counts, bin_edges = thresholded_histogram(bdfe_evo, 2, 12)
    anc_counts, anc_bin_edges = thresholded_histogram(prop_bdfe_evo, 3, 22)
    dfe_counts, dfe_bin_edges = thresholded_histogram(dfe_anc, 8, 24)
    bin_edges = bin_edges + TOP_XLIM * SHIFT_FRAC
    dfe_bin_edges = dfe_bin_edges - TOP_XLIM * SHIFT_FRAC
    anc_bin_edges = anc_bin_edges - TOP_XLIM * SHIFT_FRAC
    ylim = max(np.max(counts), np.max(anc_counts), np.max(dfe_counts)) * (
        1 + z_frac
    )
    z = ylim * z_frac
    ax_right.set_xlim(-TOP_XLIM, TOP_XLIM)
    ax_right.set_ylim(0, ylim + 10)
    ax_right.stairs(
        counts + z,
        bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved",
    )
    ax_right.add_patch(
        Rectangle(
            (-TOP_XLIM, 0),
            2 * TOP_XLIM,
            z,
            facecolor="white",
            edgecolor="none",
        )
    )
    draw_custom_segments(ax_right, TOP_XLIM, ylim)
    ax_right.stairs(
        anc_counts,
        anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor",
    )
    ax_right.stairs(
        dfe_counts,
        dfe_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=DFE_COLOR,
        lw=1.2,
        label="Ancestor DFE",
    )
    ax_right.legend(frameon=False, loc="upper left")
    ax_right.set_xlabel(r"Fitness effect $(s)$")
    ax_right.set_title("Evolved-beneficial subset")


# ---------------------------------------------------------------------------
# Limdi lower-row data and plotting
# ---------------------------------------------------------------------------

def load_limdi_fitness():
    """Load the gene x population x Green/Red corrected-effect array."""
    fitness = np.load(LIMDI_FIT_PATH).astype(float)
    if fitness.ndim != 3 or fitness.shape[1:] != (len(LIMDI_LIBRARIES), 2):
        raise ValueError(
            "Unexpected Limdi array shape "
            f"{fitness.shape}; expected (genes, {len(LIMDI_LIBRARIES)}, 2)."
        )
    return fitness


def channel_pair(fitness, population):
    """Return Green and Red estimates for one population on shared valid genes."""
    k = LIMDI_LIBRARIES.index(population)
    values = fitness[:, k, :]
    valid = np.all(values > LIMDI_MISSING, axis=1)
    return values[valid, 0], values[valid, 1]


def ancestor_evolved_pair(fitness, ancestor, evolved):
    """Return matched two-channel means for one ancestor/evolved pair."""
    i = LIMDI_LIBRARIES.index(ancestor)
    j = LIMDI_LIBRARIES.index(evolved)
    valid = np.all(fitness[:, (i, j), :] > LIMDI_MISSING, axis=(1, 2))
    ancestor_effect = np.mean(fitness[valid, i, :], axis=1)
    evolved_effect = np.mean(fitness[valid, j, :], axis=1)
    return ancestor_effect, evolved_effect


def tail_exclusion_correlations(ancestor_effect, comparison_effect):
    """Pearson r after removing exact fractions of the lowest ancestor effects."""
    order = np.argsort(ancestor_effect, kind="stable")
    results = []
    n = ancestor_effect.size
    for fraction in TAIL_EXCLUSIONS:
        n_remove = int(np.floor(fraction * n))
        kept = order[n_remove:]
        r = float(pearsonr(ancestor_effect[kept], comparison_effect[kept]).statistic)
        cutoff = (
            float("-inf")
            if n_remove == 0
            else float(ancestor_effect[order[n_remove - 1]])
        )
        results.append(
            {
                "fraction": fraction,
                "removed": n_remove,
                "kept": kept.size,
                "cutoff": cutoff,
                "r": r,
            }
        )
    return results


def draw_scatter_points(ax, x, y, marker_size):
    """Draw the retained 90%, next 5%, and lowest 5% as separate layers."""
    order = np.argsort(x, kind="stable")
    n = x.size
    n5 = int(np.floor(0.05 * n))
    n10 = int(np.floor(0.10 * n))

    groups = (
        (order[n10:], RETAINED_90_COLOR, 0.25, 1.0, "Retained 90%"),
        (order[n5:n10], NEXT_5_COLOR, 0.72, 1.25, "Next 5%"),
        (order[:n5], LOWEST_5_COLOR, 0.82, 1.45, "Lowest 5%"),
    )
    for zorder, (indices, color, alpha, size_scale, label) in enumerate(
        groups, start=3
    ):
        ax.scatter(
            x[indices],
            y[indices],
            s=marker_size * size_scale,
            color=color,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            zorder=zorder,
            label=label,
        )


def scatter_panel(ax, x, y, title, xlabel, ylabel):
    """One full-range Limdi scatter with a common moderate-effect inset."""
    lo, hi = SCATTER_LIMITS
    ax.axhline(0.0, color="grey", lw=0.75, ls="--", zorder=1)
    ax.axvline(0.0, color="grey", lw=0.75, ls="--", zorder=1)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=IDENTITY_COLOR,
        lw=1.2,
        zorder=2,
    )
    draw_scatter_points(ax, x, y, marker_size=8.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    correlations = tail_exclusion_correlations(x, y)
    marker_half_height = 0.17
    shared_marker_top = min(
        hi, correlations[-1]["cutoff"] + marker_half_height
    )
    for result in correlations[1:]:
        cutoff = result["cutoff"]
        retained = int(100 * (1 - result["fraction"]))
        marker_bottom = max(lo, cutoff - marker_half_height)
        marker_top = shared_marker_top
        ax.plot(
            [cutoff, cutoff],
            [marker_bottom, marker_top],
            color=CUTOFF_COLOR,
            lw=1.6,
            ls=(0, (4, 3)),
            zorder=6,
        )
        ax.text(
            cutoff + 0.012,
            marker_top,
            f"{retained}%",
            fontsize=15,
            color=CUTOFF_COLOR,
            ha="left",
            va="bottom",
            zorder=7,
        )

    lines = ["Pearson"] + [
        rf"$r_{{{int(100 * (1 - result['fraction']))}\%}}={result['r']:.3f}$"
        for result in correlations
    ]
    ax.text(
        lo + 0.03,
        -0.015,
        "\n".join(lines),
        transform=ax.transData,
        ha="left",
        va="top",
        fontsize=17,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
        zorder=8,
    )

    inset = ax.inset_axes([0.54, 0.08, 0.42, 0.42])
    inset.set_zorder(20)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    ilo, ihi = INSET_LIMITS
    inset.axhline(0.0, color="grey", lw=0.55, ls="--", zorder=1)
    inset.axvline(0.0, color="grey", lw=0.55, ls="--", zorder=1)
    inset.plot(
        [ilo, ihi],
        [ilo, ihi],
        color=IDENTITY_COLOR,
        lw=0.8,
        zorder=2,
    )
    density = inset.hexbin(
        x,
        y,
        gridsize=28,
        extent=(ilo, ihi, ilo, ihi),
        mincnt=1,
        cmap=DENSITY_CMAP,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    inset.set_xlim(ilo, ihi)
    inset.set_ylim(ilo, ihi)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([ilo, 0.0, ihi])
    inset.set_yticks([ilo, 0.0, ihi])
    inset.set_xticklabels([f"{ilo:.2f}", "0.00", f"{ihi:.2f}"])
    # The x-axis lower-bound label also denotes the shared square-inset limit.
    # Suppress its duplicate on y, which would otherwise collide at a corner.
    inset.set_yticklabels(["", "0.00", f"{ihi:.2f}"])
    inset.yaxis.set_ticks_position("right")
    inset.tick_params(axis="y", labelleft=False, labelright=True)
    inset.tick_params(labelsize=10, length=2, pad=1)

    return correlations, density


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)
    ax.tick_params(axis="both", which="major", length=6, width=1.35)


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="heavy",
        va="top",
        ha="left",
    )


def print_correlations(name, results):
    values = "  ".join(
        f"remove {int(100 * item['fraction']):2d}%: "
        f"r={item['r']:.4f} (n={item['kept']})"
        for item in results
    )
    print(f"{name:24s} {values}")


def main():
    couce_ancestor, couce_evolved = load_couce_pair()
    limdi_fitness = load_limdi_fitness()

    rel607_green, rel607_red = channel_pair(limdi_fitness, "REL607")
    rel607_ara2 = ancestor_evolved_pair(
        limdi_fitness, ancestor="REL607", evolved="Ara+2"
    )
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=(0.93, 1.07),
        wspace=0.28,
        hspace=0.36,
        left=0.085,
        right=0.98,
        bottom=0.075,
        top=0.965,
    )
    axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(2)]

    # Top row: Couce 0K -> 2K panels A-B.
    create_segben(axes[0], couce_ancestor, couce_evolved, labels=("0K", "2K"))
    axes[0].set_title("Same mutation, different background")
    create_overlapping_dfes(axes[1], None, couce_ancestor, couce_evolved)

    # Bottom row: full-range Limdi comparisons with nested ancestor exclusions.
    control_results, control_density = scatter_panel(
        axes[2],
        rel607_green,
        rel607_red,
        "Isogenic Control (REL607)",
        r"Fitness effect $(s)$, measurement 1",
        r"Fitness effect $(s)$, measurement 2",
    )
    ara2_results, ara2_density = scatter_panel(
        axes[3],
        rel607_ara2[0],
        rel607_ara2[1],
        "ARA+2 (50K)",
        r"Ancestral effect $(s)$",
        r"Evolved effect $(s)$",
    )
    for ax, label in zip(axes, "ABCD"):
        style_axis(ax)
        add_panel_label(ax, label)

    density_plots = (control_density, ara2_density)
    density_max = max(float(np.max(plot.get_array())) for plot in density_plots)
    shared_density_norm = mpl.colors.LogNorm(vmin=1.0, vmax=density_max)
    for plot in density_plots:
        plot.set_norm(shared_density_norm)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print_correlations("REL607 green vs red", control_results)
    print_correlations("REL607 -> Ara+2", ara2_results)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
