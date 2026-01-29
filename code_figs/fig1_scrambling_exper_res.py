import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib as mpl
from scipy.stats import ks_2samp, cramervonmises_2samp

# set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
color = sns.color_palette('CMRmap', 5)
EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.5)
ANC_FILL = (0.5, 0.5, 0.5, 0.15)
DFE_FILL = color[2]
xlim = 0.06
shift_frac = 0.025
upper_ben_limit = 0.3
lower_ben_limit = 0.005
abn = 1

# Baym data:
baym_datadir = os.path.join('..', 'data', 'alex_code')
# load and filter
Rtable = pd.read_csv(os.path.join(baym_datadir, "Rfitted_fil.txt"), sep="\t") \
              .dropna(subset=["fitted1"])
Ttable = pd.read_csv(os.path.join(baym_datadir, "2Kfitted_fil.txt"), sep="\t") \
              .dropna(subset=["fitted1"])
Ftable = pd.read_csv(os.path.join(baym_datadir, "15Kfitted_fil.txt"), sep="\t") \
              .dropna(subset=["fitted1"])
# Remove duplicates
Rtable = Rtable.drop_duplicates(subset=["fitted1"])
Ttable = Ttable.drop_duplicates(subset=["fitted1"])
Ftable = Ftable.drop_duplicates(subset=["fitted1"])
# only keep genes with abn > 1
Rtable = Rtable[(Rtable["abn"] > abn)]
Ttable = Ttable[(Ttable["abn"] > abn)]
Ftable = Ftable[(Ftable["abn"] > abn)]
# Extract values
r = Rtable.set_index('alle')['fitted1']
m = Ttable.set_index('alle')['fitted1']
k = Ftable.set_index('alle')['fitted1']
# Find common alleles
common_0_2 = r.index.intersection(m.index)
common_0_15 = r.index.intersection(k.index)
baym_dfe0K_2  = r.loc[common_0_2].values    # generation 0K
baym_dfe0K_15 = r.loc[common_0_15].values    # generation 0K
baym_dfe2K  = m.loc[common_0_2].values    # generation 2K
baym_dfe15K = k.loc[common_0_15].values    # generation 15K

#  Asencao data:
dirnum = 2
asencao_datapath = "../data/asencao_dfe_arrays"
experiment_dirs = sorted([
    d for d in os.listdir(asencao_datapath)
    if os.path.isdir(os.path.join(asencao_datapath, d))
])
exp_dir = experiment_dirs[dirnum]
exp_path = os.path.join(asencao_datapath, exp_dir)
try:
    asenc_R = np.load(os.path.join(exp_path, "R.npy"))
    asenc_S = np.load(os.path.join(exp_path, "S.npy"))
except Exception as e:
    print(f"Skipping {exp_dir}: {e}")


def thresholded_histogram(data, threshold, final_bins):
    # Step 1: Use many initial bins to capture fine structure
    init_bins = 10 * final_bins
    counts, bin_edges = np.histogram(data, bins=init_bins)

    # Step 2: Mask bins below threshold
    valid_indices = counts >= threshold
    valid_bin_edges = bin_edges[:-1][valid_indices]
    valid_data = []
    for i, keep in enumerate(valid_indices):
        if keep:
            # Get data in that bin
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i+1])
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

    evo_vs_prop_test = ks_2samp(dfe_evo, prop_bdfe_anc)
    anc_vs_prop_test = ks_2samp(dfe_anc, prop_bdfe_evo)

    # Left Panel - Forward propagate
    counts, bin_edges, _ = thresholded_histogram(data=prop_bdfe_anc, threshold=3, final_bins=25)
    anc_counts, anc_bin_edges, _ = thresholded_histogram(data=bdfe_anc, threshold=3, final_bins=20)
    dfe_counts, dfe_bin_edges, _ = thresholded_histogram(data=dfe_evo, threshold=6, final_bins=30)
    bin_edges = bin_edges - xlim * shift_frac
    dfe_bin_edges = dfe_bin_edges - xlim * shift_frac
    anc_bin_edges = anc_bin_edges + xlim * shift_frac
    max1 = np.max(counts)
    max2 = np.max(anc_counts)
    max3 = np.max(dfe_counts)
    ymax = max(max1, max2, max3)
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts_shifted = counts + z
    dfe_counts_shifted = dfe_counts + z
    ax_left.set_xlim(-xlim, xlim)
    ax_left.tick_params(labelsize=14)
    ax_left.set_ylim(0, ylim + 10)

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

    rect = Rectangle((-xlim, 0), 2*xlim, z, facecolor="white", edgecolor="none")
    ax_left.add_patch(rect)
    draw_custom_segments(ax_left, xlim, ylim)

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
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$')
    # ax_left.text(
    #     0.05, 0.95,
    #     fr'$p_{{KS}} = {evo_vs_prop_test.pvalue:.2g}$',
    #     transform=ax_left.transAxes,
    #     va="top",
    #     fontsize=12
    # )

    # Right Panel
    counts2, bin_edges2, _ = thresholded_histogram(data=bdfe_evo, threshold=2, final_bins=12)
    anc2_counts, anc2_bin_edges, _ = thresholded_histogram(data=prop_bdfe_evo, threshold=3, final_bins=22)
    dfe2_counts, dfe2_bin_edges, _ = thresholded_histogram(data=dfe_anc, threshold=8, final_bins=24)
    bin_edges2 = bin_edges2 + xlim * shift_frac
    dfe2_bin_edges = dfe2_bin_edges - xlim * shift_frac
    anc2_bin_edges = anc2_bin_edges - xlim * shift_frac
    max1 = np.max(counts2)
    max2 = np.max(anc2_counts)
    max3 = np.max(dfe2_counts)
    ymax = max(max1, max2, max3)
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts2_shifted = counts2 + z
    ax_right.set_xlim(-xlim, xlim)
    ax_right.tick_params(labelsize=14)
    ax_right.set_ylim(0, ylim + 10)

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
    rect2 = Rectangle((-xlim, 0), 2*xlim, z, facecolor="white", edgecolor="none")
    ax_right.add_patch(rect2)
    draw_custom_segments(ax_right, xlim, ylim)

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
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    # ax_right.text(
    #     0.05, 0.95,
    #     fr'$p_{{KS}} = {anc_vs_prop_test.pvalue:.2g}$',
    #     transform=ax_right.transAxes,
    #     va="top",
    #     fontsize=12
    # )

    # Adjust spines and tick positions for a cleaner look
    for ax in [ax_left, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def create_segben(ax, dfe_anc, dfe_evo, labels=(r'$t_1$', r'$t_2$')):
    # mask out non‐positive if you want
    valid_indices = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid_indices]
    dfe_evo = dfe_evo[valid_indices]

    anc_mask = (dfe_anc > lower_ben_limit) & (dfe_anc < upper_ben_limit)
    evo_mask = (dfe_evo > lower_ben_limit) & (dfe_evo < upper_ben_limit)

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
    ax.set_ylabel(r'Fitness effect $(\Delta)$')
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.tick_params(labelsize=14)
    # ax.legend(frameon=False)


def main():
    # Create a figure with 2 rows and 3 columns.
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)

    # Create the remaining four axes.
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_middle = fig.add_subplot(gs[0, 1])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_middle = fig.add_subplot(gs[1, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 2])

    create_segben(ax_top_left, baym_dfe0K_15, baym_dfe15K, labels=('0', '15K'))
    create_segben(ax_bottom_left, asenc_R, asenc_S, labels=('R', 'S'))
    create_overlapping_dfes(ax_top_middle, ax_top_right, baym_dfe0K_15, baym_dfe15K)
    create_overlapping_dfes(ax_bottom_middle, ax_bottom_right, asenc_R, asenc_S)

    # Panel labels
    labels = {
        ax_top_left: "A",
        ax_top_middle: "B",
        ax_top_right: "C",
        ax_bottom_left: "D",
        ax_bottom_middle: "E",
        ax_bottom_right: "F"
    }
    for ax, label in labels.items():
        ax.text(-0.01, 1.1, label, transform=ax.transAxes, fontweight='heavy', va='top', ha='left')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', width=1.5)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.tick_params(axis='both', which='minor', length=5, width=1.6)

    # Save the figure.
    output_dir = os.path.join('..', 'figs_paper')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig1_scrambling_exper_res.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
