import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from matplotlib.ticker import ScalarFormatter

# --- Auto-added output directory for paper figures ---
out_dir = os.path.join('..', 'figs_paper')
os.makedirs(out_dir, exist_ok=True)

# Set styling
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
abn = 1

baym_datadir = os.path.join('..', 'data', 'alex_code')


def load_baym_data():
    # Load and filter
    Rtable = pd.read_csv(os.path.join(baym_datadir, "Rfitted_fil.txt"), sep="\t").dropna(subset=["fitted1"])
    Ttable = pd.read_csv(os.path.join(baym_datadir, "2Kfitted_fil.txt"), sep="\t").dropna(subset=["fitted1"])
    Ftable = pd.read_csv(os.path.join(baym_datadir, "15Kfitted_fil.txt"), sep="\t").dropna(subset=["fitted1"])

    # Remove duplicates
    Rtable = Rtable.drop_duplicates(subset=["fitted1"])
    Ttable = Ttable.drop_duplicates(subset=["fitted1"])
    Ftable = Ftable.drop_duplicates(subset=["fitted1"])

    # Filter by abundance
    Rtable = Rtable[(Rtable["abn"] > abn)]
    Ttable = Ttable[(Ttable["abn"] > abn)]
    Ftable = Ftable[(Ftable["abn"] > abn)]

    # Extract series
    r = Rtable.set_index('alle')['fitted1']
    m = Ttable.set_index('alle')['fitted1']
    k = Ftable.set_index('alle')['fitted1']

    # Intersection 1: 0K and 2K
    common_0_2 = r.index.intersection(m.index)
    d0_vs_2 = r.loc[common_0_2].values
    d2_vs_0 = m.loc[common_0_2].values

    # Intersection 2: 2K and 15K
    common_2_15 = m.index.intersection(k.index)
    d2_vs_15 = m.loc[common_2_15].values
    d15_vs_2 = k.loc[common_2_15].values

    return (d0_vs_2, d2_vs_0), (d2_vs_15, d15_vs_2)


def thresholded_histogram(data, threshold, final_bins):
    init_bins = 10 * final_bins
    counts, bin_edges = np.histogram(data, bins=init_bins)
    valid_indices = counts >= threshold
    valid_data = []
    for i, keep in enumerate(valid_indices):
        if keep:
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
            valid_data.append(data[bin_mask])
    if not valid_data:
        # Fallback if sparse
        return np.histogram(data, bins=final_bins, density=True)[0], np.histogram(data, bins=final_bins, density=True)[
            1], data

    cleaned_data = np.concatenate(valid_data)
    final_counts, final_edges = np.histogram(cleaned_data, bins=final_bins, density=True)
    return final_counts, final_edges, cleaned_data


def create_overlapping_dfes_ben(ax_left, ax_right, dfe_anc, dfe_evo, label_anc="Anc.", label_evo="Evo."):
    # Logic for BENEFICIAL mutations (> 0)
    z_frac = 0.1
    lw_main = 1.0
    valid_indices = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid_indices]
    dfe_evo = dfe_evo[valid_indices]

    def draw_custom_segments(ax, _xlim, _ylim):
        z = _ylim * z_frac * 1.1
        ax.plot([-_xlim * 0.9, _xlim * 0.9], [z, z], linestyle="--", color="grey", lw=lw_main)
        segs = [
            ((-_xlim, -0.75), (-_xlim * 0.9, z)),
            ((_xlim, -0.75), (_xlim * 0.9, z)),
            ((-_xlim / 2, -0.75), (-_xlim / 2 * 0.9, z)),
            ((_xlim / 2, -0.75), (_xlim / 2 * 0.9, z)),
            ((0, -0.75), (0, z))
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    # Filter beneficials in ancestor background
    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]

    bdfe_anc_inds = np.where(dfe_anc > 0)
    bdfe_evo_inds = np.where(dfe_evo > 0)

    # Propagate
    prop_bdfe_anc = dfe_evo[bdfe_anc_inds]
    prop_bdfe_evo = dfe_anc[bdfe_evo_inds]

    ks_evo = ks_2samp(dfe_evo, prop_bdfe_anc)
    ks_anc = ks_2samp(dfe_anc, prop_bdfe_evo)

    # --- Left Panel (Forward Time) ---
    counts, bin_edges, _ = thresholded_histogram(data=prop_bdfe_anc, threshold=3, final_bins=25)
    anc_counts, anc_bin_edges, _ = thresholded_histogram(data=bdfe_anc, threshold=3, final_bins=20)
    dfe_counts, dfe_bin_edges, _ = thresholded_histogram(data=dfe_evo, threshold=6, final_bins=30)

    bin_edges = bin_edges - xlim * shift_frac
    dfe_bin_edges = dfe_bin_edges - xlim * shift_frac
    anc_bin_edges = anc_bin_edges + xlim * shift_frac

    ymax = max(np.max(counts), np.max(anc_counts), np.max(dfe_counts))
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac

    ax_left.set_xlim(-xlim, xlim)
    ax_left.set_ylim(0, ylim + 10)

    ax_left.stairs(values=counts + z, edges=bin_edges, baseline=0, fill=True, facecolor=EVO_FILL, edgecolor="black",
                   lw=1.1, label=label_evo)
    ax_left.stairs(values=dfe_counts + z, edges=dfe_bin_edges, baseline=0, fill=False, edgecolor=DFE_FILL, lw=1.1,
                   label=f"DFE {label_evo}")
    ax_left.add_patch(Rectangle((-xlim, 0), 2 * xlim, z, facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_left, xlim, ylim)
    ax_left.stairs(values=anc_counts, edges=anc_bin_edges, baseline=0, fill=True, facecolor=ANC_FILL, edgecolor="black",
                   lw=1.1, label=label_anc)

    ax_left.legend(frameon=False)
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$')

    # Added KS Test Text
    ax_left.text(0.05, 0.95, fr'$p_{{KS}} = {ks_evo.pvalue:.2g}$', transform=ax_left.transAxes, va="top", fontsize=12)

    # --- Right Panel (Backward Time) ---
    counts2, bin_edges2, _ = thresholded_histogram(data=bdfe_evo, threshold=2, final_bins=12)
    anc2_counts, anc2_bin_edges, _ = thresholded_histogram(data=prop_bdfe_evo, threshold=3, final_bins=22)
    dfe2_counts, dfe2_bin_edges, _ = thresholded_histogram(data=dfe_anc, threshold=8, final_bins=24)

    bin_edges2 = bin_edges2 + xlim * shift_frac
    dfe2_bin_edges = dfe2_bin_edges - xlim * shift_frac
    anc2_bin_edges = anc2_bin_edges - xlim * shift_frac

    ymax = max(np.max(counts2), np.max(anc2_counts), np.max(dfe2_counts))
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac

    ax_right.set_xlim(-xlim, xlim)
    ax_right.set_ylim(0, ylim + 10)

    ax_right.stairs(values=counts2 + z, edges=bin_edges2, baseline=0, fill=True, facecolor=EVO_FILL, edgecolor="black",
                    lw=1.1, label=label_evo)
    ax_right.add_patch(Rectangle((-xlim, 0), 2 * xlim, z, facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_right, xlim, ylim)
    ax_right.stairs(values=anc2_counts, edges=anc2_bin_edges, baseline=0, fill=True, facecolor=ANC_FILL,
                    edgecolor="black", lw=1.1, label=label_anc)
    ax_right.stairs(values=dfe2_counts, edges=dfe2_bin_edges, baseline=0, edgecolor=DFE_FILL, lw=1.1,
                    label=f"DFE {label_anc}")

    ax_right.legend(frameon=False)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')

    # Added KS Test Text
    ax_right.text(0.05, 0.95, fr'$p_{{KS}} = {ks_anc.pvalue:.2g}$', transform=ax_right.transAxes, va="top", fontsize=12)

    for ax in [ax_left, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))


def main():
    (d0, d2), (d2_new, d15) = load_baym_data()

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Row 1: 0K vs 2K
    create_overlapping_dfes_ben(ax1, ax2, d0, d2, label_anc="0K", label_evo="2K")

    # Row 2: 2K vs 15K
    create_overlapping_dfes_ben(ax3, ax4, d2_new, d15, label_anc="2K", label_evo="15K")

    # Labels
    panel_labels = ['A', 'B', 'C', 'D']
    for ax, label in zip([ax1, ax2, ax3, ax4], panel_labels):
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontweight='heavy', va='top', ha='left', fontsize=18)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_visible(True)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # use scientific notation if outside [1e-2, 1e2]
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_path = os.path.join(out_dir, "figS9_couce_scramble_ben.svg")
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()