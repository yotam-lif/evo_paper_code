import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib as mpl

# set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
color = sns.color_palette('CMRmap', 5)
EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.75)
ANC_FILL = (0.5, 0.5, 0.5, 0.4)

def create_dfe_comparison_ridgeplot(ax_container):
    """
    Creates a ridgeline plot with overlapping KDE curves.
    Each population is plotted in a transparent subplot so that
    only the KDEs overlap (not the white subplot backgrounds).
    Every facet displays an x-axis line (thicker than before). Only the bottom facet shows numeric tick labels,
    while all facets have the population label (e.g., "Anc", "Ara–1", etc.) added inside.
    """
    # --- Data Loading & Processing ---
    datapath = os.path.join('..', 'gen_data', 'anurag_data', 'Analysis',
                            'Part_3_TnSeq_analysis', 'Processed_data_for_plotting')
    dfe_data_csv = os.path.join(datapath, "dfe_data_pandas.csv")
    dfe_data = pd.read_csv(dfe_data_csv)
    if "Fitness estimate" in dfe_data.columns:
        dfe_data.rename(columns={'Fitness estimate': 'Fitness effect'}, inplace=True)

    # Combine Ara- and Ara+ gen_data
    dfe_data_minus = dfe_data[dfe_data['Ara Phenotype'] == 'Ara-']
    dfe_data_plus = dfe_data[dfe_data['Ara Phenotype'] == 'Ara+']
    dfe_data = pd.concat([dfe_data_minus, dfe_data_plus])

    # Rename populations.
    pop_names_old = ["REL606", "REL607", "Ara-1", "Ara-2", "Ara-3",
                     "Ara-4", "Ara-5", "Ara-6", "Ara+1", "Ara+2",
                     "Ara+3", "Ara+4", "Ara+5", "Ara+6"]
    libraries2 = ["Anc", "Anc*", "Ara–1", "Ara–2", "Ara–3",
                  "Ara–4", "Ara–5", "Ara–6", "Ara+1", "Ara+2",
                  "Ara+3", "Ara+4", "Ara+5", "Ara+6"]
    for i, old in enumerate(pop_names_old):
        new = libraries2[i]
        condition = dfe_data['Population'] == old
        dfe_data.loc[condition, 'Population'] = new

    # Exclude unwanted populations.
    dfe_data = dfe_data[~dfe_data['Population'].isin(['Ara–2', 'Ara+4'])]

    # Filter near-neutral fitness effects.
    dfe_ridge_data = dfe_data[(dfe_data["Fitness effect"] < 0.05) &
                              (dfe_data["Fitness effect"] > -0.1)].copy()

    # --- Define Order ---
    k_reordered = [0, 2, 4, 5, 6, 7, 1, 8, 9, 10, 12, 13]
    pop_order = [libraries2[k] for k in k_reordered if libraries2[k] not in ['Ara–2', 'Ara+4']]
    pop_order = pop_order[::-1]  # reverse order

    dfe_ridge_data['Population'] = pd.Categorical(
        dfe_ridge_data['Population'],
        categories=pop_order,
        ordered=True
    )

    color_anc = ANC_FILL
    color_other = color[0]

    # --- Create Nested Grid with Negative Vertical Spacing ---
    fig = ax_container.figure
    container_spec = ax_container.get_subplotspec()
    num_facets = len(pop_order)
    # Use negative hspace to force overlapping
    inner_gs = GridSpecFromSubplotSpec(nrows=num_facets, ncols=1,
                                       subplot_spec=container_spec,
                                       hspace=-0.5)

    # Hide the container axis
    ax_container.set_visible(False)

    # --- Plot Each Facet ---
    for i, pop in enumerate(pop_order):
        ax_facet = fig.add_subplot(inner_gs[i])
        # Make the subplot background transparent
        ax_facet.set_facecolor("none")
        # Set the zorder so that later subplots appear on top
        ax_facet.set_zorder(i)

        pop_data = dfe_ridge_data[dfe_ridge_data["Population"] == pop]

        # Determine fill color based on population
        if pop in ["Anc", "Anc*"]:
            fill_color = color_anc
        else:
            fill_color = color_other

        # Plot the filled KDE curve
        sns.kdeplot(
            data=pop_data, x="Fitness effect",
            bw_adjust=0.5, clip=(-0.1, 0.05),
            fill=True, alpha=1, linewidth=2,
            color=fill_color, ax=ax_facet
        )
        # Plot the white outline for clarity
        sns.kdeplot(
            data=pop_data, x="Fitness effect",
            bw_adjust=0.25, clip=(-0.1, 0.05),
            color="white", lw=1.25, ax=ax_facet
        )

        # Remove y-axis ticks and labels
        ax_facet.set_ylabel("")
        ax_facet.set_yticks([])
        for spine in ax_facet.spines.values():
            spine.set_visible(False)

        # Ensure the bottom spine (x-axis line) is visible and thicker
        ax_facet.spines['bottom'].set_visible(True)
        ax_facet.spines['bottom'].set_linewidth(2)

        # Make the x-axis visible on every facet.
        if i < num_facets - 1:
            # For non-bottom facets: show the x-axis line without tick marks or labels.
            ax_facet.tick_params(axis='x', which='both', length=0, labelbottom=False)
            ax_facet.set_xlabel("")
        else:
            # For the bottom facet: show tick labels (numeric values) without tick marks.
            ax_facet.set_xlim(-0.1, 0.05)
            ax_facet.set_xticks([-0.1, -0.05, 0, 0.05])
            ax_facet.tick_params(axis='x', which='both', length=0, labelbottom=True)
            ax_facet.set_xlabel(r'Fitness effect $(\Delta)$')

        # Add the population label inside each facet.
        ax_facet.text(0.03, 0.2, pop, color='black', size=18,
                      ha="left", va="center", transform=ax_facet.transAxes)

def main():
    # Create a figure with 2 rows and 3 columns.
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)

    # Merge the left-most cells ([0,0] and [1,0]) into one tall axis.
    ax_ridge = fig.add_subplot(gs[:, 0])  # spans both rows in column 0

    # Create the remaining four axes.
    ax_top_middle = fig.add_subplot(gs[0, 1])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_bottom_middle = fig.add_subplot(gs[1, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 2])

    # Plot the ridgeline (subfigure A)
    # create_dfe_comparison_ridgeplot(ax_ridge)

    # Plot segben subfigures on axes B and C.
    create_segben(ax_top_middle, ax_top_right)

    # Pass the two axes to the plotting function.
    # create_overlapping_dfes(ax_bottom_middle, ax_bottom_right)

    # Panel labels
    labels = {
        ax_ridge: "A",
        ax_top_middle: "B",
        ax_top_right: "C",
        ax_bottom_middle: "D",
        ax_bottom_right: "E"
    }
    for ax, label in labels.items():
        ax.text(-0.01, 1.1, label, transform=ax.transAxes, fontweight='heavy', va='top', ha='left')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', width=1.5)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.tick_params(axis='both', which='minor', length=5, width=1.6)
    # Title A is special
    ax_top_middle.text(-1.1, 0.125, "A", fontweight='heavy', va='top', ha='left')

    # Save the figure.
    output_dir = os.path.join('..', 'figs_paper')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baym_results.svg")
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()