import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------------
# Data Processing (from beneficials_across_backgrounds.R)
# -------------------------------

# Load datasets
Rtable = pd.read_csv("Rfitted_fil.txt", sep="\t")
Ttable = pd.read_csv("2Kfitted_fil.txt", sep="\t")
Ftable = pd.read_csv("15Kfitted_fil.txt", sep="\t")

# Remove rows with NA in 'fitted1'
Rtable = Rtable.dropna(subset=["fitted1"])
Ttable = Ttable.dropna(subset=["fitted1"])
Ftable = Ftable.dropna(subset=["fitted1"])

# Filter beneficial alleles: fitted1 in (0.015, 0.3] and abn > 1
Rben = Rtable[(Rtable["fitted1"] <= 0.3) & (Rtable["fitted1"] > 0.015) & (Rtable["abn"] > 1)].copy()
Tben = Ttable[(Ttable["fitted1"] <= 0.3) & (Ttable["fitted1"] > 0.015) & (Ttable["abn"] > 1)].copy()
Fben = Ftable[(Ftable["fitted1"] <= 0.3) & (Ftable["fitted1"] > 0.015) & (Ftable["abn"] > 1)].copy()

# Remove duplicate sites
Rben = Rben.drop_duplicates(subset=["site"])
Tben = Tben.drop_duplicates(subset=["site"])
Fben = Fben.drop_duplicates(subset=["site"])

# Build data frames for allele comparisons
Rnames = Rben["alle"].values
Repi = pd.DataFrame(np.nan, index=range(len(Rnames)), columns=["R", "M", "K"])
for i, allele in enumerate(Rnames):
    r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
    if not r_val.empty:
        Repi.at[i, "R"] = r_val.iloc[0]
    t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
    if not t_val.empty:
        Repi.at[i, "M"] = t_val.iloc[0]
    f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
    if not f_val.empty:
        Repi.at[i, "K"] = f_val.iloc[0]

Tnames = Tben["alle"].values
Tepi = pd.DataFrame(np.nan, index=range(len(Tnames)), columns=["R", "M", "K"])
for i, allele in enumerate(Tnames):
    t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
    if not t_val.empty:
        Tepi.at[i, "M"] = t_val.iloc[0]
    r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
    if not r_val.empty:
        Tepi.at[i, "R"] = r_val.iloc[0]
    f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
    if not f_val.empty:
        Tepi.at[i, "K"] = f_val.iloc[0]

Fnames = Fben["alle"].values
Fepi = pd.DataFrame(np.nan, index=range(len(Tnames)), columns=["R", "M", "K"])
for i, allele in enumerate(Fnames):
    f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
    if not f_val.empty:
        Fepi.at[i, "K"] = f_val.iloc[0]
    r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
    if not r_val.empty:
        Fepi.at[i, "R"] = r_val.iloc[0]
    t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
    if not t_val.empty:
        Fepi.at[i, "M"] = t_val.iloc[0]

# Combine data sets and keep rows with >= 2 non-NA values
Repi["back"] = 1
Tepi["back"] = 2
Fepi["back"] = 3
data = pd.concat([Repi, Tepi, Fepi], ignore_index=True)
data["nas"] = data[["R", "M", "K"]].notna().sum(axis=1)
x = data[data["nas"] >= 2].copy()

# -------------------------------
# Plotting using stairs
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(680 * 0.011, 420 * 0.011), dpi=300)
plt.rcParams["font.family"] = "sans-serif"
label_fontsize = 16
tick_fontsize = 14

# Vertical shift for the "evolved" histograms
z = 30
lw_main = 1.0

def draw_custom_segments(ax):
    ax.plot([-0.09, 0.09], [z * 1.1, z * 1.1],
            linestyle="--", color="grey", lw=lw_main)
    segs = [
        ((-0.05, -0.75), (-0.05 * 0.9, z * 1.1)),
        ((0.05, -0.75), (0.05 * 0.9, z * 1.1)),
        ((-0.1, -0.75), (-0.09, z * 1.1)),
        ((0.1, -0.75), (0.09, z * 1.1)),
        ((0, -0.75), (0, z * 1.1))
    ]
    for (x0, y0), (x1, y1) in segs:
        ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

# RGBA for fill color with alpha=0.4
PURPLE_FILL = (0.5, 0.0, 0.5, 0.4)  # "purple" but 40% opacity
GREY_FILL   = (0.5, 0.5, 0.5, 0.4)  # "grey"   but 40% opacity

# Opaque black for the edge
BLACK_EDGE  = (0.0, 0.0, 0.0, 1.0)

# --------------
# Left Panel
# --------------
ax_left = axes[0]

vals_after = np.concatenate([
    x.loc[x["back"] == 1, "M"].dropna().values,
    x.loc[x["back"] == 1, "K"].dropna().values
])
counts, bin_edges = np.histogram(vals_after, bins=30)
bin_edges = bin_edges * 0.9
counts_shifted = counts + z

ax_left.set_xlim(-0.1, 0.1)
ax_left.set_ylim(0, 500)
ax_left.set_xlabel('Fitness $(\\Delta)$', fontsize=label_fontsize)
ax_left.tick_params(labelsize=tick_fontsize)

draw_custom_segments(ax_left)

# Evolved histogram (purple fill w/ partial alpha, black edge fully opaque)
ax_left.stairs(
    values=counts_shifted,
    edges=bin_edges,
    baseline=0,
    fill=True,
    facecolor=PURPLE_FILL,   # partially transparent purple
    edgecolor=BLACK_EDGE,    # fully opaque black
    lw=1.1,
    label="Evolved"
)

# Mask bottom portion
rect = Rectangle((-0.11, 0), 0.21, z, facecolor="white", edgecolor="none")
ax_left.add_patch(rect)
draw_custom_segments(ax_left)

# Ancestor histogram (grey fill w/ partial alpha, black edge fully opaque)
vals_anc = x.loc[x["back"] == 1, "R"].dropna().values
anc_counts, anc_bin_edges = np.histogram(vals_anc, bins=24)
ax_left.stairs(
    values=anc_counts,
    edges=anc_bin_edges,
    baseline=0,
    fill=True,
    facecolor=GREY_FILL,
    edgecolor=BLACK_EDGE,
    lw=1.1,
    label="Ancestor"
)

ax_left.legend(frameon=False)

# --------------
# Right Panel
# --------------
ax_right = axes[1]

vals_after2 = np.concatenate([
    x.loc[x["back"] == 2, "M"].dropna().values,
    x.loc[x["back"] == 3, "K"].dropna().values
])
counts2, bin_edges2 = np.histogram(vals_after2, bins=10)
bin_edges2 = bin_edges2 * 0.9
counts2_shifted = counts2 + z

ax_right.set_xlim(-0.1, 0.1)
ax_right.set_ylim(0, 500)
ax_right.set_xlabel("Fitness $(\\Delta)$", fontsize=label_fontsize)
ax_right.tick_params(labelsize=tick_fontsize)

draw_custom_segments(ax_right)

# Evolved histogram (purple fill, black edge)
ax_right.stairs(
    values=counts2_shifted,
    edges=bin_edges2,
    baseline=0,
    fill=True,
    facecolor=PURPLE_FILL,
    edgecolor=BLACK_EDGE,
    lw=1.1,
    label="Evolved"
)

# Mask bottom portion
rect2 = Rectangle((-0.11, 0), 0.21, z, facecolor="white", edgecolor="none")
ax_right.add_patch(rect2)
draw_custom_segments(ax_right)

# Ancestor histogram (grey fill, black edge)
vals_anc2 = np.unique(np.concatenate([
    x.loc[x["back"].isin([2, 3]), "R"].dropna().values
]))
anc2_counts, anc2_bin_edges = np.histogram(vals_anc2, bins=30)
ax_right.stairs(
    values=anc2_counts,
    edges=anc2_bin_edges,
    baseline=0,
    fill=True,
    facecolor=GREY_FILL,
    edgecolor=BLACK_EDGE,
    lw=1.1,
    label="Ancestor"
)

ax_right.legend(frameon=False)

# Detach the axes from the plot region for each subplot
for ax in [ax_left, ax_right]:
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Offset the bottom spine (x-axis) and left spine (y-axis) outward by 10 points
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Ensure ticks only appear on the bottom and left spines
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plt.tight_layout()
plt.savefig("overlapped_DFEs.png")
plt.close(fig)
