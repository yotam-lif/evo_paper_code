import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --- Data Processing (from beneficials_across_backgrounds.R) ---

# Load datasets (tab-separated, with header)
Rtable = pd.read_csv("Rfitted_fil.txt", sep="\t")
Ttable = pd.read_csv("2Kfitted_fil.txt", sep="\t")
Ftable = pd.read_csv("15Kfitted_fil.txt", sep="\t")

# Remove rows with NA in 'fitted1'
Rtable = Rtable.dropna(subset=["fitted1"])
Ttable = Ttable.dropna(subset=["fitted1"])
Ftable = Ftable.dropna(subset=["fitted1"])

# Filter beneficial alleles (fitted1 in (0.015, 0.3] and abn > 1)
Rben = Rtable[(Rtable["fitted1"] <= 0.3) & (Rtable["fitted1"] > 0.015) & (Rtable["abn"] > 1)].copy()
Tben = Ttable[(Ttable["fitted1"] <= 0.3) & (Ttable["fitted1"] > 0.015) & (Ttable["abn"] > 1)].copy()
Fben = Ftable[(Ftable["fitted1"] <= 0.3) & (Ftable["fitted1"] > 0.015) & (Ftable["abn"] > 1)].copy()

# Remove duplicate sites to avoid counting overlaps twice
Rben = Rben.drop_duplicates(subset=["site"])
Tben = Tben.drop_duplicates(subset=["site"])
Fben = Fben.drop_duplicates(subset=["site"])

# Create data frames for allele comparisons:
# Repi: top alleles in the ancestor
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

# Tepi: top alleles in 2K
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

# Fepi: top alleles in 15K (using same number of rows as Tepi, as in the R code)
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

# --- Plotting segben.png (from beneficials_across_backgrounds.R) ---

# Figure settings: dimensions as in the R code (width=(15.63/2)*0.85 in, height=8.54/1.9 in, res=300)
fig, axes = plt.subplots(1, 2, figsize=((15.63/2)*0.85, 8.54/1.9), dpi=300)
plt.rcParams["font.family"] = "sans-serif"
label_fontsize = 16
tick_fontsize = 14

# Panel 1: Plotting top beneficial alleles in the ancestor and connecting to 2K
ax = axes[0]
ax.set_xlim(0.8, 2.2)
ax.set_ylim(-0.15, 0.1)
ax.set_ylabel("selection coeff. (s)", fontsize=label_fontsize)
ax.tick_params(labelsize=tick_fontsize)

# Plot ancestral fitness values (Repi column R) at x=1
ax.plot(np.repeat(1, len(Repi)), Repi["R"], 'o', markersize=5, color="black", markeredgewidth=0.5)
ax.axhline(0, linestyle="--", color="black")

# Draw arrows from ancestral (x=1) to 2K fitness (x=2)
for i, row in Repi.iterrows():
    if pd.notna(row["R"]) and pd.notna(row["M"]):
        start = (1, row["R"])
        end = (2, row["M"])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=(0, 0, 0, 0.5), lw=0.75)
        ax.add_patch(arrow)

# Plot 2K fitness values (from Tepi column M) at x=2
ax.plot(np.repeat(2, len(Tepi)), Tepi["M"], 'o', markersize=5, color="#cd6e6c", markeredgewidth=0.5)

# Draw arrows from 2K (x=2) back to ancestral (x=1)
for i, row in Tepi.iterrows():
    if pd.notna(row["M"]) and pd.notna(row["R"]):
        start = (2, row["M"])
        end = (1, row["R"])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=(0.8, 0.43, 0.42, 0.5), lw=0.75)
        ax.add_patch(arrow)

# Panel 2: Plotting top beneficial alleles in 2K and connecting to 15K
ax2 = axes[1]
ax2.set_xlim(0.8, 2.2)
ax2.set_ylim(-0.15, 0.1)
ax2.set_ylabel("selection coeff. (s)", fontsize=label_fontsize)
ax2.tick_params(labelsize=tick_fontsize)

# Plot 2K fitness values (Tepi column M) at x=1
ax2.plot(np.repeat(1, len(Tepi)), Tepi["M"], 'o', markersize=5, color="#cd6e6c", markeredgewidth=0.5)
ax2.axhline(0, linestyle="--", color="black")

# Draw arrows from 2K (x=1) to 15K (x=2)
for i, row in Tepi.iterrows():
    if pd.notna(row["M"]) and pd.notna(row["K"]):
        start = (1, row["M"])
        end = (2, row["K"])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=(0.8, 0.43, 0.42, 0.5), lw=0.75)
        ax2.add_patch(arrow)

# Plot 15K fitness values (Fepi column K) at x=2
ax2.plot(np.repeat(2, len(Fepi)), Fepi["K"], 'o', markersize=5, color=(0.42, 0.57, 0.8, 0.5), markeredgewidth=0.5)

# Draw arrows from 15K (x=2) back to 2K (x=1)
for i, row in Fepi.iterrows():
    if pd.notna(row["K"]) and pd.notna(row["M"]):
        start = (2, row["K"])
        end = (1, row["M"])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=(0.42, 0.57, 0.8, 0.5), lw=0.75)
        ax2.add_patch(arrow)

plt.tight_layout()
plt.savefig("segben.png")
plt.close(fig)
