import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from cmn.cmn import compute_sigma_from_hist
from cmn.cmn_fgm import Fisher
from cmn.cmn_plots import create_overlapping_dfes_sim, create_segben_sim
from cmn.cmn_sk import compute_dfe


# FGM parameters
FGM_N = 8
FGM_SIGMA = 0.05
FGM_M = 8 * 10 ** 3
FGM_RANDOM_STATE = 1
FGM_T1 = 0.75
FGM_T2 = 0.85
FGM_XLIM = 0.125

# SK parameters
SK_FILE = "N4000_rho100_beta100_repeats50.pkl"
SK_ENTRY = 0
SK_T1 = 0.1
SK_T2 = 0.5
SK_XLIM = 10

# NK parameters
NK_FILE = "N_2000_K_4_repeats_100.pkl"
NK_ENTRY = 2
NK_T1 = 0.1
NK_T2 = 0.5
NK_XLIM = 0.01

# Output parameters
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figs_paper")
OUTPUT_FILE = "fig2_scrambling_sim_res.pdf"


plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

# Set the figure and axes
fig = plt.figure(figsize=(18, 16), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)

# Add subplots for each row (FGM, SK, NK)
# FGM (First row)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# SK (Second row)
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1])
axF = fig.add_subplot(gs[1, 2])

# NK (Third row)
axG = fig.add_subplot(gs[2, 0])
axH = fig.add_subplot(gs[2, 1])
axI = fig.add_subplot(gs[2, 2])

# Technical details for each subplot
axs = [axA, axB, axC, axD, axE, axF, axG, axH, axI]
ax_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
for ax, label in zip(axs, ax_labels):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=18, fontweight="bold")
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    if ax not in (axA, axD, axG):
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# FGM simulation
fgm = Fisher(n=FGM_N, sigma=FGM_SIGMA, m=FGM_M, random_state=FGM_RANDOM_STATE)
flips, traj, dfes = fgm.relax()
ind1 = int(FGM_T1 * (len(dfes) - 1))
ind2 = int(FGM_T2 * (len(dfes) - 1))
fgm_dfe1 = dfes[ind1]
fgm_dfe2 = dfes[ind2]

# # SK data
# res_directory = os.path.join(os.path.dirname(__file__), "..", "data", "SK")
# data_file_sk = os.path.join(res_directory, SK_FILE)
# with open(data_file_sk, "rb") as f:
#     data_sk = pickle.load(f)
# data_entry = data_sk[SK_ENTRY]
# alpha_initial = data_entry["init_alpha"]
# h = data_entry["h"]
# J = data_entry["J"]
# flip_seq = data_entry["flip_seq"]
# ind1 = int(SK_T1 * (len(flip_seq) - 1))
# ind2 = int(SK_T2 * (len(flip_seq) - 1))
# sig1 = compute_sigma_from_hist(alpha_initial, flip_seq, t=ind1)
# sig2 = compute_sigma_from_hist(alpha_initial, flip_seq, t=ind2)
# sk_dfe1 = compute_dfe(sig1, h, J)
# sk_dfe2 = compute_dfe(sig2, h, J)
#
# # NK data
# res_directory = os.path.join(os.path.dirname(__file__), "..", "data", "NK")
# data_file_nk = os.path.join(res_directory, NK_FILE)
# with open(data_file_nk, "rb") as f:
#     data_nk = pickle.load(f)
# data_entry = data_nk[NK_ENTRY]
# flip_seq = data_entry["flip_seq"]
# ind1 = int(NK_T1 * (len(flip_seq) - 1))
# ind2 = int(NK_T2 * (len(flip_seq) - 1))
# nk_dfe1 = data_entry["dfes"][ind1]
# nk_dfe2 = data_entry["dfes"][ind2]

# FGM Plots
create_segben_sim(
    axA,
    fgm_dfe1,
    fgm_dfe2,
    labels=(rf"$t_1 = {int(FGM_T1 * 100)}\%$", rf"$t_2 = {int(FGM_T2 * 100)}\%$"),
)
create_overlapping_dfes_sim(axB, axC, fgm_dfe1, fgm_dfe2, xlim=FGM_XLIM)

# # SK Plots
# create_segben_sim(
#     axD,
#     sk_dfe1,
#     sk_dfe2,
#     labels=(rf"$t_1 = {int(SK_T1 * 100)}\%$", rf"$t_2 = {int(SK_T2 * 100)}\%$"),
# )
# create_overlapping_dfes_sim(axE, axF, sk_dfe1, sk_dfe2, xlim=SK_XLIM)

# # NK Plots
# create_segben_sim(
#     axG,
#     nk_dfe1,
#     nk_dfe2,
#     labels=(rf"$t_1 = {int(NK_T1 * 100)}\%$", rf"$t_2 = {int(NK_T2 * 100)}\%$"),
# )
# create_overlapping_dfes_sim(axH, axI, nk_dfe1, nk_dfe2, xlim=NK_XLIM)

# Save the figure
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE), format="pdf", bbox_inches="tight")
