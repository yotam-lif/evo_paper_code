import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from cmn.cmn import compute_sigma_from_hist
from cmn.cmn_fgm import Fisher
from cmn.cmn_plots import create_overlapping_dfes_sim, create_segben_sim
from cmn import cmn_pspin


# Every DFE is plotted as the raw (absolute) fitness effect ΔF, NOT as a selection
# coefficient -- no effect is divided by the fitness at that point in the walk. Per model:
#   * FGM   : ΔF_i = w(r + delta_i) - w(r), taken straight from the live SSWM walk
#             (dfes[k] is the DFE at phenotype traj[k]).
#   * p-spin: ΔF_i for flipping each spin i, recomputed from the stored landscape J
#             via cmn_pspin.compute_dfe(sigma, J).
#   * NK    : the landscape is NOT stored, only the precomputed ΔF DFEs. These pickles
#             predate cmn_nk's switch to an extensive fitness, so they hold INTENSIVE
#             effects (fitness = mean of f_i over loci), ~N times smaller than the
#             p-spin's extensive sum over interactions. They are multiplied by N to put
#             both models on the same scale -- see the note at the NK data block.


def auto_xlim(dfe1, dfe2, q=99.0):
    """Pick a symmetric x-limit from the beneficial (positive) effects of both DFEs.

    Uses the ``q``-th percentile so a single outlier does not blow up the box.
    Returns a positive float; falls back to 1.0 if there are no positive effects.
    """
    vals = np.concatenate([np.asarray(dfe1, float), np.asarray(dfe2, float)])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    return float(np.percentile(vals, q))


# FGM parameters
FGM_N = 4
FGM_SIGMA = 0.05
FGM_M = 4 * 10 ** 3
FGM_RANDOM_STATE = 1
FGM_T1 = 0.8
FGM_T2 = 0.9
FGM_XLIM = 0.08  # x-axis half-width of the fitness-effect (ΔF) box

# p-spin parameters
SK_FILE = "N2000_P2_pure_repeats10.pkl"
SK_ENTRY = 1
SK_T1 = 0.05
SK_T2 = 0.5
SK_XLIM = 12  # x-axis half-width of the fitness-effect (ΔF) box

# NK parameters
NK_FILE = "N_2000_K_32_repeats_100.pkl"
NK_ENTRY = 2
NK_T1 = 0.05
NK_T2 = 0.5
NK_XLIM = 40  # x-axis half-width of the fitness-effect (ΔF) box, after the xN rescale

# Output parameters
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figs_paper")
OUTPUT_FILE = "figS5_scrambling_sim_res.pdf"


plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    }
)

# Set the figure and axes
fig = plt.figure(figsize=(18, 16), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)

# Add subplots for each row (SK, NK, FGM)
# SK (First row)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# NK (Second row)
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1])
axF = fig.add_subplot(gs[1, 2])

# FGM (Third row)
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
# Raw fitness effects ΔF_i = w(r + delta_i) - w(r); dfes[k] is the DFE recorded at
# phenotype traj[k] along the walk.
fgm_dfe1 = dfes[ind1]
fgm_dfe2 = dfes[ind2]

# SK data
res_directory = os.path.join(os.path.dirname(__file__), "..", "data", "PSPIN")
data_file_sk = os.path.join(res_directory, SK_FILE)
with open(data_file_sk, "rb") as f:
    data_sk = pickle.load(f)
data_entry = data_sk[SK_ENTRY]
init_sigma = data_entry["init_sigma"]
J = data_entry["J"]
flip_seq = data_entry["flip_seq"]
ind1 = int(SK_T1 * (len(flip_seq) - 1))
ind2 = int(SK_T2 * (len(flip_seq) - 1))
sig1 = compute_sigma_from_hist(init_sigma, flip_seq, t=ind1)
sig2 = compute_sigma_from_hist(init_sigma, flip_seq, t=ind2)
# Raw fitness effects ΔF_i for flipping each spin i, at the two points in the walk.
sk_dfe1 = cmn_pspin.compute_dfe(sig1, J)
sk_dfe2 = cmn_pspin.compute_dfe(sig2, J)

# NK data
res_directory = os.path.join(os.path.dirname(__file__), "..", "data", "NK")
data_file_nk = os.path.join(res_directory, NK_FILE)
with open(data_file_nk, "rb") as f:
    data_nk = pickle.load(f)
data_entry = data_nk[NK_ENTRY]
flip_seq = data_entry["flip_seq"]
dfes = data_entry["dfes"]
ind1 = int(NK_T1 * (len(flip_seq) - 1))
ind2 = int(NK_T2 * (len(flip_seq) - 1))
# The pickles under data/NK/ were generated BEFORE cmn_nk switched to the extensive
# convention, so their effects are intensive (fitness = mean of f_i over loci) and ~N
# times smaller than the p-spin's. Rescale by N to match; each stored DFE holds one entry
# per locus, so N is just its length.
# IMPORTANT: cmn_nk.NK.compute_fitness is now EXTENSIVE. If data/NK/ is ever regenerated
# with the current code its effects are already extensive -- DROP this * nk_n rescale, or
# they will be inflated by a further factor of N.
nk_n = len(dfes[ind1])
nk_dfe1 = np.asarray(dfes[ind1]) * nk_n
nk_dfe2 = np.asarray(dfes[ind2]) * nk_n

# SK Plots
create_segben_sim(
    axA,
    sk_dfe1,
    sk_dfe2,
    labels=(r"$t_1$", rf"$t_2$"),
)
create_overlapping_dfes_sim(axB, axC, sk_dfe1, sk_dfe2, xlim=SK_XLIM)

# NK Plots
create_segben_sim(
    axD,
    nk_dfe1,
    nk_dfe2,
    labels=(rf"$t_1$", rf"$t_2$"),
)
create_overlapping_dfes_sim(axE, axF, nk_dfe1, nk_dfe2, xlim=NK_XLIM)

# FGM Plots
create_segben_sim(
    axG,
    fgm_dfe1,
    fgm_dfe2,
    labels=(r"$t_1$", r"$t_2$"),
)
create_overlapping_dfes_sim(axH, axI, fgm_dfe1, fgm_dfe2, xlim=FGM_XLIM)

# Save the figure
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE), format="pdf", bbox_inches="tight")
