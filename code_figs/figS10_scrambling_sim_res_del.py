import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from cmn.cmn import compute_sigma_from_hist
import cmn.cmn_pspin as cmn_pspin
from cmn.cmn_plots import create_segben_sim, create_overlapping_dfes_sim
import matplotlib as mpl
import pickle


# Deleterious counterpart of fig2: same selection-coefficient transform, plotted on the
# deleterious side (ben=False). Each DFE is turned into selection coefficients
# s_i = ΔF_i / F(σ), F(σ) the fitness at that point in the walk with the initial fitness
# pinned to 1. Per model:
#   * FGM   : divide the stored effects by the FGM fitness w(r) = exp(-|r|^2) at traj[k]
#             (traj[k] aligns with dfes[k]).
#   * p-spin: recompute from the stored landscape J via cmn_pspin.compute_dfe(..., sel_coeff=True);
#             the offset pins the initial fitness to 1 (cmn_pspin.compute_fit_off).
#   * NK    : the landscape is NOT stored, so reconstruct the fitness from the walk,
#             F(σ_t) = 1 + Σ_{k<t} dfes[k][flip_seq[k]], and divide.


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

# Add subplots for each row (FGM, SK, NK)
# FGM (First row)
axA = fig.add_subplot(gs[0, 0])  # Placeholder for subplot A
axB = fig.add_subplot(gs[0, 1])  # Placeholder for subplot B
axC = fig.add_subplot(gs[0, 2])  # Placeholder for subplot C

# SK (Second row)
axD = fig.add_subplot(gs[1, 0])  # Subplot D for FGM
axE = fig.add_subplot(gs[1, 1])  # Subplot E for FGM
axF = fig.add_subplot(gs[1, 2])  # Subplot F for FGM

# NK (Third row)
axG = fig.add_subplot(gs[2, 0])  # Placeholder for subplot D of SK
axH = fig.add_subplot(gs[2, 1])  # Placeholder for subplot E of SK
axI = fig.add_subplot(gs[2, 2])  # Placeholder for subplot F of SK

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

# FGM Params
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'FGM')
data_file_fgm = os.path.join(res_directory, 'fgm_rps1000_n8_sig0.05.pkl')
with open(data_file_fgm, 'rb') as f:
    data_fgm = pickle.load(f)
entry = 10
data_entry = data_fgm[entry]
flips = data_entry['flips']
traj = data_entry['traj']
dfes = data_entry['dfes']
fgm_t1 = 0.7
fgm_t2 = 0.8
ind1 = int(fgm_t1 * (len(dfes) - 1))
ind2 = int(fgm_t2 * (len(dfes) - 1))
# Convert to selection coefficients: divide each absolute effect by the FGM fitness
# w(r) = exp(-|r|^2) at that point in the trajectory (traj[k] aligns with dfes[k]).
fgm_dfe1 = np.asarray(dfes[ind1]) / np.exp(-np.dot(traj[ind1], traj[ind1]))
fgm_dfe2 = np.asarray(dfes[ind2]) / np.exp(-np.dot(traj[ind2], traj[ind2]))

# SK data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'PSPIN')
data_file_sk = os.path.join(res_directory, 'N2000_P2_pure_repeats10.pkl')
with open(data_file_sk, 'rb') as f:
    data_sk = pickle.load(f)
entry = 2
data_entry = data_sk[entry]
sigma_initial = data_entry.get('init_sigma', data_entry.get('init_alpha'))
J = data_entry['J']
flip_seq = data_entry['flip_seq']
sk_t1 = 0.3
sk_t2 = 0.5
ind1 = int(sk_t1 * (len(flip_seq) - 1))
ind2 = int(sk_t2 * (len(flip_seq) - 1))
sig1 = compute_sigma_from_hist(sigma_initial, flip_seq, t=ind1)
sig2 = compute_sigma_from_hist(sigma_initial, flip_seq, t=ind2)
# Offset so the initial configuration has fitness 1, then let compute_dfe divide each
# effect by the fitness at that time (sel_coeff=True) -> selection coefficients.
f_off_sk = cmn_pspin.compute_fit_off(sigma_initial, J)
sk_dfe1 = cmn_pspin.compute_dfe(sig1, J, f_off=f_off_sk, sel_coeff=True)
sk_dfe2 = cmn_pspin.compute_dfe(sig2, J, f_off=f_off_sk, sel_coeff=True)

# NK data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'NK')
data_file_nk = os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl')
with open(data_file_nk, 'rb') as f:
    data_nk = pickle.load(f)
entry = 0
data_entry = data_nk[entry]
flip_seq = data_entry['flip_seq']
nk_t1 = 0.1
nk_t2 = 0.5
ind1 = int(nk_t1 * (len(flip_seq) - 1))
ind2 = int(nk_t2 * (len(flip_seq) - 1))
# The NK landscape is not stored, so reconstruct the fitness from the walk itself.
# Each accepted flip k changes the fitness by dfes[k][flip_seq[k]], and the initial
# fitness is pinned to 1, so F(sigma_t) = 1 + sum_{k<t} dfes[k][flip_seq[k]].
nk_dfes = data_entry['dfes']
nk_gains = np.array([nk_dfes[k][flip_seq[k]] for k in range(len(flip_seq))])
nk_fitness = 1.0 + np.concatenate([[0.0], np.cumsum(nk_gains)])  # F(sigma_t), len == len(dfes)
nk_dfe1 = np.asarray(nk_dfes[ind1]) / nk_fitness[ind1]
nk_dfe2 = np.asarray(nk_dfes[ind2]) / nk_fitness[ind2]

# FGM Plots
create_segben_sim(axA, fgm_dfe1, fgm_dfe2, labels=(rf'$t_1$', rf'$t_2$'), ben=False)
create_overlapping_dfes_sim(axB, axC, fgm_dfe1, fgm_dfe2, xlim=0.3, ben=False)
# SK Plots
create_segben_sim(axD, sk_dfe1, sk_dfe2, labels=(rf'$t_1$', rf'$t_2$'), ben=False)
create_overlapping_dfes_sim(axE, axF, sk_dfe1, sk_dfe2, xlim=15e-3, ben=False)
# NK Plots
create_segben_sim(axG, nk_dfe1, nk_dfe2, labels=(rf'$t_1$', rf'$t_2$'), ben=False)
create_overlapping_dfes_sim(axH, axI, nk_dfe1, nk_dfe2, xlim=2e-2, ben=False)

# Save the figure
output_dir = "../figs_paper"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "figS10_scrambling_sim_res_del.pdf"), format="pdf", bbox_inches="tight")
