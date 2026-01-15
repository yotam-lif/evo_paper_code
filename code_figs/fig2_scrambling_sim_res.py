import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from cmn.cmn import compute_sigma_from_hist
from cmn.cmn_sk import compute_dfe
from cmn.cmn_plots import create_segben_sim, create_overlapping_dfes_sim
import matplotlib as mpl
import pickle


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
fgm_t1 = 0.75
fgm_t2 = 0.85
ind1 = int(fgm_t1 * (len(dfes) - 1))
ind2 = int(fgm_t2 * (len(dfes) - 1))
fgm_dfe1 = dfes[ind1]
fgm_dfe2 = dfes[ind2]

# SK data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'SK')
data_file_sk = os.path.join(res_directory, 'N4000_rho100_beta100_repeats50.pkl')
with open(data_file_sk, 'rb') as f:
    data_sk = pickle.load(f)
entry = 0
data_entry = data_sk[entry]
alpha_initial = data_entry['init_alpha']
h = data_entry['h']
J = data_entry['J']
flip_seq = data_entry['flip_seq']
sk_t1 = 0.1
sk_t2 = 0.5
ind1 = int(sk_t1 * (len(flip_seq) - 1))
ind2 = int(sk_t2 * (len(flip_seq) - 1))
sig1 = compute_sigma_from_hist(alpha_initial, flip_seq, t=ind1)
sig2 = compute_sigma_from_hist(alpha_initial, flip_seq, t=ind2)
sk_dfe1 = compute_dfe(sig1, h, J)
sk_dfe2 = compute_dfe(sig2, h, J)

# NK data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'NK')
data_file_nk = os.path.join(res_directory, 'N_2000_K_4_repeats_100.pkl')
with open(data_file_nk, 'rb') as f:
    data_nk = pickle.load(f)
entry = 2
data_entry = data_nk[entry]
flip_seq = data_entry['flip_seq']
nk_t1 = 0.1
nk_t2 = 0.5
ind1 = int(nk_t1 * (len(flip_seq) - 1))
ind2 = int(nk_t2 * (len(flip_seq) - 1))
nk_dfe1 = data_entry['dfes'][ind1]
nk_dfe2 = data_entry['dfes'][ind2]

# FGM Plots
create_segben_sim(axA, fgm_dfe1, fgm_dfe2, labels=(rf'$t_1 = {int(fgm_t1 * 100)}\%$', rf'$t_2 = {int(fgm_t2 * 100)}\%$'))
create_overlapping_dfes_sim(axB, axC, fgm_dfe1, fgm_dfe2, xlim=0.125)
# SK Plots
create_segben_sim(axD, sk_dfe1, sk_dfe2, labels=(rf'$t_1 = {int(sk_t1 * 100)}\%$', rf'$t_2 = {int(sk_t2 * 100)}\%$'))
create_overlapping_dfes_sim(axE, axF, sk_dfe1, sk_dfe2, xlim=10)
# NK Plots
create_segben_sim(axG, nk_dfe1, nk_dfe2, labels=(rf'$t_1 = {int(nk_t1 * 100)}\%$', rf'$t_2 = {int(nk_t2 * 100)}\%$'))
create_overlapping_dfes_sim(axH, axI, nk_dfe1, nk_dfe2, xlim=0.01)

# Save the figure
output_dir = "../figs_paper"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "fig2_scrambling_sim_res.svg"), format="svg", bbox_inches="tight")
