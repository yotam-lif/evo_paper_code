import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import cramervonmises_2samp

from cmn.cmn import curate_sigma_list
from cmn.cmn_sk import compute_dfe

# -----------------------------------------------------------------------------
# Global style (match other project figures)
# -----------------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

# Number of time points along the adaptive walk
NUM_POINTS = 30

# Color palettes
MODEL_COLORS = sns.color_palette("CMRmap", 3)    # FGM, SK, NK
rng = np.random.default_rng(0)  # only used if compute_random is used

# -----------------------------------------------------------------------------
# Helpers for DFE subsets
# -----------------------------------------------------------------------------
def compute_bdfe(dfe):
    """Return beneficial part of DFE and its indices."""
    dfe = np.asarray(dfe, dtype=float)
    mask = dfe > 0
    return dfe[mask], np.nonzero(mask)[0]


def compute_deleterious(dfe):
    """Return deleterious part of DFE and its indices."""
    dfe = np.asarray(dfe, dtype=float)
    mask = dfe < 0
    return dfe[mask], np.nonzero(mask)[0]


def compute_random(dfe, size):
    """Return a random subset of DFE entries and their indices."""
    dfe = np.asarray(dfe, dtype=float)
    indices = rng.choice(len(dfe), size=size, replace=False)
    return dfe[indices], indices


# -----------------------------------------------------------------------------
# Distance over time for one replicate
# -----------------------------------------------------------------------------
def compute_dfe_convergence(dfes, num_points, metric_func):
    """
    dfes: list/array of DFE arrays across time for ONE replicate.
    metric_func: e.g. compute_bdfe or compute_deleterious

    Returns
    -------
    distances : np.ndarray, shape (num_points,)
        CvM distance between
        - the full DFE at time t
        - the DFE restricted to the subset defined at time 0
          (beneficial or deleterious set at t=0)
    """
    # Subset at time 0
    initial_dfe, initial_indices = metric_func(dfes[0])

    # Choose time indices for sampling
    indx_list = np.linspace(0, len(dfes) - 2, num_points, dtype=int)
    sampled_dfes = np.asarray(dfes, dtype=object)[indx_list]

    distances = np.zeros(num_points, dtype=float)

    for i in range(num_points):
        dfe_i = np.asarray(sampled_dfes[i], dtype=float)

        # Normalize full DFE to sum to 1 (if there is any mass)
        total = dfe_i.sum()
        if total > 0:
            dfe_i = dfe_i / total

        # Apply the same subset indices as at t=0
        if len(initial_indices) > 0:
            metric_dfe_i = dfe_i[initial_indices]
        else:
            metric_dfe_i = dfe_i

        subset_total = metric_dfe_i.sum()
        if subset_total > 0:
            metric_dfe_i = metric_dfe_i / subset_total

        distances[i] = cramervonmises_2samp(dfe_i, metric_dfe_i).statistic

    return distances


# -----------------------------------------------------------------------------
# Load base data for FGM, SK, NK
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# FGM
res_directory_fgm = os.path.join(SCRIPT_DIR, "..", "data", "FGM")
data_file_fgm = os.path.join(res_directory_fgm, "fgm_rps1000_n4_sig0.05_m2000.pkl")
with open(data_file_fgm, "rb") as f:
    data_fgm = pickle.load(f)

# For FGM and NK, each entry contains a list of dfes over time; we sample NUM_POINTS
fgm_dfes = [
    np.asarray(entry["dfes"], dtype=object)[
        np.linspace(0, len(entry["dfes"]) - 2, NUM_POINTS, dtype=int)
    ]
    for entry in data_fgm
]

# SK
res_directory_sk = os.path.join(SCRIPT_DIR, "..", "data", "SK")
data_file_sk = os.path.join(res_directory_sk, "N4000_rho100_beta100_repeats50.pkl")
with open(data_file_sk, "rb") as f:
    data_sk = pickle.load(f)

sk_dfes = []
for entry in data_sk:
    sampled_sigma_list = curate_sigma_list(
        entry["init_alpha"],
        entry["flip_seq"],
        np.linspace(0, len(entry["flip_seq"]) - 2, NUM_POINTS, dtype=int),
    )
    # compute DFE at each sampled time
    sk_dfes.append(
        [compute_dfe(sigma, entry["h"], entry["J"]) for sigma in sampled_sigma_list]
    )

# NK (base K = 32)
res_directory_nk = os.path.join(SCRIPT_DIR, "..", "data", "NK")
data_file_nk = os.path.join(res_directory_nk, "N_2000_K_32_repeats_100.pkl")
with open(data_file_nk, "rb") as f:
    data_nk = pickle.load(f)

nk_dfes = [
    np.asarray(entry["dfes"], dtype=object)[
        np.linspace(0, len(entry["dfes"]) - 2, NUM_POINTS, dtype=int)
    ]
    for entry in data_nk
]


# -----------------------------------------------------------------------------
# Collect distances across replicates for each model
# -----------------------------------------------------------------------------
def collect_model_convergence(dfes_list, metric_func):
    """
    dfes_list: list of replicates; each replicate is a sequence of DFEs over time.
    metric_func: e.g. compute_bdfe or compute_deleterious

    Returns
    -------
    distances : np.ndarray, shape (n_reps, NUM_POINTS)
    """
    return np.array(
        [compute_dfe_convergence(dfes, NUM_POINTS, metric_func) for dfes in dfes_list],
        dtype=float,
    )


def per_replicate_normalized_stats(arr, start_index=0):
    """
    arr: shape (n_reps, NUM_POINTS). Each row is a distance curve for one replicate.

    Normalizes each replicate by its initial value arr[:, 0], then
    returns mean and std across replicates for the slice [start_index:].
    """
    arr = np.asarray(arr, dtype=float)
    init = arr[:, 0]  # shape (n_reps,)

    # Per-replicate normalization by initial value
    arr_norm = np.full_like(arr, np.nan)
    nonzero_mask = init != 0
    if np.any(nonzero_mask):
        arr_norm[nonzero_mask] = arr[nonzero_mask] / init[nonzero_mask, None]

    arr_norm_sliced = arr_norm[:, start_index:]
    mean_norm = np.nanmean(arr_norm_sliced, axis=0)
    std_norm = np.nanstd(arr_norm_sliced, axis=0, ddof=1)

    return mean_norm, std_norm


# -----------------------------------------------------------------------------
# Quantities for top-row (FGM/SK/NK) plots
# -----------------------------------------------------------------------------
# X-axis: percentage of the walk
x_full = np.linspace(0, 100, NUM_POINTS, dtype=int)
start_frac = 0.0
start_index = int(start_frac * NUM_POINTS)
x = x_full[start_index:]

# Beneficial subsets
fgm_ben = collect_model_convergence(fgm_dfes, compute_bdfe)
sk_ben = collect_model_convergence(sk_dfes, compute_bdfe)
nk_ben = collect_model_convergence(nk_dfes, compute_bdfe)

fgm_ben_mean, fgm_ben_std = per_replicate_normalized_stats(fgm_ben, start_index)
sk_ben_mean, sk_ben_std = per_replicate_normalized_stats(sk_ben, start_index)
nk_ben_mean, nk_ben_std = per_replicate_normalized_stats(nk_ben, start_index)

# Deleterious subsets
fgm_del = collect_model_convergence(fgm_dfes, compute_deleterious)
sk_del = collect_model_convergence(sk_dfes, compute_deleterious)
nk_del = collect_model_convergence(nk_dfes, compute_deleterious)

fgm_del_mean, fgm_del_std = per_replicate_normalized_stats(fgm_del, start_index)
sk_del_mean, sk_del_std = per_replicate_normalized_stats(sk_del, start_index)
nk_del_mean, nk_del_std = per_replicate_normalized_stats(nk_del, start_index)


# -----------------------------------------------------------------------------
# Parameter sweeps: FGM (n) and NK (K), beneficial subset only
# -----------------------------------------------------------------------------
FGM_N_VALUES = [4, 8, 16, 32]
NK_K_VALUES = [4, 8, 16, 32]


def load_fgm_dfes_for_n(n):
    """
    Load FGM DFE trajectories for a given dimensionality n.
    Assumes filenames of the form:
        fgm_rps100_n{n}_del0.001_s00.01.pkl
    in ../data/FGM.
    """
    fname = f"fgm_rps1000_n{n}_sig0.05_m2000.pkl"
    path = os.path.join(res_directory_fgm, fname)
    with open(path, "rb") as f:
        data = pickle.load(f)

    dfes_list = [
        np.asarray(entry["dfes"], dtype=object)[
            np.linspace(0, len(entry["dfes"]) - 2, NUM_POINTS, dtype=int)
        ]
        for entry in data
    ]
    return dfes_list


def load_nk_dfes_for_K(K):
    """
    Load NK DFE trajectories for a given K.
    Assumes filenames of the form:
        N_2000_K_{K}_repeats_100.pkl
    in ../data/NK.
    """
    fname = f"N_2000_K_{K}_repeats_100.pkl"
    path = os.path.join(res_directory_nk, fname)
    with open(path, "rb") as f:
        data = pickle.load(f)

    dfes_list = [
        np.asarray(entry["dfes"], dtype=object)[
            np.linspace(0, len(entry["dfes"]) - 2, NUM_POINTS, dtype=int)
        ]
        for entry in data
    ]
    return dfes_list


# Precompute mean/std for beneficial CvM vs DFE for different n (FGM) and K (NK)
fgm_n_means = {}
fgm_n_stds = {}
for n in FGM_N_VALUES:
    dfes_n = load_fgm_dfes_for_n(n)
    distances_n = collect_model_convergence(dfes_n, compute_bdfe)
    mean_n, std_n = per_replicate_normalized_stats(distances_n, start_index)
    fgm_n_means[n] = mean_n
    fgm_n_stds[n] = std_n

nk_K_means = {}
nk_K_stds = {}
for K in NK_K_VALUES:
    dfes_K = load_nk_dfes_for_K(K)
    distances_K = collect_model_convergence(dfes_K, compute_bdfe)
    mean_K, std_K = per_replicate_normalized_stats(distances_K, start_index)
    nk_K_means[K] = mean_K
    nk_K_stds[K] = std_K


# -----------------------------------------------------------------------------
# Axis formatting helper (match other project figs)
# -----------------------------------------------------------------------------
def format_axes(ax_list):
    for ax in ax_list:
        # Scientific formatting for x (always) and y (when needed)
        # ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
        # ax.xaxis.get_offset_text().set_visible(True)
        #
        # y_formatter = ScalarFormatter(useMathText=True)
        # y_formatter.set_powerlimits((-1, 1))
        # ax.yaxis.set_major_formatter(y_formatter)

        # Consistent ticks & spines
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")

        for sp in ax.spines.values():
            sp.set_linewidth(1.5)


# -----------------------------------------------------------------------------
# Plotting: 2 × 2 figure
#   Row 1:  [beneficial FGM/SK/NK]  [deleterious FGM/SK/NK]
#   Row 2:  [FGM, different n]     [NK, different K]
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

# --- Row 1: models (FGM/SK/NK) ---

# Beneficial
ax = axes[0, 0]
ax.errorbar(x, fgm_ben_mean, yerr=fgm_ben_std, fmt="-o", label="FGM", color=MODEL_COLORS[0])
ax.errorbar(x, sk_ben_mean,  yerr=sk_ben_std,  fmt="-o", label="SK",  color=MODEL_COLORS[1])
ax.errorbar(x, nk_ben_mean,  yerr=nk_ben_std,  fmt="-o", label="NK",  color=MODEL_COLORS[2])
ax.set_ylabel("CvM distance (normalized)")
ax.legend(frameon=False)

# Deleterious
ax = axes[0, 1]
ax.errorbar(x, fgm_del_mean, yerr=fgm_del_std, fmt="-o", label="FGM", color=MODEL_COLORS[0])
ax.errorbar(x, sk_del_mean,  yerr=sk_del_std,  fmt="-o", label="SK",  color=MODEL_COLORS[1])
ax.errorbar(x, nk_del_mean,  yerr=nk_del_std,  fmt="-o", label="NK",  color=MODEL_COLORS[2])
ax.legend(frameon=False)

# --- Row 2: parameter sweeps ---

# FGM: different n (beneficial vs DFE)
ax = axes[1, 0]
colors_n = sns.color_palette("CMRmap", len(FGM_N_VALUES))
for i, n in enumerate(FGM_N_VALUES):
    ax.errorbar(
        x,
        fgm_n_means[n],
        yerr=fgm_n_stds[n],
        fmt="-o",
        label=fr"$n={n}$",
        color=colors_n[i],
    )
ax.set_xlabel("Evolutionary time (%)")
ax.set_ylabel("CvM distance (normalized)")
ax.legend(frameon=False)

# NK: different K (beneficial vs DFE)
ax = axes[1, 1]
colors_K = sns.color_palette("CMRmap", len(NK_K_VALUES))
for i, K in enumerate(NK_K_VALUES):
    ax.errorbar(
        x,
        nk_K_means[K],
        yerr=nk_K_stds[K],
        fmt="-o",
        label=fr"$K={K}$",
        color=colors_K[i],
    )
ax.set_xlabel("Evolutionary time (%)")
ax.legend(frameon=False)

# Apply consistent cosmetics
format_axes(axes.flatten())

plt.tight_layout()
out_dir = os.path.join('..', 'figs_paper')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figS7_scrambling_quant.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
print(f"Saved: {out_path}")
