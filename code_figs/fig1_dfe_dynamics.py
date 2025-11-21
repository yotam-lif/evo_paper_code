import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import gaussian_kde, kstest, expon
from scipy.special import airy
from scipy.integrate import trapezoid

# ----------------------------------------------------------------
# CONFIGURATION: Speed Optimization Parameters
# ----------------------------------------------------------------
NUM_REPS_EVOL = 10  # Left panels (DFE Evolution)
NUM_REPS_FINAL = 10  # Middle panels (Final DFE)
NUM_REPS_BDFE = 50  # Right panels (BDFE analysis)

# ----------------------------------------------------------------
# Setup & Imports
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Mock imports for standalone functionality if cmn modules exist
from cmn import cmn, cmn_sk, cmn_nk
from cmn.uncmn_dfe import gen_final_dfe

# ----------------------------------------------------------------
# Global Style
# ----------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
})

# Global bin count for all BDFE histograms
CMR_COLORS = sns.color_palette('CMRmap', 5)

# ----------------------------------------------------------------
# Math & Statistics Helpers (Airy Distribution)
# ----------------------------------------------------------------
# Precomputed scale 'a' such that the mean of the normalized Airy density is 1.
AIRY_A = 0.7764582113784204
_AIRY_X_MAX = 20.0
_AIRY_GRID_SIZE = 4000
_AIRY_X = np.linspace(0.0, _AIRY_X_MAX, _AIRY_GRID_SIZE)
_AIRY_PDF = airy(AIRY_A * _AIRY_X)[0]
_AIRY_PDF[_AIRY_PDF < 0] = 0.0
_AIRY_PDF /= trapezoid(_AIRY_PDF, _AIRY_X)  # Normalize PDF
# Compute CDF for KS tests
dx = _AIRY_X[1] - _AIRY_X[0]
_AIRY_CDF = np.cumsum(_AIRY_PDF) * dx
_AIRY_CDF /= _AIRY_CDF[-1]


def airy_unit_mean_cdf(x):
    """CDF of the fixed unit-mean Airy reference distribution on x>=0."""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    if np.any(mask):
        out[mask] = np.interp(x[mask], _AIRY_X, _AIRY_CDF, left=0.0, right=1.0)
    return out


# ----------------------------------------------------------------
# Plotting Helper for BDFE Distributions with KS Test
# ----------------------------------------------------------------
def plot_bdfe_distribution(ax, samples, color, t_percent, model_type='fgm'):
    """
    Plots 'untouched' histogram of beneficial fitness effects.
    Calculates KS p-values against specific theories:
      - FGM: Exp(1) only.
      - SK/NK: Exp(1) and Airy.
    Displays p-values in the legend.
    """
    # 1. Filter for beneficial effects (Delta > 0)
    data = np.asarray(samples)
    data = data[np.isfinite(data) & (data > 0)]

    if len(data) < 3:
        return

    # 2. Normalize data by mean for shape comparison
    mu = data.mean()
    data_norm = data / mu

    # 3. Perform KS Tests
    # Test 1: Exponential (Exp(1) has mean 1)
    stat_exp, p_exp = kstest(data_norm, expon.cdf)

    if model_type in ['sk', 'nk']:
        # Test 2: Airy (Unit Mean) for SK/NK
        stat_airy, p_val_2 = kstest(data_norm, airy_unit_mean_cdf)
        label_part = rf'p_{{Exp}}={p_exp:.2g},\, p_{{Airy}}={p_val_2:.2g}'
    else:
        # FGM: Only Exponential
        label_part = rf'p_{{Exp}}={p_exp:.2g}'

    # 4. Create Label
    label = rf'$t={t_percent:.0f}\%,\; {label_part}$'

    # 5. Plot Untouched Histogram (Step style) ONLY
    sns.histplot(data, ax=ax, element="step", stat="density",
                 fill=False, color=color, label=label, common_norm=False, linewidth=2)


# ----------------------------------------------------------------
# Data Loading Section
# ----------------------------------------------------------------

def load_fgm_data():
    FGM_NS = [4, 8, 16, 32]
    fgm_data = {}
    for _n in FGM_NS:
        file_path_fgm = f'../data/FGM/fgm_rps1000_n{_n}_sig0.05_m2000.pkl'
        if os.path.exists(file_path_fgm):
            with open(file_path_fgm, 'rb') as f:
                fgm_data[_n] = pickle.load(f)
        else:
            fgm_data[_n] = []
    reps = fgm_data.get(4, [])
    final = {}
    for n, rep_list in fgm_data.items():
        all_last_dfes = []
        for rep in rep_list[:NUM_REPS_FINAL]:
            if not isinstance(rep, dict): continue
            dfes = rep.get("dfes", None)
            if dfes is None or len(dfes) == 0: continue
            all_last_dfes.extend(dfes[-1])
        final[n] = all_last_dfes
    return reps, final


def load_sk_data():
    file_path = '../data/SK/N4000_rho100_beta100_repeats50.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"SK Data file not found: {file_path}")


def load_nk_data():
    res_directory = '../data/NK'
    files = [
        (4, os.path.join(res_directory, 'N_2000_K_4_repeats_100.pkl')),
        (8, os.path.join(res_directory, 'N_2000_K_8_repeats_100.pkl')),
        (16, os.path.join(res_directory, 'N_2000_K_16_repeats_100.pkl')),
        (32, os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl'))
    ]
    data_arr = []
    K_values = [4, 8, 16, 32]
    for k, file in files:
        if os.path.exists(file):
            with open(file, 'rb') as f:
                data_arr.append(pickle.load(f))
        else:
            data_arr.append([])

    nk_single_k_data = data_arr[1]  # K=8
    return data_arr, nk_single_k_data, K_values


# ----------------------------------------------------------------
# ROW 1: FGM Panels
# ----------------------------------------------------------------

def fgm_panel_A(ax, reps, perc=(25, 50, 75, 100)):
    comb = [[] for _ in perc]
    for rep in reps[:NUM_REPS_EVOL]:
        walk_length = len(rep["dfes"])
        for i, percent in enumerate(perc):
            t_idx = int(percent * (walk_length - 1) / 100)
            comb[i].extend(rep["dfes"][t_idx])

    for i, dfe in enumerate(comb):
        dfe = np.array(dfe)
        if len(dfe) < 2: continue
        kde = gaussian_kde(dfe, bw_method=0.4)
        x = np.linspace(dfe.min(), dfe.max(), 400)
        y = kde.evaluate(x)
        ax.plot(x, y + 0.003, lw=2, color=CMR_COLORS[i % len(CMR_COLORS)], label=f"$t={perc[i]}\\%$")

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta,t)$")
    ax.legend(frameon=False)


def fgm_panel_B(ax, final):
    for i, (n, dfe) in enumerate(final.items()):
        dfe = np.asarray(dfe)
        if len(dfe) < 2: continue
        kde = gaussian_kde(dfe, bw_method=0.3)
        x = np.linspace(dfe.min(), dfe.max(), 400)
        y = kde.evaluate(x)
        ax.plot(x, y, lw=2, color=CMR_COLORS[i % len(CMR_COLORS)], label=f"$n={n}$")

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta,t=100\%)$")
    ax.set_xlim(None, 0)
    ax.legend(frameon=False, loc="upper left")


def plot_bdfe_exp_fgm(ax, reps, percents=[75, 80, 85, 90]):
    colors = CMR_COLORS
    for i, p in enumerate(percents):
        bdfe = []
        for rep in reps[:NUM_REPS_BDFE]:
            T_rep = len(rep['dfes'])
            if T_rep == 0: continue
            t_idx = int(p / 100 * (T_rep - 1))
            t_idx = max(0, min(t_idx, T_rep - 1))
            dfe_t = np.asarray(rep['dfes'][t_idx])
            bdfe.extend(dfe_t[dfe_t > 0])

        plot_bdfe_distribution(ax, bdfe, colors[i % len(colors)],
                               t_percent=p, model_type='fgm')

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')


# ----------------------------------------------------------------
# ROW 2: SK Panels
# ----------------------------------------------------------------

def sk_panel_evol(ax, data, num_points=5):
    num_repeats = min(len(data), NUM_REPS_EVOL)
    dfes = np.empty((num_repeats, num_points), dtype=object)
    percents = np.linspace(0, 100, num_points, dtype=int)

    for repeat in range(num_repeats):
        data_entry = data[repeat]
        sigma_initial = data_entry['init_alpha']
        h = data_entry['h']
        J = data_entry['J']
        flip_seq = data_entry['flip_seq']

        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in percents]
        sigma_list = cmn.curate_sigma_list(sigma_initial, flip_seq, ts)

        for idx, sigma in enumerate(sigma_list):
            dfe = cmn_sk.compute_dfe(sigma, h, J)
            if dfes[repeat, idx] is None: dfes[repeat, idx] = []
            dfes[repeat, idx].extend(dfe)

    for i in range(num_points):
        all_dfe = []
        for repeat in range(num_repeats):
            if dfes[repeat, i] is not None: all_dfe.extend(dfes[repeat, i])
        if all_dfe:
            kde = gaussian_kde(all_dfe, bw_method=0.4)
            x = np.linspace(min(all_dfe), max(all_dfe), 400)
            y = kde.evaluate(x)
            ax.plot(x, y + 0.003, lw=2, color=CMR_COLORS[i % len(CMR_COLORS)], label=f'$t={percents[i]}\\%$')

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(loc='upper left', frameon=False)


def sk_panel_fin(ax, data, N=1500, beta_arr=[0.0, 0.5, 1.0], rho=1.0):
    num_repeats = NUM_REPS_FINAL
    for i, beta in enumerate(beta_arr):
        if beta != 1.0:
            dfe = gen_final_dfe(N, beta, rho, num_repeats)
        else:
            dfe = []
            limit = min(len(data), num_repeats)
            for repeat in range(limit):
                data_entry = data[repeat]
                sigma_initial = data_entry['init_alpha']
                h = data_entry['h']
                J = data_entry['J']
                flip_seq = data_entry['flip_seq']
                sigma = cmn.compute_sigma_from_hist(sigma_initial, flip_seq)
                dfe.extend(cmn_sk.compute_dfe(sigma, h, J))

        if len(dfe) == 0: continue

        if beta == 0.0:
            dfe = np.concatenate([dfe, -dfe])
            kde = gaussian_kde(dfe, bw_method=0.5)
        else:
            kde = gaussian_kde(dfe, bw_method=0.2)

        x_grid = np.linspace(min(dfe), max(dfe), 400)
        y_small = kde.evaluate(x_grid)
        if beta == 0.0: y_small *= 2

        ax.plot(x_grid, y_small, label=f'β={beta:.1f}', color=CMR_COLORS[i % len(CMR_COLORS)], lw=2.0)

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta, t=100\%)$')
    ax.set_xlim(None, 0)
    ax.legend(loc='upper left', frameon=False)


def plot_bdfe_exp_sk(ax, data, points_lst, num_flips):
    colors = CMR_COLORS
    data_subset = data[:NUM_REPS_BDFE]

    for i, point in enumerate(points_lst):
        bdfe = []
        for repeat_idx in range(len(data_subset)):
            repeat_data = data_subset[repeat_idx]
            alphas = cmn.curate_sigma_list(
                repeat_data['init_alpha'], repeat_data['flip_seq'], [point]
            )
            h = repeat_data['h']
            J = repeat_data['J']
            bdfe_i = cmn_sk.compute_bdfe(alphas[0], h, J)[0]
            bdfe.extend(bdfe_i)

        t_percent = (point / (num_flips - 1)) * 100.0
        # SK compared to Exp and Airy
        plot_bdfe_distribution(ax, bdfe, colors[i % len(colors)],
                               t_percent=t_percent, model_type='sk')

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')


# ----------------------------------------------------------------
# ROW 3: NK Panels
# ----------------------------------------------------------------

def nk_panel_evolution_dfe(ax, nk_data):
    percents = np.linspace(0, 100, 5, dtype=int)
    combined_dfes = [[] for _ in percents]
    for repeat in nk_data[:NUM_REPS_EVOL]:
        flip_seq = repeat['flip_seq']
        num_flips = len(flip_seq)
        ts = [int(p * num_flips / 100) for p in percents]
        for i, t in enumerate(ts):
            dfe_t = repeat['dfes'][t]
            combined_dfes[i].extend(dfe_t)

    for i, combined_dfe in enumerate(combined_dfes):
        combined_dfe = np.asarray(combined_dfe)
        # Rescale dfe by N, as data was simulated for the intensive version of the model
        combined_dfe *= 2000
        if len(combined_dfe) < 2: continue
        kde = gaussian_kde(combined_dfe, bw_method=0.5)
        x_grid = np.linspace(combined_dfe.min(), combined_dfe.max(), 500)
        y_kde = kde.evaluate(x_grid)
        ax.plot(x_grid, y_kde, color=CMR_COLORS[i % len(CMR_COLORS)], lw=2.0, label=f'$t={percents[i]}\\%$')

    ax.set_xlabel(r'Fitness effect ($\Delta$)')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(frameon=False)


def nk_panel_final_dfe(ax, data_arr, K_values):
    for i, K in enumerate(K_values):
        combined = []
        for entry in data_arr[i][:NUM_REPS_FINAL]:
            combined.extend(entry['dfes'][-1])
        dfe_arr = np.asarray(combined, dtype=float)
        # Rescale dfe by N, as data was simulated for the intensive version of the model
        dfe_arr *= 2000
        if len(dfe_arr) < 2: continue
        kde = gaussian_kde(dfe_arr, bw_method=0.25)
        x_grid = np.linspace(dfe_arr.min(), 0.0, 400)
        y_kde = kde.evaluate(x_grid)
        ax.plot(x_grid, y_kde, label=f'K={int(K)}', color=CMR_COLORS[i % len(CMR_COLORS)], lw=2.0)

    ax.set_xlabel(r'Fitness effect ($\Delta$)')
    ax.set_ylabel(r'$P(\Delta, t=100\%)$')
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlim(None, 0)


def plot_bdfe_exp_nk(ax, nk_data, percents=[75, 80, 85, 90]):
    colors = CMR_COLORS
    for i, p in enumerate(percents):
        bdfe = []
        for entry in nk_data[:NUM_REPS_BDFE]:
            num_flips = len(entry['flip_seq'])
            if num_flips == 0: continue
            t_idx = int(p * num_flips / 100.0)
            t_idx = max(0, min(t_idx, num_flips - 1))
            dfe_t = np.asarray(entry['dfes'][t_idx])
            # Rescale dfe by N, as data was simulated for the intensive version of the model
            dfe_t *= 2000
            bdfe_i, _ = cmn_nk.compute_bdfe(dfe_t)
            bdfe.extend(bdfe_i)

        # NK compared to Exp and Airy
        plot_bdfe_distribution(ax, bdfe, colors[i % len(colors)],
                               t_percent=p, model_type='nk')

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------

def main():
    # 1. Load All Data
    print("Loading FGM Data...")
    fgm_reps, fgm_final = load_fgm_data()

    print("Loading SK Data...")
    sk_data = load_sk_data()

    print("Loading NK Data...")
    nk_data_arr, nk_single_k_data, nk_k_values = load_nk_data()

    # 2. Prepare specific arguments for SK plots
    # percents_array_sk = [60, 65, 70, 75]
    percents_array_sk = [75, 80, 85, 90]
    crossings_repeat_sk = 10
    num_flips_sk = len(sk_data[crossings_repeat_sk]['flip_seq'])
    percentages_C_sk = np.array(percents_array_sk)
    flip_list_sk = (percentages_C_sk / 100 * (num_flips_sk - 1)).astype(int)

    # 3. Setup Figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.34, wspace=0.34)

    # Row 0: FGM
    fgm_panel_A(axes[0, 0], fgm_reps)
    fgm_panel_B(axes[0, 1], fgm_final)
    plot_bdfe_exp_fgm(axes[0, 2], fgm_reps)

    # Row 1: SK
    sk_panel_evol(axes[1, 0], sk_data)
    sk_panel_fin(axes[1, 1], sk_data)
    plot_bdfe_exp_sk(axes[1, 2], sk_data, flip_list_sk, num_flips_sk)

    # Row 2: NK
    nk_panel_evolution_dfe(axes[2, 0], nk_single_k_data)
    nk_panel_final_dfe(axes[2, 1], nk_data_arr, nk_k_values)
    plot_bdfe_exp_nk(axes[2, 2], nk_single_k_data)

    # 4. Labeling
    panel_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    flat_axes = [ax for row_axes in axes for ax in row_axes]
    for label, ax in zip(panel_labels, flat_axes):
        ax.text(
            -0.1, 1.05, label,
            transform=ax.transAxes,
            fontsize=18, fontweight="bold",
            va="bottom", ha="left",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")

    # 5. Save
    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig1_dfe_dynamics.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()