import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import gamma

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
NUM_REPS_FINAL = 100
N_SYSTEM = 2000
WALK_PERCENTAGE = 0.6  # Point in the walk to sample from

# ----------------------------------------------------------------
# Setup & Style
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(SCRIPT_DIR):
    os.chdir(SCRIPT_DIR)

plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
})

CMR_COLORS = sns.color_palette('CMRmap', 5)


# ----------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------
def load_nk_data():
    """
    Loads NK model data from pickle files.
    """
    res_directory = '../data/NK'

    files = [
        (4, os.path.join(res_directory, 'N_2000_K_4_repeats_100.pkl')),
        (8, os.path.join(res_directory, 'N_2000_K_8_repeats_100.pkl')),
        (16, os.path.join(res_directory, 'N_2000_K_16_repeats_100.pkl')),
        (32, os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl'))
    ]

    data_map = {}
    for k_val, file_path in files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data_map[k_val] = pickle.load(f)
            except Exception as e:
                print(f"Error loading K={k_val}: {e}")
                data_map[k_val] = []
        else:
            print(f"File not found: {file_path}")
            data_map[k_val] = []

    return data_map


# ----------------------------------------------------------------
# Reflected Gamma Fitting
# ----------------------------------------------------------------
def fit_reflected_gamma(data, fixed_shape):
    """
    Fits a Gamma distribution to the NEGATIVE of the data.

    1. Clean data (remove zeros/NaNs).
    2. Invert data: x' = -x
    3. Fit Gamma(x')
    4. Return parameters.
    """
    # Clean: remove NaNs and exact zeros
    mask = np.isfinite(data)
    clean_data = data[mask]

    if len(clean_data) < 2:
        return np.nan, np.nan

    # Invert the cleaned data so the tail is on the right
    inverted_data = -1 * clean_data

    try:
        # fa = fixed shape parameter
        params = gamma.fit(inverted_data, fa=fixed_shape, method="MM")

        # params tuple is (shape, loc_inverted, scale)
        loc_inv = params[1]
        scale_fit = params[2]

        return loc_inv, scale_fit
    except Exception as e:
        print(f"Fitting error: {e}")
        return np.nan, np.nan


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------
def plot_nk_gamma_fit_with_bdfe():
    nk_data = load_nk_data()
    k_values = [4, 8, 16, 32]

    # --- Setup Two Figures ---
    # Fig 1: Full DFE
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Fig 2: BDFE (Beneficial Only)
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for i, K in enumerate(k_values):
        color = CMR_COLORS[i % len(CMR_COLORS)]

        # --- A. Prepare Simulation Data ---
        sim_repeats = nk_data.get(K, [])
        combined_dfe = []

        for rep in sim_repeats[:NUM_REPS_FINAL]:
            if 'dfes' in rep and len(rep['dfes']) > 0:
                total_steps = len(rep['dfes'])
                target_idx = int(total_steps * WALK_PERCENTAGE)
                if target_idx >= total_steps:
                    target_idx = total_steps - 1
                combined_dfe.extend(rep['dfes'][target_idx])

        if len(combined_dfe) > 1:
            dfe_arr = np.asarray(combined_dfe, dtype=float)
            dfe_arr *= N_SYSTEM

            # --- B. Plot Histogram (Full DFE on ax1) ---
            # Filter non-finite and zeros for clean plotting
            plot_data_full = dfe_arr[(np.abs(dfe_arr) > 1e-12) & np.isfinite(dfe_arr)]

            if len(plot_data_full) == 0:
                continue

            ax1.hist(plot_data_full, bins=60, density=False, histtype='step',
                     color=color, linewidth=2, alpha=0.5, label=f'K={K}')

            # --- C. Plot Histogram (BDFE on ax2) ---
            # Filter ONLY positive values
            plot_data_bdfe = plot_data_full[plot_data_full > 0]

            if len(plot_data_bdfe) > 5:  # Only plot if we have enough data points
                ax2.hist(plot_data_bdfe, bins=20, density=False, histtype='step',
                         color=color, linewidth=2, alpha=0.5, label=f'K={K}')

            # --- D. Fit Reflected Gamma (Global Fit) ---
            shape_fixed = K + 1
            loc_inv, scale_fit = fit_reflected_gamma(dfe_arr, fixed_shape=shape_fixed)

            if np.isnan(scale_fit):
                continue

            print(f"K={K:2d} | Shape={shape_fixed} | Loc(inv)={loc_inv:.3f} | Scale={scale_fit:.3f}")

            # --- E. Draw Curves ---

            # --- Curve 1: Full DFE (ax1) ---
            x_min_full, x_max_full = np.min(plot_data_full), np.max(plot_data_full)
            x_plot_full = np.linspace(x_min_full, x_max_full, 500)

            # PDF calculation: invert x, use parameters, get y
            y_vals_full = gamma.pdf(-x_plot_full, a=shape_fixed, loc=loc_inv, scale=scale_fit)
            ax1.plot(x_plot_full, y_vals_full, color=color, linestyle='-', linewidth=2.5)

            # --- Curve 2: BDFE Zoom (ax2) ---
            # We use the SAME fit parameters, but evaluate only for x > 0
            # If there are no beneficials in data, we can still show the theoretical tail
            if len(plot_data_bdfe) > 0:
                x_max_bdfe = np.max(plot_data_bdfe)
            else:
                x_max_bdfe = scale_fit * 2  # arbitrary view if no data

            # Start from 0 up to max positive observed
            x_plot_bdfe = np.linspace(0, x_max_bdfe, 200)

            # Evaluate using the same global parameters
            y_vals_bdfe = gamma.pdf(-x_plot_bdfe, a=shape_fixed, loc=loc_inv, scale=scale_fit)

            # Only plot curve if we have BDFE data or want to show the tail existence
            if len(plot_data_bdfe) > 5:
                ax2.plot(x_plot_bdfe, y_vals_bdfe, color=color, linestyle='-', linewidth=2.5)

    # --- Formatting Figure 1 (Full) ---
    ax1.set_xlabel(r'Fitness effect ($\Delta$)')
    ax1.set_ylabel(r'Density')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title(f"Full DFE (Walk {int(WALK_PERCENTAGE * 100)}%)\nFit: Reflected Gamma ($k=K+1$)")

    # Legend 1
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=CMR_COLORS[i], lw=2) for i in range(4)]
    legend_labels = [f"K={k}" for k in k_values]
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', lw=1, alpha=0.5, label="Sim."))
    legend_labels.append("Sim.")
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', lw=2.5, label="Fit"))
    legend_labels.append("Gamma Fit")
    ax1.legend(legend_handles, legend_labels, frameon=False, loc='upper left')

    # --- Formatting Figure 2 (BDFE) ---
    ax2.set_xlabel(r'Beneficial Fitness effect ($\Delta > 0$)')
    ax2.set_ylabel(r'Density')
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title(f"Beneficial Tail (Walk {int(WALK_PERCENTAGE * 100)}%)\nFit: Zoom of Global Gamma")

    # Legend 2
    ax2.legend(legend_handles, legend_labels, frameon=False, loc='upper right')

    # --- Saving ---
    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)

    path1 = os.path.join(out_dir, "fig_nk_gamma_full_95.svg")
    fig1.savefig(path1, format="svg", bbox_inches="tight")
    print(f"Saved Full DFE figure to {path1}")

    path2 = os.path.join(out_dir, "fig_nk_gamma_bdfe_95.svg")
    fig2.savefig(path2, format="svg", bbox_inches="tight")
    print(f"Saved BDFE figure to {path2}")

    plt.show()


if __name__ == "__main__":
    plot_nk_gamma_fit_with_bdfe()