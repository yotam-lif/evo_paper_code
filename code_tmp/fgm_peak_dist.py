import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
SIGMA = 0.05
FGM_NS = [4, 8, 16, 32]
NUM_REPS_MAX = 100  # Load as many as possible for better histograms

# ----------------------------------------------------------------
# Style & Setup (Matched to fig1_dfe_dynamics.py)
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Global Style
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
def load_fgm_final_distances():
    """
    Loads FGM data and extracts the squared distance from origin
    normalized by sigma^2 for the final state of the simulation.
    """
    fgm_distances = {}

    for _n in FGM_NS:
        # Path based on your original script's relative structure
        file_path_fgm = f'../data/FGM/fgm_rps1000_n{_n}_sig0.05_m2000.pkl'

        normalized_sq_distances = []

        if os.path.exists(file_path_fgm):
            print(f"Loading n={_n} from {file_path_fgm}...")
            with open(file_path_fgm, 'rb') as f:
                rep_list = pickle.load(f)

            # Iterate through replicates
            count = 0
            for rep in rep_list:
                if count >= NUM_REPS_MAX:
                    break
                if not isinstance(rep, dict):
                    continue

                # Extract trajectory list and get the final position vector
                if 'traj' in rep:
                    traj = rep['traj']
                    if len(traj) > 0:
                        r_final = np.array(traj[-1])

                        # Calculate ||r_final||^2 / sigma^2
                        r_sq = np.sum(r_final ** 2)
                        val = r_sq / (_n * SIGMA ** 2)
                        normalized_sq_distances.append(val)
                        count += 1
                else:
                    # Fallback or error logging if 'traj' is missing
                    if count == 0:
                        print(f"  WARNING: Key 'traj' not found in rep. Keys found: {list(rep.keys())}")
                    continue

        else:
            print(f"Warning: File not found: {file_path_fgm}")

        fgm_distances[_n] = normalized_sq_distances

    return fgm_distances


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------
def plot_distance_distributions(data):
    # Create a single figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Iterate through N values to plot
    for i, n in enumerate(FGM_NS):
        values = data.get(n, [])
        if len(values) == 0:
            continue

        values = np.array(values)
        color = CMR_COLORS[i % len(CMR_COLORS)]
        label = f"$n={n}$"

        # Plot Histogram (Step style, matching BDFE panels from Fig 1)
        sns.histplot(values, ax=ax, element="step", stat="density", bins=10,
                     fill=False, color=color, label=label,
                     common_norm=False, linewidth=2)

    # Styling
    ax.set_xlabel(r"$||r_{final}||^2 / n\sigma^2$")
    ax.set_ylabel(r"Density")
    ax.legend(frameon=False, loc="upper right")

    # Beautify spines (from original script)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")

    plt.tight_layout()

    # Save
    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS_fgm_distance_to_peak.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    data = load_fgm_final_distances()
    plot_distance_distributions(data)