import os
import sys
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde


NUM_REPS_EVOL = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
for path in (SCRIPT_DIR, REPO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cmn import cmn, cmn_sk  # noqa: E402


plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)


def load_fgm_data():
    n_val = 8
    candidate_paths = [
        f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05_m2000.pkl",
    ]

    for file_path in candidate_paths:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(
        "FGM data file not found. Checked: " + ", ".join(candidate_paths)
    )


def load_sk_data():
    file_path = "../data/SK/N4000_rho100_beta100_repeats50.pkl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SK Data file not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_nk_data():
    file_path = "../data/NK/N_2000_K_8_repeats_100.pkl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NK Data file not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def fgm_panel(ax, reps, percents=(25, 50, 75, 100)):
    combined_dfes = [[] for _ in percents]
    for rep in reps[:NUM_REPS_EVOL]:
        walk_length = len(rep["dfes"])
        for idx, percent in enumerate(percents):
            t_idx = int(percent * (walk_length - 1) / 100)
            combined_dfes[idx].extend(rep["dfes"][t_idx])

    for idx, dfe in enumerate(combined_dfes):
        dfe = np.asarray(dfe)
        if len(dfe) < 2:
            continue
        kde = gaussian_kde(dfe, bw_method=0.4)
        x_grid = np.linspace(dfe.min(), dfe.max(), 400)
        y_grid = kde.evaluate(x_grid)
        ax.plot(
            x_grid,
            y_grid + 0.003,
            lw=2,
            color=CMR_COLORS[idx % len(CMR_COLORS)],
            label=f"$t={percents[idx]}\\%$",
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta, t)$")
    ax.legend(loc="upper left", frameon=False)

def sk_panel(ax, data, num_points=5):
    num_repeats = min(len(data), NUM_REPS_EVOL)
    dfes = np.empty((num_repeats, num_points), dtype=object)
    percents = np.linspace(0, 100, num_points, dtype=int)

    for repeat in range(num_repeats):
        data_entry = data[repeat]
        sigma_initial = data_entry["init_alpha"]
        h = data_entry["h"]
        J = data_entry["J"]
        flip_seq = data_entry["flip_seq"]

        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in percents]
        sigma_list = cmn.curate_sigma_list(sigma_initial, flip_seq, ts)

        for idx, sigma in enumerate(sigma_list):
            dfe = cmn_sk.compute_dfe(sigma, h, J)
            if dfes[repeat, idx] is None:
                dfes[repeat, idx] = []
            dfes[repeat, idx].extend(dfe)

    for idx in range(num_points):
        all_dfe = []
        for repeat in range(num_repeats):
            if dfes[repeat, idx] is not None:
                all_dfe.extend(dfes[repeat, idx])
        if all_dfe:
            kde = gaussian_kde(all_dfe, bw_method=0.4)
            x_grid = np.linspace(min(all_dfe), max(all_dfe), 400)
            y_grid = kde.evaluate(x_grid)
            ax.plot(
                x_grid,
                y_grid + 0.003,
                lw=2,
                color=CMR_COLORS[idx % len(CMR_COLORS)],
                label=f"$t={percents[idx]}\\%$",
            )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.legend(loc="upper left", frameon=False)


def nk_panel(ax, nk_data):
    percents = np.linspace(0, 100, 5, dtype=int)
    combined_dfes = [[] for _ in percents]
    for repeat in nk_data[:NUM_REPS_EVOL]:
        flip_seq = repeat["flip_seq"]
        num_flips = len(flip_seq)
        ts = [int(percent * num_flips / 100) for percent in percents]
        for idx, t_idx in enumerate(ts):
            dfe_t = repeat["dfes"][t_idx]
            combined_dfes[idx].extend(dfe_t)

    for idx, combined_dfe in enumerate(combined_dfes):
        combined_dfe = np.asarray(combined_dfe)
        combined_dfe *= 2000
        if len(combined_dfe) < 2:
            continue
        kde = gaussian_kde(combined_dfe, bw_method=0.5)
        x_grid = np.linspace(combined_dfe.min(), combined_dfe.max(), 500)
        y_grid = kde.evaluate(x_grid)
        ax.plot(
            x_grid,
            y_grid,
            color=CMR_COLORS[idx % len(CMR_COLORS)],
            lw=2.0,
            label=f"$t={percents[idx]}\\%$",
        )

    ax.set_xlabel(r"Fitness effect ($\Delta$)")
    ax.legend(loc="upper left", frameon=False)


def style_axis(ax, title):
    ax.set_title(title, fontsize=18, pad=10)
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")


def main():
    print("Loading FGM Data...")
    fgm_reps = load_fgm_data()

    print("Loading SK Data...")
    sk_data = load_sk_data()

    print("Loading NK Data...")
    nk_data = load_nk_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.2)

    fgm_panel(axes[0], fgm_reps)
    sk_panel(axes[1], sk_data)
    nk_panel(axes[2], nk_data)

    for ax, title in zip(axes, ["FGM", "SK", "NK"]):
        style_axis(ax, title)

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig3_dfes.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
