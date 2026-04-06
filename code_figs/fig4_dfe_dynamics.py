import os
import sys
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde


NUM_REPS_EVOL = 10
NUM_REPS_FINAL = 10


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
for path in (SCRIPT_DIR, REPO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cmn import cmn, cmn_pspin


plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)


def plot_kde(
    ax,
    samples,
    color,
    label,
    bw_method,
    offset=0.0,
    num_points=400,
    reflect_negative=False,
):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        return

    if reflect_negative:
        # Use reflection at Delta=0 to reduce KDE boundary bias for one-sided final DFEs.
        samples = samples[samples <= 0]
        if samples.size < 2:
            return
        mirrored_samples = np.concatenate([samples, -samples])
        if np.allclose(mirrored_samples.min(), mirrored_samples.max()):
            return
        kde = gaussian_kde(mirrored_samples, bw_method=bw_method)
        x_grid = np.linspace(samples.min(), 0.0, num_points)
        y_grid = 2.0 * kde.evaluate(x_grid)
    else:
        if np.allclose(samples.min(), samples.max()):
            return
        kde = gaussian_kde(samples, bw_method=bw_method)
        x_grid = np.linspace(samples.min(), samples.max(), num_points)
        y_grid = kde.evaluate(x_grid)

    ax.plot(x_grid, y_grid + offset, lw=2, color=color, label=label)


def load_fgm_data():
    fgm_ns = [4, 8, 16, 32]
    fgm_data = {}

    for n_val in fgm_ns:
        candidate_paths = [
            f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05_m2000.pkl",
            f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05.pkl",
        ]
        for file_path in candidate_paths:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    fgm_data[n_val] = pickle.load(f)
                break
        else:
            fgm_data[n_val] = []

    reps = fgm_data.get(4, [])
    final = {}
    for n_val, rep_list in fgm_data.items():
        all_last_dfes = []
        for rep in rep_list[:NUM_REPS_FINAL]:
            if not isinstance(rep, dict):
                continue
            dfes = rep.get("dfes")
            if dfes is None or len(dfes) == 0:
                continue
            all_last_dfes.extend(dfes[-1])
        final[n_val] = all_last_dfes

    return reps, final


def load_pspin_data():
    file_paths = {
        1: "../data/PSPIN/N400_P1_mixed_repeats10.pkl",
        2: "../data/PSPIN/N400_P2_mixed_repeats10.pkl",
        3: "../data/PSPIN/N400_P3_mixed_repeats10.pkl",
    }

    pspin_data = {}
    for order, file_path in file_paths.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PSPIN data file not found: {file_path}")
        with open(file_path, "rb") as f:
            pspin_data[order] = pickle.load(f)

    return pspin_data


def load_nk_data():
    res_directory = "../data/NK"
    files = [
        (4, os.path.join(res_directory, "N_2000_K_4_repeats_100.pkl")),
        (8, os.path.join(res_directory, "N_2000_K_8_repeats_100.pkl")),
        (16, os.path.join(res_directory, "N_2000_K_16_repeats_100.pkl")),
        (32, os.path.join(res_directory, "N_2000_K_32_repeats_100.pkl")),
    ]

    data_arr = []
    k_values = [4, 8, 16, 32]
    for _, file_path in files:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data_arr.append(pickle.load(f))
        else:
            data_arr.append([])

    nk_single_k_data = data_arr[1]
    return data_arr, nk_single_k_data, k_values


def fgm_evolution_panel(ax, reps, percents=(25, 50, 75, 100)):
    combined_dfes = [[] for _ in percents]
    for rep in reps[:NUM_REPS_EVOL]:
        walk_length = len(rep["dfes"])
        for idx, percent in enumerate(percents):
            t_idx = int(percent * (walk_length - 1) / 100)
            combined_dfes[idx].extend(rep["dfes"][t_idx])

    for idx, dfe in enumerate(combined_dfes):
        plot_kde(
            ax,
            dfe,
            CMR_COLORS[idx % len(CMR_COLORS)],
            f"$t={percents[idx]}\\%$",
            bw_method=0.5,
            offset=0.003,
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta, t)$")
    ax.legend(frameon=False, loc="upper left")


def fgm_final_panel(ax, final):
    for idx, (n_val, dfe) in enumerate(final.items()):
        plot_kde(
            ax,
            dfe,
            CMR_COLORS[idx % len(CMR_COLORS)],
            f"$n={n_val}$",
            bw_method=0.3,
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    ax.set_xlim(None, 0)
    ax.legend(frameon=False, loc="upper left")


def pspin_evolution_panel(ax, data, num_points=5):
    num_repeats = min(len(data), NUM_REPS_EVOL)
    dfes = np.empty((num_repeats, num_points), dtype=object)
    percents = np.linspace(0, 100, num_points, dtype=int)

    for repeat in range(num_repeats):
        data_entry = data[repeat]
        sigma_initial = data_entry["init_sigma"]
        J = data_entry["J"]
        flip_seq = data_entry["flip_seq"]

        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in percents]
        sigma_list = cmn.curate_sigma_list(sigma_initial, flip_seq, ts)

        for idx, sigma in enumerate(sigma_list):
            dfe = cmn_pspin.compute_dfe(sigma, J)
            if dfes[repeat, idx] is None:
                dfes[repeat, idx] = []
            dfes[repeat, idx].extend(dfe)

    for idx in range(num_points):
        all_dfe = []
        for repeat in range(num_repeats):
            if dfes[repeat, idx] is not None:
                all_dfe.extend(dfes[repeat, idx])
        plot_kde(
            ax,
            all_dfe,
            CMR_COLORS[idx % len(CMR_COLORS)],
            f"$t={percents[idx]}\\%$",
            bw_method=0.4,
            offset=0.003,
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t)$")
    ax.legend(loc="upper left", frameon=False)


def pspin_final_panel(ax, pspin_data):
    for idx, order in enumerate(sorted(pspin_data)):
        dfe = []
        for data_entry in pspin_data[order][:NUM_REPS_FINAL]:
            sigma = cmn.compute_sigma_from_hist(
                data_entry["init_sigma"], data_entry["flip_seq"]
            )
            dfe.extend(cmn_pspin.compute_dfe(sigma, data_entry["J"]))

        plot_kde(
            ax,
            dfe,
            CMR_COLORS[idx % len(CMR_COLORS)],
            f"$P={order}$",
            bw_method=0.4,
            reflect_negative=(order == 1),
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    ax.set_xlim(None, 0)
    ax.legend(loc="upper left", frameon=False)


def nk_evolution_panel(ax, nk_data):
    percents = np.linspace(0, 100, 5, dtype=int)
    combined_dfes = [[] for _ in percents]
    for repeat in nk_data[:NUM_REPS_EVOL]:
        flip_seq = repeat["flip_seq"]
        num_flips = len(flip_seq)
        ts = [int(percent * num_flips / 100) for percent in percents]
        for idx, t_idx in enumerate(ts):
            combined_dfes[idx].extend(repeat["dfes"][t_idx])

    for idx, combined_dfe in enumerate(combined_dfes):
        combined_dfe = np.asarray(combined_dfe, dtype=float) * 2000
        plot_kde(
            ax,
            combined_dfe,
            CMR_COLORS[idx % len(CMR_COLORS)],
            f"$t={percents[idx]}\\%$",
            bw_method=0.5,
            num_points=500,
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t)$")
    ax.legend(frameon=False, loc="upper left")


def nk_final_panel(ax, data_arr, k_values):
    for idx, k_val in enumerate(k_values):
        combined = []
        for entry in data_arr[idx][:NUM_REPS_FINAL]:
            combined.extend(entry["dfes"][-1])
        dfe_arr = np.asarray(combined, dtype=float) * 2000
        plot_kde(
            ax,
            dfe_arr,
            CMR_COLORS[idx % len(CMR_COLORS)],
            f"$K={k_val}$",
            bw_method=0.25,
        )

    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    ax.set_xlim(None, 0)
    ax.legend(frameon=False, loc="upper left")


def style_axis(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")


def main():
    print("Loading FGM data...")
    fgm_reps, fgm_final = load_fgm_data()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data_arr, nk_single_k_data, nk_k_values = load_nk_data()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.34, wspace=0.25)

    fgm_evolution_panel(axes[0, 0], fgm_reps)
    pspin_evolution_panel(axes[0, 1], pspin_data[2])
    nk_evolution_panel(axes[0, 2], nk_single_k_data)

    fgm_final_panel(axes[1, 0], fgm_final)
    pspin_final_panel(axes[1, 1], pspin_data)
    nk_final_panel(axes[1, 2], nk_data_arr, nk_k_values)

    for ax, title in zip(axes[0], ["FGM", "SK", "NK"]):
        ax.set_title(title, fontsize=18, pad=10)

    panel_labels = ["A", "B", "C", "D", "E", "F"]
    for label, ax in zip(panel_labels, axes.flat):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        style_axis(ax)

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig4_dfe_dynamics.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
