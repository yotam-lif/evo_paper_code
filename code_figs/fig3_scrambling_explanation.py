import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
    }
)


def gaussian_fitness(r1, r2):
    return np.exp(-0.5 * (r1**2 + r2**2))


def style_axis(ax):
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")


def main():
    grid_lim = 3.0
    n_points = 300
    r = np.linspace(-grid_lim, grid_lim, n_points)
    r1, r2 = np.meshgrid(r, r)
    fitness = gaussian_fitness(r1, r2)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10), constrained_layout=True)
    norm = mpl.colors.PowerNorm(gamma=1, vmin=0.0, vmax=1.0)

    panel_labels = ["A", "B"]
    for ax, panel_label in zip(axes, panel_labels):
        im = ax.imshow(
            fitness,
            extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
            origin="lower",
            cmap="plasma",
            norm=norm,
        )
        ax.text(
            -0.1,
            1.05,
            panel_label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
        )
        style_axis(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Fitness", rotation=90, labelpad=16, fontsize=16)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.tick_params(width=1.5, length=6, which="major", labelsize=14)
        cbar.ax.tick_params(width=1.5, length=3, which="minor")
        cbar.outline.set_linewidth(1.5)

    out_dir = os.path.join(REPO_DIR, "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig3_scrambling_explanation.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
