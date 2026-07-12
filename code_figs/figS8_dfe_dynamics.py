import os
import sys
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns


NUM_REPS_EVOL = 10

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
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)
PERCENTS = [0, 25, 50, 75, 100]


def waterfall_plot_panel(ax, time_datasets, colors, time_values,
                         elev=15, azim=-75, title=None, xlim=None, bins=20):
    """Draw a 3D waterfall of the DFE time evolution on a 3d matplotlib axes.

    Each time slice is placed at its own depth (y = time percentage) so the
    slices recede into the page: t=0% sits at the back, t=100% is drawn in
    front nearest the viewer. Each slice is a step-style density histogram
    (``bins`` bins), sampled finely as a piecewise-constant curve.

    ``xlim`` optionally overrides the auto-computed fitness-effect range with a
    manual ``(xmin, xmax)`` for this panel; when None the panel spans its own
    data.
    """
    ax.computed_zorder = False

    profiles = []
    xmin_g, xmax_g = np.inf, -np.inf
    max_y = 0.0

    for i, data in enumerate(time_datasets):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data.min(), data.max()):
            profiles.append(None)
            continue
        # Anchor the bin edges on integer multiples of the bin width so that
        # 0 is always an edge (never inside a bin straddling beneficial and
        # deleterious effects). ``bins`` sets the width across the data span.
        lo, hi = data.min(), data.max()
        width = (hi - lo) / bins
        edges = np.arange(np.floor(lo / width), np.ceil(hi / width) + 1) * width
        counts, edges = np.histogram(data, bins=edges, density=True)
        x_pts = np.linspace(edges[0], edges[-1], 600)
        idx = np.clip(np.digitize(x_pts, edges) - 1, 0, len(counts) - 1)
        y_pts = counts[idx]
        profiles.append((x_pts, y_pts))
        xmin_g = min(xmin_g, x_pts.min())
        xmax_g = max(xmax_g, x_pts.max())
        max_y = max(max_y, y_pts.max())

    if max_y == 0.0 or not np.isfinite(xmin_g):
        return

    if xlim is not None:
        xmin_g, xmax_g = xlim

    # Dotted reference line at zero fitness effect (Delta=0) on the (Delta, t)
    # floor. Drawn first so the histogram slices render on top of it.
    ax.plot([0.0, 0.0], [time_values[0], time_values[-1]], [0.0, 0.0],
            color="grey", ls=":", lw=2, zorder=-1)

    x_full = np.linspace(xmin_g, xmax_g, 600)
    for i, (profile, color) in enumerate(zip(profiles, colors)):
        y_t = time_values[i]
        if profile is None:
            ax.plot([xmin_g, xmax_g], [y_t, y_t], [0.0, 0.0],
                    color=color, lw=3, zorder=i)
            continue
        x_k, y_k = profile
        y_full = np.interp(x_full, x_k, y_k, left=0.0, right=0.0)
        verts = [(x_full[0], y_t, 0.0)]
        verts += list(zip(x_full, np.full_like(x_full, y_t), y_full))
        verts += [(x_full[-1], y_t, 0.0)]
        fill = Poly3DCollection([verts], facecolors=[(*mpl.colors.to_rgb(color), 0.30)],
                                edgecolors="none")
        fill.set_zorder(i - 0.1)
        ax.add_collection3d(fill)
        ax.plot(x_full, np.full_like(x_full, y_t), y_full,
                color=color, lw=3, zorder=i)

    ax.set_xlim(xmin_g, xmax_g)
    ax.set_ylim(time_values[-1], time_values[0])  # t=100% toward the viewer
    ax.set_zlim(0.0, None)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=1.4)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_yticks(list(time_values))
    ax.set_yticklabels([])
    ax.set_zticks([])
    ax.tick_params(axis="both", labelsize=16, pad=0.01)
    ax.tick_params(width=3, length=10, which="major")
    ax.tick_params(width=3, length=5, which="minor")
    # Offset the time labels just past the right edge of the data, scaled to
    # the panel's own fitness-effect range (an absolute offset would overflow
    # into the neighbouring panel for the small-range models, e.g. FGM).
    dx = 0.03 * (xmax_g - xmin_g)
    for v in time_values:
        ax.text(xmax_g + dx, v, 0.0, f" {v}%", fontsize=16, ha="left", va="center")
    ax.set_xlabel(r"Fitness effect $(s)$", labelpad=2)
    ax.set_ylabel(r"$t$", labelpad=7)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_linewidth(1.5)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=16, pad=10)


def extract_fgm_ridge_data(reps):
    combined = [[] for _ in PERCENTS]
    for rep in reps[:NUM_REPS_EVOL]:
        if not isinstance(rep, dict):
            continue
        walk_length = len(rep["dfes"])
        # Re-anchor the reference position: the walk's state after 75% of the
        # simulation becomes the new t=0%, and the remaining 75%->100% span is
        # rescaled onto PERCENTS (so the old 75% slice is the new 0% slice).
        start_idx = int(85 * (walk_length - 1) / 100)
        span = (walk_length - 1) - start_idx
        for idx, pct in enumerate(PERCENTS):
            t_idx = start_idx + int(pct * span / 100)
            combined[idx].extend(rep["dfes"][t_idx])
    return combined


def extract_pspin_ridge_data(data):
    num_repeats = min(len(data), NUM_REPS_EVOL)
    combined = [[] for _ in PERCENTS]
    for repeat in range(num_repeats):
        entry = data[repeat]
        flip_seq = entry["flip_seq"]
        # Slice up to len(flip_seq) (not len-1): compute_sigma_from_hist applies
        # hist[:t], so t = len(flip_seq) is the final local optimum. Using len-1
        # would stop one flip short, leaving the last pending beneficial mutation
        # as a spurious positive tail in the 100% slice.
        ts = [int(len(flip_seq) * pct / 100) for pct in PERCENTS]
        sigma_list = cmn.curate_sigma_list(entry["init_sigma"], flip_seq, ts)
        for idx, sigma in enumerate(sigma_list):
            combined[idx].extend(cmn_pspin.compute_dfe(sigma, entry["J"]))
    return combined


def extract_nk_ridge_data(nk_data):
    combined = [[] for _ in PERCENTS]
    for repeat in nk_data[:NUM_REPS_EVOL]:
        flip_seq = repeat["flip_seq"]
        num_flips = len(flip_seq)
        ts = [int(pct * num_flips / 100) for pct in PERCENTS]
        for idx, t_idx in enumerate(ts):
            combined[idx].extend(repeat["dfes"][t_idx])
    return [np.asarray(d, dtype=float) * 2000 for d in combined]


def load_fgm_reps():
    for path in [
        "../data/FGM/fgm_rps1000_n4_sig0.05_m2000.pkl",
        "../data/FGM/fgm_rps1000_n4_sig0.05.pkl",
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return []


def load_pspin_data():
    path = "../data/PSPIN/N1500_P2_pure_repeats10.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"PSPIN data file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_nk_single_k():
    path = "../data/NK/N_2000_K_8_repeats_100.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def _content_bbox(fig, pad=0.1):
    """Tight crop box (in inches) computed from the rendered pixels.

    `bbox_inches="tight"` crops a 3d axes at its subplot-cell edge and ignores
    `ax.text` artists placed in data coordinates, so the rightmost panel's time
    labels get clipped. Measuring the non-white bounding box of the drawn canvas
    instead captures every visible mark (labels included), regardless of artist
    type, and is converted back to an inch-space Bbox for savefig.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    h, w = buf.shape[:2]
    nonwhite = (buf[:, :, :3] < 250).any(axis=2)
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    dpi = fig.dpi
    x0, x1 = cols.min() / dpi, cols.max() / dpi
    # canvas rows run top→bottom; inch-space y runs bottom→top.
    y0, y1 = (h - rows.max()) / dpi, (h - rows.min()) / dpi
    return Bbox([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]])


def main():
    print("Loading FGM data...")
    fgm_reps = load_fgm_reps()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data = load_nk_single_k()

    fgm_datasets = extract_fgm_ridge_data(fgm_reps)
    pspin_datasets = extract_pspin_ridge_data(pspin_data)
    nk_datasets = extract_nk_ridge_data(nk_data)

    fig = plt.figure(figsize=(16, 14))
    # left margin gives the FGM panel's 3d content room to overflow its cell
    # without being clipped at the canvas edge (the tight crop can only recover
    # pixels that were actually drawn on-canvas).
    # 2 panels on the top row, 1 centred on the bottom row: a 2x4 grid where
    # the top panels each span 2 columns and the bottom panel spans the middle
    # 2 columns (cols 1-2), leaving it horizontally centred.
    gs = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.30, left=0.04, right=0.90)
    ax1 = fig.add_subplot(gs[0, 0:2], projection="3d")
    ax2 = fig.add_subplot(gs[0, 2:4], projection="3d")
    ax3 = fig.add_subplot(gs[1, 1:3], projection="3d")

    waterfall_plot_panel(ax1, fgm_datasets, CMR_COLORS, PERCENTS,
                         title="FGM", xlim=(-0.06, 0.04))
    waterfall_plot_panel(ax2, pspin_datasets, CMR_COLORS, PERCENTS,
                         title="p-spin", xlim=(-15.0, 13.0))
    waterfall_plot_panel(ax3, nk_datasets, CMR_COLORS, PERCENTS,
                         title="NK", xlim=(-25.0, 18.0))

    label_kw = dict(fontsize=18, fontweight="bold", va="bottom", ha="left")
    for panel_label, ax in zip(["A", "B", "C"], [ax1, ax2, ax3]):
        ax.text2D(-0.05, 1.05, panel_label, transform=ax.transAxes, **label_kw)

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS8_dfe_dynamics.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches=_content_bbox(fig, pad=0.2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()