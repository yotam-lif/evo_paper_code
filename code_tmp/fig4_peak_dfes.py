import json
import os
import sys
import pickle

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)
PEAK4_COLORS = cmr.take_cmap_colors('cmr.emerald', 4, cmap_range=(0.3, 1.0))
PEAK3_COLORS = cmr.take_cmap_colors('cmr.emerald', 3, cmap_range=(0.3, 1.0))
PERCENTS = [0, 25, 50, 75, 100]


# ── 3D waterfall panel ────────────────────────────────────────────────────────
def _two_sided_kde(data, bw_method, num_points=400, min_fraction=0.02):
    """Boundary-corrected KDE on each side of 0, weighted by empirical fraction.

    Each side is estimated with the reflection trick so there is no KDE leakage
    across 0. The density values just below and just above 0 are independent,
    making any discontinuity at 0 visible in the resulting curve.
    A side is only plotted if it contains at least min_fraction of the data,
    preventing artefact bumps from a handful of near-zero outliers.
    """
    neg = data[data <= 0]
    pos = data[data > 0]
    n = len(data)
    parts_x, parts_y = [], []

    if neg.size / n >= min_fraction and neg.size >= 2 and not np.allclose(neg.min(), neg.max()):
        kde = gaussian_kde(np.concatenate([neg, -neg]), bw_method=bw_method)
        x = np.linspace(neg.min(), 0.0, num_points // 2)
        parts_x.append(x)
        parts_y.append(2.0 * kde.evaluate(x) * (neg.size / n))

    if pos.size / n >= min_fraction and pos.size >= 2 and not np.allclose(pos.min(), pos.max()):
        kde = gaussian_kde(np.concatenate([pos, -pos]), bw_method=bw_method)
        x = np.linspace(0.0, pos.max(), num_points // 2)
        # Skip x=0 when negative side already claimed it
        skip = 1 if parts_x else 0
        parts_x.append(x[skip:])
        parts_y.append(2.0 * kde.evaluate(x)[skip:] * (pos.size / n))

    if not parts_x:
        return None
    return np.concatenate(parts_x), np.concatenate(parts_y)


def waterfall_plot_panel(ax, time_datasets, colors, time_values,
                         bw_method=0.4, elev=15, azim=-75):
    """Draw a 3D waterfall of the DFE time evolution on a 3d matplotlib axes.

    Same per-curve design as the previous ridge plot (white-backed fill at
    0.75-alpha colour, black outline + baseline), but instead of stacking the
    curves with a vertical offset they are placed on a real time axis
    (`time_values`) that recedes into the page: t=0% sits at the back and t=100%
    is drawn in front, nearest the viewer. The time slices are identified by a
    legend rather than by the (unlabelled) depth axis.

    `ax` must be a 3d axes (``projection="3d"``). x is the fitness effect, the
    receding y axis is time, and z is the probability density.
    """
    # Respect the explicit per-curve zorder we set below instead of letting
    # mpl3d re-sort by projected depth, so later times always draw on top.
    ax.computed_zorder = False

    kdes = []
    xmin_g, xmax_g = np.inf, -np.inf
    max_y = 0.0

    for i, data in enumerate(time_datasets):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data.min(), data.max()):
            kdes.append(None)
            continue
        if i == 0:
            kde = gaussian_kde(data, bw_method=bw_method)
            x_pts = np.linspace(data.min(), data.max(), 400)
            y_pts = kde.evaluate(x_pts)
        else:
            result = _two_sided_kde(data, bw_method)
            if result is None:
                kdes.append(None)
                continue
            x_pts, y_pts = result
        kdes.append((x_pts, y_pts))
        xmin_g = min(xmin_g, x_pts.min())
        xmax_g = max(xmax_g, x_pts.max())
        max_y = max(max_y, y_pts.max())

    if max_y == 0.0 or not np.isfinite(xmin_g):
        return

    # One curve per time slice, placed at its own depth (y = time), drawn in the
    # slice's own colour with no fill. The curve spans the full fitness-effect
    # range: where the slice has no density it runs flat along z=0, giving every
    # slice a coloured horizontal baseline from end to end with the KDE bump
    # rising above it. zorder increases with t so later slices draw on top.
    # Evaluate every slice over the full fitness-effect range: outside its KDE
    # support the density is 0, so each curve is defined (and plotted) across the
    # whole axis, running flat along z=0 where it has no mass.
    x_full = np.linspace(xmin_g, xmax_g, 600)
    for i, (kde_result, color) in enumerate(zip(kdes, colors)):
        y_t = time_values[i]
        if kde_result is None:
            ax.plot([xmin_g, xmax_g], [y_t, y_t], [0.0, 0.0],
                    color=color, lw=3, zorder=i)
            continue
        x_k, y_k = kde_result
        y_full = np.interp(x_full, x_k, y_k, left=0.0, right=0.0)
        # translucent fill under the curve, in the slice's own colour, sitting
        # just behind its line (zorder i-0.1) so later slices still occlude it.
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
    ax.set_box_aspect(None, zoom=1.4)  # fill the cell to match the 2d panels

    # Thin out the fitness-effect axis and let its label fall at matplotlib's
    # default 3d position (so it tracks azim/elev). Draw the time labels by hand,
    # each anchored exactly at its slice's baseline terminus (xmax_g, v, 0). The
    # alignment is in screen space, so a left-aligned label always sits just off
    # the line end whatever the viewing angle — it tracks azim/elev too.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_yticks(list(time_values))
    ax.set_yticklabels([])
    ax.set_zticks([])  # density axis hidden, as in the original ridge panel
    ax.tick_params(axis="both", labelsize=16, pad=0.01)
    ax.tick_params(width=3, length=10, which="major")
    ax.tick_params(width=3, length=5, which="minor")
    for v in time_values:
        ax.text(xmax_g + 0.4, v, 0.0, f" {v}%", fontsize=16, ha="left", va="center")
    ax.set_xlabel(r"Fitness effect $(s)$", labelpad=2)
    ax.set_ylabel(r"$t$", labelpad=7)

    # Keep the box clean: translucent panes, no grid. Match the 2d panels'
    # spine weight (1.5) on the 3d axis lines.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_linewidth(1.5)
    ax.grid(False)
    ax.set_title("Additive model", fontsize=16, pad=10)


# ── KDE line plot (used for final-DFE panels) ─────────────────────────────────

def plot_kde(ax, samples, color, label, bw_method, offset=0.0,
             num_points=400, reflect_negative=False):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        return

    if reflect_negative:
        samples = samples[samples <= 0]
        if samples.size < 2:
            return
        mirrored = np.concatenate([samples, -samples])
        if np.allclose(mirrored.min(), mirrored.max()):
            return
        kde = gaussian_kde(mirrored, bw_method=bw_method)
        x_grid = np.linspace(samples.min(), 0.0, num_points)
        y_grid = 2.0 * kde.evaluate(x_grid)
    else:
        if np.allclose(samples.min(), samples.max()):
            return
        kde = gaussian_kde(samples, bw_method=bw_method)
        x_grid = np.linspace(samples.min(), samples.max(), num_points)
        y_grid = kde.evaluate(x_grid)

    x_plot = np.concatenate([[x_grid[0]], x_grid, [x_grid[-1]]])
    y_plot = np.concatenate([[0.0], y_grid, [0.0]])
    ax.plot(x_plot, y_plot + offset, lw=3, color=color, label=label)


# ── Scaling inset: floor exponent α vs panel parameter (n / p / K) ─────────────
# Each main panel shows the final (peak) DFE for several values of its parameter
# (n / K / p) at one large system size. The inset distils the finite-size
# behaviour into a single number per parameter value: the boundary floor
# p0(N) = density of the peak DFE at Δ=0 vanishes (or not) with system size as
# p0(N) ∝ N^{-α}, and the inset plots that floor exponent α as a function of the
# panel parameter. α is inferred with the Bayesian floor model (the near-zero DFE
# is p0(N) + c u^θ, extended-Poisson likelihood pooled over the N-sweep; θ free
# except the known SK value θ=1). Values are precomputed by
# cmn/cmn_bayes.py (`python cmn/cmn_bayes.py alpha`) and cached in
# data/floor_alpha_by_param.json.
# α=0 marks a persistent (N-independent) floor; α>0 a floor that vanishes.
ALPHA_CACHE_PATH = os.path.join(REPO_DIR, "data", "cache", "floor_alpha_by_param.json")
_ALPHA_CACHE = None
_PARAM_SYMBOL = {"FGM": "n", "NK": "K", "PSPIN": "p"}


def alpha_cache():
    """Lazily load (and memoise) the precomputed α-vs-parameter cache."""
    global _ALPHA_CACHE
    if _ALPHA_CACHE is None:
        if not os.path.exists(ALPHA_CACHE_PATH):
            raise FileNotFoundError(
                f"{ALPHA_CACHE_PATH} not found — run "
                "`python cmn/cmn_bayes.py alpha` first to generate it."
            )
        with open(ALPHA_CACHE_PATH) as f:
            _ALPHA_CACHE = json.load(f)
    return _ALPHA_CACHE


def add_scaling_inset(ax, model_key, items, bounds=(0.165, 0.50, 0.45, 0.47)):
    """Inset of the floor exponent α vs the panel parameter (n / p / K).

    `items` is a list of (param_value, color, label) tuples whose param_value
    keys into the cached α for `model_key` ("FGM" / "NK" / "PSPIN"). Each point
    is the posterior median floor exponent with a 95% credible-interval bar,
    coloured to match its main-panel DFE curve. The dotted line at α=0 marks a
    persistent (N-independent) floor; α>0 a floor that vanishes as N→∞.
    """
    data = alpha_cache()[model_key]
    axins = ax.inset_axes(bounds)

    xs, los, his = [], [], []
    for param_value, color, _label in items:
        s = data.get(str(param_value))
        if s is None:
            continue
        med, lo, hi = s
        axins.errorbar(param_value, med, yerr=[[med - lo], [hi - med]],
                       fmt="o", ms=5, color=color, mec=color, mfc=color,
                       ecolor=color, elinewidth=1.3, capsize=2.5, zorder=3)
        xs.append(param_value); los.append(lo); his.append(hi)

    xs = np.asarray(xs, dtype=float)
    order = np.argsort(xs)
    meds = np.array([data[str(int(x))][0] for x in xs])
    axins.plot(xs[order], meds[order], "-", color="0.5", lw=0.9, alpha=0.7, zorder=1)
    axins.axhline(0.0, color="green", ls=":", lw=0.9, alpha=0.8, zorder=0)

    sym = _PARAM_SYMBOL[model_key]
    if model_key in ("FGM", "NK"):
        axins.set_xscale("log", base=2)
        axins.set_xlim(xs.min() / 1.45, xs.max() * 1.45)
    else:
        axins.set_xlim(xs.min() - 0.6, xs.max() + 0.6)
    axins.set_xticks(xs)
    axins.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
    axins.xaxis.set_minor_formatter(NullFormatter())

    ylo = min(0.0, min(los)); yhi = max(his); pad = 0.12 * (yhi - ylo)
    axins.set_ylim(ylo - pad, yhi + pad)
    axins.set_xlabel(rf"${sym}$", fontsize=12, labelpad=0)
    axins.set_ylabel(r"$\alpha$", fontsize=12, labelpad=0)
    axins.tick_params(labelsize=9, width=1.0, length=3, which="major")
    axins.tick_params(width=1.0, length=2, which="minor")
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)
    axins.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    return axins


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_fgm_data():
    fgm_ns = [4, 8, 16, 32]
    fgm_data = {}
    for n_val in fgm_ns:
        for path in [
            f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05_m2000.pkl",
            f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05.pkl",
        ]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    fgm_data[n_val] = pickle.load(f)
                break
        else:
            fgm_data[n_val] = []

    final = {}
    for n_val, rep_list in fgm_data.items():
        all_last = []
        for rep in rep_list[:NUM_REPS_FINAL]:
            if not isinstance(rep, dict):
                continue
            dfes = rep.get("dfes")
            if dfes:
                all_last.extend(dfes[-1])
        final[n_val] = all_last
    return final


def load_pspin_data():
    file_paths = {
        1: "../data/PSPIN/N400_P1_pure_repeats10.pkl",
        2: "../data/PSPIN/N400_P2_pure_repeats10.pkl",
        3: "../data/PSPIN/N400_P3_pure_repeats10.pkl",
    }
    pspin_data = {}
    for order, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"PSPIN data file not found: {path}")
        with open(path, "rb") as f:
            pspin_data[order] = pickle.load(f)
    return pspin_data


def load_nk_data():
    res_directory = "../data/NK"
    k_values = [4, 8, 16, 32]
    data_arr = []
    for k in k_values:
        path = os.path.join(res_directory, f"N_2000_K_{k}_repeats_100.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data_arr.append(pickle.load(f))
        else:
            data_arr.append([])
    return data_arr, k_values


# ── Panel builders ────────────────────────────────────────────────────────────

def pspin_p1_waterfall_panel(ax, pspin_data):
    num_repeats = min(len(pspin_data[1]), NUM_REPS_EVOL)
    combined = [[] for _ in PERCENTS]
    for repeat in range(num_repeats):
        entry = pspin_data[1][repeat]
        flip_seq = entry["flip_seq"]
        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in PERCENTS]
        sigma_list = cmn.curate_sigma_list(entry["init_sigma"], flip_seq, ts)
        for idx, sigma in enumerate(sigma_list):
            combined[idx].extend(cmn_pspin.compute_dfe(sigma, entry["J"]))

    waterfall_plot_panel(ax, combined, CMR_COLORS, PERCENTS, bw_method=0.4)


def _set_xlim_with_epsilon(ax):
    """Set right xlim to 15% of the data range past 0."""
    xmin = ax.get_xlim()[0]
    ax.set_xlim(None, 0.1 * abs(xmin))


def fgm_final_panel(ax, final):
    items = []
    for idx, (n_val, dfe) in enumerate(final.items()):
        color = PEAK4_COLORS[idx % len(PEAK4_COLORS)]
        label = f"$n={n_val}$"
        plot_kde(ax, dfe, color, label, bw_method=0.3)
        items.append((n_val, color, label))
    ax.set_xlabel(r"Fitness effect $(s)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    _set_xlim_with_epsilon(ax)
    ax.legend(frameon=True, loc="lower left")
    add_scaling_inset(ax, "FGM", items)


def pspin_final_panel(ax, pspin_data):
    items = []
    for idx, order in enumerate(sorted(pspin_data)):
        dfe = []
        for entry in pspin_data[order][:NUM_REPS_FINAL]:
            sigma = cmn.compute_sigma_from_hist(entry["init_sigma"], entry["flip_seq"])
            dfe.extend(cmn_pspin.compute_dfe(sigma, entry["J"]))
        color = PEAK3_COLORS[idx % len(PEAK3_COLORS)]
        label = f"$p={order}$"
        plot_kde(ax, dfe, color, label, bw_method=0.4,
                 reflect_negative=(order == 1))
        if order != 1:  # p=1 (additive) has no N-sweep -> omitted from the inset
            items.append((order, color, label))
    ax.set_xlabel(r"Fitness effect $(s)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    _set_xlim_with_epsilon(ax)
    ax.legend(loc="lower left", frameon=True)
    add_scaling_inset(ax, "PSPIN", items)


def nk_final_panel(ax, data_arr, k_values):
    items = []
    for idx, k_val in enumerate(k_values):
        combined = []
        for entry in data_arr[idx][:NUM_REPS_FINAL]:
            combined.extend(entry["dfes"][-1])
        dfe_arr = np.asarray(combined, dtype=float) * 2000
        color = PEAK4_COLORS[idx % len(PEAK4_COLORS)]
        label = f"$K={k_val}$"
        plot_kde(ax, dfe_arr, color, label, bw_method=0.25)
        items.append((k_val, color, label))
    ax.set_xlabel(r"Fitness effect $(s)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    _set_xlim_with_epsilon(ax)
    ax.legend(frameon=True, loc="lower left")
    add_scaling_inset(ax, "NK", items)


class FixedPowerFormatter(ScalarFormatter):
    """ScalarFormatter that locks the y-axis exponent to a chosen power."""
    def __init__(self, power):
        super().__init__(useMathText=True)
        self._fixed_power = power
        self.set_scientific(True)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_power

    def _set_orderOfMagnitude(self, range_):
        self.orderOfMagnitude = self._fixed_power

    def __call__(self, x, pos=None):
        return f"{x / 10**self._fixed_power:.0f}"


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
    fgm_final = load_fgm_data()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data_arr, nk_k_values = load_nk_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.38, wspace=0.15)

    # A: additive (p-spin P=1) DFE evolution as a 3D waterfall over time.
    # The top-left cell is replaced by a 3d axes.
    axes[0, 0].remove()
    ax_a = fig.add_subplot(2, 2, 1, projection="3d")
    pspin_p1_waterfall_panel(ax_a, pspin_data)

    # B, C, D: final DFEs — FGM, SK, NK
    fgm_final_panel(axes[0, 1], fgm_final)
    axes[0, 1].set_title("FGM", fontsize=16, pad=10)

    pspin_final_panel(axes[1, 0], pspin_data)
    axes[1, 0].set_title("p-spin", fontsize=16, pad=10)

    nk_final_panel(axes[1, 1], nk_data_arr, nk_k_values)
    axes[1, 1].set_title("NK", fontsize=16, pad=10)

    # B–D: standard 2d panel labels at the top-left of each data area.
    label_kw = dict(fontsize=18, fontweight="bold", va="bottom", ha="left")
    for panel_label, ax in zip(["B", "C", "D"], [axes[0, 1], axes[1, 0], axes[1, 1]]):
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes, **label_kw)
        style_axis(ax)

    # Panel A is a 3d axes whose bounding box is much larger than a 2d data area,
    # so transAxes(-0.1, 1.05) would land in the wrong place. Align "A" with the
    # grid by borrowing C's x (it sits directly below A) and B's y (same row).
    inv = fig.transFigure.inverted()
    x_ref = inv.transform(axes[1, 0].transAxes.transform((-0.1, 1.05)))[0]
    y_ref = inv.transform(axes[0, 1].transAxes.transform((-0.1, 1.05)))[1]
    fig.text(x_ref, y_ref, "A", **label_kw)

    axes[1, 1].yaxis.set_major_formatter(FixedPowerFormatter(-2))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig4_peak_dfes.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
