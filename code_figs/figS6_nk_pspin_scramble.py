import seaborn as sns

r"""Collapse of the subset DFE-autocorrelation across interaction strength, at two anchor points,
for the NK model (top row) and the pure p-spin model (bottom row).

A 2x2 figure. Each panel plots the log subset-autocorrelation log rho of the DFE against a
connectivity-rescaled step, and overlays a single line -a*tau fit (through the origin) by pooling
all series in the panel; the fitted prefactor a is shown in the legend.

  A  NK, anchored at the start of the walk (t_0 = 0%),  x = t*K/N,  K in {4, 8, 16, 32, 64}, N=500.
  B  NK, anchored near the optimum   (t_0 = 75%),       x = t*K/N.
  C  p-spin, t_0 = 0%,   x = t*(p-1)/N,  p in {2, 3, 4} (N=500 for p=2,3; N=300 for p=4).
  D  p-spin, t_0 = 75%,  x = t*(p-1)/N.

The connectivity is K for NK and p-1 for the p-spin (the number of other spins coupled to a given
site in one interaction). A clean collapse onto -a*tau means the decorrelation timescale is N/K
(resp. N/(p-1)). Both rows are read (never recomputed) from the pre-built subset-autocorrelation
caches under data/cache/ (anchor fractions [0,0.25,0.5,0.75]): the NK curves from
figS9_nk_pearson_cache.pkl and the p-spin curves from figS8_sk_pearson_cache.pkl.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter


class _FixedOrderFormatter(ScalarFormatter):
    """ScalarFormatter that pins the shared 10^n multiplier to a fixed power (e.g. 10^-2), so
    ticks read 2, 4, 6 with a '×10⁻²' offset rather than 0.02, 0.04, 0.06."""

    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self._fixed_order = order

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order


# Minimum number of walks that must still reach a step for it to be plotted (used to trim each
# anchor series). Matches the convention of the sibling scrambling scripts.
PEARSON_MIN_REPS = 3


def apply_axis_style(ax, label):
    """Bold panel letter at the top-left plus the shared spine/tick styling."""
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="bottom", ha="left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# ───────────────────────────────────── Configuration ─────────────────────────────────────
# NK: N=500, several K; connectivity = K. Read from the pre-built subset-autocorrelation cache.
NK_N = 500
K_VALUES = [4, 8, 16, 32, 64]
NK_CACHE = "../data/cache/figS9_nk_pearson_cache.pkl"
NK_KEY = {k: f"nk_pearson_anchors_N{NK_N}_K{k}_0-0.25-0.5-0.75" for k in K_VALUES}
K_COLORS = dict(zip(K_VALUES, sns.color_palette("viridis", len(K_VALUES))))

# p-spin: connectivity = p-1. Read from the figS8 subset-autocorrelation cache.
PSPIN_CACHE = "../data/cache/figS8_sk_pearson_cache.pkl"
PSPIN = {2: 500, 3: 500, 4: 300}     # p -> N of the cached run
PSPIN_KEY = {p: f"pearson_anchors_p{p}_N{N}_0-0.25-0.5-0.75" for p, N in PSPIN.items()}
P_COLORS = dict(zip(sorted(PSPIN), sns.color_palette("CMRmap", len(PSPIN) + 1)))

# Anchors shown (must exist in the [0,0.25,0.5,0.75] fraction list of both caches).
PANEL_FRACS = [0.0, 0.75]
ANCHOR_IDX = {0.0: 0, 0.75: 3}
# Fixed x-window for both p-spin (bottom-row) panels, and the forced x-axis 10^n multiplier.
PSPIN_XMAX = 0.16
PSPIN_XORDER = -1
# Same explicit control for the NK (top-row) panels, per anchor fraction (their ranges differ).
NK_XMAX = {0.0: 0.4, 0.75: 0.4}
NK_XORDER = {0.0: -1, 0.75: -1}
# Fixed (not fitted) NK decorrelation timescale per anchor: tau_s = NK_TAU_CONST * N/K, so the
# reference line has slope -1/NK_TAU_CONST in x = tK/N units.
NK_TAU_CONST = {0.0: 3.0 / 4.0, 0.75: 1.0}
NK_TAU_LABEL = {0.0: r"$\tau_s = 3N/4K$", 0.75: r"$\tau_s = N/K$"}
# Forced 10^n y-axis multiplier per anchor fraction (shared by both rows: same log-rho range per
# column). t_0=0 spans ~[-4,0] -> order 0 (plain integers, no multiplier); t_0=75% spans ~[-0.6,0]
# -> order -1 (ticks read as x10^-1).
YORDER = {0.0: 0, 0.75: -1}

OUT_PATH = "../figs_paper/figS6_nk_pspin_scramble.pdf"


def _trim_anchor(anchors, anchor_idx):
    """Return (t, mean_logv) for one anchor, trimmed to steps reached by >= PEARSON_MIN_REPS reps."""
    a = anchors[anchor_idx]
    counts = a["counts"]
    enough = np.flatnonzero(counts >= PEARSON_MIN_REPS)
    last = int(enough[-1]) if enough.size else 0
    t = np.arange(last + 1)
    return t, a["mean_logv"][:last + 1]


def plot_panel(ax, series, a, title, xlabel, tau_text, xmax, show_ylabel=True, xorder=None,
               yorder=None):
    series_handles = []
    for label, tau, y, color in series:
        m = np.isfinite(y) & (tau <= xmax)
        (h,) = ax.plot(tau[m], y[m], color=color, lw=2.0, marker="o", markersize=3.5, label=label)
        series_handles.append(h)

    xs = np.linspace(0.0, xmax, 40)
    (fit_h,) = ax.plot(xs, -a * xs, color="black", lw=2.0, ls="--", label=r"$-t/\tau_s$")

    ax.set_xlim(0, xmax)
    # Shared 10^n x-axis multiplier. xorder pins the power (e.g. -2); otherwise auto sci notation.
    if xorder is not None:
        fmt = _FixedOrderFormatter(xorder, useMathText=True)
        fmt.set_scientific(True)
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(13)
    # Same shared 10^n multiplier on the y-axis. yorder pins the power; otherwise auto sci notation.
    if yorder is not None:
        fmt = _FixedOrderFormatter(yorder, useMathText=True)
        fmt.set_scientific(True)
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(13)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(r"$\log\,\rho(t_0, t_0 + t)$")
    ax.set_title(title)

    # Series (K / p) legend, single column, lower left.
    series_leg = ax.legend(handles=series_handles, frameon=False, loc="lower left", ncol=1)
    ax.add_artist(series_leg)
    # Fit legend: a single "-t/tau_s" line (so the dashed handle sits level with it) at the top
    # right, with the explicit tau_s written as text just beneath the legend box.
    fit_leg = ax.legend(handles=[fit_h], frameon=False, loc="upper right")
    ax.figure.canvas.draw()
    bb = fit_leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x1, bb.y0 - 0.005, tau_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=13)


# ───────────────────────────────────── Data assembly ─────────────────────────────────────

def _nk_series(frac):
    """(series, xmax) for the NK panel at anchor fraction ``frac``. x = t*K/N.

    Read (never recomputed) from the pre-built NK subset-autocorrelation cache.
    """
    if not os.path.exists(NK_CACHE):
        raise FileNotFoundError(f"{NK_CACHE} not found.")
    with open(NK_CACHE, "rb") as f:
        cache = pickle.load(f)

    series = []
    for k in K_VALUES:
        key = NK_KEY[k]
        if key not in cache:
            raise KeyError(f"{key} missing from {NK_CACHE}.")
        anchors = cache[key]["value"]["anchors"]
        t, y = _trim_anchor(anchors, ANCHOR_IDX[frac])
        series.append((rf"$K={k}$", t * k / NK_N, y, K_COLORS[k]))

    return series, NK_XMAX[frac]


def _pspin_series(frac):
    """(series, xmax, xlabel, tau_expr) for the p-spin panel at anchor fraction ``frac``.

    Read (never recomputed) from the p-spin subset-autocorrelation cache. The collapse variable
    folds in the expected p-spin decorrelation rate, which differs by regime:
      * early (t_0=0):    x = 2(p-1) t / N   (unflipped-subset rate near the start)
      * late  (t_0=0.75): x = 2 p     t / N   (rate near the optimum)
    so a clean collapse onto -x confirms the expected rate.
    """
    if not os.path.exists(PSPIN_CACHE):
        raise FileNotFoundError(f"{PSPIN_CACHE} not found.")
    with open(PSPIN_CACHE, "rb") as f:
        cache = pickle.load(f)

    early = (frac == 0.0)
    factor = (lambda pp: 2 * (pp - 1)) if early else (lambda pp: 2 * pp)
    xlabel = r"$2(p-1)\,t/N$" if early else r"$2p\,t/N$"
    # Fixed theory reference line (slope -1 in these rescaled units); no fitting for p-spin.
    tau_expr = r"$\tau_s = N/2(p-1)$" if early else r"$\tau_s = N/2p$"

    series = []
    for p in sorted(PSPIN):
        n = PSPIN[p]
        key = PSPIN_KEY[p]
        if key not in cache:
            raise KeyError(f"{key} missing from {PSPIN_CACHE}.")
        anchors = cache[key]["value"]["anchors"]
        t, y = _trim_anchor(anchors, ANCHOR_IDX[frac])
        tau = t * factor(p) / n
        label = rf"$p={p}$" if n == 500 else rf"$p={p}$ ($N={n}$)"
        series.append((label, tau, y, P_COLORS[p]))

    return series, PSPIN_XMAX, xlabel, tau_expr


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    fig.subplots_adjust(wspace=0.25, hspace=0.32)
    for ax, label in zip(axes.flat, ("A", "B", "C", "D")):
        apply_axis_style(ax, label)

    # Top row: NK. Bottom row: p-spin.
    for col, frac in enumerate(PANEL_FRACS):
        nk_series, nk_xmax = _nk_series(frac)
        # NK: fixed (not fitted) timescale tau_s = NK_TAU_CONST * N/K, i.e. decay exp(-t/tau_s) =
        # exp(-a x) with a = 1/NK_TAU_CONST and x = tK/N.
        a = 1.0 / NK_TAU_CONST[frac]
        nk_tau = NK_TAU_LABEL[frac]
        plot_panel(axes[0, col], nk_series, a,
                   title=rf"$t_0={100 * frac:g}\%$",
                   xlabel=r"$t\,K/N$", tau_text=nk_tau,
                   xmax=nk_xmax, show_ylabel=(col == 0), xorder=NK_XORDER[frac],
                   yorder=YORDER[frac])

        # Bottom row: p-spin, no fitting -- draw the theory line (slope -1 in rescaled units).
        ps_series, ps_xmax, ps_xlabel, ps_tau = _pspin_series(frac)
        plot_panel(axes[1, col], ps_series, 1.0,
                   title=rf"$t_0={100 * frac:g}\%$",
                   xlabel=ps_xlabel, tau_text=ps_tau,
                   xmax=ps_xmax, show_ylabel=(col == 0), xorder=PSPIN_XORDER,
                   yorder=YORDER[frac])

    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
