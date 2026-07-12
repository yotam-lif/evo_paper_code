r"""Pearson autocorrelation of the DFE along unconstrained FGM adaptive walks, in the
far-field and near-field regimes, compared to the angular scrambling timescale (figS4).

This is the experimental analogue of the model's directional autocorrelation
C_{r-hat}(0,t): for a fixed pool of mutations we track the Pearson correlation of their
fitness effects between a reference background r(0) and a later one r(t),
    rho(0,t) = corr(s_0, s_t),
over the real stored SSWM walks in data/FGM/fgm_rps1000_n*_sig0.05.pkl. Here
s = Delta / w(r), w(r) = exp(-||r||^2), is the selection coefficient. Pearson is invariant
to the per-background rescaling by w(r), but at large n the stored absolute effect
Delta ~ e^{-n} is tiny (~1e-14 at the far-field start for n=32), so we correlate s for
numerical robustness. The pool is fixed along a walk (cmn_fgm.Fisher never flips the
deltas), so column i of every dfes[k] is the same mutation; rho(0) = 1.

Each walk starts far-field at R~(0) = sqrt(n)/sigma = 20 sqrt(n) and descends to its
optimum. The far-field column anchors at the start; the near-field column re-anchors at
the first step where R~ = ||r||/sigma drops to a small reference radius (figS3's idea).

Theory (sections_si/pearson.tex): rho(0,t) ~ C_{r-hat}(0,t)
    = exp( -(n-1)/2 * integral_0^t dt'/R~(t')^2 ).
  * Far field  -- substitute the SSWM radial law R~(t) = R~(0) - t sqrt(pi/2) and integrate,
        log rho = -(n-1)/2 * t / ( R~(0) * (R~(0) - t sqrt(pi/2)) ),
    a closed form that bends down and diverges at the collapse time t* = R~(0)/sqrt(pi/2).
  * Near field -- the linear angular timescale, log rho = -t / tau, tau = 2 R~(0)^2/(n-1).

Rows are the dimensionalities in N_VALUES; columns are far / near. Each panel is shown only
down to log rho = LOG_CUTOFF. Aggregated curves are cached so the multi-GB walk files are
read at most once.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# ---- Configuration ------------------------------------------------------------
SIGMA = 0.05
N_VALUES = [8, 32]            # one row per dimensionality
NEAR_RS = [4.0, 10.0]         # near-field anchor radius R~(0), paired with N_VALUES
REPS = 200                    # replicates averaged per panel
MIN_REPS = 30                 # drop steps sampled by fewer replicates
MIN_SEG_STEPS = 3             # discard a re-anchored segment shorter than this
PEARSON_FLOOR = 1e-3          # clip rho to this before taking the log
LOG_CUTOFF = -0.8             # show each panel only down to this log-correlation
SQRT_HALF_PI = np.sqrt(np.pi / 2.0)   # |dR~/dt| for the SSWM radial law

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR.parent / "figs_paper" / "figS3_fgm_autocorr.pdf"
CACHE_PATH = DATA_DIR / "figS4_pearson_angular_cache.pkl"

FAR_COLOR = "steelblue"
NEAR_COLOR = "darkorange"
THEORY_COLOR = "black"
CLOSED_LABEL = "Theory (Eq. *)"     # far field: time-dependent closed form
LINEAR_LABEL = "Theory (Eq. **)"    # near field: linear angular timescale


# ---- Theory -------------------------------------------------------------------
def pearson_theory_logrho(t, r0_tilde, n):
    r"""log of the time-dependent Pearson closed form,
        log rho(0,t) = -(n-1)/2 * t / ( R~(0) * (R~(0) - t sqrt(pi/2)) ),
    NaN at/after the radial-collapse time t* = R~(0)/sqrt(pi/2)."""
    t = np.asarray(t, dtype=float)
    denom = r0_tilde * (r0_tilde - SQRT_HALF_PI * t)
    out = np.full(t.shape, np.nan)
    valid = denom > 0.0
    out[valid] = -0.5 * (n - 1) * t[valid] / denom[valid]
    return out


# ---- Data ---------------------------------------------------------------------
def anchored_pearson(dfes, rad, r_ref):
    r"""Per-step DFE autocorrelation rho(t) = corr(s_0, s_t) of one walk.

    r_ref is None for the far-field anchor (anchor at step 0), otherwise the first step
    with R~ <= r_ref. Returns (seg, R~_anchor) with seg[0] = 1, or (None, None) if the
    walk never reaches r_ref or the remaining segment is too short.
    """
    if r_ref is None:
        k0 = 0
    else:
        hits = np.nonzero(rad <= r_ref)[0]
        if hits.size == 0:
            return None, None
        k0 = int(hits[0])
    if len(dfes) - k0 < MIN_SEG_STEPS:
        return None, None

    def selection_coeff(k):
        return np.asarray(dfes[k], dtype=float) / np.exp(-(rad[k] * SIGMA) ** 2)

    s0 = selection_coeff(k0)
    if np.std(s0) <= 1e-12:
        return None, None

    seg = np.full(len(dfes) - k0, np.nan)
    seg[0] = 1.0
    for j, k in enumerate(range(k0 + 1, len(dfes)), start=1):
        sk = selection_coeff(k)
        if np.std(sk) > 1e-12:
            seg[j] = float(np.corrcoef(s0, sk)[0, 1])
    return seg, float(rad[k0])


def stack_padded(traces):
    """Pad a list of 1D arrays into a (reps x maxlen) array, NaN past each end."""
    out = np.full((len(traces), max(len(t) for t in traces)), np.nan)
    for i, t in enumerate(traces):
        out[i, :len(t)] = t
    return out


def compute_n_anchors(n, anchors):
    """Load walk file for dimensionality n once and compute the re-anchored Pearson traces
    for every requested anchor. `anchors` maps a cache key to its reference radius (None =
    far/start). Returns {key: {arr, r0, n_reps}}."""
    path = DATA_DIR / "FGM" / f"fgm_rps1000_n{n}_sig0.05.pkl"
    size_gb = path.stat().st_size / 1024 ** 3
    print(f"  n={n}: loading {path.name} ({size_gb:.2f} GB), up to {REPS} reps ...",
          flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)

    bucket = {key: {"traces": [], "r0": []} for key in anchors}
    for rep in data[:REPS]:
        dfes = rep["dfes"]
        rad = np.linalg.norm(np.asarray(rep["traj"], dtype=float)[:len(dfes)], axis=1) / SIGMA
        for key, r_ref in anchors.items():
            seg, r0 = anchored_pearson(dfes, rad, r_ref)
            if seg is not None:
                bucket[key]["traces"].append(seg)
                bucket[key]["r0"].append(r0)
    del data  # free this (multi-GB) file before the next n is loaded

    out = {}
    for key, b in bucket.items():
        out[key] = {
            "arr": stack_padded(b["traces"]),
            "r0": float(np.mean(b["r0"])),
            "n_reps": len(b["traces"]),
            "reps": REPS,                 # requested cap, for cache validity
        }
        print(f"  n={n} [{key}]: {len(b['traces'])} replicates, "
              f"mean anchor R~(0)={out[key]['r0']:.1f}.")
    return out


def load_curves(panels):
    """Return the per-panel Pearson traces, using the cache where possible and loading each
    n's walk file at most once for the rest."""
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    by_n = {}
    for p in panels:
        by_n.setdefault(p["n"], {})[p["key"]] = p["r_ref"]

    curves, dirty = {}, False
    for n, anchors in by_n.items():
        needed = {}
        for key, r_ref in anchors.items():
            cached = cache.get(key)
            if cached is not None and cached.get("reps", 0) >= REPS:
                curves[key] = cached
            else:
                needed[key] = r_ref
        if not needed:
            continue
        for key, result in compute_n_anchors(n, needed).items():
            curves[key] = cache[key] = result
            dirty = True

    if dirty:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    return curves


# ---- Figure -------------------------------------------------------------------
def apply_axis_style(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="bottom", ha="left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)


def subsample(last, n_markers=8):
    return np.arange(0, last + 1, max(1, (last + 1) // n_markers))


def plot_panel(ax, panel, curve):
    """One regime panel: <log rho(t)> vs steps with its theory overlay (closed form for the
    far field, linear -t/tau for the near field), shown down to log rho = LOG_CUTOFF."""
    n = panel["n"]
    r0 = curve["r0"]
    near = panel["field"] == "near"
    color = NEAR_COLOR if near else FAR_COLOR

    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.clip(curve["arr"], PEARSON_FLOOR, None))
    logv[~np.isfinite(curve["arr"]) | (curve["arr"] <= 0)] = np.nan

    enough = np.nonzero(np.sum(np.isfinite(logv), axis=0) >= MIN_REPS)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_logv = np.nanmean(logv, axis=0)
        std_logv = np.nanstd(logv, axis=0)

    # Stop at the first step that reaches LOG_CUTOFF; else at the last well-sampled step.
    last = int(enough[-1]) if enough.size else 1
    below = np.nonzero((mean_logv <= LOG_CUTOFF) & np.isfinite(mean_logv))[0]
    if below.size:
        last = min(last, int(below[0]))

    t = np.arange(last + 1)
    ax.plot(t, mean_logv[:last + 1], color=color, lw=2.2, label="Simulation")
    mk = subsample(last)
    ax.errorbar(t[mk], mean_logv[mk], yerr=std_logv[mk], fmt="o", color=color,
                markersize=4, capsize=3, elinewidth=1.0, alpha=0.85)

    if near:
        tau = 2.0 * r0 ** 2 / (n - 1)
        t_line = np.linspace(0.0, float(last), 50)
        ax.plot(t_line, -t_line / tau, color=THEORY_COLOR, lw=2.0, ls="--", label=LINEAR_LABEL)
    else:
        t_line = np.linspace(0.0, min(float(last), 0.999 * r0 / SQRT_HALF_PI), 300)
        ax.plot(t_line, pearson_theory_logrho(t_line, r0, n), color=THEORY_COLOR, lw=2.0,
                ls="--", label=CLOSED_LABEL)

    ax.set_xlim(0, last)
    ax.set_ylim(LOG_CUTOFF - 0.05, 0.05)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r"$\log\,\rho(0,\,t)$")
    ax.set_title(rf"$n={n}$, $\tilde R(0)={r0:.0f}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(frameon=False, loc="lower left")


def build_panels():
    """Two panels per n, in row-major order: far field (anchored at the start) then near
    field (re-anchored at R~(0) = near_r)."""
    panels = []
    for n, near_r in zip(N_VALUES, NEAR_RS):
        panels.append({"n": n, "field": "far", "r_ref": None, "key": f"n{n}_far"})
        panels.append({"n": n, "field": "near", "r_ref": near_r, "key": f"n{n}_near{near_r:g}"})
    return panels


def make_figure():
    panels = build_panels()
    curves = load_curves(panels)

    fig, axes = plt.subplots(len(N_VALUES), 2, figsize=(12.4, 5.0 * len(N_VALUES)),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.30, hspace=0.40)
    for idx, panel in enumerate(panels):
        ax = axes.flat[idx]
        apply_axis_style(ax, chr(ord("A") + idx))
        plot_panel(ax, panel, curves[panel["key"]])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {OUT_PATH}")


if __name__ == "__main__":
    make_figure()
