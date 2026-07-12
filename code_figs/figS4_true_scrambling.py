r"""Subset-DFE -> total-DFE EMD decay along *unconstrained* FGM adaptive walks,
measured in two regimes -- far field (radial scrambling) and near field (angular
scrambling) -- by re-anchoring the same walks at a chosen starting radius R~(0).

Companion to figS1_radial_scrambling.py (radial scrambling, isolated by a wedge
constraint) and figS2_angular_scrambling.py (angular scrambling, isolated by a
constant-radius constraint). Here neither constraint is imposed: we reanalyze the
ordinary SSWM adaptive walks stored in data/FGM/fgm_rps1000_n*_sig0.05.pkl.

Measure (identical to figS1/figS2)
----------------------------------
Fix a subset M of mutations: those beneficial at the anchor step. The stored mutation
pool is fixed across the whole walk (cmn_fgm.Fisher never flips the deltas), so column
i of every dfes[k] is the same mutation -- the anchor mask therefore tracks the same
mutations at all later steps. At each step we measure the earth-mover's distance
between the DFE of that fixed subset M and the total DFE, normalized to its anchor-step
value and averaged across replicates. All panels use the scale-free selection
coefficient s = dfe / w(r), w(r)=exp(-||r||^2), recovered from the stored position
traj[k] (otherwise the global e^{-||r||^2} prefactor, which changes by ~e^n over the
descent, would swamp the EMD).

Re-anchoring (the key step)
---------------------------
Every production walk *starts* far-field at R~(0) = sqrt(n)/sigma = 20 sqrt(n) and
descends monotonically to its optimum, so a single walk is neither a far- nor a
near-field experiment. To probe a chosen regime we re-anchor each walk at the first
step where its radius R~ = ||r||/sigma drops to a reference value R~(0) (figS1's
realign-to-a-common-radius idea), redefine the subset M there, and track the EMD from
that step.

Theory (2x2 figure)
-------------------
  * Panel A -- far field, EMD for every n (linear y), against the radial law. Radial
    descent dominates: the SSWM radial speed is dR~/dt = -sqrt(pi/2), and the
    normalized EMD tracks the radius,
        EMD(t)/EMD(0) = R~(t)/R~(0) = 1 - sqrt(pi/2) * t / R~(0).

  * Panels B, C -- near field (n=8, n=16), EMD (log y), against the combined law (black),
        log[ EMD(t)/EMD(0) ] = -t / tau,
        tau^-1 = tau_ang^-1 + tau_rad^-1,
        tau_ang = 2 R~(0)^2 / (n-1),   tau_rad = R~(0) / sqrt(pi/2).
    tau_ang is figS2's azimuthal (constant-radius) timescale and tau_rad = R~(0)/sqrt(pi/2)
    is figS1's radial-descent time. Neither constraint is imposed here, so along an
    unconstrained walk both processes scramble the subset DFE simultaneously and their
    *rates* add: the small-R~(0) anchors keep descending radially (dR~/dt = -sqrt(pi/2))
    while they rotate, so neither pure law fits but their summed rate (the combined tau)
    does.

  * Panel D -- near field (n=32), EMD (log y), against the pure-angular law -t/tau_ang
    (black). At large n / large R~(0) the radial term is negligible and tau -> tau_ang, so
    the angular law alone fits.

  The near-panel time axis is NOT set by any timescale: each panel runs out to x_max, the
  first step at which its drawn law or the simulation reaches NEAR_LOG_THRESHOLD = -1.5.

Each near panel uses a directly specified anchor radius R~(0) (--r-near, paired with
NEAR_PANEL_NS), NOT scaled by sqrt(n), and is drawn in black. All anchor radii are
command-line parameters (--r-far, --r-near).
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import cmasher as cmr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import wasserstein_distance

# ----------------------------------------------------------------
# 1. VISUAL STYLE (matched to figS1 / figS2)
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

SIGMA = 0.05
DEFAULT_N_VALUES = [4, 8, 16, 32]
DEFAULT_REPS = 100

DEFAULT_R_FAR = 40.0         # far-field anchor: a large radius every walk passes through
FAR_MAX_STEPS = 40           # cap the panel-A (far-field) time axis at this many steps
NEAR_PANEL_NS = (8, 16, 32)   # dimensionalities n shown in the near-field panels B, C, D
DEFAULT_R_NEAR = (4.0, 6.0, 7.0)  # absolute near-field anchor radius R~(0) for panels B, C, D
                             # (NOT scaled by sqrt(n)); paired with NEAR_PANEL_NS
# Each near panel draws ONE reference law (in black), keyed by n: the combined law for
# B (n=8) and C (n=16), the pure-angular law for D (n=32).
NEAR_PANEL_LAW = {8: "combined", 16: "combined", 32: "angular"}

MIN_SEG_STEPS = 3        # discard a re-anchored segment shorter than this
NEAR_MIN_REPS = 30       # near-field panels: drop steps with fewer qualifying replicates
NEAR_LOG_THRESHOLD = -1.5  # near panels span until the drawn law or simulation reaches this log-EMD

# Figure text, centralized so labels / equation references can be edited in one place.
XLABEL = "Time (steps)"
EMD_YLABEL = r"$\mathrm{EMD}(t)$"
LOGEMD_YLABEL = r"$\log[\mathrm{EMD}(t)]$"
SIM_LABEL = "Simulation"
RADIAL_LABEL = "Theory (Eq. *)"                   # far-field radial law (panel A)
COMBINED_LABEL = "Theory (Eq. **)"                # near-field combined law (panels B, C)
ANGULAR_LABEL = "Theory (Eq. ***)"   # near-field pure-angular law (panel D)
NEAR_COLOR = "black"          # near-field panels B, C, D: data and reference law
# Linestyle + label for each near-panel law (all drawn in NEAR_COLOR).
NEAR_LAW_STYLE = {"combined": ("--", COMBINED_LABEL), "angular": (":", ANGULAR_LABEL)}


def apply_axis_style(ax, label):
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)


# ----------------------------------------------------------------
# 2. DATA LOADING
# ----------------------------------------------------------------
def resolve_fgm_data_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir.parent / "data" / "FGM", script_dir.parent / "data" / "fgm"]
    return next((p for p in candidates if p.exists()), candidates[0])


def resolve_fgm_path(n):
    path = resolve_fgm_data_dir() / f"fgm_rps1000_n{n}_sig0.05.pkl"
    return path if path.exists() else None


# ----------------------------------------------------------------
# 3. PER-REPLICATE RE-ANCHORED EMD TRACE
# ----------------------------------------------------------------
def anchored_emd(dfes, rad, r_ref, scale_free=True):
    r"""Normalized subset-DFE EMD trace of one walk, re-anchored at radius r_ref.

    Finds the first step k0 with R~ <= r_ref, fixes the subset M = {mutations
    beneficial there}, and returns EMD(x[M], x) / EMD_0 for k = k0, k0+1, ... Returns
    None if the walk never reaches r_ref, the remaining segment is too short, the
    subset is empty, or the anchor-step EMD is unusable for normalization.

    scale_free selects the DFE values x used in the EMD:
      True  -> selection coefficient  s = dfe / w(r) = w(r+delta)/w(r) - 1, recovered
               from the stored radius. Needed whenever R~ varies over the segment, where
               w(r)=e^{-||r||^2} would otherwise swamp the EMD with scale. (Used by all
               panels here.)
      False -> absolute fitness effect  Delta = w(r+delta) - w(r)  (the raw stored dfes).
    """
    L = len(dfes)
    hits = np.nonzero(rad <= r_ref)[0]
    if hits.size == 0:
        return None
    k0 = int(hits[0])
    if L - k0 < MIN_SEG_STEPS:
        return None
    mask = np.asarray(dfes[k0], dtype=float) > 0.0
    if not np.any(mask):
        return None

    seg = np.empty(L - k0)
    for j, k in enumerate(range(k0, L)):
        x = np.asarray(dfes[k], dtype=float)           # absolute effect w(r+d) - w(r)
        if scale_free:
            w0 = np.exp(-(rad[k] * SIGMA) ** 2)        # w(r) = exp(-||r||^2)
            if not np.isfinite(w0) or w0 <= 0.0:
                seg[j] = np.nan
                continue
            x = x / w0                                 # selection coefficient
        seg[j] = wasserstein_distance(x[mask], x)
    if not np.isfinite(seg[0]) or seg[0] <= 0.0:
        return None
    return seg / seg[0]


def stack_padded(traces):
    """Pad a list of 1D arrays into a (reps x maxlen) array, NaN past each end."""
    max_len = max(len(t) for t in traces)
    out = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        out[i, :len(t)] = t
    return out


def compute_n_curve(n, reps, r_far, r_near):
    r"""Load one n and return its re-anchored EMD traces (cached so the multi-GB files
    are never reloaded unless reps/anchors change).

    r_near is the absolute near-field anchor radius for this n, or None to compute the
    far-field trace only (used for n that appear in panel A but not in a near panel).

    Returns a dict (or None if the file is missing/unusable):
      far_arr : (reps x T) normalized EMD per step, anchored at R~(0) = r_far
      near_arr: (reps x T) normalized EMD per step, anchored at R~(0) = r_near (omitted
                when r_near is None)
      r_far   : far-field anchor radius (common across n)
      r_near  : near-field anchor radius (present only when r_near is given)
      n_reps  : number of replicates contributing to the far-field anchor
    """
    path = resolve_fgm_path(n)
    if path is None:
        print(f"  n={n}: no rps1000 file found -- skipping.")
        return None

    size_gb = os.path.getsize(path) / 1024 ** 3
    near_msg = f", near R~(0)={r_near:g}" if r_near is not None else ""
    print(f"  n={n}: loading {path.name} ({size_gb:.1f} GB), up to {reps} reps "
          f"[far R~(0)={r_far:g}{near_msg}] ...", flush=True)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except MemoryError:
        print(f"  n={n}: MemoryError on {size_gb:.1f} GB file -- skipping.")
        return None

    far_traces, near_traces = [], []
    for rep in data[:reps]:
        if not isinstance(rep, dict) or "dfes" not in rep or "traj" not in rep:
            continue
        dfes = rep["dfes"]
        traj = np.asarray(rep["traj"], dtype=float)
        L = len(dfes)
        rad = np.linalg.norm(traj[:L], axis=1) / SIGMA
        far = anchored_emd(dfes, rad, r_far, scale_free=True)         # far: selection coeff.
        if far is not None:
            far_traces.append(far)
        if r_near is not None:
            near = anchored_emd(dfes, rad, r_near, scale_free=True)   # near: selection coeff.
            if near is not None:
                near_traces.append(near)
    del data  # free this (multi-GB) file before the next, larger n is loaded

    if not far_traces:
        print(f"  n={n}: no usable replicates -- skipping.")
        return None

    result = {
        "far_arr": stack_padded(far_traces),
        "r_far": float(r_far),
        "n_reps": len(far_traces),
    }
    msg = f"{len(far_traces)} far"
    if r_near is not None and near_traces:
        result["near_arr"] = stack_padded(near_traces)
        result["r_near"] = float(r_near)
        msg += f" / {len(near_traces)} near"
    print(f"  n={n}: {msg} replicates.")
    return result


def load_curves(n_values, reps, r_far, near_anchors, cache_path, refresh=False):
    """Compute (and cache) the per-n re-anchored traces. `near_anchors` maps n -> its
    absolute near-field anchor radius (n absent from it gets a far-field trace only). A
    cache entry is reused only if built with the same anchors and at least `reps` reps;
    keying each n by its own near anchor means tuning one panel reloads only that n."""
    cache = {}
    if cache_path and os.path.exists(cache_path) and not refresh:
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    def key(n):
        rn = near_anchors.get(n)
        return f"n{n}_far{r_far:g}_rn{'None' if rn is None else f'{rn:g}'}"

    curves, dirty = {}, False
    for n in n_values:
        cached = cache.get(key(n))
        if cached is not None and cached.get("n_reps", 0) >= reps and not refresh:
            print(f"  n={n}: using cached curve ({cached['n_reps']} reps).")
            curves[n] = cached
            continue
        result = compute_n_curve(n, reps, r_far, near_anchors.get(n))
        if result is not None:
            curves[n] = result
            cache[key(n)] = result
            dirty = True

    if cache_path and dirty:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  cache written to {cache_path}")
    return curves


# ----------------------------------------------------------------
# 4. FIGURE
# ----------------------------------------------------------------
def _subsample(last, n_markers=9):
    step = max(1, (last + 1) // n_markers)
    return np.arange(0, last + 1, step)


def plot_near_log_panel(ax, n, curve, law):
    """Near-field panel for one dimensionality: <log[ EMD(t)/EMD(0) ]> vs steps since the
    near anchor, against a single straight reference law -t/tau (black), where `law` selects
        "combined":  tau,      tau^-1 = tau_ang^-1 + tau_rad^-1   (figS1 radial + figS2 angular)
        "angular":   tau_ang = 2 R~(0)^2 / (n-1)                  (figS2, pure azimuthal)
    with tau_rad = R~(0) / sqrt(pi/2). The combined timescale adds the angular and radial
    *rates*, capturing that an unconstrained walk scrambles its subset DFE angularly (figS2)
    while still descending radially (figS1); at large n / large R~(0) the radial term is
    negligible and tau -> tau_ang. The time axis is NOT set by a timescale: it runs to x_max,
    the first step at which the drawn law or the simulation reaches NEAR_LOG_THRESHOLD = -1.5,
    and the law is continued across that full axis."""
    r_near = curve["r_near"]
    tau_ang = 2.0 * r_near ** 2 / (n - 1)           # figS2 azimuthal timescale
    tau_rad = r_near / np.sqrt(np.pi / 2.0)         # figS1 radial-descent time
    tau = 1.0 / (1.0 / tau_ang + 1.0 / tau_rad)     # combined: rates add
    tau_law = tau if law == "combined" else tau_ang
    ls, label = NEAR_LAW_STYLE[law]

    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.clip(curve["near_arr"], 1e-3, None))
    logv[~np.isfinite(curve["near_arr"]) | (curve["near_arr"] <= 0)] = np.nan
    enough = np.nonzero(np.sum(np.isfinite(logv), axis=0) >= NEAR_MIN_REPS)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_logv = np.nanmean(logv, axis=0)
        std_logv = np.nanstd(logv, axis=0)

    # Horizontal extent: the first step at which the drawn law or the simulation reaches
    # NEAR_LOG_THRESHOLD (= -1.5) -- decoupled from the timescale. The law -t/tau_law crosses
    # at t = -thr*tau_law; the simulation crosses at its first step with mean log-EMD <= thr.
    # x_max is the earlier of the two, rounded up to a whole number of steps.
    line_cross = -NEAR_LOG_THRESHOLD * tau_law
    sim_hits = np.nonzero(mean_logv <= NEAR_LOG_THRESHOLD)[0]
    sim_cross = float(sim_hits[0]) if sim_hits.size else np.inf
    x_max = max(1, int(np.ceil(min(line_cross, sim_cross))))
    eps = 0.1

    if enough.size:
        last = min(int(enough[-1]), x_max)             # show data out to x_max steps
        t = np.arange(last + 1)
        mk = _subsample(last, n_markers=6)
        ax.plot(t, mean_logv[:last + 1], color=NEAR_COLOR, lw=0.0, marker="o",
                markersize=4.5, label=SIM_LABEL)
        ax.errorbar(t[mk], mean_logv[mk], yerr=std_logv[mk], fmt="none",
                    ecolor=NEAR_COLOR, capsize=2, elinewidth=1.0, alpha=0.4)

    # Draw the reference law -t/tau_law (black), continued across the full axis (0 .. x_max).
    t_line = np.linspace(0.0, x_max + eps, 50)
    ax.plot(t_line, -t_line / tau_law, color=NEAR_COLOR, lw=2.0, ls=ls, label=label)

    ax.set_xlim(0, x_max + eps)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(LOGEMD_YLABEL)
    ax.set_title(rf"$n = {n}$,  $\tilde R(0) = {r_near:.1f}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="lower left")


def make_figure(n_values, reps, r_far, near_anchors, out_path, cache_path, refresh=False):
    print("--- Subset-DFE EMD decay along re-anchored unconstrained FGM walks ---")
    print(f"n values: {n_values}   reps (max) per n: {reps}")
    near_str = ", ".join(f"n={n}: R~(0)={r:g}" for n, r in sorted(near_anchors.items()))
    print(f"far-field anchor R~(0) = {r_far:g}   near-field anchors [{near_str}]")

    curves = load_curves(n_values, reps, r_far, near_anchors, cache_path, refresh=refresh)
    if not curves:
        raise RuntimeError("No FGM data could be loaded for any requested n.")

    n_list = sorted(curves)
    colors = cmr.take_cmap_colors("cmr.emerald", len(n_list), cmap_range=(0.3, 1.0))
    color_by_n = dict(zip(n_list, colors))
    slope = np.sqrt(np.pi / 2.0)

    # Layout (2x2): A = far-field EMD (all n); B, C, D = near-field EMD for NEAR_PANEL_NS.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    fig.subplots_adjust(wspace=0.28, hspace=0.30)
    ax_far, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    apply_axis_style(ax_far, "A")
    apply_axis_style(ax_b, "B")
    apply_axis_style(ax_c, "C")
    apply_axis_style(ax_d, "D")

    # ---- Panel A: far field, EMD for every n, against the radial law
    # 1 - sqrt(pi/2) t / R~(0).
    far_tmax = 1
    for n in n_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_far = np.nanmean(curves[n]["far_arr"], axis=0)
            std_far = np.nanstd(curves[n]["far_arr"], axis=0)
        finite = np.where(np.isfinite(mean_far))[0]
        if not finite.size:
            continue
        last = min(int(finite[-1]), FAR_MAX_STEPS)     # cap the far-field axis
        far_tmax = max(far_tmax, last)
        t = np.arange(last + 1)
        ax_far.plot(t, mean_far[:last + 1], color=color_by_n[n], lw=2.0)
        idx = _subsample(last)
        ax_far.errorbar(t[idx], mean_far[idx], yerr=std_far[idx], fmt="o",
                        color=color_by_n[n], markersize=4, capsize=3, label=fr"$n = {n}$")

    t_far = np.arange(0, far_tmax + 1)
    radial = 1.0 - slope * t_far / r_far
    radial[radial < 0] = np.nan
    ax_far.plot(t_far, radial, color="black", lw=2.2, ls="--", label=RADIAL_LABEL)
    ax_far.set_xlim(0, far_tmax)
    ax_far.set_ylim(-0.02, 1.05)
    ax_far.set_xlabel(XLABEL)
    ax_far.set_ylabel(EMD_YLABEL)
    ax_far.set_title(rf"$\tilde R(0) = {r_far:g}$")
    ax_far.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_far.legend(frameon=False, loc="upper right")

    # ---- Panels B, C, D: near field, one dimensionality each (NEAR_PANEL_NS), each against
    # a single black reference law per NEAR_PANEL_LAW (combined for B, C; angular for D).
    for ax, n in zip((ax_b, ax_c, ax_d), NEAR_PANEL_NS):
        if n in curves and "near_arr" in curves[n]:
            plot_near_log_panel(ax, n, curves[n], NEAR_PANEL_LAW.get(n, "combined"))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(out_path)[1].lstrip(".").lower() or "pdf"
    fig.savefig(out_path, format=ext, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subset-DFE -> total-DFE EMD decay along unconstrained FGM SSWM "
                    "adaptive walks, re-anchored to probe the far-field (radial) and "
                    "near-field (angular) regimes. Reads data/FGM/fgm_rps1000_n*.pkl."
    )
    parser.add_argument("--n-values", default=",".join(str(n) for n in DEFAULT_N_VALUES),
                        help="Comma-separated dimensionalities n (curves in panel A).")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="Max replicates per n used for averaging.")
    parser.add_argument("--r-far", type=float, default=DEFAULT_R_FAR,
                        help="Far-field anchor radius R~(0) (common across n).")
    parser.add_argument("--r-near", default=",".join(f"{r:g}" for r in DEFAULT_R_NEAR),
                        help="Absolute near-field anchor radii R~(0) for the near panels "
                             f"(comma-separated, paired with n={','.join(map(str, NEAR_PANEL_NS))}; "
                             "NOT scaled by sqrt(n)).")
    parser.add_argument("--out", default=os.path.join("..", "figs_paper",
                        "figS4_true_scrambling.pdf"), help="Output figure path.")
    parser.add_argument("--cache", default=os.path.join("..", "data",
                        "figS3_emd_regimes_cache.pkl"),
                        help="Aggregated-curve cache (set empty to disable).")
    parser.add_argument("--refresh", action="store_true",
                        help="Recompute curves even if a cache entry exists.")
    args = parser.parse_args()
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    r_near_vals = [float(x) for x in args.r_near.split(",") if x.strip()]
    near_anchors = dict(zip(NEAR_PANEL_NS, r_near_vals))
    cache_path = args.cache if args.cache else None
    make_figure(n_values, args.reps, args.r_far, near_anchors, args.out, cache_path,
                refresh=args.refresh)
