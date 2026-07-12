r"""Subset->total selection-coefficient scrambling for the pure p-spin (SK) adaptive walk (figS7).

A 2x3 figure (rows = interaction order, columns = re-anchoring angle):
  A,B,C  Subset->total selection-coefficient earth-mover's-distance decay (figS3-style), p=2
         (N=500), one re-anchoring angle per panel (EMD_ANCHOR_FRACS x theta_0).
  D,E,F  Same subset->total EMD decay for p=3 (N=500).

The per-spin selection coefficient is s_i(t) = Delta F_i(t) / F(t): the fitness change of flipping
spin i (the DFE entry) divided by the current total fitness F(t). At each anchor we fix M = the
spins beneficial (Delta F > 0) at the anchor and track the normalized earth-mover's distance
EMD(s_t[M], s_t) / EMD(anchor) between that subset's selection coefficients and the full
distribution as the background scrambles. The anchors are the steps where the angle to the final
config, theta(t) = arccos(sigma(t).sigma_f / N), reaches each fraction in EMD_ANCHOR_FRACS of
theta_0 (theta_0 = the start). Angular-scrambling reference timescales are overlaid. Expensive
quantities (including the multi-GB p=3 walk file) are cached to data/.
"""

import os
import pickle
import sys
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy.stats import wasserstein_distance

# Repo root on path so we can drive the native p-spin DFE updates.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_pspin

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ───────────────────────────────────── Configuration ─────────────────────────────────────
CACHE_PATH = "../data/cache/figS7_sk_scrambling_cache.pkl"

# Subset->total DFE EMD panels: figS3-style, one anchor angle per panel. Row A,B,C uses p=2
# (N=500); row D,E,F is the same measure for p=3 (N=500).
# Tune EMD_ANCHOR_FRACS to move the anchor angle theta_i = frac * theta_0; labels and the
# cache key derive from it automatically, so changing it here is all that's needed.
EMD_FILE = "../data/PSPIN/N500_P2_pure_repeats10.pkl"  # row A,B,C (p=2)
EMD_FILE_P3 = "../data/PSPIN/N500_P3_pure_repeats10.pkl"  # row D,E,F (p=3)
EMD_ANCHOR_FRACS = [1.0, 0.5, 0.3]
EMD_WINDOW = 20
# Velocity used for the angular timescale tau = sqrt(N) sin(theta_i) / V.
EMD_TAU_VELOCITY = 0.62

# Display.
LOG_FLOOR = 1e-3          # clip the normalized EMD before taking the log
MIN_REPS = 3              # plot only steps still reached by at least this many walks
FITNESS_FLOOR = 1e-9      # guard the s = dF/F division against a near-zero fitness
ANCHOR_COLOR_BY_FRAC = dict(zip(EMD_ANCHOR_FRACS,
                                [plt.get_cmap("viridis")(x) for x in (0.12, 0.50, 0.84)]))


def apply_axis_style(ax, label):
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)


# ───────────────────────────────────── Aggregation helpers ─────────────────────────────────────

def _finite_mean_std(values):
    """Columnwise mean/std ignoring NaNs, without all-NaN runtime warnings."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")

    mask = np.isfinite(values)
    counts = mask.sum(axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=float)
    std = np.full(values.shape[1], np.nan, dtype=float)

    valid = counts > 0
    if np.any(valid):
        safe_vals = np.where(mask[:, valid], values[:, valid], 0.0)
        mean_valid = safe_vals.sum(axis=0) / counts[valid]
        diff = np.where(mask[:, valid], values[:, valid] - mean_valid, 0.0)
        std_valid = np.sqrt((diff ** 2).sum(axis=0) / counts[valid])
        mean[valid] = mean_valid
        std[valid] = std_valid

    return mean, std, counts


def _pad_traces(traces):
    if not traces:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(trace) for trace in traces)
    padded = np.full((len(traces), max_len), np.nan, dtype=float)
    for idx, trace in enumerate(traces):
        padded[idx, :len(trace)] = trace
    return padded


def _subsample(last, n_markers=8):
    return np.arange(0, last + 1, max(1, (last + 1) // n_markers))


def _theta_frac_label(frac):
    """LaTeX label for a fraction of theta_0 (1 -> theta_0, 0.75 -> 3 theta_0/4, 0.5 -> theta_0/2)."""
    fr = Fraction(frac).limit_denominator(100)
    if fr == 1:
        return r"$\theta_0$"
    num, den = fr.numerator, fr.denominator
    return rf"$\theta_0/{den}$" if num == 1 else rf"${num}\theta_0/{den}$"


def _fracs_tag(fracs):
    """Stable cache-key fragment encoding a list of anchor fractions."""
    return "-".join(f"{f:g}" for f in fracs)


# ───────────────────────────────────── Replay & EMD ─────────────────────────────────────

def _replay_walk(entry):
    """Replay one stored walk: return (sel_hist, dfe_hist, theta, theta0).

    dfe_hist[t] is the full DFE (the fitness change Delta F_i of flipping each spin, via the native
    incremental p-spin updates) and sel_hist[t] is the per-spin selection coefficient
    s_i(t) = Delta F_i(t) / F(t), with F(t) the current total fitness. theta(t) =
    arccos(sigma(t).sigma_f / N) is the angle to the final config, and theta0 = theta(0).
    """
    sigma0 = np.asarray(entry.get("init_sigma", entry.get("init_alpha")), dtype=np.int8)
    J = entry["J"]
    flip_seq = np.asarray(entry["flip_seq"], dtype=int)
    N = sigma0.shape[0]
    T = len(flip_seq)

    state = cmn_pspin._initialize_relaxation_state(sigma0, J)
    sig_hist = np.empty((T + 1, N), dtype=np.int8)
    dfe_hist = np.empty((T + 1, N), dtype=np.float64)
    fit_hist = np.empty(T + 1, dtype=np.float64)
    sig_hist[0] = state["sigma"]
    dfe_hist[0] = state["dfe"]
    fit_hist[0] = state["fitness"]
    for j, site in enumerate(flip_seq, start=1):
        cmn_pspin._apply_flip(state, J, int(site))
        sig_hist[j] = state["sigma"]
        dfe_hist[j] = state["dfe"]
        fit_hist[j] = state["fitness"]

    # Per-spin selection coefficient s_i = Delta F_i / F (guard a near-zero total fitness).
    safe_fit = np.where(np.abs(fit_hist) < FITNESS_FLOOR, np.nan, fit_hist)
    sel_hist = dfe_hist / safe_fit[:, None]

    # Angle to the final config (overlap angle; int32 avoids int8 overflow in the dot).
    sig_f = sig_hist[-1].astype(np.int32)
    theta = np.arccos(np.clip(sig_hist.astype(np.int32) @ sig_f / N, -1.0, 1.0))
    return sel_hist, dfe_hist, theta, float(theta[0])


def _subset_emd_from_anchor(sel_hist, dfe_hist, k0):
    """Normalized subset->total selection-coefficient EMD trace from anchor step k0 (figS3-style).

    M = spins beneficial (Delta F > 0) at the anchor, with fixed indices tracked forward.
    seg[dt] = EMD(s_{k0+dt}[M], s_{k0+dt}) / EMD_anchor over the selection coefficients
    s = Delta F / F, so seg[0] = 1. Returns all-NaN if the anchor has no beneficial spins, a
    degenerate anchor EMD, or a step with non-finite selection coefficients (near-zero fitness).
    """
    T = sel_hist.shape[0] - 1
    mask = dfe_hist[k0] > 0.0
    n_steps = T - k0 + 1
    if mask.sum() < 1:
        return np.full(n_steps, np.nan)

    seg = np.full(n_steps, np.nan)
    for dt, t in enumerate(range(k0, T + 1)):
        x = sel_hist[t]
        if np.all(np.isfinite(x)):
            seg[dt] = wasserstein_distance(x[mask], x)
    if not np.isfinite(seg[0]) or seg[0] <= 0.0:
        return np.full(n_steps, np.nan)
    return seg / seg[0]


def compute_emd_subset(file_path, n_repeats):
    """Subset->total selection-coefficient EMD decay re-anchored at the EMD_ANCHOR_FRACS angles.

    figS3-style: at each anchor we fix M = spins beneficial at the anchor and track the normalized
    earth-mover's distance EMD(s_t[M], s_t) / EMD(anchor) between that subset's selection
    coefficients (s = Delta F / F) and the full distribution as the background scrambles. Returns
    one aggregated curve (mean/std of log[EMD(t)/EMD(0)] vs steps-since-anchor, plus replicate
    counts) per anchor.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    size_gb = os.path.getsize(file_path) / 1024 ** 3
    print(f"  loading {os.path.basename(file_path)} ({size_gb:.2f} GB) for subset EMD, "
          f"up to {n_repeats} reps ...", flush=True)
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    n = min(n_repeats, len(data))
    anchor_traces = [[] for _ in EMD_ANCHOR_FRACS]
    theta0_vals = []

    for k in range(n):
        sel_hist, dfe_hist, theta, theta0 = _replay_walk(data[k])
        theta0_vals.append(theta0)
        for a, frac in enumerate(EMD_ANCHOR_FRACS):
            hits = np.flatnonzero(theta <= frac * theta0)
            k0 = int(hits[0]) if hits.size else 0
            anchor_traces[a].append(_subset_emd_from_anchor(sel_hist, dfe_hist, k0))
    del data  # free the walk file before returning

    anchors = []
    for a, frac in enumerate(EMD_ANCHOR_FRACS):
        arr = _pad_traces(anchor_traces[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(np.clip(arr, LOG_FLOOR, None))
        logv[~np.isfinite(arr) | (arr <= 0)] = np.nan
        mean_logv, std_logv, counts = _finite_mean_std(logv)
        anchors.append({
            "frac": frac,
            "label": _theta_frac_label(frac),
            "mean_logv": mean_logv,
            "std_logv": std_logv,
            "counts": counts,
            "n_reps": len(anchor_traces[a]),
        })

    return {"anchors": anchors, "theta0_mean": float(np.mean(theta0_vals))}


# ───────────────────────────────────── Cache ─────────────────────────────────────

def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _cached(cache, key, n_repeats, compute_fn):
    """Return cache[key]['value'], recomputing if absent or generated with a different n_repeats."""
    entry = cache.get(key)
    if entry is not None and entry.get("n_repeats") == n_repeats:
        return cache[key]["value"], False
    value = compute_fn()
    cache[key] = {"n_repeats": n_repeats, "value": value}
    return value, True


# ───────────────────────────────────── Plotting ─────────────────────────────────────

def plot_panel_emd(ax, anchor, N, theta0_mean, p):
    """Log normalized subset->total DFE EMD vs steps-since-anchor (one anchor).

    Two angular-scrambling reference lines are overlaid: the velocity form
    tau = sqrt(N) sin(theta_i) / V (EMD_TAU_VELOCITY) and the diffusive form tau = N sin^2(theta_i)/2.
    """
    mean_logv = anchor["mean_logv"]
    std_logv = anchor["std_logv"]
    counts = anchor["counts"]
    color = ANCHOR_COLOR_BY_FRAC.get(anchor["frac"], "steelblue")

    enough = np.flatnonzero(counts >= MIN_REPS)
    last = int(enough[-1]) if enough.size else 1
    last = min(last, EMD_WINDOW)

    t = np.arange(last + 1)
    y = mean_logv[:last + 1]
    ax.plot(t, y, color=color, lw=2.0, label="Simulation")
    mk = _subsample(last, n_markers=6)
    ax.errorbar(t[mk], mean_logv[mk], yerr=std_logv[mk], fmt="o", color=color,
                markersize=3.5, capsize=2.5, elinewidth=0.9, alpha=0.85)

    # Linear fit through the origin, log[EMD(t)/EMD(0)] = -(m/N) t (the curve is 0 at t=0).
    finite = np.isfinite(y)
    denom = float(np.sum(t[finite] ** 2))
    m = -N * float(np.sum(t[finite] * y[finite])) / denom if denom > 0 else np.nan
    ax.plot(t, -(m / N) * t, color="black", lw=1.8, ls="--",
            label=rf"$-(m/N)\,t,\ m={m:.1f}$")
    ax.plot(t, -2.0 * (p + 1) / N * t, color="darkorange", lw=2.0, ls="--",
            label=r"$-2(p+1)\,t/N$")

    # Angular-scrambling timescales at the anchor angle theta_i = frac * theta_0, both forms.
    theta_i = anchor["frac"] * theta0_mean
    # Velocity form: R = sqrt(N) sin(theta_i), tau = sqrt(N) R^2 / (2 V sqrt(N - R^2)).
    R = np.sqrt(N) * np.sin(theta_i)
    tau_vel = np.sqrt(N) * R ** 2 / (2.0 * EMD_TAU_VELOCITY * np.sqrt(N - R ** 2))
    ax.plot(t, -t / tau_vel, color="crimson", lw=2.0, ls=":",
            label=rf"$-t/\tau,\ \tau=\sqrt{{N}}\,R^2/(2v\sqrt{{N-R^2}}),\ v={EMD_TAU_VELOCITY:g}$")
    # Diffusive form: tau = N sin^2(theta_i)/2 (perpendicular radius sqrt(N) sin(theta_i), tau = R^2/2).
    tau_diff = N * np.sin(theta_i) ** 2 / 2.0
    ax.plot(t, -t / tau_diff, color="darkgreen", lw=2.0, ls=(0, (3, 1, 1, 1)),
            label=r"$-t/\tau,\ \tau=N\sin^2\!\theta_i/2$")
    # Combined rate: tau_3^-1 = tau_1^-1 + tau_2^-1 (tau_1 = velocity, tau_2 = diffusive).
    tau_comb = 1.0 / (1.0 / tau_vel + 1.0 / tau_diff)
    ax.plot(t, -t / tau_comb, color="navy", lw=2.0, ls="--",
            label=r"$-t/\tau,\ \tau^{-1}=\tau_1^{-1}+\tau_2^{-1}$")

    ax.set_xlim(0, last)
    ax.set_xlabel("Steps since anchor")
    ax.set_ylabel(r"$\log[\mathrm{EMD}(t)/\mathrm{EMD}(0)]$")
    ax.set_title(rf"{anchor['label']},  $p={p}$,  $N={N}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(frameon=False, loc="lower left")


def make_figure(emd_rows, out_path):
    """Assemble the 2x3 figure: rows = interaction order (p=2 then p=3), columns = anchor angle."""
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    fig.subplots_adjust(wspace=0.34, hspace=0.40)

    labels = [["A", "B", "C"], ["D", "E", "F"]]
    for r in range(2):
        for c in range(3):
            apply_axis_style(axes[r, c], labels[r][c])

    for row_idx, emd in enumerate(emd_rows):
        for ax, anchor in zip(axes[row_idx], emd["anchors"]):
            plot_panel_emd(ax, anchor, emd["N"], emd["theta0_mean"], emd["p"])
        for ax in axes[row_idx][len(emd["anchors"]):]:
            ax.axis("off")

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {out_path}")


# ───────────────────────────────────── Main ─────────────────────────────────────

def _file_N(file_path):
    """Extract the integer N from a '..N1234_P..' PSPIN filename."""
    base = os.path.basename(file_path)
    return int(base.split("_")[0][1:])


def main(n_repeats=10):
    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)

    cache = _load_cache()
    dirty = False

    emd_tag = _fracs_tag(EMD_ANCHOR_FRACS)
    emd_rows = []
    for p, emd_file in ((2, EMD_FILE), (3, EMD_FILE_P3)):
        res, d = _cached(cache, f"emd_subset_sel_p{p}_N{_file_N(emd_file)}_{emd_tag}", n_repeats,
                         lambda emd_file=emd_file: compute_emd_subset(emd_file, n_repeats))
        dirty |= d
        emd_rows.append({**res, "N": _file_N(emd_file), "p": p})

    if dirty:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    out_path = os.path.join(out_dir, "figS7_sk_scrambling.pdf")
    make_figure(emd_rows, out_path)


if __name__ == "__main__":
    main(n_repeats=10)
