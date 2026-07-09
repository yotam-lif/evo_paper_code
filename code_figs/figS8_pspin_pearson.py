import seaborn as sns

r"""Pearson autocorrelation of the selection-coefficient DFE along the pure p-spin (SK)
adaptive walk (figS8_sk_pearson).

A 2x2 figure:
  A  Subset autocorrelation of the distribution of selection coefficients for p=2 (N=500),
     re-anchored at 4 points along the walk, tracking only the spins still in their
     anchor state.
  B  Same subset autocorrelation for p=3 (N=500).
  C  Same subset autocorrelation for p=4 (N=300).
  D  Start/end variance ratio Var[s](0)/Var[s](t) rescaled by (p-1)/p, for each p in
     {2, 3, 4}, collapsing onto a horizontal line at 1.

For each walk we replay the stored flip sequence, recording, at every step t, the spin
configuration sigma(t), the full DFE (the fitness change Delta F_i of flipping each spin via the
native incremental p-spin updates), and the total fitness F(t). The per-spin selection coefficient
is s_i(t) = Delta F_i(t) / F(t). For panels A-C the anchors are the steps at fractions t_0 = k0/T
of walk completion (T = total walk length), placed at t_0 in {0%, 25%, 50%, 85%}. From each anchor
we track only the spins still in their anchor state (flipped an even number of times since the
anchor) and correlate their anchor selection coefficients against their current ones:
rho(dt) = corr(s_anchor[subset], s_{anchor+dt}[subset]), rho(0) = 1. Dashed reference lines
-2(p-1)t/N and -2p t/N are overlaid. Panel D reports, per interaction order p, the across-spin
variance ratio Var(Delta F_i at t=0) / Var(Delta F_i at t=T) rescaled by (p-1)/p, averaged over
repeats, which collapses onto a horizontal line at 1.
"""

import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Repo root on path so we can drive the native p-spin DFE updates.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_pspin

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ───────────────────────────────────── Configuration ─────────────────────────────────────
# Highest available N per interaction order, used for the two Pearson DFE panels.
# A path may be a single pickle holding a list of walk entries, or a directory
# holding one pickle per repeat (see _iter_entries); p=4 uses the latter because
# each N=300 repeat is ~9 GB and must be streamed one at a time.
PEARSON_FILES = {
    2: "../data/PSPIN/N500_P2_pure_repeats10.pkl",
    3: "../data/PSPIN/N500_P3_pure_repeats10.pkl",
    4: "../data/PSPIN/N300_P4_pure_repeats5",
}
CACHE_PATH = "../data/cache/figS8_sk_pearson_cache.pkl"

# Pearson display.
PEARSON_FLOOR = 1e-3      # clip rho before taking the log
PEARSON_MIN_REPS = 3      # plot only steps still reached by at least this many walks
FITNESS_FLOOR = 1e-9      # guard the s = dF/F division against a near-zero fitness
# Re-anchor the subset autocorrelation at 4 points along the walk, defined by these fractions
# of walk completion. Anchor step k0 = round(frac * T), where T is the total walk length; each
# curve's legend label is its t_0 = k0/T (percent of walk completed at the anchor).
ANCHOR_FRACS = [0.0, 0.25, 0.5, 0.75]
ANCHOR_COLORS = sns.color_palette("CMRmap", 4)
# Common steps-since-anchor window shown for every anchor curve, per interaction order.
PEARSON_WINDOW = {2: 20, 3: 20, 4: 20}


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


def _t_frac_label(frac):
    """LaTeX label showing the anchor's t_0 as percent of walk completed (0 -> 0%, 0.25 -> 25%)."""
    return rf"$t_0={100 * frac:g}\%$"


def _fracs_tag(fracs):
    """Stable cache-key fragment encoding a list of anchor fractions."""
    return "-".join(f"{f:g}" for f in fracs)


# ───────────────────────────────────── Replay & autocorrelation ─────────────────────────────────────

def _replay_walk(entry):
    """Replay one stored walk: return (sig_hist, sel_hist, dfe_hist).

    sig_hist[t] is the spin configuration at step t, dfe_hist[t] is the full DFE (the fitness
    effect Delta F_i(t) of flipping each spin via the native incremental p-spin updates), and
    sel_hist[t] is the per-spin selection coefficient s_i(t) = Delta F_i(t) / F(t), where F(t)
    is the total fitness.
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
    return sig_hist, sel_hist, dfe_hist


def _subset_autocorr_from_anchor(sig_hist, sel_hist, k0, track_all=False):
    """Subset selection-coefficient autocorrelation starting from anchor step k0.

    seg[dt] = corr(s_{k0}[subset], s_{k0+dt}[subset]). With track_all=False the subset is the
    spins still in their anchor state (equal to sigma(k0), i.e. flipped an even number of times
    since k0); with track_all=True the subset is *all* spins. seg[0] = 1.
    """
    T = sig_hist.shape[0] - 1
    s_anchor = sel_hist[k0]
    sig_anchor = sig_hist[k0]

    seg = np.full(T - k0 + 1, np.nan)
    seg[0] = 1.0
    for dt, t in enumerate(range(k0 + 1, T + 1), start=1):
        subset = np.ones(sig_hist.shape[1], dtype=bool) if track_all else (sig_hist[t] == sig_anchor)
        if subset.sum() >= 2:
            a = s_anchor[subset]
            b = sel_hist[t][subset]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) \
                    and np.std(a) > 1e-12 and np.std(b) > 1e-12:
                seg[dt] = float(np.corrcoef(a, b)[0, 1])
    return seg


def _rep_sort_key(name):
    """Order per-repeat filenames by their 'rep<k>' index when present, else lexically."""
    for part in os.path.splitext(name)[0].split("_"):
        if part.startswith("rep"):
            try:
                return (0, int(part[3:]))
            except ValueError:
                break
    return (1, name)


def _iter_entries(file_path, n_repeats):
    """Yield up to n_repeats stored walk entries, loading lazily to bound memory.

    Two on-disk layouts are supported:
      * a single pickle holding a list of entries (loaded once, indexed);
      * a directory holding one pickle per repeat (loaded and freed one at a
        time, so only a single ~9 GB walk is resident at once).
    """
    if os.path.isdir(file_path):
        files = sorted(
            (f for f in os.listdir(file_path) if f.endswith(".pkl")),
            key=_rep_sort_key,
        )
        n = min(n_repeats, len(files))
        for name in files[:n]:
            path = os.path.join(file_path, name)
            size_gb = os.path.getsize(path) / 1024 ** 3
            print(f"    loading {name} ({size_gb:.2f} GB) ...", flush=True)
            with open(path, "rb") as f:
                entry = pickle.load(f)
            yield entry
            del entry
    else:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        n = min(n_repeats, len(data))
        for k in range(n):
            yield data[k]
        del data


def compute_pearson_dfe(file_path, n_repeats, track_all=False):
    """Subset selection-coefficient autocorrelation re-anchored at 4 walk-completion fractions.

    With track_all=False each anchor curve tracks only the spins still in their anchor state;
    with track_all=True it tracks all spins. Returns one aggregated curve (columnwise mean/std
    of log rho vs steps-since-anchor, plus replicate counts) per anchor.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    print(f"  computing Pearson DFE from {os.path.basename(file_path)}, "
          f"up to {n_repeats} reps ...", flush=True)

    anchor_traces = [[] for _ in ANCHOR_FRACS]

    for entry in _iter_entries(file_path, n_repeats):
        sig_hist, sel_hist, _ = _replay_walk(entry)
        T = sig_hist.shape[0] - 1
        for a, frac in enumerate(ANCHOR_FRACS):
            k0 = min(T, max(0, int(round(frac * T))))
            anchor_traces[a].append(
                _subset_autocorr_from_anchor(sig_hist, sel_hist, k0, track_all=track_all))

    anchors = []
    for a, frac in enumerate(ANCHOR_FRACS):
        arr = _pad_traces(anchor_traces[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(np.clip(arr, PEARSON_FLOOR, None))
        logv[~np.isfinite(arr) | (arr <= 0)] = np.nan
        mean_logv, std_logv, counts = _finite_mean_std(logv)
        anchors.append({
            "frac": frac,
            "label": _t_frac_label(frac),
            "mean_logv": mean_logv,
            "std_logv": std_logv,
            "counts": counts,
            "n_reps": len(anchor_traces[a]),
        })

    return {"anchors": anchors}


def compute_var_ratio(file_path, n_repeats):
    """Per-repeat across-spin variance of the fitness effect (DFE) at the start and end of the walk.

    For each stored walk, compute the across-spin variance of the fitness effect Delta F_i at
    step 0 (start) and at the final step T (local optimum). Both variances are returned per repeat
    ({"var_start": [...], "var_end": [...]}) so the ratio (and its direction) can be formed at plot
    time without recomputing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    print(f"  computing DFE variance ratio from {os.path.basename(file_path)}, "
          f"up to {n_repeats} reps ...", flush=True)

    var_start_list, var_end_list = [], []
    for entry in _iter_entries(file_path, n_repeats):
        _, _, dfe_hist = _replay_walk(entry)
        var_start = np.nanvar(dfe_hist[0])
        var_end = np.nanvar(dfe_hist[-1])
        if np.isfinite(var_start) and np.isfinite(var_end) and var_start > 0 and var_end > 0:
            var_start_list.append(float(var_start))
            var_end_list.append(float(var_end))
    return {"var_start": var_start_list, "var_end": var_end_list}


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

def plot_panel_pearson(ax, result, p, N, track_all=False, show_ylabel=True):
    """Log subset selection-coefficient autocorrelation vs steps-since-anchor, one curve per anchor.

    Every anchor is shown over the same window (PEARSON_WINDOW[p] steps); a curve ends earlier
    only if the walk runs out of steps after that anchor (the near-optimum anchors). The lower
    dashed reference line is -2(p-1)t/N when tracking the unflipped subset (track_all=False) and
    -2(p+1)t/N when tracking all spins (track_all=True).
    """
    window = PEARSON_WINDOW[p]
    for a, anchor in enumerate(result["anchors"]):
        mean_logv = anchor["mean_logv"]
        std_logv = anchor["std_logv"]
        counts = anchor["counts"]
        color = ANCHOR_COLORS[a]

        # Common window, but never past the last step still reached by >= PEARSON_MIN_REPS walks.
        enough = np.flatnonzero(counts >= PEARSON_MIN_REPS)
        last = int(enough[-1]) if enough.size else 1
        last = min(last, window)

        t = np.arange(last + 1)
        ax.plot(t, mean_logv[:last + 1], color=color, lw=2.0, label=anchor["label"])
        mk = _subsample(last, n_markers=6)
        ax.errorbar(t[mk], mean_logv[mk], yerr=std_logv[mk], fmt="o", color=color,
                    markersize=3.5, capsize=2.5, elinewidth=0.9, alpha=0.85)

    # Reference lines from the anchor.
    t_line = np.linspace(0.0, float(window), 20)
    if track_all:
        ax.plot(t_line, -2.0 * (p + 1) / N * t_line, color="grey", lw=2.0, ls="--",
                label=r"$-2(p+1)\,t/N$")
    else:
        ax.plot(t_line, -2.0 * (p - 1) / N * t_line, color="grey", lw=2.0, ls="--",
                label=r"$-2(p-1)\,t/N$")
    ax.plot(t_line, -2.0 * p / N * t_line, color="black", lw=2.0, ls="--",
            label=r"$-2p\,t/N$")

    ax.set_xlim(0, window)
    ax.set_xlabel("t (# of mutations)")
    if show_ylabel:
        ax.set_ylabel(r"$\log\,\rho(t_0, t_0 + t)$")
    ax.set_title(rf"$p={p}$, $N={N}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # Two separate legends: the t_0 anchors at the lower left, and the reference fits in a
    # distinct legend at the top right.
    handles, labels = ax.get_legend_handles_labels()
    n_anchors = len(result["anchors"])
    anchor_leg = ax.legend(handles[:n_anchors], labels[:n_anchors],
                           frameon=False, loc="lower left")
    ax.add_artist(anchor_leg)
    ax.legend(handles[n_anchors:], labels[n_anchors:],
              frameon=False, loc="upper right")


def plot_panel_var_ratio(ax, per_p_data):
    """Start/end variance ratio Var[s](0)/Var[s](t) * (p-1)/p vs p, with a horizontal line at 1.

    per_p_data maps p -> {"var_start": [...], "var_end": [...]} of per-repeat across-spin DFE
    variances. Each point is the mean over repeats of (var_start/var_end) * (p-1)/p; error bars
    are the std over repeats. The rescaling collapses the ratio onto the dashed line at 1.
    """
    ps = sorted(per_p_data)
    xs, ys, yerr = [], [], []
    for p in ps:
        var_start = np.asarray(per_p_data[p]["var_start"], dtype=float)
        var_end = np.asarray(per_p_data[p]["var_end"], dtype=float)
        r = (var_end / var_start ) * p / (p - 1.0)
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        xs.append(p)
        ys.append(float(np.mean(r)))
        yerr.append(float(np.std(r)))

    ax.axhline(1.0, color="grey", lw=2.0, ls="--")
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", color="black", markersize=8,
                capsize=4, elinewidth=1.2, lw=0)

    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\frac{\mathrm{Var}[s](t)}{\mathrm{Var}[s](0)}\times\frac{p}{p-1}$")
    ax.set_xticks(ps)


def make_figure(pearson, out_path):
    """Assemble the 2x2 figure: subset selection-coefficient autocorrelation for p=2, p=3, p=4
    (panels A-C, unflipped subset), and the start/end fitness-effect (DFE) variance ratio
    Var(DF_0)/Var(DF_t) versus p, with the (p-1)/p reference curve (panel D)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.0))
    fig.subplots_adjust(wspace=0.30, hspace=0.30)

    for ax, label in zip(axes.flat, ("A", "B", "C", "D")):
        apply_axis_style(ax, label)

    # Panels A-C: unflipped subset autocorrelation (y-label only on the leftmost column).
    plot_panel_pearson(axes[0, 0], pearson[2]["curve"], 2, pearson[2]["N"], track_all=False, show_ylabel=True)
    plot_panel_pearson(axes[0, 1], pearson[3]["curve"], 3, pearson[3]["N"], track_all=False, show_ylabel=False)
    plot_panel_pearson(axes[1, 0], pearson[4]["curve"], 4, pearson[4]["N"], track_all=False, show_ylabel=True)
    # Panel D: start/end fitness-effect variance ratio Var(DF_0)/Var(DF_t) versus p.
    per_p_data = {p: pearson[p]["var_ratio"] for p in (2, 3, 4)}
    plot_panel_var_ratio(axes[1, 1], per_p_data)

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

    pearson = {}
    pearson_tag = _fracs_tag(ANCHOR_FRACS)
    for p in (2, 3, 4):
        pf = PEARSON_FILES[p]
        N = _file_N(pf)
        entry = {"N": N}
        value, d = _cached(
            cache, f"pearson_anchors_p{p}_N{N}_{pearson_tag}", n_repeats,
            lambda pf=pf: compute_pearson_dfe(pf, n_repeats, track_all=False))
        dirty |= d
        entry["curve"] = value

        value, d = _cached(
            cache, f"var_dfe_startend_p{p}_N{N}", n_repeats,
            lambda pf=pf: compute_var_ratio(pf, n_repeats))
        dirty |= d
        entry["var_ratio"] = value

        pearson[p] = entry

    if dirty:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    out_path = os.path.join(out_dir, "figS8_pspin_pearson.pdf")
    make_figure(pearson, out_path)


if __name__ == "__main__":
    main(n_repeats=10)
