r"""Angular scrambling for the pure p-spin (SK) adaptive walk (figS7_sk_angular).

A 2x2 figure (rows = interaction order, columns = re-anchoring shell):
  A  Far-field angular scrambling (p=2): log of the in-shell direction autocorrelation
     <u_hat(t_ref).u_hat(t)>, re-anchored at theta_0 = pi/4 (N=1000).
  B  Near-field angular scrambling (p=2), re-anchored at theta_0 = pi/12 (N=1000).
  C  Same as A but for p=3 (highest available N=500).
  D  Same as B but for p=3 (N=500).

For each walk the spin configuration sigma(t) is replayed from the stored flip sequence.
Writing r_t = sigma(t) and rhat_f = sigma_f/||sigma_f||, the angle to the final config is
theta(t) = arccos(r_t.rhat_f / ||r_t||) and u_hat(t) is the unit in-shell (perpendicular to
rhat_f) direction. We re-anchor at the first step whose perpendicular radius
R = sqrt(N) sin(theta) drops to sqrt(N) sin(theta_0), and track the direction autocorrelation
<u_hat(t_ref).u_hat(t)> from there, with a -t/tau (tau = R^2/2) diffusive reference.

Note: the autocorrelation is built directly from the spin configurations and is invariant under
any orthonormal change of basis (e.g. the J-eigenbasis used for p=2), so the same construction
applies unchanged to p=3, where no single pairwise coupling matrix exists.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ───────────────────────────────────── Configuration ─────────────────────────────────────
# Highest available N per interaction order used for the angular panels.
ANGULAR_FILES = {
    2: "../data/PSPIN/N1000_P2_pure_repeats10.pkl",
    3: "../data/PSPIN/N500_P3_pure_repeats10.pkl",
}
CACHE_PATH = "../data/cache/figS7_sk_angular_cache.pkl"

colors = sns.color_palette("CMRmap", 6)

# Angular-scrambling shells (re-anchor radius), far field then near field.
SHELL_THETAS = [np.pi / 4.0, np.pi / 12.0]
SHELL_TITLES = [r'$\theta_0 = \pi/4$', r'$\theta_0 = \pi/12$']
SHELL_COLOR = "m"
SHELL_TRUNC_THRESHOLD = -1.0


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


# ───────────────────────────────────── Data Loading ─────────────────────────────────────

def load_walks(file_path, n_repeats=None):
    """Load the first n_repeats stored walks as (sigma_initial, flip_seq) pairs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if n_repeats is None:
        n_repeats = len(data)
    else:
        n_repeats = min(n_repeats, len(data))

    runs = []
    for k in range(n_repeats):
        entry = data[k]
        sigma_initial = np.asarray(entry.get("init_sigma", entry.get("init_alpha")), dtype=int)
        flip_seq = np.asarray(entry["flip_seq"], dtype=int)
        runs.append((sigma_initial, flip_seq))

    return runs


# ───────────────────────────────────── Core Geometry ─────────────────────────────────────

def _shell_autocorr(sigma_initial, flip_seq, target_radii, eps=1e-10):
    """Shell-anchored direction autocorrelation along one walk.

    shell_corr[m, dt] = <u_hat(t_ref_m), u_hat(t_ref_m + dt)>, where t_ref_m is the first step
    whose perpendicular radius R = sqrt(N) sin(theta) drops to target_radii[m] and u_hat is the
    unit in-shell direction (sigma(t) minus its projection on sigma_f, normalized). Works for any
    interaction order: it depends only on the spin configurations, not on any coupling matrix.
    """
    T = len(flip_seq)
    N = sigma_initial.shape[0]
    sphere_radius = np.sqrt(N)
    target_radii = np.asarray(target_radii, dtype=float)
    n_targets = target_radii.shape[0]

    # Final configuration and its (unit) direction.
    sigma_f = sigma_initial.astype(np.int64).copy()
    for i in flip_seq:
        sigma_f[int(i)] *= -1
    rf = sigma_f.astype(float)
    nrf = np.linalg.norm(rf)
    if nrf < eps:
        return np.full((n_targets, T + 1), np.nan), np.full(n_targets, np.nan)
    rhat_f = rf / nrf

    def _theta_uhat(r):
        nr = np.linalg.norm(r)
        if nr < eps:
            return np.nan, None
        cos_theta = np.clip(np.dot(r, rhat_f) / nr, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        u_vec = r - np.dot(r, rhat_f) * rhat_f
        nu = np.linalg.norm(u_vec)
        if nu < eps:
            return theta, None
        return theta, u_vec / nu

    shell_corr = np.full((n_targets, T + 1), np.nan)
    shell_ref_set = np.zeros(n_targets, dtype=bool)
    shell_ref_steps = np.full(n_targets, -1, dtype=int)
    shell_ref_vecs = np.zeros((n_targets, N), dtype=float)
    shell_ref_radii = np.full(n_targets, np.nan, dtype=float)

    sigma_t = sigma_initial.astype(float).copy()

    def _register(idx, radius, uhat):
        for m, target_radius in enumerate(target_radii):
            if (not shell_ref_set[m]) and np.isfinite(radius) and (radius <= target_radius):
                shell_ref_set[m] = True
                shell_ref_steps[m] = idx
                shell_ref_vecs[m, :] = uhat
                shell_ref_radii[m] = radius
            if shell_ref_set[m]:
                dt = idx - shell_ref_steps[m]
                shell_corr[m, dt] = np.dot(shell_ref_vecs[m, :], uhat)

    theta, uhat = _theta_uhat(sigma_t)
    if uhat is not None:
        _register(0, sphere_radius * np.sin(theta), uhat)

    for t in range(T):
        i = int(flip_seq[t])
        sigma_t[i] *= -1.0
        theta, uhat = _theta_uhat(sigma_t)
        if uhat is None:
            continue
        _register(t + 1, sphere_radius * np.sin(theta), uhat)

    return shell_corr, shell_ref_radii


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


def summarize_log_traces(values, tiny=1e-12):
    mean, std, counts = _finite_mean_std(values)
    log_mean = np.log(np.clip(mean, tiny, None))
    log_lower = np.log(np.clip(mean - std, tiny, None))
    log_upper = np.log(np.clip(mean + std, tiny, None))
    return log_mean, log_lower, log_upper, mean, std, counts


def _pad_traces(traces):
    if not traces:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(trace) for trace in traces)
    padded = np.full((len(traces), max_len), np.nan, dtype=float)
    for idx, trace in enumerate(traces):
        padded[idx, :len(trace)] = trace
    return padded


def _truncate_log_trace(log_mean, *arrays, threshold=-1.1):
    """Truncate arrays at the first index where log_mean <= threshold, inclusive."""
    log_mean = np.asarray(log_mean, dtype=float)
    stop = len(log_mean)

    finite = np.flatnonzero(np.isfinite(log_mean))
    if finite.size:
        crossed = np.flatnonzero(log_mean[finite] <= threshold)
        if crossed.size:
            stop = finite[crossed[0]] + 1
        else:
            stop = finite[-1] + 1
    else:
        stop = 0

    truncated = [log_mean[:stop]]
    truncated.extend(np.asarray(arr)[:stop] for arr in arrays)
    return tuple(truncated)


# ───────────────────────────────────── Computations ─────────────────────────────────────

def compute_shell_panels(file_path, p, n_repeats):
    """Angular-scrambling panels for one walk file (one interaction order p)."""
    runs = load_walks(file_path, n_repeats=n_repeats)
    if not runs:
        raise RuntimeError(f"No SK runs were loaded from {file_path}.")

    n_dim = runs[0][0].size
    target_radii = np.sqrt(n_dim) * np.sin(np.asarray(SHELL_THETAS, dtype=float))

    shell_traces = [[] for _ in target_radii]
    shell_ref_radii = [[] for _ in target_radii]

    for sigma0, flip_seq in runs:
        shell_corr, ref_radii = _shell_autocorr(sigma0, flip_seq, target_radii)
        for m in range(len(target_radii)):
            valid_idx = np.flatnonzero(np.isfinite(shell_corr[m]))
            if valid_idx.size:
                shell_traces[m].append(shell_corr[m, :valid_idx[-1] + 1].copy())
                shell_ref_radii[m].append(ref_radii[m])

    panels = []
    for m in range(len(target_radii)):
        shell_stack = _pad_traces(shell_traces[m])
        if shell_stack.size == 0:
            print(f"Warning: no shell-aligned traces for shell index {m} (p={p}); skipping.")
            panels.append(None)
            continue

        log_mean, log_lower, log_upper, _, _, _ = summarize_log_traces(shell_stack)
        log_mean, log_lower, log_upper, time = _truncate_log_trace(
            log_mean, log_lower, log_upper, np.arange(shell_stack.shape[1]),
            threshold=SHELL_TRUNC_THRESHOLD,
        )

        mean_ref_radius = float(np.nanmean(shell_ref_radii[m]))
        if not np.isfinite(mean_ref_radius) or mean_ref_radius <= 0.0:
            print(f"Warning: invalid reference radius for shell index {m} (p={p}); skipping.")
            panels.append(None)
            continue

        panels.append({
            "title": rf"$p={p}$, {SHELL_TITLES[m]}",
            "time": time,
            "log_mean": log_mean,
            "log_lower": log_lower,
            "log_upper": log_upper,
            "tau_theory": mean_ref_radius ** 2 / 2.0,
            "color": SHELL_COLOR,
        })

    return panels


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

def plot_panel_shell(ax, panel):
    """Log in-shell direction autocorrelation with the linear (diffusive) angular timescale."""
    ax.plot(panel["time"], panel["log_mean"], lw=2.5, color=panel["color"], label="Simulation")
    ax.fill_between(panel["time"], panel["log_lower"], panel["log_upper"],
                    color=panel["color"], alpha=0.30, linewidth=0)
    ax.plot(panel["time"], -panel["time"] / panel["tau_theory"],
            color="black", lw=2.0, ls=":", label=r"Theory (***)")
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r'$\log (\hat{\boldsymbol{u}}(t_\mathrm{ref}) \cdot \hat{\boldsymbol{u}}(t))$')
    ax.set_title(panel["title"])
    ax.legend(frameon=False, loc="lower left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def make_figure(shell_rows, out_path):
    """Assemble the 2x2 figure: rows = interaction order (p=2 then p=3), columns = shells."""
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.0))
    fig.subplots_adjust(wspace=0.30, hspace=0.34)

    labels = [["A", "B"], ["C", "D"]]
    for r in range(2):
        for c in range(2):
            apply_axis_style(axes[r, c], labels[r][c])

    for r, panels in enumerate(shell_rows):
        for c in range(2):
            panel = panels[c] if c < len(panels) else None
            if panel is None:
                axes[r, c].axis("off")
            else:
                plot_panel_shell(axes[r, c], panel)

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

    shell_rows = []
    for p in (2, 3):
        pf = ANGULAR_FILES[p]
        panels, d = _cached(cache, f"shells_p{p}_N{_file_N(pf)}", n_repeats,
                            lambda pf=pf, p=p: compute_shell_panels(pf, p, n_repeats))
        dirty |= d
        shell_rows.append(panels)

    if dirty:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    out_path = os.path.join(out_dir, "figS7_sk_angular.pdf")
    make_figure(shell_rows, out_path)


if __name__ == "__main__":
    main(n_repeats=10)
