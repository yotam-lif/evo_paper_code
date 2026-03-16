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

# ───────────────────────────────────── Data Loading ─────────────────────────────────────
FILE_PATH = "../data/SK/N4000_rho100_beta100_repeats50.pkl"
colors = sns.color_palette("CMRmap", 6)

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


def load_sk_runs(n_repeats=None):
    """Load the first n_repeats SK simulation trajectories."""
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"{FILE_PATH} not found. Ensure data is present.")

    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    if n_repeats is None:
        n_repeats = len(data)
    elif n_repeats > len(data):
        raise ValueError(f"Requested n_repeats={n_repeats}, but file has only {len(data)} runs.")

    runs = []
    for k in range(n_repeats):
        entry = data[k]
        sigma_initial = np.asarray(entry["init_alpha"], dtype=int)
        J = np.asarray(entry["J"], dtype=float)
        flip_seq = np.asarray(entry["flip_seq"], dtype=int)
        runs.append((sigma_initial, J, flip_seq))

    return runs


# ───────────────────────────────────── Core Geometry ─────────────────────────────────────

def _theta_and_uhat(r_t, rhat_f, eps=1e-10):
    """
    Decompose r_t as:
        r_t = ||r_t|| (cos(theta) rhat_f + sin(theta) uhat),
    where uhat is orthogonal to rhat_f.
    """
    nr = np.linalg.norm(r_t)
    if nr < eps:
        return np.nan, None

    cos_theta = np.dot(r_t, rhat_f) / nr
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    u_vec = r_t - np.dot(r_t, rhat_f) * rhat_f
    nu = np.linalg.norm(u_vec)
    if nu < eps:
        return theta, None
    return theta, u_vec / nu


def analyze_run_time_series(sigma_initial, J, flip_seq, ref_percents, target_radii=None, eps=1e-10):
    """
    For one run, compute time series vs percent-walk:
      - theta(t): geodesic angle from r_f.
      - corr_j(t): <uhat(t), uhat(t_ref_j)> for reference percents ref_percents.
      - shell_corr_m(t): <uhat(t_ref_m), uhat(t_ref_m + t)> where the
        reference time is the first step at which R <= target_radii[m].
    """
    T = len(flip_seq)
    steps = np.arange(T + 1)
    if T == 0:
        frac = np.array([0.0], dtype=float)
    else:
        frac = steps / T * 100.0

    _, eigvecs = np.linalg.eigh(J)

    sigma_t = sigma_initial.astype(float).copy()
    r0 = eigvecs.T @ sigma_t
    n_dim = r0.shape[0]
    sphere_radius = np.sqrt(n_dim)
    target_radii = np.asarray([] if target_radii is None else target_radii, dtype=float)

    # Compute terminal point r_f in one pass.
    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()
    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0
    rf = r_t

    nrf = np.linalg.norm(rf)
    if nrf < eps:
        theta = np.full(T + 1, np.nan, dtype=float)
        corr = np.full((len(ref_percents), T + 1), np.nan, dtype=float)
        shell_corr = np.full((len(target_radii), T + 1), np.nan, dtype=float)
        shell_ref_radii = np.full(len(target_radii), np.nan, dtype=float)
        return frac, theta, corr, shell_corr, shell_ref_radii

    rhat_f = rf / nrf

    theta = np.full(T + 1, np.nan, dtype=float)
    corr = np.full((len(ref_percents), T + 1), np.nan, dtype=float)
    shell_corr = np.full((len(target_radii), T + 1), np.nan, dtype=float)

    ref_percents = np.asarray(ref_percents, dtype=float)
    ref_set = np.zeros(len(ref_percents), dtype=bool)
    ref_vecs = np.zeros((len(ref_percents), r0.shape[0]), dtype=float)
    shell_ref_set = np.zeros(len(target_radii), dtype=bool)
    shell_ref_vecs = np.zeros((len(target_radii), r0.shape[0]), dtype=float)
    shell_ref_steps = np.full(len(target_radii), -1, dtype=int)
    shell_ref_radii = np.full(len(target_radii), np.nan, dtype=float)

    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()

    theta[0], uhat = _theta_and_uhat(r_t, rhat_f, eps=eps)
    radius = sphere_radius * np.sin(theta[0]) if np.isfinite(theta[0]) else np.nan

    if uhat is not None:
        for j, rp in enumerate(ref_percents):
            if (not ref_set[j]) and (frac[0] >= rp):
                ref_set[j] = True
                ref_vecs[j, :] = uhat
                corr[j, 0] = 1.0

        for m, target_radius in enumerate(target_radii):
            if (not shell_ref_set[m]) and np.isfinite(radius) and (radius <= target_radius):
                shell_ref_set[m] = True
                shell_ref_steps[m] = 0
                shell_ref_vecs[m, :] = uhat
                shell_ref_radii[m] = radius
                shell_corr[m, 0] = 1.0

    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0

        idx = t + 1
        theta[idx], uhat = _theta_and_uhat(r_t, rhat_f, eps=eps)
        radius = sphere_radius * np.sin(theta[idx]) if np.isfinite(theta[idx]) else np.nan

        if uhat is not None:
            for j, rp in enumerate(ref_percents):
                if (not ref_set[j]) and (frac[idx] >= rp):
                    ref_set[j] = True
                    ref_vecs[j, :] = uhat
                    corr[j, idx] = 1.0

            if np.any(ref_set):
                corr[ref_set, idx] = ref_vecs[ref_set, :] @ uhat

            for m, target_radius in enumerate(target_radii):
                if (not shell_ref_set[m]) and np.isfinite(radius) and (radius <= target_radius):
                    shell_ref_set[m] = True
                    shell_ref_steps[m] = idx
                    shell_ref_vecs[m, :] = uhat
                    shell_ref_radii[m] = radius

                if shell_ref_set[m]:
                    dt = idx - shell_ref_steps[m]
                    shell_corr[m, dt] = np.dot(shell_ref_vecs[m, :], uhat)

    return frac, theta, corr, shell_corr, shell_ref_radii


def _interp_to_grid(x, y, xgrid):
    """Interpolate 1D y(x) to xgrid; preserves NaNs by masking."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return np.full_like(xgrid, np.nan, dtype=float)

    xi = x[mask]
    yi = y[mask]
    order = np.argsort(xi)
    xi = xi[order]
    yi = yi[order]

    out = np.interp(xgrid, xi, yi)

    # For xgrid outside [min(xi), max(xi)] we should mark as NaN (since interp clamps).
    out[(xgrid < xi.min()) | (xgrid > xi.max())] = np.nan
    return out


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


# ───────────────────────────────────── Plotting ─────────────────────────────────────

def make_figure(avg_cos_theta, std_cos_theta, u_corr_single, xgrid, ref_percents, shell_panels, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    fig.subplots_adjust(wspace=0.28, hspace=0.35)

    ax_theta = axes[0, 0]
    ax_u = axes[0, 1]

    # Panel A: averaged cos(theta(t))
    apply_axis_style(ax_theta, "A")
    ax_theta.plot(xgrid, avg_cos_theta, lw=2.5, color=colors[0])
    ax_theta.fill_between(
        xgrid,
        np.clip(avg_cos_theta - std_cos_theta, -1.0, 1.0),
        np.clip(avg_cos_theta + std_cos_theta, -1.0, 1.0),
        color=colors[0],
        alpha=0.25,
        linewidth=0,
    )
    ax_theta.set_xlabel('Walk completed (%)')
    ax_theta.set_ylabel(r'$\cos\theta(t)$')

    # Panel B: azimuthal memory using u-hat
    apply_axis_style(ax_u, "B")
    for j, rp in enumerate(ref_percents):
        ax_u.plot(xgrid, u_corr_single[j], lw=2.0, color=colors[j+1], label=fr'$t_\mathrm{{ref}}={rp}\%$')

    ax_u.set_xlabel('Walk completed (%)')
    ax_u.set_ylabel(r'$C_{\boldsymbol{\hat u}}(t_{\text{ref}}, t)$')
    ax_u.legend(frameon=False, handlelength=2.2, columnspacing=1.0, loc='lower left')

    for ax, panel in zip(axes[1], shell_panels):
        apply_axis_style(ax, panel["label"])
        ax.plot(panel["time"], panel["log_mean"], lw=2.5, color=panel["color"], label="Simulation")
        ax.fill_between(
            panel["time"],
            panel["log_lower"],
            panel["log_upper"],
            color=panel["color"],
            alpha=0.30,
            linewidth=0,
        )
        ax.plot(
            panel["time"],
            -panel["time"] / panel["tau_theory"],
            color="black",
            lw=2.0,
            ls=":",
            label=r"Theory (*)",
        )
        ax.set_xlabel("Time (steps)")
        ax.set_title(panel["title"])
        ax.legend(frameon=False, loc="lower left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1, 0].set_ylabel(r'$\log C_{\boldsymbol{\hat u}}(0, t)$')

    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved figure to {out_path}")


# ───────────────────────────────────── Main ─────────────────────────────────────

def main(n_repeats=20):
    ref_percents = [0, 20, 40, 60, 80]

    runs = load_sk_runs(n_repeats=n_repeats)
    if not runs:
        raise RuntimeError("No SK runs were loaded.")

    # Common percent grid for averaging
    xgrid = np.linspace(0.0, 100.0, 1001)
    n_dim = runs[0][0].size
    target_thetas = np.array([np.pi / 4.0, np.pi / 40.0], dtype=float)
    target_radii = np.sqrt(n_dim) * np.sin(target_thetas)

    cos_series = []
    shell_traces = [[] for _ in target_radii]
    shell_ref_radii = [[] for _ in target_radii]
    u_corr_single = None  # Panel B uses the first run, like the current figA3 panel.

    for k, (sigma0, J, flip_seq) in enumerate(runs):
        frac, theta, corr, shell_corr, ref_radii = analyze_run_time_series(
            sigma0,
            J,
            flip_seq,
            ref_percents=ref_percents,
            target_radii=target_radii,
        )

        cos_series.append(_interp_to_grid(frac, np.cos(theta), xgrid))

        corr_interp = np.vstack([_interp_to_grid(frac, corr[j], xgrid) for j in range(len(ref_percents))])

        if k == 0:
            u_corr_single = corr_interp

        for m in range(len(target_radii)):
            valid_idx = np.flatnonzero(np.isfinite(shell_corr[m]))
            if valid_idx.size:
                shell_traces[m].append(shell_corr[m, :valid_idx[-1] + 1].copy())
                shell_ref_radii[m].append(ref_radii[m])

    cos_stack = np.vstack(cos_series)
    avg_cos_theta, std_cos_theta, _ = _finite_mean_std(cos_stack)

    shell_titles = [
        r'$\theta_0 = \pi/4$',
        r'$\theta_0 = \pi/40$',
    ]
    shell_labels = ["C", "D"]
    shell_colors = ["m", "m"]
    shell_panels = []
    for m in range(len(target_radii)):
        shell_stack = _pad_traces(shell_traces[m])
        if shell_stack.size == 0:
            raise RuntimeError(f"No valid shell-aligned traces were found for target radius index {m}.")

        log_mean, log_lower, log_upper, _, _, _ = summarize_log_traces(shell_stack)
        log_mean, log_lower, log_upper, time = _truncate_log_trace(
            log_mean,
            log_lower,
            log_upper,
            np.arange(shell_stack.shape[1]),
            threshold=-1.1,
        )
        mean_ref_radius = float(np.nanmean(shell_ref_radii[m]))
        if not np.isfinite(mean_ref_radius) or mean_ref_radius <= 0.0:
            raise RuntimeError(f"Invalid reference radius for shell index {m}.")

        shell_panels.append({
            "label": shell_labels[m],
            "title": shell_titles[m],
            "time": time,
            "log_mean": log_mean,
            "log_lower": log_lower,
            "log_upper": log_upper,
            "tau_theory": mean_ref_radius ** 2 / 2.0,
            "color": shell_colors[m],
        })

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figA3_SK_scrambling.pdf")

    if u_corr_single is None:
        raise RuntimeError('u_corr_single was not set; no runs loaded?')

    make_figure(avg_cos_theta, std_cos_theta, u_corr_single, xgrid, ref_percents, shell_panels, out_path)

if __name__ == "__main__":
    main()
