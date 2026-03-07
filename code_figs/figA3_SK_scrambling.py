import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})

# ───────────────────────────────────── Data Loading ─────────────────────────────────────
FILE_PATH = "../data/SK/N4000_rho100_beta100_repeats50.pkl"
colors = sns.color_palette("CMRmap", 7)

def load_sk_runs(n_repeats=10):
    """Load the first n_repeats SK simulation trajectories."""
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"{FILE_PATH} not found. Ensure data is present.")

    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    if n_repeats > len(data):
        raise ValueError(f"Requested n_repeats={n_repeats}, but file has only {len(data)} runs.")

    runs = []
    for k in range(n_repeats):
        entry = data[k]
        sigma_initial = np.array(entry["init_alpha"], dtype=int)
        J = np.array(entry["J"], dtype=float)
        flip_seq = np.array(entry["flip_seq"], dtype=int)
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


def analyze_run_time_series(sigma_initial, J, flip_seq, ref_percents, eps=1e-10):
    """
    For one run, compute time series vs percent-walk:
      - theta(t): geodesic angle from r_f.
      - corr_j(t): <uhat(t), uhat(t_ref_j)> for reference percents ref_percents.
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
        return frac, theta, corr

    rhat_f = rf / nrf

    theta = np.full(T + 1, np.nan, dtype=float)
    corr = np.full((len(ref_percents), T + 1), np.nan, dtype=float)

    ref_percents = np.asarray(ref_percents, dtype=float)
    ref_set = np.zeros(len(ref_percents), dtype=bool)
    ref_vecs = np.zeros((len(ref_percents), r0.shape[0]), dtype=float)

    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()

    theta[0], uhat = _theta_and_uhat(r_t, rhat_f, eps=eps)

    if uhat is not None:
        for j, rp in enumerate(ref_percents):
            if (not ref_set[j]) and (frac[0] >= rp):
                ref_set[j] = True
                ref_vecs[j, :] = uhat
                corr[j, 0] = 1.0

    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0

        idx = t + 1
        theta[idx], uhat = _theta_and_uhat(r_t, rhat_f, eps=eps)

        if uhat is not None:
            for j, rp in enumerate(ref_percents):
                if (not ref_set[j]) and (frac[idx] >= rp):
                    ref_set[j] = True
                    ref_vecs[j, :] = uhat
                    corr[j, idx] = 1.0

            if np.any(ref_set):
                corr[ref_set, idx] = ref_vecs[ref_set, :] @ uhat

    return frac, theta, corr


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


# ───────────────────────────────────── Plotting ─────────────────────────────────────

def make_figure(avg_theta, u_corr_single, xgrid, ref_percents, out_path):
    fig, (ax_theta, ax_u) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Panel A: averaged theta(t)
    ax_theta.plot(xgrid, avg_theta, lw=2.5, color=colors[0])
    ax_theta.set_xlabel('Walk completed (%)')
    ax_theta.set_ylabel(r'$\langle \theta(t) \rangle$')
    ax_theta.set_ylim(bottom=0.0)

    # Panel B: azimuthal memory using u-hat
    for j, rp in enumerate(ref_percents):
        ax_u.plot(xgrid, u_corr_single[j], lw=2.0, color=colors[j], label=fr'$t_\mathrm{{ref}}={rp}\%$')

    ax_u.set_xlabel('Walk completed (%)')
    ax_u.set_ylabel(r'$\hat{\mathbf{u}}(t)\cdot\hat{\mathbf{u}}(t_{\mathrm{ref}})$')
    ax_u.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1.2)
    ax_u.set_ylim(-1.05, 1.05)
    ax_u.legend(frameon=False, handlelength=2.2, columnspacing=1.0, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved figure to {out_path}")


# ───────────────────────────────────── Main ─────────────────────────────────────

def main():
    n_repeats = 10
    ref_percents = [0, 0.1, 1, 5, 20, 50, 70]

    runs = load_sk_runs(n_repeats=n_repeats)

    # Common percent grid for averaging
    xgrid = np.linspace(0.0, 100.0, 1001)

    theta_series = []
    u_corr_single = None  # Panel B uses the first run, like the old bottom-right panel.

    for k, (sigma0, J, flip_seq) in enumerate(runs):
        frac, theta, corr = analyze_run_time_series(
            sigma0, J, flip_seq, ref_percents=ref_percents
        )

        theta_series.append(_interp_to_grid(frac, theta, xgrid))

        corr_interp = np.vstack([_interp_to_grid(frac, corr[j], xgrid) for j in range(len(ref_percents))])

        if k == 0:
            u_corr_single = corr_interp

    theta_stack = np.vstack(theta_series)
    avg_theta = np.nanmean(theta_stack, axis=0)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figA3_SK_scrambling.pdf")

    if u_corr_single is None:
        raise RuntimeError('u_corr_single was not set; no runs loaded?')

    make_figure(avg_theta, u_corr_single, xgrid, ref_percents, out_path)


if __name__ == "__main__":
    main()
