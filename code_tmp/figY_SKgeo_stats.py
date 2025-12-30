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

def _compute_u1_u2(r0, rf, eps=1e-14):
    """Orthonormal basis for the principal plane span{r0, rf}."""
    n0 = np.linalg.norm(r0)
    if n0 < eps:
        raise ValueError("r0 has near-zero norm; cannot define principal plane.")
    u1 = r0 / n0

    rf_perp = rf - np.dot(rf, u1) * u1
    n2 = np.linalg.norm(rf_perp)
    if n2 < eps:
        return u1, None, True
    u2 = rf_perp / n2
    return u1, u2, False


def analyze_run_time_series(sigma_initial, J, flip_seq, ref_percents, eps=1e-10):
    """
    For one run, compute time series vs percent-walk:
      - cos_memory(t) = cos(angle between r(t) and r(0))
      - eta(t) = ||a_perp(t)||^2 / ||r(t)||^2, where a_perp is orthogonal to span{r0, rf}
      - corr_j(t) = <a_hat(t), a_hat(t_ref_j)> for reference percents ref_percents

    Notes:
      - r(t) is in the eigenbasis of J: r = V^T sigma.
      - References are chosen at the first step with percent >= ref_percent
        where ||a_perp|| > eps; otherwise they remain undefined (NaN).
    """
    T = len(flip_seq)
    steps = np.arange(T + 1)
    frac = steps / steps[-1] * 100.0

    # eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(J)

    sigma_t = sigma_initial.astype(float).copy()
    r0 = eigvecs.T @ sigma_t

    # compute rf in a single pass
    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()
    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0
    rf = r_t

    # principal plane
    u1, u2, degenerate = _compute_u1_u2(r0, rf)
    if degenerate:
        # Return NaNs for azimuthal metrics
        cos_memory = np.full(T + 1, np.nan)
        cos_memory[0] = 1.0
        eta = np.full(T + 1, np.nan)
        corr = np.full((len(ref_percents), T + 1), np.nan)
        return frac, cos_memory, eta, corr, eigvals, eigvecs, r0, rf

    def azimuth_component(x):
        return x - np.dot(x, u1) * u1 - np.dot(x, u2) * u2

    # initialize series
    cos_memory = np.zeros(T + 1, dtype=float)
    cos_memory[0] = 1.0
    eta = np.zeros(T + 1, dtype=float)
    corr = np.full((len(ref_percents), T + 1), np.nan, dtype=float)

    # reference bookkeeping
    ref_percents = np.asarray(ref_percents, dtype=float)
    ref_set = np.zeros(len(ref_percents), dtype=bool)
    ref_vecs = np.zeros((len(ref_percents), r0.shape[0]), dtype=float)

    # time 0 metrics
    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()
    norm0 = np.linalg.norm(r0) + 1e-12

    a_t = azimuth_component(r_t)
    nr = np.linalg.norm(r_t) + 1e-12
    na = np.linalg.norm(a_t)
    eta[0] = (na * na) / (nr * nr)

    if na > eps:
        a_hat = a_t / na
    else:
        a_hat = None

    # set any refs at t=0
    for j, rp in enumerate(ref_percents):
        if (not ref_set[j]) and (frac[0] >= rp) and (a_hat is not None):
            ref_set[j] = True
            ref_vecs[j, :] = a_hat
            corr[j, 0] = 1.0

    # iterate
    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0

        idx = t + 1

        # memory loss
        cos_memory[idx] = np.dot(r_t, r0) / ((np.linalg.norm(r_t) + 1e-12) * norm0)

        # azimuthal amplitude + direction
        a_t = azimuth_component(r_t)
        nr = np.linalg.norm(r_t) + 1e-12
        na = np.linalg.norm(a_t)
        eta[idx] = (na * na) / (nr * nr)

        if na > eps:
            a_hat = a_t / na
        else:
            a_hat = None

        # set references when first reached
        if a_hat is not None:
            for j, rp in enumerate(ref_percents):
                if (not ref_set[j]) and (frac[idx] >= rp):
                    ref_set[j] = True
                    ref_vecs[j, :] = a_hat
                    corr[j, idx] = 1.0

        # correlations for already-set refs
        if a_hat is not None:
            # dot against all refs that are set
            if np.any(ref_set):
                corr[ref_set, idx] = ref_vecs[ref_set, :] @ a_hat

    return frac, cos_memory, eta, corr, eigvals, eigvecs, r0, rf


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

def make_figure(avg_A, avg_C, D_single, xgrid, proj_init, proj_fin, ref_percents, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: averaged cos_memory
    axA.plot(xgrid, avg_A, lw=2.5, color=colors[0])
    axA.set_xlabel('$t$')
    axA.set_ylabel(r'$r(t) \cdot r_0 / N $')
    axA.set_ylim(-0.1, 1.1)
    axA.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1.2)
    axA.axhline(1, color='gray', linestyle=':', alpha=0.5, lw=1.0)

    # Panel B: spectral structure (single run; do not average)
    ranks = np.arange(1, len(proj_fin) + 1)
    axB.scatter(ranks, proj_init, s=6, alpha=0.35, label='Initial', color=colors[3])
    axB.scatter(ranks, proj_fin,  s=6, alpha=0.35, label='Final', color=colors[0])
    axB.set_xscale('log')
    axB.set_yscale('log')
    axB.set_xlabel('Eigenmode rank (1 = largest eigenvalue)')
    axB.set_ylabel(r'$ r_i / N $')
    axB.legend(frameon=False)

    # Panel C: averaged eta
    axC.plot(xgrid, avg_C, lw=2.5, color=colors[0])
    axC.set_xlabel('$t$')
    axC.set_ylabel(r'$\eta(t)=\|a_\perp(t)\|^2/\|r(t)\|^2$')
    axC.set_ylim(bottom=0)

    # Panel D: averaged correlations for multiple t_ref
    for j, rp in enumerate(ref_percents):
        axD.plot(xgrid, D_single[j], lw=2.0, color=colors[j], label=fr'$t_\mathrm{{ref}}={rp}\%$')

    axD.set_xlabel('$t$')
    axD.set_ylabel(r'$\boldsymbol{ \hat{a}}_\perp(t) \cdot \boldsymbol{ \hat{a}_\perp}(t_{\mathrm{ref}}) $')
    axD.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1.2)
    axD.set_ylim(0, 1.05)
    axD.legend(frameon=False, handlelength=2.2, columnspacing=1.0, loc='upper right')

    # No panel titles (per request)

    plt.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved figure to {out_path}")


# ───────────────────────────────────── Main ─────────────────────────────────────

def main():
    n_repeats = 10
    ref_percents = [0, 0.1, 1, 5, 20, 50, 70]

    runs = load_sk_runs(n_repeats=n_repeats)

    # Common percent grid for averaging
    xgrid = np.linspace(0.0, 100.0, 1001)

    # Accumulate interpolated series
    A_series = []
    C_series = []
    D_single = None  # Panel D: single run (repeat 0)

    # Keep Panel B from the first run only
    proj_init = None
    proj_fin = None

    for k, (sigma0, J, flip_seq) in enumerate(runs):
        frac, cos_memory, eta, corr, eigvals, eigvecs, r0, rf = analyze_run_time_series(
            sigma0, J, flip_seq, ref_percents=ref_percents
        )

        A_series.append(_interp_to_grid(frac, cos_memory, xgrid))
        C_series.append(_interp_to_grid(frac, eta, xgrid))

        corr_interp = np.vstack([_interp_to_grid(frac, corr[j], xgrid) for j in range(len(ref_percents))])

        if k == 0:
            D_single = corr_interp
            # spectral projections for Panel B
            # r is in eigenbasis with eigvals ascending; rank 1 corresponds to largest eigval = index N-1
            proj_init = (r0[::-1] ** 2) / (r0 @ r0)
            proj_fin = (rf[::-1] ** 2) / (rf @ rf)

    A_stack = np.vstack(A_series)
    C_stack = np.vstack(C_series)

    avg_A = np.nanmean(A_stack, axis=0)
    avg_C = np.nanmean(C_stack, axis=0)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_sk_geometry_azimuthal_avg10.svg")

    if D_single is None:
        raise RuntimeError('D_single was not set; no runs loaded?')

    make_figure(avg_A, avg_C, D_single, xgrid, proj_init, proj_fin, ref_percents, out_path)


if __name__ == "__main__":
    main()
