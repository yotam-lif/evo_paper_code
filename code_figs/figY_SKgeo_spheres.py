#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ───────────────────────────────────── Data Loading ─────────────────────────────────────
FILE_PATH = "../data/SK/N4000_rho100_beta100_repeats50.pkl"

def load_sk_run(repeat_idx=0, file_path=FILE_PATH):
    """Load a single SK simulation trajectory."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} not found. Update FILE_PATH to point to your SK pickle."
        )

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if repeat_idx < 0 or repeat_idx >= len(data):
        raise IndexError(f"repeat_idx {repeat_idx} out of range (len={len(data)}).")

    entry = data[repeat_idx]
    sigma_initial = np.array(entry["init_alpha"], dtype=int)
    J = np.array(entry["J"], dtype=float)
    flip_seq = np.array(entry["flip_seq"], dtype=int)
    return sigma_initial, J, flip_seq

# ───────────────────────────────────── Geometry Helpers ─────────────────────────────────────
def _unit_sphere_wireframe(ax, n=24):
    """Draw a light wireframe unit sphere."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.5, alpha=0.25)

def _normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms

def _downsample_indices(n, max_points):
    if n <= max_points:
        return np.arange(n)
    idx = np.linspace(0, n-1, max_points).round().astype(int)
    # ensure strictly increasing
    idx = np.unique(idx)
    if idx[-1] != n-1:
        idx = np.r_[idx, n-1]
    return idx

def compute_projected_trajectory(sigma0, J, flip_seq, dims, max_points=2500):
    """
    Compute r(t) projected to 3 selected eigenbasis coordinates (dims),
    and normalize each projected point to the unit sphere in R^3.

    Notes:
    - We work in the eigenbasis: r = V^T sigma.
    - We DO NOT build the full r(t) in R^N; we update only the selected dims.
    """
    N = sigma0.shape[0]
    if any(d < 0 or d >= N for d in dims):
        raise ValueError(f"dims must be within [0, {N-1}]")

    # Diagonalize J (ascending eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(J)

    # Initial projected coordinates in eigenbasis: r_d = v_d · sigma
    Vsel = eigvecs[:, dims]                   # shape (N, 3)
    r0 = Vsel.T @ sigma0                      # shape (3,)

    # Reconstruct projected path incrementally
    sigma_t = sigma0.copy()
    T = len(flip_seq)
    traj = np.zeros((T + 1, 3), dtype=float)
    traj[0] = r0
    r_t = r0.copy()

    for t in range(T):
        i = int(flip_seq[t])
        # delta r_d = -2 * sigma_i * v_{i,d}
        r_t = r_t - 2.0 * sigma_t[i] * Vsel[i, :]
        traj[t + 1] = r_t
        sigma_t[i] *= -1

    # Normalize to the unit sphere for visualization
    traj_s = _normalize_rows(traj)

    # Downsample for plotting performance
    idx = _downsample_indices(len(traj_s), max_points=max_points)
    traj_s = traj_s[idx]

    return traj_s, eigvals

def plot_3d_projection_grid(sigma0, J, flip_seq, repeat_idx=0,
                            seed=123, n_panels=9, max_points=2500,
                            elev=18, azim=35):
    """
    Create a 3x3 grid of 3D spherical projections.
    Each panel chooses 3 random eigenbasis coordinates (dims).
    """
    N = sigma0.shape[0]
    rng = np.random.default_rng(seed)

    # Sample 9 triplets of distinct dims (with replacement across panels)
    dim_sets = []
    for _ in range(n_panels):
        dims = rng.choice(N, size=3, replace=False)
        dim_sets.append(tuple(int(d) for d in dims))

    fig = plt.figure(figsize=(13, 13))

    for p, dims in enumerate(dim_sets, start=1):
        ax = fig.add_subplot(3, 3, p, projection='3d')

        traj_s, eigvals = compute_projected_trajectory(
            sigma0, J, flip_seq, dims=dims, max_points=max_points
        )

        # Sphere
        _unit_sphere_wireframe(ax, n=26)

        # Endpoints on the sphere
        r0 = traj_s[0]
        rf = traj_s[-1]

        # Make "near vs far" clearer:
        # Treat +z as near to the viewer (given our chosen view).
        # Plot near-side segments solid; far-side segments faint/dashed.
        z = traj_s[:, 2]
        for i in range(len(traj_s) - 1):
            seg = traj_s[i:i+2]
            near = (z[i] >= 0) and (z[i+1] >= 0)
            if near:
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=1, alpha=0.9, color='blue')
            else:
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=0.7, alpha=0.18, linestyle='--', color='blue')

        # Endpoints (red) + labels
        ax.scatter([r0[0]], [r0[1]], [r0[2]], s=10, c='red', depthshade=False)
        ax.scatter([rf[0]], [rf[1]], [rf[2]], s=10, c='red', depthshade=False)

        # Label slightly offset
        def _label_point(pt, text, dx=0.3):
            ax.text(pt[0] + dx,
                    pt[1],
                    pt[2],
                    text,
                    color='red',
                    fontsize=10)

        _label_point(r0, r"$r_0$")
        _label_point(rf, r"$r_f$")

        # Axis labels: user-requested lambda_index notation.
        # Here the index is the eigenbasis coordinate index (0..N-1, ascending eigenvalues).
        d1, d2, d3 = dims
        ax.set_xlabel(rf"$\lambda_{{{d1}}}$", labelpad=2)
        ax.set_ylabel(rf"$\lambda_{{{d2}}}$", labelpad=2)
        ax.set_zlabel(rf"$\lambda_{{{d3}}}$", labelpad=2)

        # Title: show dims + their eigenvalues (useful context, short)
        ax.set_title(rf"$\lambda=({eigvals[d1]:.3g},\,{eigvals[d2]:.3g},\,{eigvals[d3]:.3g})$", fontsize=10)

        # Styling / view
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)

        # Reduce clutter (ticks are not informative here)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

# ───────────────────────────────────── Main ─────────────────────────────────────
def main():
    repeat_idx = 0
    sigma0, J, flip_seq = load_sk_run(repeat_idx)

    fig = plot_3d_projection_grid(
        sigma0, J, flip_seq,
        repeat_idx=repeat_idx,
        seed=123,
        n_panels=9,
        max_points=2500,
        elev=18,
        azim=35,
    )

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_sk_3d_projection_grid.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
