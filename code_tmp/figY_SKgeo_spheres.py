#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────── Data Loading ─────────────────────────────────────
FILE_PATH = "../data/SK/N4000_rho100_beta100_repeats50.pkl"


def load_sk_run(repeat_idx=0, file_path=FILE_PATH):
    """Load a single SK simulation trajectory."""
    # --- MOCK DATA GENERATION FOR DEMONSTRATION IF FILE MISSING ---
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating dummy data for demonstration.")
        N = 100
        sigma_initial = np.random.choice([-1, 1], size=N)
        # Create a random symmetric matrix J
        A = np.random.randn(N, N)
        J = (A + A.T) / np.sqrt(2 * N)
        # Random flip sequence
        flip_seq = np.random.randint(0, N, size=2000)
        return sigma_initial, J, flip_seq
    # --------------------------------------------------------------

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
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.5, color='gray', alpha=0.15)


def _normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def _downsample_indices(n, max_points):
    if n <= max_points:
        return np.arange(n)
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    return idx


def compute_projected_trajectory(sigma0, J, flip_seq, dims, max_points=2500):
    N = sigma0.shape[0]

    # Diagonalize J (ascending eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(J)

    # Initial projected coordinates in eigenbasis: r_d = v_d · sigma
    Vsel = eigvecs[:, dims]
    r0 = Vsel.T @ sigma0

    # Reconstruct projected path incrementally
    sigma_t = sigma0.copy()
    T = len(flip_seq)
    traj = np.zeros((T + 1, 3), dtype=float)
    traj[0] = r0
    r_t = r0.copy()

    for t in range(T):
        i = int(flip_seq[t])
        r_t = r_t - 2.0 * sigma_t[i] * Vsel[i, :]
        traj[t + 1] = r_t
        sigma_t[i] *= -1

    traj_s = _normalize_rows(traj)
    idx = _downsample_indices(len(traj_s), max_points=max_points)
    traj_s = traj_s[idx]

    return traj_s, eigvals


def plot_3d_projection_grid(sigma0, J, flip_seq, repeat_idx=0,
                            seed=123, n_panels=9, max_points=2500,
                            elev=18, azim=35):
    N = sigma0.shape[0]
    rng = np.random.default_rng(seed)

    dim_sets = []
    for _ in range(n_panels):
        dims = rng.choice(N, size=3, replace=False)
        dim_sets.append(tuple(int(d) for d in dims))

    fig = plt.figure(figsize=(15, 15))  # Slightly larger to accommodate labels

    # Pick a colormap for time evolution (e.g., plasma, viridis, cool)
    cmap = plt.get_cmap('magma')

    for p, dims in enumerate(dim_sets, start=1):
        ax = fig.add_subplot(3, 3, p, projection='3d')

        # Adjust camera distance to prevent label clipping
        ax.dist = 11

        traj_s, eigvals = compute_projected_trajectory(
            sigma0, J, flip_seq, dims=dims, max_points=max_points
        )

        # Draw Sphere
        _unit_sphere_wireframe(ax, n=26)

        # Draw Trajectory with Gradient and Depth Cuing
        z = traj_s[:, 2]
        num_segments = len(traj_s) - 1

        for i in range(num_segments):
            seg = traj_s[i:i + 2]

            # 1. Determine Color based on time (i / total)
            progress = i / num_segments
            color = cmap(progress)

            # 2. Determine Style based on Depth (Near vs Far)
            # Check if both points are in front of the z=0 plane
            near = (z[i] >= -0.05) and (z[i + 1] >= -0.05)  # slight buffer

            if near:
                # Front: Solid, vibrant, slightly thicker
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        lw=1.5, alpha=0.9, color=color)
            else:
                # Back: Dashed, fainter, thinner
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        lw=1.0, alpha=0.3, linestyle='--', color=color)

        # Endpoints
        r0 = traj_s[0]
        rf = traj_s[-1]

        # Start point (use first color of cmap)
        ax.scatter([r0[0]], [r0[1]], [r0[2]], s=20, color='green', edgecolors='k', zorder=10)
        # End point (use last color of cmap)
        ax.scatter([rf[0]], [rf[1]], [rf[2]], s=20, color='green', edgecolors='k', zorder=10)

        # Labels for start/end
        ax.text(r0[0] - 0.075, r0[1], r0[2], r"$r_0$", color='green', fontsize=10, fontweight='bold')
        ax.text(rf[0] - 0.075, rf[1], rf[2], r"$r_f$", color='green', fontsize=10, fontweight='bold')

        # Axis Labels with Eigenvalues
        # Note: labelpad is increased to prevent clipping
        d1, d2, d3 = dims

        lbl_x = rf"$\lambda_{{{d1}}} = {eigvals[d1]:.2f}$"
        lbl_y = rf"$\lambda_{{{d2}}} = {eigvals[d2]:.2f}$"
        lbl_z = rf"$\lambda_{{{d3}}} = {eigvals[d3]:.2f}$"

        ax.set_xlabel(lbl_x, labelpad=10, fontsize=9)
        ax.set_ylabel(lbl_y, labelpad=10, fontsize=9)
        ax.set_zlabel(lbl_z, labelpad=12, fontsize=9, rotation=90)
        ax.zaxis.set_rotate_label(False)

        # Styling
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)

        # Remove ticks to keep it clean
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Transparent background for panels
        ax.patch.set_alpha(0)

    plt.subplots_adjust(wspace=0.2, hspace=0.2, right=0.9, left=0.05, bottom=0.05, top=0.95)
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
        max_points=2500,  # Adjust if too slow
        elev=20,
        azim=45,
    )

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_sk_3d_projection_gradient.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()