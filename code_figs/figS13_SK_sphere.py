import numpy as np
from scipy.stats import ks_2samp
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)
colors = sns.color_palette("CMRmap", 3)

# ---------- Core helpers ----------

def random_unit_vectors(m, d, rng):
    v = rng.normal(size=(m, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def small_random_rotation(V, rng, theta_std=0.05):
    """
    Apply a small random 2D rotation (in a random plane) to all rows of V.
    V: shape (M, d)
    """
    M, d = V.shape

    # Random 2D subspace
    basis = rng.normal(size=(d, 2))
    q, _ = np.linalg.qr(basis)
    e_a = q[:, 0]      # shape (d,)
    e_b = q[:, 1]

    # Components in that plane
    Va = V @ e_a       # shape (M,)
    Vb = V @ e_b

    # Small random angle
    theta = rng.normal(scale=theta_std)
    ct = np.cos(theta)
    st = np.sin(theta)

    Va_new = ct * Va - st * Vb
    Vb_new = st * Va + ct * Vb

    # Update V only in the 2D plane
    V = V + np.outer(Va_new - Va, e_a) + np.outer(Vb_new - Vb, e_b)

    # Renormalize to avoid drift
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    return V


def sample_uniform_negative_projections(num, d, n, rng):
    """
    Sample 'num' projections onto n from unit vectors
    uniformly distributed over the hemisphere {v · n <= 0}.
    """
    proj_list = []
    batch = max(2 * num, 1000)

    while len(proj_list) < num:
        v = random_unit_vectors(batch, d, rng)
        proj = v @ n
        mask = proj <= 0
        proj_list.append(proj[mask])

    proj_all = np.concatenate(proj_list)
    return proj_all[:num]


def run_process(
    d=50,
    M=None,
    theta_std=0.05,
    max_steps=10 ** 6,
    seed=0,
    verbose=True
):
    """
    Simulate the process:
      - M random vectors on S^{d-1}
      - Track by sign of initial projection on n
      - repeat:
        * small random rotation
        * choose one positive-projection vector and flip it
      - stop when no positive-projection vectors remain
    """
    rng = np.random.default_rng(seed)

    if M is None:
        M = 10 * d  # e.g. 10 vectors per dimension

    # Fixed normal n
    n = np.zeros(d)
    n[0] = 1.0

    # Initial cloud
    V = random_unit_vectors(M, d, rng)

    # Initial sets
    proj0 = V @ n
    V_plus_mask = proj0 > 0
    V_minus_mask = proj0 <= 0

    if verbose:
        print(f"d={d}, M={M}")
        print(f"Initial: {V_plus_mask.sum()} V_plus, {V_minus_mask.sum()} V_minus")

    # Dynamics
    steps = 0
    while steps < max_steps:
        steps += 1

        # 1. small random rotation
        V = small_random_rotation(V, rng, theta_std=theta_std)

        # 2. flip one vector with positive projection (if any)
        proj = V @ n
        pos_indices = np.where(proj > 0)[0]
        if len(pos_indices) == 0:
            break

        i = rng.choice(pos_indices)
        V[i] = -V[i]  # flip

    # Final projections
    proj_final = V @ n

    if verbose:
        print(f"Finished in {steps} steps")

    # Final projections for the two sets
    V_plus_proj_final = proj_final[V_plus_mask]
    V_minus_proj_final = proj_final[V_minus_mask]

    return {
        "V_final": V,
        "n": n,
        "V_plus_mask": V_plus_mask,
        "V_minus_mask": V_minus_mask,
        "V_plus_proj_final": V_plus_proj_final,
        "V_minus_proj_final": V_minus_proj_final,
        "steps": steps,
        "rng": rng,
        "d": d,
        "M": M,
        "theta_std": theta_std,
    }


# ---------- Analysis + plotting ----------

def analyze_once(
    d=50,
    M=None,
    theta_std=0.05,
    seed=0
):
    res = run_process(d=d, M=M, theta_std=theta_std, seed=seed)

    V_plus = res["V_plus_proj_final"]
    V_minus = res["V_minus_proj_final"]
    n = res["n"]
    rng = res["rng"]
    d = res["d"]
    M = res["M"]
    theta_std = res["theta_std"]

    # Uniform reference from a negative hemisphere
    uniform_proj = sample_uniform_negative_projections(
        num=len(V_plus),
        d=d,
        n=n,
        rng=rng,
    )

    # KS tests
    ks_pmi = ks_2samp(V_plus, V_minus)
    ks_pu = ks_2samp(V_plus, uniform_proj)
    ks_mu = ks_2samp(V_minus, uniform_proj)

    fig, ax = plt.subplots(figsize=(8, 6))

    bins = 15
    # All are ≤ 0 by construction; focus on a little range around the bulk
    xmin = min(V_plus.min(), V_minus.min(), uniform_proj.min())
    xmax = 0.0

    sns.histplot(
        V_plus,
        bins=bins,
        binrange=(xmin, xmax),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=r"$V_{+}$",
        color=colors[0],
        linewidth=1.5,
    )
    sns.histplot(
        V_minus,
        bins=bins,
        binrange=(xmin, xmax),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=r"$V_{-}$",
        color=colors[1],
        linewidth=1.5,
    )
    sns.histplot(
        uniform_proj,
        bins=bins,
        binrange=(xmin, xmax),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label="uniform",
        color=colors[2],
        linewidth=1.5,
    )

    ax.set_xlabel(r"projection on $n$")
    ax.set_ylabel("density")
    ax.set_title(fr"$d={d}, M={M}, \theta={theta_std}$")
    ax.set_xlim(xmin, xmax)
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")

    # KS p-values as text on the figure
    ax.text(
        0.05, 0.95,
        rf"$p_{{\mathrm{{KS}}}}(V_+, V_-) = {ks_pmi.pvalue:.2g}$",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.05, 0.90,
        rf"$p_{{\mathrm{{KS}}}}(V_+, \mathrm{{unif}}) = {ks_pu.pvalue:.2g}$",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.05, 0.85,
        rf"$p_{{\mathrm{{KS}}}}(V_-, \mathrm{{unif}}) = {ks_mu.pvalue:.2g}$",
        transform=ax.transAxes,
        va="top",
    )

    plt.tight_layout()
    out_dir = os.path.join('..', 'figs_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figS13_SK_sphere.svg')
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved: {out_path}")

    return res


if __name__ == "__main__":
    # Example run
    analyze_once(d=1000, M=1000, theta_std=0.1, seed=1)
