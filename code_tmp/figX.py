import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import ks_2samp

# ───────────────────────────────────── style ─────────────────────────────────────

plt.rcParams["font.family"] = "sans-serif"
color = sns.color_palette("CMRmap", 3)
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

# Same data path pattern as sk_results.py
FILE_PATH = "../data/SK/N4000_rho100_beta100_repeats50.pkl"


# ───────────────────────────── core SK / geometric helpers ─────────────────────────────

def diagonalize_J(J):
    """
    J: (N, N) symmetric SK matrix

    returns:
        eigvals: (N,)
        eigvecs: (N, N) with eigvecs[i, k] = v^{(k)}_i

    IMPORTANT:
        - We do NOT reorder eigenpairs; Λ_k must match the k-th column of eigvecs.
    """
    eigvals, eigvecs = np.linalg.eigh(J)
    return eigvals, eigvecs


def spins_to_r(sigma, eigvecs):
    """
    Map spins σ to geometric coordinates r.

    sigma: (N,)
    eigvecs: (N, N) with eigvecs[i, k] = v^{(k)}_i

    r_k = sum_i v^{(k)}_i * sigma_i = (V^T sigma)_k
    """
    return eigvecs.T @ sigma


def r_to_grad(r, eigvals):
    """
    Gradient in geometric picture: g_k = 2 * λ_k * r_k

    r: (N,)
    eigvals: (N,)
    """
    return 2.0 * eigvals * r


def reconstruct_geometric_path(sigma_initial, flip_indices, eigvals, eigvecs):
    """
    Efficient reconstruction of the SK geometric picture along the adaptive walk.

    We:
        - compute r_0 and g_0,
        - update r_t and g_t incrementally with each single-spin flip,
        - track the angle between consecutive gradient directions,
        - track the displacement ||r_t - r_0||,
        - track the norm ||r_t||,
        - keep g_0 and ĝ_final.

    Parameters
    ----------
    sigma_initial : (N,)
        Initial spin configuration (±1).
    flip_indices : (T,)
        Sequence of spin indices flipped at each step.
    eigvals : (N,)
        Eigenvalues of J (unsorted, paired with eigvecs).
    eigvecs : (N, N)
        Eigenvectors of J, eigvecs[i, k] = v^{(k)}_i.

    Returns
    -------
    g0 : (N,)
        Gradient in geometric picture at t=0.
    ghat_final : (N,)
        Final unit gradient direction after all flips.
    theta_path : (T,)
        Angle between consecutive gradient directions:
            θ_t = arccos( ĝ(t) · ĝ(t-1) ).
    disp_path : (T+1,)
        Displacement ||r_t - r_0|| as a function of time (t = 0..T).
    rnorm_path : (T+1,)
        Norm ||r_t|| as a function of time (t = 0..T).
    """
    sigma_t = np.asarray(sigma_initial, dtype=int).copy()
    flip_indices = np.asarray(flip_indices, dtype=int)
    N = sigma_t.shape[0]
    T = flip_indices.shape[0]

    # t = 0: full geometric construction
    r_t = spins_to_r(sigma_t, eigvecs)  # O(N^2) once
    r0 = r_t.copy()
    g_t = r_to_grad(r_t, eigvals)       # O(N)
    g0 = g_t.copy()

    ghat_t = g_t / np.linalg.norm(g_t)
    theta_path = np.zeros(T, dtype=float)
    disp_path = np.zeros(T + 1, dtype=float)
    rnorm_path = np.zeros(T + 1, dtype=float)

    disp_path[0] = 0.0
    rnorm_path[0] = np.linalg.norm(r_t)

    # incremental updates
    for t in range(T):
        i = flip_indices[t]

        # δr_k = -2 σ_i v^{(k)}_i  (row i of eigvecs is v^{(k)}_i across k)
        delta_r = -2.0 * sigma_t[i] * eigvecs[i, :]   # (N,)
        r_t = r_t + delta_r
        g_t = g_t + 2.0 * eigvals * delta_r

        sigma_t[i] *= -1  # update spin configuration for consistency

        # displacement and norm
        disp_path[t + 1] = np.linalg.norm(r_t - r0)
        rnorm_path[t + 1] = np.linalg.norm(r_t)

        ghat_new = g_t / np.linalg.norm(g_t)
        theta_path[t] = np.arccos(
            np.clip(np.dot(ghat_new, ghat_t), -1.0, 1.0)
        )
        ghat_t = ghat_new

    ghat_final = ghat_t
    return g0, ghat_final, theta_path, disp_path, rnorm_path


def compute_all_initial_deltas(sigma0, eigvecs):
    """
    True geometric mutation vectors δ^{(i)} at t=0.

    From the SK geometric representation:
        δ^{(i)}_k = -2 σ_i v^{(k)}_i.

    With eigvecs[i, k] = v^{(k)}_i, we have:

        deltas0[i, k] = -2 * sigma0[i] * eigvecs[i, k]

    sigma0: (N,)
    eigvecs: (N, N)

    returns:
        deltas0: (N, N)
        delta_norms0: (N,)
    """
    sigma0 = np.asarray(sigma0, dtype=float)
    N = sigma0.shape[0]
    deltas0 = np.zeros((N, N), dtype=float)

    for i in range(N):
        deltas0[i, :] = -2.0 * sigma0[i] * eigvecs[i, :]

    delta_norms0 = np.linalg.norm(deltas0, axis=1)
    return deltas0, delta_norms0


def compute_initial_fitness_effects(J, sigma0):
    """
    Exact ΔF_i = F(σ^(i)) - F(σ0) for flipping spin i, using the local field.

    SK energy:
        F(σ) = 0.5 σ^T J σ
    Local field:
        h = J σ
    For flipping spin i:
        ΔF_i = F(σ^(i)) - F(σ) = -2 σ_i h_i.

    J      : (N, N)
    sigma0 : (N,)
    """
    sigma0 = np.asarray(sigma0, dtype=float)
    h = J @ sigma0          # O(N^2)
    deltaF = -2.0 * sigma0 * h   # O(N)
    return deltaF


def classify_initial_mutations(deltaF0):
    """
    deltaF0: (N,) exact fitness effects at t=0

    returns:
        V_plus_idx  : indices of beneficial mutations (ΔF > 0)
        V_minus_idx : indices of non-beneficial (ΔF <= 0)
    """
    deltaF0 = np.asarray(deltaF0, dtype=float)
    V_plus_idx = np.where(deltaF0 > 0)[0]
    V_minus_idx = np.where(deltaF0 <= 0)[0]
    return V_plus_idx, V_minus_idx


def prepare_figA_data(deltas0, g0, deltaF0):
    """
    Data for orientation-related panels.

    We compute:
        cosθ_i   = (g0 · δ_i) / (||g0|| ||δ_i||)
        ~ΔF_i    = ΔF_i / (||g0|| ||δ_i||)
    """
    deltas0 = np.asarray(deltas0, dtype=float)
    g0 = np.asarray(g0, dtype=float)
    deltaF0 = np.asarray(deltaF0, dtype=float)

    g0_norm = np.linalg.norm(g0)
    delta_norms0 = np.linalg.norm(deltas0, axis=1)
    grad_proj = deltas0 @ g0

    denom = (delta_norms0 * g0_norm) + 1e-12
    cos_thetas = grad_proj / denom
    normed_deltaF = deltaF0 / denom

    return {
        "cos_thetas": cos_thetas,
        "normed_deltaF": normed_deltaF,
        "deltaF0": deltaF0,
        "grad_proj": grad_proj,
    }


def compute_gradient_rotation(theta_path):
    """
    Turn the θ_t array into (thetas, frac) where frac is % of flips completed.

    theta_path: (T,) angles between ĝ(t) and ĝ(t-1)

    returns:
        thetas: (T,)
        frac  : (T,) t / T
    """
    theta_path = np.asarray(theta_path, dtype=float)
    T = theta_path.shape[0]
    if T == 0:
        return theta_path, np.array([])

    frac = np.arange(T, dtype=float) / float(T)
    return theta_path, frac


def compute_final_projections(deltas0, V_plus_idx, V_minus_idx, ghat_final):
    """
    Use the true initial δ^{(i)} and the final gradient direction to compute:

        final_proj_plus  : projections of normalized δ^{(i)} for i in V_+ onto ĝ_final
        final_proj_minus : same for V_-
        all_proj         : projections for all normalized δ^{(i)}

    deltas0     : (N, N)
    V_plus_idx  : indices of initially beneficial mutations
    V_minus_idx : indices of initially deleterious/non-beneficial mutations
    ghat_final  : (N,) final unit gradient
    """
    deltas0 = np.asarray(deltas0, dtype=float)
    ghat_final = np.asarray(ghat_final, dtype=float)
    N, d = deltas0.shape
    assert d == N

    delta_norms0 = np.linalg.norm(deltas0, axis=1)
    u = deltas0 / (delta_norms0[:, None] + 1e-12)  # normalized δ^{(i)}

    proj_final = u @ ghat_final  # (N,)

    final_proj_plus = proj_final[V_plus_idx]
    final_proj_minus = proj_final[V_minus_idx]
    all_proj = proj_final

    return final_proj_plus, final_proj_minus, all_proj


def compute_ks_tests(V_plus_proj, V_minus_proj, all_proj):
    """
    Run 2-sample KS tests:
        V_+ vs V_-
        V_+ vs All
        V_- vs All
    """
    ks_pm = ks_2samp(V_plus_proj, V_minus_proj)
    ks_pa = ks_2samp(V_plus_proj, all_proj)
    ks_ma = ks_2samp(V_minus_proj, all_proj)
    return {"pm": ks_pm, "pa": ks_pa, "ma": ks_ma}


# ───────────────────────────────────── plotting ─────────────────────────────────────

def plot_all_five_panels(
    figA_data,
    thetas, frac,
    V_plus_proj, V_minus_proj, all_proj,
    ks_results,
    disp_path,
    rnorm_path,
):
    """
    Create a 2×3 figure:

    A (0,0):  histogram of cos(theta) between δ and ∇F(r0).
    B (0,1):  scatter of cosθ vs ΔF.
    C (1,0):  gradient rotation per flip vs % flips.
    D (1,1):  final projections of V_+, V_-, and all onto ĝ_final.
    E (1,2):  displacement ||r_t - r_0|| and ||r_t|| vs % flips.
    The remaining axis (0,2) is hidden.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axA = axes[0, 0]
    axB = axes[0, 1]
    axC = axes[1, 0]
    axD = axes[1, 1]
    axE = axes[1, 2]
    ax_unused = axes[0, 2]

    # Hide the unused top-right axis
    ax_unused.axis("off")

    cos_thetas = figA_data["cos_thetas"]
    deltaF0 = figA_data["deltaF0"]

    # ── Panel A: histogram of cos(theta) ──
    bins_cos = 30
    axA.hist(
        cos_thetas,
        bins=bins_cos,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=color[0],
    )
    axA.set_xlabel(r"$\cos(\theta)$ between $\delta$ and $\nabla F(r_0)$")
    axA.set_ylabel("density")
    axA.set_title("A: Initial orientation distribution")

    # ── Panel B: cosθ vs ΔF (true fitness effect) ──
    axB.scatter(cos_thetas, deltaF0, s=8, alpha=0.5, color=color[1])
    axB.set_xlabel(r"$\cos(\theta)$ between $\delta$ and $\nabla F(r_0)$")
    axB.set_ylabel(r"$\Delta F$")
    axB.set_title("B: Initial fitness effect vs orientation")

    # ── Panel C: θ per flip vs % flips ──
    if len(thetas) > 0:
        axC.plot(frac * 100.0, thetas, lw=1.5, color=color[2])
    axC.set_xlabel("% of flips completed")
    axC.set_ylabel(r"$\angle(\hat g_{t}, \hat g_{t-1})$")
    axC.set_title("C: Gradient rotation per flip")

    # ── Panel D: final projections ──
    bins_proj = 15
    xmin = min(V_plus_proj.min(), V_minus_proj.min(), all_proj.min())
    xmax = 0.0  # focus on negative-side projections; adjust if desired

    sns.histplot(
        V_plus_proj,
        bins=bins_proj,
        binrange=(xmin, xmax),
        stat="density",
        element="step",
        fill=False,
        ax=axD,
        label=r"$V_{+}$",
        color=color[0],
        linewidth=1.5,
    )
    sns.histplot(
        V_minus_proj,
        bins=bins_proj,
        binrange=(xmin, xmax),
        stat="density",
        element="step",
        fill=False,
        ax=axD,
        label=r"$V_{-}$",
        color=color[1],
        linewidth=1.5,
    )
    sns.histplot(
        all_proj,
        bins=bins_proj,
        binrange=(xmin, xmax),
        stat="density",
        element="step",
        fill=False,
        ax=axD,
        label="all",
        color=color[2],
        linewidth=1.5,
    )

    axD.set_xlabel(r"projection on $\hat g_{\mathrm{final}}$")
    axD.set_ylabel("density")
    axD.set_title("D: Scrambling of initial $V_{+}$ / $V_{-}$")

    axD.text(
        0.05,
        0.95,
        rf"$p_{{\mathrm{{KS}}}}(V_+, V_-) = {ks_results['pm'].pvalue:.2g}$",
        transform=axD.transAxes,
        va="top",
    )
    axD.text(
        0.05,
        0.90,
        rf"$p_{{\mathrm{{KS}}}}(V_+, \mathrm{{all}}) = {ks_results['pa'].pvalue:.2g}$",
        transform=axD.transAxes,
        va="top",
    )
    axD.text(
        0.05,
        0.85,
        rf"$p_{{\mathrm{{KS}}}}(V_-, \mathrm{{all}}) = {ks_results['ma'].pvalue:.2g}$",
        transform=axD.transAxes,
        va="top",
    )

    # ── Panel E: displacement ||r_t - r_0|| and ||r_t|| vs % flips ──
    disp_path = np.asarray(disp_path, dtype=float)
    rnorm_path = np.asarray(rnorm_path, dtype=float)
    T = len(disp_path) - 1
    if T > 0:
        frac_disp = np.arange(T + 1, dtype=float) / float(T)
        x_vals = frac_disp * 100.0
        axE.plot(x_vals, disp_path, lw=1.5, color=color[0], label=r"$\|\boldsymbol{r}(t) - \boldsymbol{r}(0)\|$")
        axE.plot(x_vals, rnorm_path, lw=1.5, color=color[1], label=r"$\|\boldsymbol{r}(t)\|$")

    axE.set_xlabel("% of flips completed")
    axE.set_ylabel("magnitude")
    axE.set_title("E: Displacement and $|r_t|$ over time")
    axE.legend(frameon=False)

    # Global styling like sk_results.py
    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-1, 1))
    formatter_y.format = lambda x, _: f"{x:.1f}"

    for ax in [axA, axB, axC, axD, axE]:
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        # Only format y-axis with ScalarFormatter; x-axes use default (no scientific formatting)
        ax.yaxis.set_major_formatter(formatter_y)

    axD.legend(frameon=False)

    plt.tight_layout()
    return fig


def save_figure(fig, filename):
    output_dir = "../figs_paper"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved: {out_path}")


# ───────────────────────────────────── main driver ─────────────────────────────────────

def analyze_single_SK_run(repeat_idx=0, seed=0):
    """
    Run the full geometric analysis for a single SK simulation (one repeat)
    from the SK data file and produce a 2×3 figure.

    repeat_idx: which repeat in the pickle to use (0-based)
    """
    rng = np.random.default_rng(seed)

    # Load SK data (same file structure as in sk_results.py)
    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    if repeat_idx < 0 or repeat_idx >= len(data):
        raise IndexError(f"repeat_idx {repeat_idx} out of range (len={len(data)})")

    entry = data[repeat_idx]
    sigma_initial = np.array(entry["init_alpha"], dtype=int)
    J = np.array(entry["J"], dtype=float)
    flip_seq = np.array(entry["flip_seq"], dtype=int)

    N = sigma_initial.shape[0]
    print(f"Analyzing SK run #{repeat_idx} with N={N}, T={len(flip_seq)} flips.")

    # Eigen-decomposition (once)
    eigvals, eigvecs = diagonalize_J(J)

    # Geometric path: g0, ghat_final, θ per flip, displacement path, norm path
    g0, ghat_final, theta_path, disp_path, rnorm_path = reconstruct_geometric_path(
        sigma_initial, flip_seq, eigvals, eigvecs
    )
    thetas, frac = compute_gradient_rotation(theta_path)

    # Initial true deltas and true ΔF at t=0
    deltas0, delta_norms0 = compute_all_initial_deltas(sigma_initial, eigvecs)
    deltaF0 = compute_initial_fitness_effects(J, sigma_initial)
    V_plus_idx, V_minus_idx = classify_initial_mutations(deltaF0)

    # Orientation-related data
    figA_data = prepare_figA_data(deltas0, g0, deltaF0)

    # Final projections + KS tests
    final_proj_plus, final_proj_minus, all_proj = compute_final_projections(
        deltas0, V_plus_idx, V_minus_idx, ghat_final
    )
    ks_results = compute_ks_tests(final_proj_plus, final_proj_minus, all_proj)

    # Plot and save 2×3 figure
    fig = plot_all_five_panels(
        figA_data,
        thetas,
        frac,
        final_proj_plus,
        final_proj_minus,
        all_proj,
        ks_results,
        disp_path,
        rnorm_path,
    )
    save_figure(fig, "figX_SK_geom_from_simulation.svg")

    return {
        "g0": g0,
        "ghat_final": ghat_final,
        "theta_path": theta_path,
        "disp_path": disp_path,
        "rnorm_path": rnorm_path,
        "deltas0": deltas0,
        "deltaF0": deltaF0,
        "V_plus_idx": V_plus_idx,
        "V_minus_idx": V_minus_idx,
        "final_proj_plus": final_proj_plus,
        "final_proj_minus": final_proj_minus,
        "all_proj": all_proj,
        "ks_results": ks_results,
    }


if __name__ == "__main__":
    # You can change repeat_idx to analyze a different SK trajectory from the file
    analyze_single_SK_run(repeat_idx=0, seed=1)
