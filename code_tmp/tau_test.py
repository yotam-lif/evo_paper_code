import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, pearsonr, norm
from scipy import integrate
import multiprocessing
import time


# ----------------------------------------------------------------
# Helper: Expected Max of M Gaussians
# ----------------------------------------------------------------
def expected_max_gaussian(m, sigma):
    """
    Calculates E[max(X1, ..., Xm)] where Xi ~ N(0, sigma^2).
    """

    # PDF of max: m * phi(x) * Phi(x)^(m-1)
    # We integrate x * pdf(x) from -inf to inf (effectively -5sigma to 5sigma is enough)

    def integrand(x):
        # working with standard normal z = x/sigma
        z = x / sigma
        pdf_z = norm.pdf(z)
        cdf_z = norm.cdf(z)
        # Avoid numerical underflow for cdf^(m-1)
        return x * m * pdf_z * (cdf_z ** (m - 1)) / sigma

    # Integration bounds: mean is roughly sigma * sqrt(2 ln m)
    # We integrate generously around that peak
    peak_est = sigma * np.sqrt(2 * np.log(m))
    val, err = integrate.quad(integrand, -sigma, peak_est + 5 * sigma)
    return val


# ----------------------------------------------------------------
# Model Classes
# ----------------------------------------------------------------

class FisherModel:
    def __init__(self, n, sigma, m, R0, seed=None):
        self.n = int(n)
        self.sigma = float(sigma)
        self.m = int(m)
        self.rng = np.random.default_rng(seed)
        self.deltas = self.rng.normal(loc=0.0, scale=self.sigma, size=(self.m, self.n))
        self.R0 = float(R0)
        self.r = np.zeros(n)
        self.r[0] = self.R0

    def compute_fitness(self, r):
        return np.exp(-0.5 * np.dot(r, r))

    def compute_dfe(self, r):
        w0 = self.compute_fitness(r)
        r_new = r + self.deltas
        r2_new = np.einsum('ij,ij->i', r_new, r_new)
        w_new = np.exp(-0.5 * r2_new)
        return w_new - w0


class FisherConstantRadius(FisherModel):
    def step(self, epsilon):
        r_candidates = self.r + self.deltas
        dists = np.linalg.norm(r_candidates, axis=1)
        valid_mask = (dists >= (self.R0 - epsilon)) & (dists <= (self.R0 + epsilon))
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) == 0: return False

        choice = self.rng.choice(valid_indices)
        self.r += self.deltas[choice]
        return True


class FisherSSWM(FisherModel):
    def step(self):
        dfe = self.compute_dfe(self.r)
        beneficial_mask = dfe > 0
        if not np.any(beneficial_mask):
            return False

        ben_indices = np.nonzero(beneficial_mask)[0]
        ben_effects = dfe[ben_indices]
        probs = ben_effects / np.sum(ben_effects)

        choice = self.rng.choice(ben_indices, p=probs)
        self.r += self.deltas[choice]
        self.deltas[choice] *= -1
        return True


# ----------------------------------------------------------------
# Parallel Worker
# ----------------------------------------------------------------
def run_single_replicate(args):
    mode, seed, N, SIGMA, M, R0, params, MAX_T, time_points = args

    if mode == 'constant':
        model = FisherConstantRadius(N, SIGMA, M, R0, seed=seed)
        epsilon = params['epsilon']
    elif mode == 'sswm':
        model = FisherSSWM(N, SIGMA, M, R0, seed=seed)

    dfe_0 = model.compute_dfe(model.r)

    correlations = np.full(len(time_points), np.nan)  # Initialize with NaN
    radii = np.full(len(time_points), np.nan)

    current_t_idx = 0
    correlations[0] = 1.0
    radii[0] = np.linalg.norm(model.r)

    time_points_set = set(time_points)

    for t in range(1, MAX_T + 1):
        if mode == 'constant':
            success = model.step(epsilon)
        else:
            success = model.step()

        if not success:
            break  # Stop recording if stuck/peak reached

        if t in time_points_set:
            current_t_idx += 1
            dfe_t = model.compute_dfe(model.r)
            corr, _ = pearsonr(dfe_0, dfe_t)
            correlations[current_t_idx] = corr
            radii[current_t_idx] = np.linalg.norm(model.r)

    return correlations, radii


# ----------------------------------------------------------------
# Main Experiment
# ----------------------------------------------------------------
def run_experiment():
    # --- Parameters ---
    N = 15
    SIGMA = 0.01
    M = 2 * 10 ** 4
    REPS = 100
    R0 = SIGMA * 100

    # --- Constant Radius ---
    EPSILON = 1.5 * SIGMA
    TAU_CONST = (2 * R0 ** 2) / ((N - 1) * SIGMA ** 2)

    # --- Velocity Calculation ---
    # Theoretical radial velocity based on max of M Gaussians
    V_THEORY = expected_max_gaussian(M, SIGMA)
    print(f"Theoretical Max Velocity (v): {V_THEORY:.5f} (approx {V_THEORY / SIGMA:.2f} sigma)")

    # Estimated walk length for SSWM
    # Walk ends when R(t) -> 0. T_end = R0 / V_THEORY
    T_END_SSWM = int(R0 / V_THEORY * 1.5)  # Add buffer

    # Durations
    MAX_T_CONST = int(2.5 * TAU_CONST)
    MAX_T_SSWM = max(20, T_END_SSWM)  # Ensure at least some steps

    print(f"--- Configuration ---")
    print(f"N={N}, Sigma={SIGMA}, R0={R0:.4f}")
    print(f"Estimated SSWM Walk Length: {int(R0 / V_THEORY)} steps")

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    base_seed = np.random.randint(0, 1e6)

    tp_const = np.arange(0, MAX_T_CONST + 1, max(1, MAX_T_CONST // 50))
    tp_sswm = np.arange(0, MAX_T_SSWM + 1, max(1, MAX_T_SSWM // 20))

    tasks_const = []
    tasks_sswm = []

    for i in range(REPS):
        tasks_const.append(('constant', base_seed + i, N, SIGMA, M, R0, {'epsilon': EPSILON}, MAX_T_CONST, tp_const))
        tasks_sswm.append(('sswm', base_seed + i + REPS, N, SIGMA, M, R0, {}, MAX_T_SSWM, tp_sswm))

    num_proc = min(multiprocessing.cpu_count(), REPS)
    with multiprocessing.Pool(processes=num_proc) as pool:
        res_const = pool.map(run_single_replicate, tasks_const)
        res_sswm = pool.map(run_single_replicate, tasks_sswm)

    corr_const = np.array([r[0] for r in res_const])
    corr_sswm = np.array([r[0] for r in res_sswm])

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # === Subplot 1: Constant Radius ===
    ax = axes[0]
    mean_c = np.nanmean(corr_const, axis=0)
    mask = (mean_c > 0.001) & (~np.isnan(mean_c))
    x = tp_const[mask]
    y = np.log(mean_c[mask])

    ax.plot(x, -x / TAU_CONST, 'r-', lw=2, label=f'Theory: $\\tau_0 = {TAU_CONST:.1f}$')
    sns.regplot(x=x, y=y, ax=ax, scatter=False, ci=95,
                line_kws={'color': 'black', 'linestyle': '--', 'label': 'Sim Fit'})
    y_err = np.nanstd(corr_const, axis=0)[mask] / mean_c[mask]
    ax.errorbar(x, y, yerr=y_err, fmt='o', color='purple', alpha=0.5, ms=4)

    ax.set_title(f'Constant Radius ($R \\approx R_0$)\nCheck of Eq A6')
    ax.set_xlabel('Time (steps)')
    ax.set_ylabel('ln(Pearson Correlation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === Subplot 2: Variable Radius (Standard SSWM) ===
    ax = axes[1]
    mean_c_sswm = np.nanmean(corr_sswm, axis=0)

    mask_s = (mean_c_sswm > 0.001) & (~np.isnan(mean_c_sswm))
    x_s = tp_sswm[mask_s]
    y_s = np.log(mean_c_sswm[mask_s])

    # --- Theory Eq A8 using V_THEORY ---
    # R(t) = R0 - v*t
    # Limit calculation to valid domain where R > 0
    valid_theory_mask = (R0 - V_THEORY * x_s) > 1e-9
    x_th = x_s[valid_theory_mask]

    numerator = (N - 1) * SIGMA ** 2 * x_th
    denominator = 2 * R0 * (R0 - V_THEORY * x_th)
    y_theory_closed = - numerator / denominator

    ax.plot(x_th, y_theory_closed, 'g--', lw=2.5,
            label=rf'Eq. A8 ($v \approx {V_THEORY / SIGMA:.1f}\sigma$)')

    # Sim Data
    ax.plot(x_s, y_s, 'ko--', alpha=0.4, label='Simulation')
    y_err_s = np.nanstd(corr_sswm, axis=0)[mask_s] / mean_c_sswm[mask_s]
    ax.fill_between(x_s, y_s - y_err_s, y_s + y_err_s, color='purple', alpha=0.2)

    ax.set_title(f'Variable Radius (Standard SSWM)\nCheck of Eq A8')
    ax.set_xlabel('Time (steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()