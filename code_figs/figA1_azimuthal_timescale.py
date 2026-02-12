import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import linregress, pearsonr, norm
from scipy import integrate
from scipy.integrate import odeint, cumulative_trapezoid
import multiprocessing
import os
import pandas as pd

# ----------------------------------------------------------------
# 1. VISUAL STYLE CONFIGURATION
# ----------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
CMR_COLORS = sns.color_palette('CMRmap', 5)


def apply_axis_style(ax, label):
    """Applies specific styling: No grid, bold label, thicker spines."""
    ax.text(
        -0.1, 1.05, label,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="bottom", ha="left",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    ax.grid(False)


# ----------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ----------------------------------------------------------------
def expected_max_gaussian(m, sigma):
    """Calculates E[max(X1, ..., Xm)] where Xi ~ N(0, sigma^2)."""

    def integrand(x):
        z = x / sigma
        pdf_z = norm.pdf(z)
        cdf_z = norm.cdf(z)
        return x * m * pdf_z * (cdf_z ** (m - 1)) / sigma

    peak_est = sigma * np.sqrt(2 * np.log(m))
    val, err = integrate.quad(integrand, -sigma, peak_est + 5 * sigma)
    return val


# ----------------------------------------------------------------
# 3. MODEL CLASSES
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
        if not np.any(beneficial_mask): return False
        ben_indices = np.nonzero(beneficial_mask)[0]
        ben_effects = dfe[ben_indices]
        probs = ben_effects / np.sum(ben_effects)
        choice = self.rng.choice(ben_indices, p=probs)
        self.r += self.deltas[choice]
        self.deltas[choice] *= -1
        return True


# ----------------------------------------------------------------
# 4. SIMULATION WORKER
# ----------------------------------------------------------------
def run_single_replicate(args):
    mode, seed, N, SIGMA, M, R0, params, MAX_T, time_points = args

    # Extract Termination Condition (R_final) if present
    R_FINAL = params.get('R_final', 0.0)

    if mode == 'constant':
        model = FisherConstantRadius(N, SIGMA, M, R0, seed=seed)
        epsilon = params['epsilon']
    elif mode == 'sswm':
        model = FisherSSWM(N, SIGMA, M, R0, seed=seed)

    dfe_0 = model.compute_dfe(model.r)

    correlations = np.full(len(time_points), np.nan)
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

        # Stop if stuck
        if not success: break

        # Stop if reached Target Radius (for SSWM)
        current_R = np.linalg.norm(model.r)
        if mode == 'sswm' and current_R < R_FINAL:
            break

        if t in time_points_set:
            current_t_idx += 1
            dfe_t = model.compute_dfe(model.r)
            corr, _ = pearsonr(dfe_0, dfe_t)
            correlations[current_t_idx] = corr
            radii[current_t_idx] = current_R

    return correlations, radii


# ----------------------------------------------------------------
# 5. MAIN EXPERIMENT
# ----------------------------------------------------------------
def run_experiment():
    # --- Shared Params ---
    N = 16
    SIGMA = 0.025
    M = 3 * 10 ** 4
    REPS = 500

    # --- Configuration for Subplots ---

    # PANEL A: Constant Radius
    R0_A = 100 * SIGMA

    # PANEL B: Variable Radius (Large -> Medium)
    R0_BD = 50 * SIGMA
    RF_BD = 30 * SIGMA

    # PANEL C: Variable Radius (Large -> Peak)
    R0_C = 130 * SIGMA
    RF_C = 25 * SIGMA

    # --- Theory Calculations ---
    # 1. Weighted SSWM Velocity (v_eff)
    V_SSWM = SIGMA * np.sqrt(np.pi / 2)

    # Panel A Tau (Eq A16)
    TAU_A = (2 * R0_A ** 2) / ((N - 1) * SIGMA ** 2)

    # Panel B Tau (Constant Tau approx using R0_BD)
    TAU_B = (2 * R0_BD ** 2) / ((N - 1) * SIGMA ** 2)

    # --- Time Setup ---

    # Panel A: Run for 2 * Tau
    MAX_T_A = int(2 * TAU_A)
    tp_A = np.arange(0, MAX_T_A + 1, max(1, MAX_T_A // 50))

    # Panels B: Estimate time
    EST_STEPS_BD = int((R0_BD - RF_BD) / V_SSWM)
    MAX_T_BD_SAFE = int(EST_STEPS_BD * 1.5)
    tp_BD = np.arange(0, MAX_T_BD_SAFE + 1, max(1, MAX_T_BD_SAFE // 100))

    # Panel C: Run deeper
    EST_STEPS_C = int((R0_C - RF_C) / V_SSWM)
    MAX_T_C_SAFE = int(EST_STEPS_C * 1.5)
    tp_C = np.arange(0, MAX_T_C_SAFE + 1, max(1, MAX_T_C_SAFE // 100))

    print(f"--- Configuration ---")
    print(f"N={N}, Sigma={SIGMA}, M={M}")
    print(f"V_SSWM (Theoretical) = {V_SSWM:.4f}")
    print(f"Panel A: R0={R0_A:.2f}")
    print(f"Panel B: R0={R0_BD:.2f} -> Rf={RF_BD:.2f}")
    print(f"Panel C: R0={R0_C:.2f} -> Rf={RF_C:.2f}")

    # --- Parallel Simulation ---
    base_seed = np.random.randint(0, 1e6)
    tasks = []

    # Task Set A
    for i in range(REPS):
        tasks.append(('constant', base_seed + i, N, SIGMA, M, R0_A,
                      {'epsilon': SIGMA, 'R_final': 0}, MAX_T_A, tp_A))

    # Task Set B
    for i in range(REPS):
        tasks.append(('sswm', base_seed + i + REPS, N, SIGMA, M, R0_BD,
                      {'R_final': RF_BD}, MAX_T_BD_SAFE, tp_BD))

    # Task Set C
    for i in range(REPS):
        tasks.append(('sswm', base_seed + i + 2 * REPS, N, SIGMA, M, R0_C,
                      {'R_final': RF_C}, MAX_T_C_SAFE, tp_C))

    num_proc = min(multiprocessing.cpu_count(), len(tasks))
    with multiprocessing.Pool(processes=num_proc) as pool:
        results = pool.map(run_single_replicate, tasks)

    # Unpack
    res_A = results[:REPS]
    res_B = results[REPS:2 * REPS]
    res_C = results[2 * REPS:]

    # --- Data Processing ---
    def results_to_df(res_list, t_points):
        data = []
        radii_list = []
        for rep_idx, (corr, rad) in enumerate(res_list):
            radii_list.append(rad)
            for t, c in zip(t_points, corr):
                if c > 0.001 and not np.isnan(c):
                    data.append({'Time': t, 'Log Correlation': np.log(c), 'Rep': rep_idx})
        return pd.DataFrame(data), radii_list

    df_A, _ = results_to_df(res_A, tp_A)
    df_B, _ = results_to_df(res_B, tp_BD)
    df_C, radii_C = results_to_df(res_C, tp_C)

    # --- Plotting 1x3 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.25)

    # ==========================
    # Panel A: Constant Radius
    # ==========================
    ax = axes[0]
    apply_axis_style(ax, "A")
    x_A = np.linspace(df_A['Time'].min(), df_A['Time'].max(), 100)

    # Sim
    sns.lineplot(data=df_A, x='Time', y='Log Correlation', errorbar='sd', ax=ax,
                 color=CMR_COLORS[1], lw=2, label='Simulation')

    # Theory
    ax.plot(x_A, -x_A / TAU_A, color='red', lw=2, ls='--',
            label=rf'Constant R approx. (eq. A16)')

    # Fit (Forced through origin: y = ax)
    x_dat = df_A['Time'].values
    y_dat = df_A['Log Correlation'].values
    # Least squares solution for y = ax is sum(xy) / sum(xx)
    slope_zero_intercept = np.sum(x_dat * y_dat) / np.sum(x_dat ** 2)

    ax.plot(x_A, slope_zero_intercept * x_A, color=CMR_COLORS[0], lw=2, ls='--',
            label=rf'Linear Fit')

    ax.set_ylabel("Log Correlation")
    ax.set_xlabel("Time (steps)")
    ax.legend(frameon=False, loc='lower left')

    # ==========================
    # Panel B: Variable Radius vs Constant Tau Approx
    # ==========================
    ax = axes[1]
    apply_axis_style(ax, "B")
    x_B = np.linspace(0, df_B['Time'].max(), 100)

    # ADDED: y='Log Correlation' (this was missing)
    sns.lineplot(data=df_B, x='Time', y='Log Correlation', errorbar='sd', ax=ax,
                 color=CMR_COLORS[1], lw=2, label='Simulation')

    ax.plot(x_B, -x_B / TAU_B, color='red', lw=2.5, ls='--',
            label=rf'Constant R approx. (eq. A16)')

    ax.set_ylabel("")  # Explicitly remove Y label
    ax.set_xlabel("Time (steps)")
    ax.legend(frameon=False, loc='lower left')

    # ==========================
    # Panel C: Dynamic R Theory (Simulated vs Analytical ODE)
    # ==========================
    ax = axes[2]
    apply_axis_style(ax, "C")

    # 1. Theory using Simulation Radii (Integrate Eq A17 with real R)
    # ----------------------------------------------------------------
    # Calculate avg R(t) from simulation
    max_len = max(len(r) for r in radii_C)
    r_mat = np.full((REPS, max_len), np.nan)
    for i, r in enumerate(radii_C): r_mat[i, :len(r)] = r
    mean_r_C = np.nanmean(r_mat, axis=0)

    # Align time points
    common_len = min(len(tp_C), len(mean_r_C))
    # Cut off if radius gets too small (unstable numerics/singularity)
    valid_mask = mean_r_C[:common_len] > 0.5 * SIGMA
    tp_C_cut = tp_C[:common_len][valid_mask]
    r_C_cut = mean_r_C[:common_len][valid_mask]

    rate_sim_R = - ((N - 1) * SIGMA ** 2) / (2 * r_C_cut ** 2)
    y_theory_sim_R = cumulative_trapezoid(rate_sim_R, tp_C_cut, initial=0)

    # 2. Theory using Analytical ODE for R(t)
    # ----------------------------------------------------------------
    # ODE: dR/dt = (n-1)sigma^2 / 2R - v_eff
    def radial_dynamics(r, t, v_eff, n, sigma):
        if r <= 1e-6: return 0
        repulsion = ((n - 1) * sigma ** 2) / (2 * r)
        d_rdt = repulsion - v_eff
        return d_rdt

    # Integrate ODE
    t_ode = np.linspace(0, tp_C_cut[-1], 200)
    R_ode = odeint(
        radial_dynamics,
        y0=R0_C,
        t=t_ode,
        args=(V_SSWM, N, SIGMA)
    ).flatten()

    # Avoid div by zero
    R_ode[R_ode < 1e-6] = 1e-6

    # Calculate Log Corr
    rate_ode = - ((N - 1) * SIGMA ** 2) / (2 * R_ode ** 2)
    y_theory_ode = cumulative_trapezoid(rate_ode, t_ode, initial=0)

    # 3. Plotting
    # ----------------------------------------------------------------
    # Simulation Data - ADDED: y='Log Correlation' (this was missing)
    sns.lineplot(data=df_C, x='Time', y='Log Correlation', errorbar='sd', ax=ax,
                 color=CMR_COLORS[1], lw=2, label='Simulation')

    # Theory (Sim R) - "Perfect" Theory
    ax.plot(tp_C_cut, y_theory_sim_R, color=CMR_COLORS[0], lw=2.5, ls='--',
            label=r'eq. A17 (using Simulated $R(t)$)')

    # Theory (ODE R) - "Predictive" Theory
    ax.plot(t_ode, y_theory_ode, color=CMR_COLORS[2], lw=2.5, ls='--',
            label=r'eq. A17 (using eq. A19 for $R(t)$)')

    ax.set_ylabel("")  # Explicitly remove Y label
    ax.set_xlabel("Time (steps)")
    ax.legend(frameon=False, loc='lower left')

    # Save
    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figA1_azimuthal_timescale.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    run_experiment()