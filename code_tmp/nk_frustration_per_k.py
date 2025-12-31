import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class NKModel:
    def __init__(self, N, K, seed=None):
        self.N = N
        self.K = K
        self.rng = np.random.default_rng(seed)
        # Cache for fitness values to emulate infinite landscape
        self._fitness_cache = [{} for _ in range(N)]

    def get_fitness_contribution(self, locus_idx, local_config_val):
        """Lazy load fitness contribution from N(0,1)"""
        if local_config_val not in self._fitness_cache[locus_idx]:
            self._fitness_cache[locus_idx][local_config_val] = self.rng.normal(0, 1)
        return self._fitness_cache[locus_idx][local_config_val]

    def get_local_config_val(self, sigma, i):
        """Convert local neighbors (i, i+1... i+K) to int"""
        val = 0
        for offset in range(self.K + 1):
            neighbor_idx = (i + offset) % self.N
            bit = 1 if sigma[neighbor_idx] > 0 else 0
            val = (val << 1) | bit
        return val

    def compute_total_fitness(self, sigma):
        total_fit = 0.0
        for i in range(self.N):
            val = self.get_local_config_val(sigma, i)
            total_fit += self.get_fitness_contribution(i, val)
        return total_fit / self.N

    def get_local_contributions(self, sigma):
        """Extract individual f_i values for a specific genome"""
        contributions = np.zeros(self.N)
        for i in range(self.N):
            val = self.get_local_config_val(sigma, i)
            contributions[i] = self.get_fitness_contribution(i, val)
        return contributions


def evolve_to_peak(model):
    """Evolve using SSWM dynamics until a local peak is reached"""
    sigma = 2 * model.rng.integers(0, 2, model.N) - 1

    while True:
        # 1. Compute DFE (Distribution of Fitness Effects)
        current_fit = model.compute_total_fitness(sigma)
        deltas = np.zeros(model.N)

        # Check all N possible single-bit flips
        for i in range(model.N):
            sigma[i] *= -1
            new_fit = model.compute_total_fitness(sigma)
            deltas[i] = new_fit - current_fit
            sigma[i] *= -1  # Revert

        # 2. Identify Beneficial Mutations
        beneficial = np.where(deltas > 0)[0]

        if len(beneficial) == 0:
            break  # Local peak reached

        # 3. Select Mutation (Probability linear in fitness gain)
        gains = deltas[beneficial]
        probs = gains / np.sum(gains)

        choice = model.rng.choice(beneficial, p=probs)
        sigma[choice] *= -1

    return sigma


def run_analysis(N=400, K=3):
    print(f"Running NK Model Analysis (N={N}, K={K})...")
    model = NKModel(N, K, seed=42)

    # A. Run Dynamics
    peak_sigma = evolve_to_peak(model)

    # B. Extract Data (f_i values at peak)
    peak_values = model.get_local_contributions(peak_sigma)

    # C. Compute Theoretical Distribution (No Simulation)
    # PDF of Max(X1..XM) where X ~ N(0,1) and M = 2^(K+1)
    M = 2 ** (K + 1)
    x_grid = np.linspace(-3, 5, 1000)

    pdf_norm = norm.pdf(x_grid)
    cdf_norm = norm.cdf(x_grid)

    # The Exact Formula: M * phi(x) * Phi(x)^(M-1)
    pdf_max = M * pdf_norm * (cdf_norm ** (M - 1))

    # D. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(peak_values, bins=50, density=True, alpha=0.6, color='skyblue',
             edgecolor='black', label=f'Actual $f_i$ at Peak')
    plt.plot(x_grid, pdf_max, 'r-', linewidth=2.5,
             label=f'Theoretical Independent Max ($M=2^{{{K + 1}}}$)')

    plt.xlabel('Fitness Contribution ($f_i$)')
    plt.ylabel('Probability Density')
    plt.title(f'Frustration in NK Model: Peak vs. Theory (N={N}, K={K})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_analysis(N=400, K=8)