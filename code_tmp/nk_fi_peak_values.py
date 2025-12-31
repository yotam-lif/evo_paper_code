import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, linregress


class NKModel:
    def __init__(self, N, K, seed=None):
        self.N = N
        self.K = K
        # Use a high-quality RNG
        self.rng = np.random.default_rng(seed)

        # Lazy-loading cache for fitness values.
        # Structure: list of N dicts. _fitness_cache[i][config_int] = fitness_value
        self._fitness_cache = [{} for _ in range(N)]

    def get_fitness_contribution(self, locus_idx, local_config_val):
        """
        Retrieve f_i for a specific local configuration value (int).
        If not present in cache, sample it from N(0,1) and store it.
        """
        if local_config_val not in self._fitness_cache[locus_idx]:
            self._fitness_cache[locus_idx][local_config_val] = self.rng.normal(0, 1)
        return self._fitness_cache[locus_idx][local_config_val]

    def get_local_config_val(self, sigma, i):
        """
        Get the integer representation of the neighbors of i.
        Neighbors: i, i+1, ..., i+K (cyclic).
        Returns a Python integer.
        """
        val = 0
        # Iterate through the K+1 dependencies
        for offset in range(self.K + 1):
            neighbor_idx = (i + offset) % self.N
            # Map spin -1 -> 0, +1 -> 1
            bit = 1 if sigma[neighbor_idx] > 0 else 0
            val = (val << 1) | bit
        return val

    def compute_total_fitness(self, sigma):
        """Computes the mean fitness of the genome."""
        total_fit = 0.0
        for i in range(self.N):
            val = self.get_local_config_val(sigma, i)
            total_fit += self.get_fitness_contribution(i, val)
        return total_fit / self.N

    def get_local_contributions(self, sigma):
        """Returns the array of individual f_i values for the genome."""
        contributions = np.zeros(self.N)
        for i in range(self.N):
            val = self.get_local_config_val(sigma, i)
            contributions[i] = self.get_fitness_contribution(i, val)
        return contributions


def evolve_to_peak(model):
    """
    Evolve a random genome using SSWM (Strong Selection Weak Mutation) dynamics
    until a local peak is reached.
    """
    # Initialize random genome
    sigma = 2 * model.rng.integers(0, 2, model.N) - 1
    steps = 0

    while True:
        current_fit = model.compute_total_fitness(sigma)

        # Identify beneficial mutations (check all N neighbors)
        deltas = []
        beneficial_indices = []

        for i in range(model.N):
            # Flip
            sigma[i] *= -1
            new_fit = model.compute_total_fitness(sigma)
            delta = new_fit - current_fit

            if delta > 0:
                deltas.append(delta)
                beneficial_indices.append(i)

            # Flip back
            sigma[i] *= -1

        # Check termination condition (Local Peak)
        if not beneficial_indices:
            break

        # SSWM Selection: Probability proportional to fitness gain
        deltas = np.array(deltas)
        probs = deltas / np.sum(deltas)

        # Choose and mutate
        choice_idx = model.rng.choice(len(beneficial_indices), p=probs)
        site_to_flip = beneficial_indices[choice_idx]

        sigma[site_to_flip] *= -1
        steps += 1

    return sigma, steps


def run_simulation_and_plot():
    # Parameters
    N = 600
    K_values = [4, 8, 12, 16, 20, 24]

    # Store means for the scaling plot
    peak_means = []

    # --- Figure 1: Distributions ---
    plt.figure(figsize=(12, 8))

    # Use a colormap to distinguish K values clearly
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(K_values)))

    print(f"Starting simulations for N={N}...")

    for idx, K in enumerate(K_values):
        print(f"  Processing K={K}...", end="", flush=True)

        # Initialize and Run
        model = NKModel(N, K, seed=42 + K)  # Different seed per K for variety
        peak_sigma, steps = evolve_to_peak(model)

        # Extract data: f_i values at the peak
        data = model.get_local_contributions(peak_sigma)

        # 1. Statistics & Fit
        mean_val = np.mean(data)
        variance = np.var(data)
        mu, std = norm.fit(data)  # fit returns (mean, std)

        peak_means.append(mean_val)

        # 2. Goodness of Fit Test (Kolmogorov-Smirnov)
        ks_stat, p_value = kstest(data, 'norm', args=(mu, std))

        print(f" Peak reached in {steps} steps. Mean: {mean_val:.3f}, Var: {variance:.3f}, KS p-val: {p_value:.2e}")

        # --- Plotting Histogram ---
        color = colors[idx]

        # Label with Mean, Variance, and KS p-value
        label_text = (f"K={K} | $\mu$={mean_val:.2f} | $\sigma^2$={variance:.2f} | $p_{{KS}}$={p_value:.1e}")

        # Histogram (Step style)
        counts, bins = np.histogram(data, bins=30, density=True)
        plt.hist(data, bins=30, density=True, histtype='step',
                 color=color, linewidth=2, label=label_text, alpha=0.9)

        # Gaussian Fit Curve
        xmin, xmax = bins[0], bins[-1]
        x_plot = np.linspace(xmin - 0.5, xmax + 0.5, 100)
        p_plot = norm.pdf(x_plot, mu, std)

        plt.plot(x_plot, p_plot, '--', color=color, linewidth=1.5, alpha=0.6)

    plt.title(f"Distribution of Local Fitness Values ($f_i$) at Peak\n(N={N}, SSWM Dynamics)")
    plt.xlabel("Fitness Contribution Value")
    plt.ylabel("Probability Density")
    plt.legend(title="Interaction (K) | Mean | Var | Goodness of Fit", loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Scaling Analysis (Log-Log) ---
    plt.figure(figsize=(8, 6))

    # We plot Mean vs (K+1) based on the theoretical discussion
    x_vals = np.array(K_values) + 1
    y_vals = np.array(peak_means)

    # Log-Log transformation for linear fit
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)

    # Perform Linear Regression: log(y) = slope * log(x) + intercept
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    # Plot Data Points
    plt.scatter(x_vals, y_vals, s=100, color='blue', label='Simulation Means', zorder=5)

    # Plot Fit Line
    # y = exp(intercept) * x^slope
    fit_y = np.exp(intercept) * (x_vals ** slope)
    plt.plot(x_vals, fit_y, 'r--', linewidth=2, label=f'Fit Slope = {slope:.3f}')

    # Add theoretical reference line (Slope -0.5) for visual comparison
    # We anchor it to the first data point for alignment
    c_ref = y_vals[0] / (x_vals[0] ** (-0.5))
    ref_y = c_ref * (x_vals ** (-0.5))
    plt.plot(x_vals, ref_y, 'k:', alpha=0.6, label='Theoretical Slope -0.5')

    plt.xscale('log')
    plt.yscale('log')

    # Formatting
    plt.xlabel("Interaction Term $(K+1)$")
    plt.ylabel("Mean Fitness Contribution $\langle f_i \\rangle$")
    plt.title("Scaling of Mean Fitness at Peak vs. Complexity")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Print slope analysis
    print("\n--- Scaling Analysis ---")
    print(f"Fit equation: log(Mean) = {slope:.4f} * log(K+1) + {intercept:.4f}")
    print(f"Measured Slope: {slope:.4f}")
    print(f"Theoretical Slope (if 1/sqrt(K+1)): -0.5")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation_and_plot()