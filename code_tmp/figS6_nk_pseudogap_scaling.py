import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

# ----------------------------------------------------------------
# Global plot settings (match other project figures)
# ----------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12


def power_law(Delta, M, theta, c):
    """Power-law with offset: P(s) = Θ s^M + c."""
    return M * Delta ** theta + c


def _apply_axis_style(ax):
    """Apply consistent tick and spine styles."""
    ax.tick_params(axis='both', which='major', length=10, width=1.5, labelsize=14)
    ax.tick_params(axis='both', which='minor', length=5, width=1.6, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def _make_sci_formatter():
    """Return a ScalarFormatter that uses 10^x notation (×10^{...})."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    return formatter


def analyze_dfe_power_law(data_files, k_values, lower_thr, upper_thr, num_bins=50):
    """
    Fit and plot P(Delta) = M Delta^Θ + c in a 2×2 figure.

    Top-left:  small-Delta DFEs with fits for all K on the same axes.
    Top-right: Θ(K)
    Bottom-left: M(K)
    Bottom-right: c(K)
    """
    if not data_files:
        print("No gen_data files found.")
        return

    # Set up 2×2 figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for ax in axs.flatten():
        _apply_axis_style(ax)

    ax_main = axs[0, 0]
    ax_theta = axs[0, 1]
    ax_M = axs[1, 0]
    ax_c = axs[1, 1]

    # Panel labels A–D
    panel_labels = ['A', 'B', 'C', 'D']
    for lab, ax in zip(panel_labels, axs.flatten()):
        ax.text(-0.12, 1.08, lab, transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    num_k = len(k_values)
    colors = sns.color_palette('CMRmap', n_colors=num_k)

    # Storage for parameters vs K (only for successful fits)
    K_fit = []
    Theta_vals = []
    M_vals = []
    c_vals = []
    r2_vals = []

    # ----------------------------------------------------------------
    # Loop over K and fit the small-Delta pseudogap region
    # ----------------------------------------------------------------
    for idx, (path, K) in enumerate(zip(data_files, k_values)):
        if not os.path.isfile(path):
            print(f"File not found for K={K}: {path}")
            continue

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Extract final DFE values at the end of the walk (absolute values)
        final_vals = []
        for rep in data:
            final_vals.extend(np.abs(rep['dfes'][-1]))
        final_vals = np.asarray(final_vals, dtype=float)

        # Optional diagnostic: how empty is the very small-s region?
        vals_for_diagnostic = final_vals[final_vals <= 0.02]
        if vals_for_diagnostic.size > 0:
            counts_diag, edges_diag = np.histogram(vals_for_diagnostic, bins=300, density=True)
            left_most_bin_count = counts_diag[0]
            print(
                f'K={K}, left-most bin density (Delta in '
                f'[{edges_diag[0]:.2e}, {edges_diag[1]:.2e}]): {left_most_bin_count:.3e}'
            )

        # Restrict to the fitting window [lower_thr, upper_thr]
        s = final_vals[(final_vals >= lower_thr) & (final_vals <= upper_thr)]
        if s.size == 0:
            print(f"No values in range [{lower_thr}, {upper_thr}] for K={K}")
            continue

        # Histogram to estimate density in the window
        counts, edges = np.histogram(s, bins=num_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = counts > 0
        xdata = centers[mask]
        ydata = counts[mask]

        if xdata.size < 3:
            print(f"Not enough nonzero bins to fit for K={K}")
            continue

        # Fit Θ s^M + c with bounds (same logic as before, just renamed parameters)
        lower_bounds = [1e3, 0, 0]      # M, Θ, c >= ...
        upper_bounds = [1e11, 3, 1e4]   # M, Θ, c <= ...
        popt, _ = curve_fit(
            power_law,
            xdata,
            ydata,
            p0=[1e4, 1, 1e3],
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
        M, Theta, c_param = popt

        # Goodness of fit (R^2)
        y_pred = power_law(xdata, *popt)
        ss_res = np.sum((ydata - y_pred) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)

        # Store parameters
        K_fit.append(K)
        M_vals.append(M)
        Theta_vals.append(Theta)
        c_vals.append(c_param)
        r2_vals.append(r2)

        # Plot this K on the main panel
        ax_main.plot(
            xdata,
            y_pred,
            color=colors[idx],
            linewidth=2.0,
            label=rf'$K={K}, R^2={r2:.2f}$'
        )
        ax_main.scatter(xdata, ydata, color=colors[idx], s=30)

    # If no fits succeeded, just bail out
    if not K_fit:
        print("No successful fits; nothing to plot.")
        plt.close(fig)
        return

    # Sort by K for the parameter panels
    K_fit = np.asarray(K_fit)
    order = np.argsort(K_fit)
    K_sorted = K_fit[order]
    M_sorted = np.asarray(M_vals)[order]
    Theta_sorted = np.asarray(Theta_vals)[order]
    c_sorted = np.asarray(c_vals)[order]

    # ----------------------------------------------------------------
    # Main panel styling: scientific notation on both axes
    # ----------------------------------------------------------------
    sci_formatter_x = _make_sci_formatter()
    sci_formatter_y = _make_sci_formatter()
    ax_main.xaxis.set_major_formatter(sci_formatter_x)
    ax_main.yaxis.set_major_formatter(sci_formatter_y)

    ax_main.set_xlabel(r'$|\Delta|$')
    ax_main.set_ylabel(r'$P(\Delta)$')
    ax_main.legend(frameon=False)

    # ----------------------------------------------------------------
    # Parameter panels: M(K), Θ(K), c(K)
    # ----------------------------------------------------------------
    for axis in (ax_theta, ax_M, ax_c):
        axis.set_xticks(K_sorted)
        axis.set_xlabel(r'$K$')

    ax_M.plot(K_sorted, M_sorted, marker='o', linewidth=2.0)
    ax_M.set_ylabel(r'$M(K)$')
    ax_theta.yaxis.set_major_formatter(_make_sci_formatter())

    ax_theta.plot(K_sorted, Theta_sorted, marker='o', linewidth=2.0)
    ax_theta.set_ylabel(r'$\Theta(K)$')

    ax_c.plot(K_sorted, c_sorted, marker='o', linewidth=2.0)
    ax_c.set_ylabel(r'$c(K)$')
    ax_c.yaxis.set_major_formatter(_make_sci_formatter())

    plt.tight_layout()
    # Save SVG to the same paper folder convention
    out_dir = os.path.join('..', 'figs_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figS6_nk_pseudogap_scaling.svg')
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    # Configuration for the small-s fitting window
    UPPER_THRESHOLD = 1.2 * 1e-3
    LOWER_THRESHOLD = 1e-10
    NUM_BINS = 15
    K_VALUES = [4, 8, 16, 32]

    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'NK')
    DATA_FILES = [os.path.join(base, f'N_2000_K_{k}_repeats_100.pkl') for k in K_VALUES]

    analyze_dfe_power_law(
        DATA_FILES,
        K_VALUES,
        lower_thr=LOWER_THRESHOLD,
        upper_thr=UPPER_THRESHOLD,
        num_bins=NUM_BINS
    )
