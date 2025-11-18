import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
import pickle
import seaborn as sns

# Global bin count for all BDFE histograms
BDFE_BINS = 30

# Load FGM replicate data for all n values used in the original figure.
# We keep everything external: no new simulations are run here.
FGM_NS = [4, 8, 16, 32]
FGM_DATA = {}
for _n in FGM_NS:
    file_path_fgm = f'../data/FGM/fgm_rps1000_n{_n}_sig0.05_m2000.pkl'
    with open(file_path_fgm, 'rb') as f:
        FGM_DATA[_n] = pickle.load(f)

# Backwards-compatible alias: the n=4 replicate list,
# used as the "reps" input to FGM panel functions.
data_fgm = FGM_DATA[4]

# Ensure we run relative to this file so data paths used in the original modules still work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the original result modules (these keep the original rcParams/style)
import fgm_results as fgm_mod
import sk_results as sk_mod
import nk_results as nk_mod
from cmn.cmn_fgm import Fisher

percents_array_fgm = [80, 85, 90, 95]
percents_array_sk = [60, 65, 70, 75]
percents_array_nk = [60, 65, 70, 75]
# === EMD helper: empirical vs Exp(1) ===
def _emd_to_exp1(samples, grid_size=None):
    """
    1-Wasserstein (EMD) between empirical samples and Exp(1), using quantiles.
    'samples' are assumed to be already normalized by their mean (mean = 1).
    """
    x = np.asarray(samples)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n == 0:
        return np.nan
    x.sort()
    if grid_size is None or grid_size == n:
        u = (np.arange(1, n+1) - 0.5) / n
        q_exp1 = -np.log1p(-u)
        return np.mean(np.abs(x - q_exp1))
    else:
        u = (np.arange(1, grid_size+1) - 0.5) / grid_size
        try:
            q_emp = np.quantile(x, u, method='inverted_cdf')
        except TypeError:
            q_emp = np.quantile(x, u, interpolation='nearest')
        q_exp1 = -np.log1p(-u)
        return np.mean(np.abs(q_emp - q_exp1))
  # used to generate FGM inputs exactly like the original


# === New helpers: Exponential-fit BDFE panels (replacing the previous log-histograms) ===
import math

def _ks_exp_pvalue(samples, lam):
    '''One-sample KS test against Exp(lam). Returns (D, p_approx).'''
    if len(samples) == 0 or not np.isfinite(lam) or lam <= 0:
        return np.nan, np.nan
    x = np.sort(samples)
    n = len(x)
    # Empirical CDF at each observed x
    ecdf = (np.arange(1, n+1))/n
    # Theoretical CDF for exponential
    tcdf = 1.0 - np.exp(-lam * x)
    d_plus = np.max(ecdf - tcdf)
    d_minus = np.max(tcdf - (np.arange(0, n)/n))
    D = max(d_plus, d_minus)
    # Asymptotic p-value approximation
    en = math.sqrt(n)
    if not np.isfinite(D) or D <= 0:
        return D, 1.0
    # Kolmogorov distribution complementary CDF approximation
    s = 0.0
    for j in range(1, 101):
        s += (-1)**(j-1) * math.exp(-2 * (j**2) * (en*D)**2)
    p = max(0.0, min(1.0, 2*s))
    return D, p



# === Airy unit-mean reference distribution and generic KS helper ===

# Precomputed scale 'a' such that the mean of the normalized Airy density
# p(x) ∝ Ai(a x) on x>0 is 1.
AIRY_A = 0.7764582113784204

# Precompute Airy PDF/CDF on a grid for x >= 0
_AIRY_X_MAX = 20.0
_AIRY_GRID_SIZE = 4000
_AIRY_X = np.linspace(0.0, _AIRY_X_MAX, _AIRY_GRID_SIZE)
_AIRY_PDF = airy(AIRY_A * _AIRY_X)[0]
_AIRY_PDF[_AIRY_PDF < 0] = 0.0
dx = _AIRY_X[1] - _AIRY_X[0]
# Normalize to integrate to 1
_AIRY_PDF /= np.trapz(_AIRY_PDF, _AIRY_X)
_AIRY_CDF = np.cumsum(_AIRY_PDF) * dx
_AIRY_CDF /= _AIRY_CDF[-1]

def airy_unit_mean_cdf(x):
    """CDF of the fixed unit-mean Airy reference distribution on x>=0."""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    if np.any(mask):
        out[mask] = np.interp(x[mask], _AIRY_X, _AIRY_CDF, left=0.0, right=1.0)
    return out

def _ks_pvalue_against_cdf(samples, cdf_fn):
    """One-sample KS statistic and asymptotic p-value vs an arbitrary CDF.

    samples are 1D real numbers; cdf_fn(x) should return the model CDF at x (vectorized).
    """
    x = np.asarray(samples)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan
    x.sort()
    ecdf = (np.arange(1, n+1)) / n
    tcdf = cdf_fn(x)
    D = np.max(np.abs(ecdf - tcdf))
    en = math.sqrt(n)
    if not np.isfinite(D) or D <= 0:
        return D, 1.0
    s = 0.0
    for j in range(1, 101):
        s += (-1)**(j-1) * math.exp(-2 * (j**2) * (en*D)**2)
    p = max(0.0, min(1.0, 2*s))
    return D, p



def _plot_empirical_vs_exp(ax, samples, color, t_percent, bins=BDFE_BINS):
    """Mean-normalize positive BDFE samples and compare to Exp(1) and unit-mean Airy.

    - Takes raw samples (can contain negatives / NaNs).
    - Keeps only positive, finite values.
    - Divides by the sample mean so the resulting sample has mean 1.
    - Plots the empirical PDF as a dashed line.
    - Overlays the Exp(1) and unit-mean Airy reference PDFs.
    - Returns (p_exp, p_airy) = KS p-values vs Exp(1) and Airy.
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size < 3:
        return np.nan, np.nan

    mu = samples.mean()
    if not np.isfinite(mu) or mu <= 0:
        return np.nan, np.nan

    samples_norm = samples / mu  # mean-normalization

    counts, edges = np.histogram(samples_norm, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if centers.size == 0:
        return np.nan, np.nan

    # KS vs Exp(1) (unit-mean exponential) and Airy
    _, p_exp = _ks_pvalue_against_cdf(samples_norm, lambda x: 1.0 - np.exp(-np.asarray(x)))
    _, p_airy = _ks_pvalue_against_cdf(samples_norm, airy_unit_mean_cdf)

    # Legend label for this time point
    if t_percent is None:
        label = rf'$p_{{\exp}}={p_exp:.2g},\,p_{{\mathrm{{Ai}}}}={p_airy:.2g}$'
    else:
        label = (
            rf'$t={t_percent:.0f}\%,\,p_{{\exp}}={p_exp:.2g},'
            rf'\,p_{{\mathrm{{Ai}}}}={p_airy:.2g}$'
        )

    # Empirical PDF as dashed line (will carry the label)
    line, = ax.plot(centers, counts, ls='--', lw=2, color=color, label=label)

    # Theoretical PDFs on the same x-range
    # x_max = centers[-1]
    # x = np.linspace(0.0, x_max * 1.05, 400)
    # y_exp = np.exp(-x)
    # y_airy = np.interp(x, _AIRY_X, _AIRY_PDF, left=_AIRY_PDF[0], right=0.0)
    #
    # ax.plot(x, y_exp, lw=2, color=color, alpha=0.8)
    # ax.plot(x, y_airy, lw=2, color=color, ls=':', alpha=0.8)

    return p_exp, p_airy



def plot_bdfe_exp_fgm(ax, reps, percents=percents_array_fgm):
    """Recompute BDFE from FGM reps and compare to Exp/Airy via KS."""
    colors = getattr(fgm_mod, 'COLOR', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for i, p in enumerate(percents):
        bdfe = []
        for rep in reps:
            T_rep = len(rep['dfes'])
            if T_rep == 0:
                continue
            t_idx = int(p / 100 * (T_rep - 1))
            t_idx = max(0, min(t_idx, T_rep - 1))
            dfe_t = np.asarray(rep['dfes'][t_idx])
            bdfe.extend(dfe_t[dfe_t > 0])
        _plot_empirical_vs_exp(ax, bdfe, colors[i % len(colors)], t_percent=p, bins=BDFE_BINS)
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')



def plot_bdfe_exp_sk(ax, points_lst, num_flips):
    """Use SK cached data to recompute BDFE and compare to Exp/Airy via KS."""
    colors = sns.color_palette('CMRmap', 5)
    num = len(points_lst)
    for i, point in enumerate(points_lst):
        bdfe = []
        for repeat_idx in range(len(sk_mod.data)):
            repeat_data = sk_mod.data[repeat_idx]
            alphas = sk_mod.cmn.curate_sigma_list(
                repeat_data['init_alpha'],
                repeat_data['flip_seq'],
                [point],
            )
            h = repeat_data['h']
            J = repeat_data['J']
            bdfe_i = sk_mod.cmn_sk.compute_bdfe(alphas[0], h, J)[0]
            bdfe.extend(bdfe_i)
        t_percent = (point / (num_flips - 1)) * 100.0
        _plot_empirical_vs_exp(ax, bdfe, colors[i % len(colors)], t_percent=t_percent, bins=BDFE_BINS)
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')



def plot_bdfe_exp_nk(ax, percents):
    """Use NK cached nk_data to recompute BDFE and compare to Exp/Airy via KS."""
    colors = getattr(nk_mod, 'color', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for i, p in enumerate(percents):
        bdfe = []
        for entry in nk_mod.nk_data:
            num_flips = len(entry['flip_seq'])
            if num_flips == 0:
                continue
            t_idx = int(p * num_flips / 100.0)
            t_idx = max(0, min(t_idx, num_flips - 1))
            dfe_t = np.asarray(entry['dfes'][t_idx])
            bdfe_i, _ = nk_mod.cmn_nk.compute_bdfe(dfe_t)
            bdfe.extend(bdfe_i)
        _plot_empirical_vs_exp(ax, bdfe, colors[i % len(colors)], t_percent=p, bins=BDFE_BINS)
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')


def build_fgm_inputs():
    """
    Build the inputs ``reps`` and ``final`` for the FGM row
    using only the externally-loaded FGM_DATA pickle files.

    - ``reps`` is taken from the n=4 dataset (as in the original script).
    - ``final`` aggregates the last-timepoint DFE for each n in FGM_NS.
    """
    # Reps: use n=4 replicate series for the time-evolution panels.
    reps = FGM_DATA[4]

    # Final DFEs for each n: collect the last-timepoint DFE across replicates.
    final = {}
    for n, rep_list in FGM_DATA.items():
        all_last_dfes = []
        for rep in rep_list:
            if not isinstance(rep, dict):
                continue
            dfes = rep.get("dfes", None)
            if dfes is None or len(dfes) == 0:
                continue
            all_last_dfes.extend(dfes[-1])
        final[n] = all_last_dfes

    return reps, final
def build_sk_inputs():
    """
    Recreate the arguments used by sk_results.py for panels A–C.
    We use the same choices as in the original main():
      - Panel A: num_points=5, num_repeats=10
      - Panel B: N=1500, beta_arr=[0.0, 0.5, 1.0], rho=1.0, num_repeats=10
      - Panel C: percentages 70,75,80,85% of the flip sequence, num_bins = BDFE_BINS
    """
    num_repeats = 10
    # Determine number of flips from one repeat (original used repeat index 10)
    crossings_repeat = 10
    num_flips = len(sk_mod.data[crossings_repeat]['flip_seq'])
    percentages_C = np.array(percents_array_sk)
    flip_list = (percentages_C / 100 * (num_flips - 1)).astype(int)
    return {
        "num_repeats": num_repeats,
        "N": 1500,
        "beta_arr": [0.0, 0.5, 1.0],
        "rho": 1.0,
        "flip_list": flip_list,
        "num_bins": 16,
        "num_flips": num_flips,
    }

def build_nk_inputs():
    """
    nk_results.py exposes panel functions that carry their own state from import,
    so there are no extra inputs beyond the percent choices used in the original.
    """
    return {"percents": percents_array_nk}

def plot_column1(ax, dataset, reps, sk_args):
    """Column 1: DFE evolution panel for a given dataset."""
    if dataset == "FGM":
        # FGM: reuse panel_A from fgm_results.py
        fgm_mod.panel_A(ax, reps)
    elif dataset == "SK":
        # SK: evolution DFE from sk_results.py
        sk_mod.create_fig_dfe_evol(
            ax,
            num_points=5,
            num_repeats=sk_args["num_repeats"],
        )
    elif dataset == "NK":
        # NK: evolution DFE from nk_results.py
        nk_mod.create_fig_evolution_dfe(ax)
    else:
        raise ValueError(f"Unknown dataset label for column 1: {dataset!r}")


def plot_column2(ax, dataset, final_fgm, sk_args):
    """Column 2: final DFE panel for a given dataset."""
    if dataset == "FGM":
        # FGM: final DFE vs n
        fgm_mod.panel_B(ax, final_fgm)
    elif dataset == "SK":
        # SK: final DFE for different betas
        sk_mod.create_fig_dfe_fin(
            ax,
            N=sk_args["N"],
            beta_arr=sk_args["beta_arr"],
            rho=sk_args["rho"],
            num_repeats=sk_args["num_repeats"] * 5,
        )
    elif dataset == "NK":
        # NK: final DFE from nk_results.py
        nk_mod.create_fig_final_dfe(ax)
    else:
        raise ValueError(f"Unknown dataset label for column 2: {dataset!r}")


def plot_column3(ax, dataset, reps, sk_args, nk_args):
    """Column 3: BDFE panel for a given dataset."""
    if dataset == "FGM":
        # FGM: BDFE at different times using the FGM replicate data
        plot_bdfe_exp_fgm(ax, reps, percents=percents_array_fgm)
    elif dataset == "SK":
        # SK: BDFE at different time-points along the flip sequence
        plot_bdfe_exp_sk(
            ax,
            points_lst=sk_args["flip_list"],
            num_flips=sk_args["num_flips"],
        )
    elif dataset == "NK":
        # NK: BDFE at different times using nk_data
        plot_bdfe_exp_nk(
            ax,
            percents=nk_args["percents"],
        )
    else:
        raise ValueError(f"Unknown dataset label for column 3: {dataset!r}")


def main():
    # Build inputs
    reps, final = build_fgm_inputs()
    sk_args = build_sk_inputs()
    nk_args = build_nk_inputs()

    # 3x3 figure with shared layout
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.34, wspace=0.34)

    datasets = ["FGM", "SK", "NK"]

    # Create each row using the same three column functions
    for row, ds in enumerate(datasets):
        ax_evol = axes[row, 0]
        ax_final = axes[row, 1]
        ax_bdfe = axes[row, 2]

        plot_column1(ax_evol, ds, reps, sk_args)
        plot_column2(ax_final, ds, final, sk_args)
        plot_column3(ax_bdfe, ds, reps, sk_args, nk_args)

    # Label subfigures A–I in row-major order, as before
    panel_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    flat_axes = [ax for row_axes in axes for ax in row_axes]
    for label, ax in zip(panel_labels, flat_axes):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Keep spine widths and ticks consistent with the originals
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")

    # Save SVG
    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig1_dfe_dynamics.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved: {out_path}")
