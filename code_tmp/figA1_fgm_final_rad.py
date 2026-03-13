import argparse
import math
import pickle
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

N_PATTERN = re.compile(r"_n(\d+)_")
N_PERCENT_POINTS = 100
KDE_BW_ADJUST = 1.6
SIGMA = 0.05
PERCENT_START_NEAR = 85

# Match style in scrambling simulation figure scripts.
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
    }
)
CMR_COLORS = sns.color_palette("CMRmap", 5)


def find_fgm_files(data_root: Path) -> list[Path]:
    """Find all FGM simulation pickle files and sort them by n."""
    if not data_root.exists():
        return []

    files = [p for p in data_root.glob("*.pkl") if "fgm" in p.name.lower()]

    def extract_n(path: Path) -> int:
        match = N_PATTERN.search(path.name)
        return int(match.group(1)) if match else 10**9

    return sorted(files, key=lambda p: (extract_n(p), p.name))


def extract_n_from_name(path: Path) -> int:
    match = N_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not parse n from filename: {path.name}")
    return int(match.group(1))


def sample_radius_vs_percent(traj: list, n_points: int = N_PERCENT_POINTS) -> np.ndarray:
    """
    Coarse-grain one walk into n_points at integer percentages.
    Each point is the closest trajectory entry to p% of the walk length.
    """
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.full(n_points, np.nan)

    radii = np.linalg.norm(arr, axis=1)
    total_steps = len(radii) - 1
    percents = np.arange(1, n_points + 1, dtype=float)

    if total_steps <= 0:
        return np.full(n_points, radii[0], dtype=float)

    targets = (percents / 100.0) * total_steps
    idx = np.rint(targets).astype(int)
    idx = np.clip(idx, 0, total_steps)
    return radii[idx]


def load_fgm_metrics(path: Path) -> tuple[int, np.ndarray, np.ndarray, float]:
    """
    Load one FGM file and extract:
    - final radii
    - raw (unnormalized) radius coarse-grained to 100 percent points for each repeat
    - mean walk length in fixed mutations (mean(len(traj)-1))
    """
    n = extract_n_from_name(path)

    with path.open("rb") as handle:
        repeats = pickle.load(handle)

    final_radii = []
    percent_radii_raw = []
    step_counts = []
    for rep in repeats:
        if not isinstance(rep, dict):
            continue
        traj = rep.get("traj")
        if traj is None or len(traj) == 0:
            continue

        sampled_raw = sample_radius_vs_percent(traj, n_points=N_PERCENT_POINTS)
        if np.all(np.isfinite(sampled_raw)):
            percent_radii_raw.append(sampled_raw)
            final_radii.append(sampled_raw[-1])
            step_counts.append(len(traj) - 1)

    final_arr = np.asarray(final_radii, dtype=float)
    percent_arr = (
        np.vstack(percent_radii_raw)
        if len(percent_radii_raw) > 0
        else np.empty((0, N_PERCENT_POINTS), dtype=float)
    )
    mean_steps = float(np.mean(step_counts)) if len(step_counts) > 0 else np.nan
    return n, final_arr, percent_arr, mean_steps


def collect_all_metrics(
    files: list[Path],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    """
    Collect all n-specific metrics from all FGM files.
    Returns:
    - final radii by n
    - coarse-grained raw radius trajectories by n (n_reps x 100)
    - mean walk length by n (in fixed mutations)
    """
    final_by_n: dict[int, np.ndarray] = {}
    coarse_by_n: dict[int, np.ndarray] = {}
    mean_steps_by_n: dict[int, float] = {}

    for path in files:
        n, final_arr, coarse_arr, mean_steps = load_fgm_metrics(path)
        final_by_n[n] = final_arr
        coarse_by_n[n] = coarse_arr
        mean_steps_by_n[n] = mean_steps
        print(
            f"Loaded n={n}: {len(final_arr)} repeats, mean steps={mean_steps:.2f}, from {path.name}"
        )

    return (
        dict(sorted(final_by_n.items())),
        dict(sorted(coarse_by_n.items())),
        dict(sorted(mean_steps_by_n.items())),
    )


def cv_over_percent_radius(coarse: np.ndarray) -> np.ndarray:
    """Compute CV^2 of radius at each percent across realizations."""
    if coarse.size == 0:
        return np.full(N_PERCENT_POINTS, np.nan, dtype=float)

    mean = np.nanmean(coarse, axis=0)
    std = np.nanstd(coarse, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv2 = (std / mean) ** 2
    cv2[~np.isfinite(cv2)] = np.nan
    return cv2


def mean_nondim_radius_over_percent(coarse: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """Compute mean non-dimensional radius R/sigma across realizations."""
    if coarse.size == 0:
        return np.full(N_PERCENT_POINTS, np.nan, dtype=float)
    return np.nanmean(coarse / sigma, axis=0)


def solve_mean_radius_ode(
    n: int,
    r0: float,
    t_end: float,
    n_points: int = N_PERCENT_POINTS,
    internal_steps: int = 10000,
) -> np.ndarray:
    """
    Solve the compact radial-dynamics ODE for non-dimensional radius \\tilde R:
        d\\tilde R/dt =
            -sqrt(1 + n/(2\\tilde R^2))
             * [((1+lambda^2)Phi(-lambda) - lambda*phi(lambda))
                / (phi(lambda) - lambda*Phi(-lambda))]
        lambda = n / sqrt(4\\tilde R^2 + 2n)

    We integrate on t in [0, t_end], where t_end is the mean walk length
    (in fixed substitutions) for this n, and then sample at 100 percent points.
    """
    eps = 1e-9
    t_start = 0.0
    if not np.isfinite(t_end) or t_end <= 0:
        return np.full(n_points, np.nan, dtype=float)
    t_out = np.linspace(t_start, t_end, n_points)

    def drift(r_curr: float) -> float:
        r_safe = max(r_curr, eps)
        lam = n / math.sqrt(4.0 * r_safe * r_safe + 2.0 * n)
        phi_lam = math.exp(-0.5 * lam * lam) / math.sqrt(2.0 * math.pi)
        phi_minus_lam = 0.5 * (1.0 - math.erf(lam / math.sqrt(2.0)))
        numer = (1.0 + lam * lam) * phi_minus_lam - lam * phi_lam
        denom = phi_lam - lam * phi_minus_lam
        if abs(denom) < eps:
            return math.nan
        pref = math.sqrt(1.0 + n / (2.0 * r_safe * r_safe))
        return -pref * (numer / denom)

    internal_steps = max(int(internal_steps), 1)
    dt = (t_end - t_start) / internal_steps

    t_vals = [t_start]
    r_vals = [max(float(r0), eps)]

    for _ in range(internal_steps):
        r_curr = r_vals[-1]

        # RK4 integration for better stability/accuracy on the fine grid.
        k1 = drift(r_curr)
        if not math.isfinite(k1):
            break
        k2 = drift(r_curr + 0.5 * dt * k1)
        if not math.isfinite(k2):
            break
        k3 = drift(r_curr + 0.5 * dt * k2)
        if not math.isfinite(k3):
            break
        k4 = drift(r_curr + dt * k3)
        if not math.isfinite(k4):
            break

        r_next = r_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not math.isfinite(r_next):
            break

        t_vals.append(t_vals[-1] + dt)
        r_vals.append(max(r_next, eps))

    r_out = np.full(n_points, np.nan, dtype=float)
    t_vals_arr = np.asarray(t_vals, dtype=float)
    r_vals_arr = np.asarray(r_vals, dtype=float)
    last_t = t_vals_arr[-1]

    valid_mask = t_out <= last_t
    if np.any(valid_mask):
        r_out[valid_mask] = np.interp(t_out[valid_mask], t_vals_arr, r_vals_arr)

    return r_out


def solve_linear_radius_approx(
    r0: float,
    t_end: float,
    n_points: int = N_PERCENT_POINTS,
) -> np.ndarray:
    """
    Rough approximation:
        \\tilde R(t) = \\tilde R(0) - sqrt(pi/2) * t
    sampled on the same time window used for the ODE.
    """
    if not np.isfinite(t_end) or t_end <= 0:
        return np.full(n_points, np.nan, dtype=float)

    t = np.linspace(0.0, t_end, n_points)
    r = float(r0) - np.sqrt(np.pi / 2.0) * t

    # Keep physical branch (non-negative radius) and stop line after crossing 0.
    first_nonpos = np.flatnonzero(r <= 0.0)
    if first_nonpos.size > 0:
        idx = int(first_nonpos[0])
        r[idx] = 0.0
        if idx + 1 < n_points:
            r[idx + 1 :] = np.nan
    return r


def solve_near_field_approx(
    r_t0: float,
    t0: float,
    t_values: np.ndarray,
) -> np.ndarray:
    """
    Near-field approximation:
        R(t) = sqrt(R(t0)^2 - 4*(t-t0))
    """
    t_values = np.asarray(t_values, dtype=float)
    out = np.full_like(t_values, np.nan, dtype=float)
    if not np.isfinite(r_t0):
        return out

    mask = t_values >= t0
    arg = r_t0 * r_t0 - 4.0 * (t_values[mask] - t0)
    valid = arg >= 0.0
    out_vals = np.full(mask.sum(), np.nan, dtype=float)
    out_vals[valid] = np.sqrt(arg[valid])
    out[mask] = out_vals
    return out


def style_axis(ax: plt.Axes, scientific: bool = True) -> None:
    """Apply scrambling-figure axis styling."""
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_five_panels(
    final_by_n: dict[int, np.ndarray],
    coarse_by_n: dict[int, np.ndarray],
    mean_steps_by_n: dict[int, float],
) -> plt.Figure:
    """
    Plot five subplots:
    A: final-radius KDE
    B: CV^2 vs percent
    C: mean R/sigma + complex ODE
    D: mean R/sigma + linear approximation
    E: from 85% onward, mean R/sigma + near-field approximation
    """
    if not final_by_n:
        raise ValueError("No FGM data found to plot.")

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(34, 6), constrained_layout=True)
    colors = CMR_COLORS[: len(final_by_n)]
    percent_axis = np.arange(1, N_PERCENT_POINTS + 1)

    for color, (n, values) in zip(colors, final_by_n.items()):
        if values.size == 0:
            continue
        scale = (n-1) * SIGMA / 2
        if scale <= 0:
            continue
        values_scaled = values / scale
        sns.kdeplot(
            values_scaled,
            ax=ax1,
            color=color,
            fill=False,
            linewidth=2.0,
            bw_adjust=KDE_BW_ADJUST,
            label=rf"$n={n}$",
        )
        # ax1.axvline(np.mean(values_scaled), color=color, linestyle=":", linewidth=2.0)

    for color, (n, coarse) in zip(colors, coarse_by_n.items()):
        cv = cv_over_percent_radius(coarse)
        if np.all(np.isnan(cv)):
            continue
        ax2.plot(percent_axis, cv, color=color, linewidth=2.0, label=rf"$n={n}$")

    for color, (n, coarse) in zip(colors, coarse_by_n.items()):
        mean_r = mean_nondim_radius_over_percent(coarse, sigma=SIGMA)
        if np.all(np.isnan(mean_r)):
            continue

        # Panel C: simulation + complex ODE
        ax3.plot(percent_axis, mean_r, color=color, linewidth=2.0, label=rf"$n={n}$")

        # Initial condition for the non-dimensional radius: R0 = sqrt(n) / sigma.
        r0 = np.sqrt(n) / SIGMA
        t_end = mean_steps_by_n.get(n, np.nan)
        mean_pred = solve_mean_radius_ode(
            n=n,
            r0=r0,
            t_end=t_end,
            n_points=N_PERCENT_POINTS,
        )
        if not np.all(np.isnan(mean_pred)):
            ax3.plot(percent_axis, mean_pred, color=color, linewidth=2.0, linestyle=":")

        # Panel D: simulation + linear approximation
        ax4.plot(percent_axis, mean_r, color=color, linewidth=2.0, label=rf"$n={n}$")

        linear_pred = solve_linear_radius_approx(
            r0=r0,
            t_end=t_end,
            n_points=N_PERCENT_POINTS,
        )
        if not np.all(np.isnan(linear_pred)):
            # Block-dotted style for the rough linear approximation.
            ax4.plot(
                percent_axis,
                linear_pred,
                color=color,
                linewidth=2.0,
                linestyle=(0, (4, 2, 1, 2)),
            )

        # Panel E: simulation + near-field approximation, from 85% onward.
        start_idx = max(PERCENT_START_NEAR - 1, 0)
        p_sub = percent_axis[start_idx:]
        mean_sub = mean_r[start_idx:]
        ax5.plot(p_sub, mean_sub, color=color, linewidth=2.0, label=rf"$n={n}$")

        if np.isfinite(t_end) and t_end > 0:
            finite_idx = np.flatnonzero(np.isfinite(mean_sub))
            if finite_idx.size > 0:
                k0 = int(finite_idx[0])
                p0 = p_sub[k0]
                t0 = (p0 / 100.0) * t_end
                r_t0 = mean_sub[k0]
                t_sub = (p_sub / 100.0) * t_end
                near_pred = solve_near_field_approx(r_t0=r_t0, t0=t0, t_values=t_sub)
                if not np.all(np.isnan(near_pred)):
                    ax5.plot(
                        p_sub,
                        near_pred,
                        color=color,
                        linewidth=2.0,
                        linestyle=(0, (2, 2)),
                    )

    ax1.set_xlabel(r"$ 2\tilde{R}(t=100\%) / (n-1)$")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False, loc="upper left")

    ax2.set_xlabel("Walk progress (%)")
    ax2.set_ylabel(r"$CV^2(\tilde{R})$")
    ax2.legend(frameon=False, loc="upper left")

    ax3.set_xlabel("Walk progress (%)")
    ax3.set_ylabel(r"$\langle R / \sigma \rangle$")
    ax3.legend(frameon=False, loc="best")

    ax4.set_xlabel("Walk progress (%)")
    ax4.set_ylabel(r"$\langle R / \sigma \rangle$")
    ax4.legend(frameon=False, loc="best")

    ax5.set_xlabel("Walk progress (%)")
    ax5.set_ylabel(r"$\langle R / \sigma \rangle$")
    ax5.legend(frameon=False, loc="best")

    ax1.text(-0.12, 1.04, "A", transform=ax1.transAxes, fontsize=18, fontweight="bold")
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=18, fontweight="bold")
    ax3.text(-0.12, 1.04, "C", transform=ax3.transAxes, fontsize=18, fontweight="bold")
    ax4.text(-0.12, 1.04, "D", transform=ax4.transAxes, fontsize=18, fontweight="bold")
    ax5.text(-0.12, 1.04, "E", transform=ax5.transAxes, fontsize=18, fontweight="bold")
    style_axis(ax1)
    style_axis(ax2, scientific=False)
    style_axis(ax3, scientific=False)
    style_axis(ax4, scientific=False)
    style_axis(ax5, scientific=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FGM final-radius KDE, CV^2, and three mean R/sigma theory comparisons."
        )
    )
    parser.add_argument(
        "--save",
        type=str,
        default="../figs_paper/figA1_fgm_final_rad.pdf",
        help="Output path for figure (default: ../figs_paper/figA1_fgm_final_rad.pdf).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_candidates = [script_dir.parent / "data" / "fgm", script_dir.parent / "data" / "FGM"]
    data_dir = next((p for p in data_candidates if p.exists()), data_candidates[-1])

    files = find_fgm_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No FGM .pkl files found in {data_dir}")

    final_by_n, coarse_by_n, mean_steps_by_n = collect_all_metrics(files)
    fig = plot_five_panels(final_by_n, coarse_by_n, mean_steps_by_n)

    save_path = (
        (script_dir / args.save).resolve()
        if not Path(args.save).is_absolute()
        else Path(args.save)
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf")
    print(f"Saved figure to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
