import argparse
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
LEFT_SIGMA = 0.05

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


def load_fgm_metrics(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Load one FGM file and extract:
    - final radii
    - raw (unnormalized) radius coarse-grained to 100 percent points for each repeat
    """
    n = extract_n_from_name(path)

    with path.open("rb") as handle:
        repeats = pickle.load(handle)

    final_radii = []
    percent_radii_raw = []
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

    final_arr = np.asarray(final_radii, dtype=float)
    percent_arr = (
        np.vstack(percent_radii_raw)
        if len(percent_radii_raw) > 0
        else np.empty((0, N_PERCENT_POINTS), dtype=float)
    )
    return n, final_arr, percent_arr


def collect_all_metrics(files: list[Path]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Collect all n-specific metrics from all FGM files.
    Returns:
    - final radii by n
    - coarse-grained raw radius trajectories by n (n_reps x 100)
    """
    final_by_n: dict[int, np.ndarray] = {}
    coarse_by_n: dict[int, np.ndarray] = {}

    for path in files:
        n, final_arr, coarse_arr = load_fgm_metrics(path)
        final_by_n[n] = final_arr
        coarse_by_n[n] = coarse_arr
        print(
            f"Loaded n={n}: {len(final_arr)} repeats with usable trajectories from {path.name}"
        )

    return dict(sorted(final_by_n.items())), dict(sorted(coarse_by_n.items()))


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


def style_axis(ax: plt.Axes, scientific: bool = True) -> None:
    """Apply scrambling-figure axis styling."""
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_two_panels(
    final_by_n: dict[int, np.ndarray],
    coarse_by_n: dict[int, np.ndarray],
) -> plt.Figure:
    """Plot two subplots: final-radius KDE and CV vs walk percent."""
    if not final_by_n:
        raise ValueError("No FGM data found to plot.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = CMR_COLORS[: len(final_by_n)]
    percent_axis = np.arange(1, N_PERCENT_POINTS + 1)

    for color, (n, values) in zip(colors, final_by_n.items()):
        if values.size == 0:
            continue
        scale = (n - 1) * LEFT_SIGMA / 2
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

    ax1.set_xlabel(r"$ 2\tilde{R}(t=100\%) / (n-1)$")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False, loc="upper left")

    ax2.set_xlabel("Walk progress (%)")
    ax2.set_ylabel(r"$CV^2(\tilde{R})$")
    ax2.legend(frameon=False, loc="upper left")

    ax1.text(-0.12, 1.04, "A", transform=ax1.transAxes, fontsize=18, fontweight="bold")
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=18, fontweight="bold")
    style_axis(ax1)
    style_axis(ax2, scientific=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FGM final-radius KDE and walk-progress CV curves for all n files."
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

    final_by_n, coarse_by_n = collect_all_metrics(files)
    fig = plot_two_panels(final_by_n, coarse_by_n)

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
