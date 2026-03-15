import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

CMR_COLORS = sns.color_palette("CMRmap", 4)
N_PATTERN = re.compile(r"_n(\d+)_")
N_PERCENT_POINTS = 100

def find_fgm_files(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    files = [p for p in data_root.glob("*.pkl") if "fgm" in p.name.lower()]

    def extract_n(path: Path) -> int:
        match = N_PATTERN.search(path.name)
        return int(match.group(1)) if match else 10 ** 9

    return sorted(files, key=lambda p: (extract_n(p), p.name))


def extract_n_from_name(path: Path) -> int:
    match = N_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not parse n from filename: {path.name}")
    return int(match.group(1))


def sample_radius_vs_percent(traj, n_points: int = N_PERCENT_POINTS) -> np.ndarray:
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


def load_fgm_cv2_metrics() -> dict[int, np.ndarray]:
    script_dir = Path(__file__).resolve().parent
    data_candidates = [script_dir.parent / "data" / "fgm", script_dir.parent / "data" / "FGM"]
    data_dir = next((p for p in data_candidates if p.exists()), data_candidates[-1])

    coarse_by_n = {}
    for path in find_fgm_files(data_dir):
        n = extract_n_from_name(path)
        with path.open("rb") as handle:
            repeats = pickle.load(handle)

        percent_radii = []
        for rep in repeats:
            if not isinstance(rep, dict):
                continue
            traj = rep.get("traj")
            if traj is None or len(traj) == 0:
                continue
            sampled = sample_radius_vs_percent(traj, n_points=N_PERCENT_POINTS)
            if np.all(np.isfinite(sampled)):
                percent_radii.append(sampled)

        coarse_by_n[n] = (
            np.vstack(percent_radii)
            if len(percent_radii) > 0
            else np.empty((0, N_PERCENT_POINTS), dtype=float)
        )

    return dict(sorted(coarse_by_n.items()))


def cv2_over_percent_radius(coarse: np.ndarray) -> np.ndarray:
    if coarse.size == 0:
        return np.full(N_PERCENT_POINTS, np.nan, dtype=float)
    mean = np.nanmean(coarse, axis=0)
    std = np.nanstd(coarse, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv2 = (std / mean) ** 2
    cv2[~np.isfinite(cv2)] = np.nan
    return cv2


def main():
    fgm_cv2_by_n = load_fgm_cv2_metrics()

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    percent_axis = np.arange(1, N_PERCENT_POINTS + 1)
    for color, (n_val, coarse) in zip(CMR_COLORS, fgm_cv2_by_n.items()):
        cv2 = cv2_over_percent_radius(coarse)
        if np.all(np.isnan(cv2)):
            continue
        ax.plot(percent_axis, cv2, color=color, lw=2.3, label=rf"$n={n_val}$")

    ax.set_xlabel("Walk progress (%)")
    ax.set_ylabel(r"$CV^2(R)$")
    ax.legend(frameon=False, loc="best")
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figA1_R_CV.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
