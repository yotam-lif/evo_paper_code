import argparse
import os
import pickle
import re
import warnings
import multiprocessing
from pathlib import Path

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import cramervonmises_2samp, ks_2samp, wasserstein_distance

# ----------------------------------------------------------------
# 1. VISUAL STYLE CONFIGURATION
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

CMR_COLORS = sns.color_palette("CMRmap", 4)
DEFAULT_SUBSET_DISTANCE_METRIC = "emd"

N_PATTERN = re.compile(r"_n(\d+)_")
# Finite-size sweep files (e.g. fgm_rps10_n4_N500_sig0.05.pkl) carry only ~10
# replicates each and belong to the floor study; the high-statistics production
# runs (fgm_rps1000_n*_sig*.pkl) have ~1000 replicates. Exclude the _N###_ sweep
# files so the CV^2 panel uses the smooth 1000-rep data, not the noisy 10-rep one.
NSWEEP_PATTERN = re.compile(r"_N\d+_")
SIG_PATTERN = re.compile(r"sig([0-9]*\.?[0-9]+)")
N_PERCENT_POINTS = 100


def apply_axis_style(ax, label):
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)


def normalize_distance_metric(metric):
    metric_key = metric.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cvm": "cvm",
        "cramervonmises": "cvm",
        "cramer_von_mises": "cvm",
        "emd": "emd",
        "earth_movers_distance": "emd",
        "earth_mover_distance": "emd",
        "earthmovers": "emd",
        "wasserstein": "emd",
        "ks": "ks",
        "ks_2samp": "ks",
        "kolmogorov_smirnov": "ks",
    }
    if metric_key not in aliases:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from cvm, emd, or ks.")
    return aliases[metric_key]


def distance_metric_label(metric):
    metric_key = normalize_distance_metric(metric)
    labels = {
        "cvm": "CvM distance",
        "emd": "Earth mover's distance",
        "ks": "KS statistic",
    }
    return labels[metric_key]


def compute_distance_metric(values_a, values_b, metric):
    metric_key = normalize_distance_metric(metric)
    if metric_key == "cvm":
        return cramervonmises_2samp(values_a, values_b).statistic
    if metric_key == "emd":
        return wasserstein_distance(values_a, values_b)
    if metric_key == "ks":
        return ks_2samp(values_a, values_b).statistic
    raise ValueError(f"Unsupported metric '{metric}'")


# ----------------------------------------------------------------
# 1b. FGM RADIUS-CV^2 PANEL (loaded from saved trajectories)
# ----------------------------------------------------------------
def find_fgm_files(data_root):
    if not data_root.exists():
        return []
    files = [p for p in data_root.glob("*.pkl")
             if "fgm" in p.name.lower() and not NSWEEP_PATTERN.search(p.name)]

    def extract_n(path):
        match = N_PATTERN.search(path.name)
        return int(match.group(1)) if match else 10 ** 9

    return sorted(files, key=lambda p: (extract_n(p), p.name))


def extract_n_from_name(path):
    match = N_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not parse n from filename: {path.name}")
    return int(match.group(1))


def extract_sigma_from_name(path, default=0.05):
    match = SIG_PATTERN.search(path.name)
    return float(match.group(1)) if match else default


def resolve_fgm_data_dir():
    script_dir = Path(__file__).resolve().parent
    data_candidates = [script_dir.parent / "data" / "fgm", script_dir.parent / "data" / "FGM"]
    return next((p for p in data_candidates if p.exists()), data_candidates[-1])


def sample_radius_vs_percent(traj, n_points=N_PERCENT_POINTS):
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


def load_fgm_cv2_metrics():
    data_dir = resolve_fgm_data_dir()

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


def cv2_over_percent_radius(coarse):
    if coarse.size == 0:
        return np.full(N_PERCENT_POINTS, np.nan, dtype=float)
    mean = np.nanmean(coarse, axis=0)
    std = np.nanstd(coarse, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv2 = (std / mean) ** 2
    cv2[~np.isfinite(cv2)] = np.nan
    return cv2


def plot_fgm_cv2_panel(ax):
    """Panel A: squared coefficient of variation of the FGM walk radius across
    replicates, versus walk progress, one curve per dimensionality n."""
    fgm_cv2_by_n = load_fgm_cv2_metrics()
    percent_axis = np.arange(1, N_PERCENT_POINTS + 1)
    for color, (n_val, coarse) in zip(CMR_COLORS, fgm_cv2_by_n.items()):
        cv2 = cv2_over_percent_radius(coarse)
        if np.all(np.isnan(cv2)):
            continue
        ax.plot(percent_axis, cv2, color=color, lw=2.3, label=rf"$n={n_val}$")

    ax.set_xlabel("Walk progress (%)")
    ax.set_ylabel(r"$CV^2(R(t))$")
    ax.legend(frameon=False, loc="best")


# ----------------------------------------------------------------
# 1c. FGM RADIUS-VS-TIME PANEL (mean descent vs far-field ODE)
# ----------------------------------------------------------------
def load_fgm_radius_vs_time():
    """Per dimensionality n, the non-dimensional walk radius
    tilde_r = ||r|| / sigma for every replicate as a function of walk step,
    read from the high-statistics (1000-rep) production trajectories. Returns
    {n: array of shape (n_reps, max_steps)} with NaN padding past each walk's end.
    """
    data_dir = resolve_fgm_data_dir()
    radius_by_n = {}
    for path in find_fgm_files(data_dir):
        n = extract_n_from_name(path)
        sigma = extract_sigma_from_name(path)
        with path.open("rb") as handle:
            repeats = pickle.load(handle)

        tilde_r_traces = []
        for rep in repeats:
            if not isinstance(rep, dict):
                continue
            traj = rep.get("traj")
            if traj is None or len(traj) == 0:
                continue
            arr = np.asarray(traj, dtype=float)
            if arr.ndim != 2:
                continue
            tilde_r_traces.append(np.linalg.norm(arr, axis=1) / sigma)

        if not tilde_r_traces:
            continue

        max_steps = max(len(trace) for trace in tilde_r_traces)
        padded = np.full((len(tilde_r_traces), max_steps), np.nan)
        for i, trace in enumerate(tilde_r_traces):
            padded[i, :len(trace)] = trace
        radius_by_n[n] = padded

    return dict(sorted(radius_by_n.items()))


def realign_held_from_radius(radii, r_ref):
    """Re-segment each replicate so t'=0 is its first step with radius <= r_ref,
    then forward-fill the post-termination peak floor. Every larger-n walk passes
    through r_ref on its way down, so this places all n on a *common* start radius
    for a like-for-like descent. In SSWM a terminated walk has reached a local
    optimum and freezes there, so holding it at its final radius is exact, not a
    fudge. Returns a (n_reps, max_seg_len) array, each row starting near r_ref."""
    segments = []
    for row in radii:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        hits = np.nonzero(finite <= r_ref)[0]
        idx0 = int(hits[0]) if hits.size else 0
        segments.append(finite[idx0:])
    if not segments:
        return np.empty((0, 0))

    max_len = max(len(s) for s in segments)
    held = np.empty((len(segments), max_len))
    for i, s in enumerate(segments):
        held[i, :len(s)] = s
        held[i, len(s):] = s[-1]          # freeze at the peak floor
    return held


def plot_fgm_radius_panel(ax):
    """Panel B: mean FGM walk radius vs time, re-segmented so every n starts from
    a *common* initial radius r_ref = the n=4 initial tilde_r_0 (~40). Each curve
    is then r(t)/r_ref, so the far-field ODE  d||r||/dt = -sqrt(pi/2) sigma
    collapses to a single line  r(t)/r_ref = 1 - sqrt(pi/2) t / tilde_r_0 shared
    by all n. The curves track it together and peel off (decelerate) once they
    reach tilde_r = sqrt(n) -- larger n peeling earlier, since sqrt(n) is hit at
    a larger radius. One curve per n, colored as in panel A."""
    radius_by_n = load_fgm_radius_vs_time()
    slope = np.sqrt(np.pi / 2.0)

    # Common start radius: the smallest-n (n=4) initial tilde_r_0. Every larger-n
    # walk passes through it on the way down to the peak.
    smallest_n = min(radius_by_n)
    r_ref = float(np.nanmean(radius_by_n[smallest_n][:, 0]))

    right_edges = []
    for color, (n_val, radii) in zip(CMR_COLORS, radius_by_n.items()):
        held = realign_held_from_radius(radii, r_ref)
        if held.size == 0:
            continue
        mean_r = held.mean(axis=0)
        std_r = held.std(axis=0)
        n_steps = held.shape[1]
        t = np.arange(n_steps)

        # Normalize by the common start radius r_ref so all n share one theory line.
        norm_mean = mean_r / r_ref
        norm_std = std_r / r_ref

        ax.plot(t, norm_mean, color=color, lw=2.2, label=rf"$n = {n_val}$")
        ax.fill_between(t, norm_mean - norm_std, norm_mean + norm_std,
                        color=color, alpha=0.18, lw=0)
        right_edges.append(n_steps - 1)

    # One shared far-field line, from the common start radius r_ref; mask the part
    # below zero so the dashed line stops at the peak.
    t_max = max(right_edges) if right_edges else 1
    t_th = np.arange(0, t_max + 1)
    theory = 1.0 - slope * t_th / r_ref
    theory[theory < 0] = np.nan
    ax.plot(t_th, theory, color="black", lw=1.5, ls="--", alpha=0.9,
            label="Theory (Eq. *)")

    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r"$\langle \tilde{R}(t) \rangle / \tilde{R}(0)$")
    ax.set_title(rf"$\tilde{{R}}(0) = {r_ref:.0f}$")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, t_max + 2)
    ax.legend(frameon=False, loc="best")


# ----------------------------------------------------------------
# 2. MODEL CLASSES
# ----------------------------------------------------------------
class FisherModel:
    def __init__(self, n, sigma, m, R0, seed=None):
        self.n = int(n)
        self.sigma = float(sigma)
        self.m = int(m)
        self.rng = np.random.default_rng(seed)
        self.deltas = self.rng.normal(loc=0.0, scale=self.sigma, size=(self.m, self.n))
        self.R0 = float(R0)
        self.r = np.zeros(self.n)
        self.r[0] = self.R0

    def compute_fitness(self, r):
        return np.exp(-0.5 * np.dot(r, r))

    def compute_dfe(self, r):
        w0 = self.compute_fitness(r)
        r_new = r + self.deltas
        r2_new = np.einsum("ij,ij->i", r_new, r_new)
        w_new = np.exp(-0.5 * r2_new)
        return w_new - w0

    @staticmethod
    def normalize(vec):
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return np.zeros_like(vec)
        return vec / norm


class FisherRadialWedge(FisherModel):
    """SSWM adaptive walk confined to a fixed angular wedge around the initial
    direction, isolating *radial* scrambling.

    The wedge is defined once, by the starting position r0: it is the tube of
    points x whose squared perpendicular distance from the fixed axis
    rhat0 = r0 / ||r0|| satisfies
        || x_perp ||^2  <=  sin^2(phi) * ||r0||^2,
    where phi (in radians) sets the wedge half-width at the initial radius
    (sin^2(phi) ||r0||^2 is the squared-perpendicular area). A mutation is
    admissible iff the landing point r + delta stays inside this region. Among
    the admissible AND beneficial candidates, one is chosen with probability
    proportional to its fitness effect (the FGM SSWM rule). Because admissible
    moves cannot wander in orientation, all progress is radial, and the walk
    descends along rhat0 toward the peak.
    """

    def __init__(self, n, sigma, m, R0, phi, seed=None):
        super().__init__(n, sigma, m, R0, seed=seed)
        self.axis = FisherModel.normalize(self.r)              # rhat0 (fixed for all t)
        self.perp_threshold = np.sin(float(phi)) ** 2 * R0 ** 2  # squared-perpendicular area

    def step(self):
        r_candidates = self.r + self.deltas
        proj = r_candidates @ self.axis                  # parallel component along rhat0
        perp_sq = np.einsum("ij,ij->i", r_candidates, r_candidates) - proj ** 2
        in_wedge = perp_sq <= self.perp_threshold

        dfe = self.compute_dfe(self.r)
        beneficial = dfe > 0

        valid_indices = np.nonzero(in_wedge & beneficial)[0]
        if len(valid_indices) == 0:
            return False

        effects = dfe[valid_indices]
        probs = effects / np.sum(effects)
        choice = self.rng.choice(valid_indices, p=probs)
        self.r += self.deltas[choice]
        # In FGM, if a mutation fixes, the forward mutation flips sign.
        self.deltas[choice] *= -1
        return True


# ----------------------------------------------------------------
# 3. SIMULATION WORKER
# ----------------------------------------------------------------
def run_single_replicate(args):
    seed, n, sigma, m, R0, params, max_t, time_points = args

    phi = params["phi"]
    subset_metric = normalize_distance_metric(
        params.get("subset_metric", DEFAULT_SUBSET_DISTANCE_METRIC)
    )

    model = FisherRadialWedge(n, sigma, m, R0, phi, seed=seed)

    # Static copy of the exact initial mutations to track the decorrelation of
    # those same mutational fitness effects in time (independent of stepping).
    initial_deltas = model.deltas.copy()
    delta_norms_sq = np.sum(initial_deltas ** 2, axis=1)

    def get_malthusian(r):
        return -0.5 * delta_norms_sq - np.dot(initial_deltas, r)

    def get_fitness_effects(r):
        # Selection coefficient s = w(r+delta)/w(r) - 1 = expm1(malthusian).
        # Scale-free: unlike the absolute difference w(r+delta) - w(r), this drops
        # the global exp(-||r||^2/2) prefactor, so EMD compares DFE *shapes* across
        # radii rather than being swamped by the exponentially varying scale.
        return np.expm1(get_malthusian(r))

    dfe0 = get_fitness_effects(model.r)
    # The tracked subset M is fixed at t=0: the mutations initially beneficial.
    initial_beneficial_mask = dfe0 > 0

    def get_subset_distance(dfe_t):
        if not np.any(initial_beneficial_mask):
            return np.nan
        tracked_dfe_t = dfe_t[initial_beneficial_mask]
        return compute_distance_metric(tracked_dfe_t, dfe_t, subset_metric)

    subset_distance0 = get_subset_distance(dfe0)
    std_dfe0 = np.std(dfe0)

    radii = np.full(len(time_points), np.nan)
    pearsons = np.full(len(time_points), np.nan)
    subset_distances = np.full(len(time_points), np.nan)

    radii[0] = np.linalg.norm(model.r)
    pearsons[0] = 1.0
    if np.isfinite(subset_distance0):
        subset_distances[0] = 1.0

    current_t_idx = 0
    time_points_set = set(int(t) for t in time_points)

    for t in range(1, max_t + 1):
        success = model.step()
        if not success:
            break

        if t in time_points_set:
            current_t_idx += 1
            radii[current_t_idx] = np.linalg.norm(model.r)

            dfe_t = get_fitness_effects(model.r)
            if std_dfe0 > 1e-12 and np.std(dfe_t) > 1e-12:
                pearsons[current_t_idx] = np.corrcoef(dfe0, dfe_t)[0, 1]
            if np.isfinite(subset_distance0) and subset_distance0 > 1e-12:
                subset_distance_t = get_subset_distance(dfe_t)
                if np.isfinite(subset_distance_t):
                    subset_distances[current_t_idx] = subset_distance_t / subset_distance0

    return pearsons, subset_distances, radii


# ----------------------------------------------------------------
# 4. HELPERS
# ----------------------------------------------------------------
def stack_results(res_list):
    pearsons = np.array([res[0] for res in res_list], dtype=float)
    subset_distances = np.array([res[1] for res in res_list], dtype=float)
    radii = np.array([res[2] for res in res_list], dtype=float)
    return pearsons, subset_distances, radii


# ----------------------------------------------------------------
# 5. MAIN EXPERIMENT
# ----------------------------------------------------------------
def run_experiment(n_values, r0_tilde_values, phi,
                   subset_metric=DEFAULT_SUBSET_DISTANCE_METRIC):
    subset_metric = normalize_distance_metric(subset_metric)
    subset_metric_label = distance_metric_label(subset_metric)

    sigma = 0.05
    reps = 50
    m = 5 * 10 ** 4

    print("--- Configuration (radial scrambling, wedge-constrained SSWM) ---")
    print(f"Subset distance metric: {subset_metric_label} ({subset_metric})")
    print(f"sigma={sigma}, m={m}, reps={reps}, phi={phi} rad")
    print(f"Panels (R0_tilde): {r0_tilde_values}")
    print(f"Curves (n): {n_values}")

    base_seed = np.random.randint(0, 1_000_000)

    # Each (panel R0_tilde, curve n) gets `reps` replicates.
    tasks = []
    spans = {}
    for p, R0_tilde in enumerate(r0_tilde_values):
        R0 = R0_tilde * sigma
        # Larger starting radius => longer radial descent; scale the horizon.
        max_t = int(4 * R0_tilde) + 20
        time_points = np.arange(0, max_t + 1)
        for k, n in enumerate(n_values):
            start = len(tasks)
            for i in range(reps):
                tasks.append((
                    base_seed + 100_000 * (p + 1) + 10_000 * (k + 1) + i,
                    int(n),
                    sigma,
                    m,
                    R0,
                    {"phi": phi, "subset_metric": subset_metric},
                    max_t,
                    time_points,
                ))
            spans[(p, k)] = (start, len(tasks), time_points)

    num_proc = min(multiprocessing.cpu_count(), len(tasks))
    with multiprocessing.Pool(processes=num_proc) as pool:
        results = pool.map(run_single_replicate, tasks)

    # ------------------------------
    # Plotting: panel A = FGM radius CV^2; panel B = FGM mean radius vs time
    # (far-field ODE test); panels C.. = one EMD panel per R0_tilde. Laid out on
    # a 2-column grid (a clean 2x2 for the default two R0_tilde panels).
    # ------------------------------
    n_panels = len(r0_tilde_values)
    total_panels = 2 + n_panels                # A: CV^2, B: radius-vs-time, C..: EMD per R0_tilde
    ncols = 2
    nrows = int(np.ceil(total_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows), squeeze=False)
    fig.subplots_adjust(wspace=0.28, hspace=0.32)
    panel_axes = axes.flatten()

    # Panel A: FGM radius CV^2 vs walk progress (loaded from saved trajectories).
    apply_axis_style(panel_axes[0], "A")
    plot_fgm_cv2_panel(panel_axes[0])

    # Panel B: FGM mean non-dimensional radius vs time against the far-field ODE.
    apply_axis_style(panel_axes[1], "B")
    plot_fgm_radius_panel(panel_axes[1])

    # EMD n-curves use cmr.emerald (matching fig4_peak_dfes), sampled over
    # cmap_range=(0.3, 1.0) to skip the near-white low end.
    n_colors = cmr.take_cmap_colors("cmr.emerald", len(n_values), cmap_range=(0.3, 1.0))

    for p, R0_tilde in enumerate(r0_tilde_values):
        ax = panel_axes[p + 2]               # EMD panels follow panels A and B
        apply_axis_style(ax, chr(ord("C") + p))
        max_reached = 1

        for k, n in enumerate(n_values):
            start, end, time_points = spans[(p, k)]
            _pear, subset, _radii = stack_results(results[start:end])

            # Time points past the longest walk are all-NaN columns; ignore the
            # resulting "empty slice" warnings (those points are trimmed by xlim).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_subset = np.nanmean(subset, axis=0)
                std_subset = np.nanstd(subset, axis=0)

            finite_t = np.where(np.isfinite(mean_subset))[0]
            last = int(finite_t[-1]) if len(finite_t) else 0
            if len(finite_t):
                max_reached = max(max_reached, int(time_points[last]))

            color = n_colors[k]
            # Mean EMD trace plus error bars (std across replicates).
            ax.plot(time_points, mean_subset, color=color, lw=2.0)
            # Subsample markers across the *populated* range, not the full horizon,
            # so the error bars span the visible curve rather than collapsing to a
            # few points near t=0.
            step = max(1, (last + 1) // 10)
            marker_idx = np.arange(0, last + 1, step)
            ax.errorbar(
                time_points[marker_idx],
                mean_subset[marker_idx],
                yerr=std_subset[marker_idx],
                fmt="o",
                color=color,
                markersize=4,
                capsize=3,
                label=fr"$n = {int(n)}$",
            )

        # Far-field theory: the radial descent is linear at the SSWM speed
        # d<R~>/dt = -sqrt(pi/2), and the normalized scrambling tracks the
        # normalized radius, EMD(t) ~ R~(t)/R~0 = 1 - sqrt(pi/2) * t / R~0.
        tp_panel = spans[(p, 0)][2]
        theory = np.clip(1.0 - (np.sqrt(np.pi / 2) / R0_tilde) * tp_panel, 0.0, None)
        ax.plot(tp_panel, theory, color="black", lw=2.0, ls="--",
                label="Theory (Eq. *)")

        ax.set_xlim(0, max_reached)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel(f"{subset_metric.upper()} (norm.)")
        ax.set_title(rf"$\tilde{{R}}(0) = {R0_tilde:g}$,  $\phi = {phi:g}$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False, loc="best", title=None)

    # Hide any leftover axes when the panel count does not fill the grid.
    for ax in panel_axes[total_panels:]:
        ax.set_visible(False)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS7_radial_scrambling.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radial scrambling: SSWM adaptive walk confined to a fixed angular "
                    "wedge around the initial direction, measuring convergence of the "
                    "initially-beneficial subset DFE to the overall DFE across n values."
    )
    parser.add_argument(
        "--subset-metric",
        default=DEFAULT_SUBSET_DISTANCE_METRIC,
        help="Distance metric for comparing the t=0 beneficial subset to the full DFE: cvm, emd, or ks.",
    )
    parser.add_argument(
        "--n-values",
        default="4,8,16,32",
        help="Comma-separated dimensionalities n, overlaid as EMD curves in every panel.",
    )
    parser.add_argument(
        "--r0-values",
        default="10,40",
        help="Comma-separated initial R0_tilde values, one panel each.",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.6,
        help="Wedge half-angle in radians, fixed at the initial radius: admissible "
             "iff ||x_perp||^2 <= sin^2(phi) * ||r0||^2. sin^2(phi) must exceed "
             "~n*sigma^2/R0^2 for the largest n at the smallest R0, else those "
             "walks cannot fit any mutation in the wedge and get stuck.",
    )
    args = parser.parse_args()
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    r0_tilde_values = [float(x) for x in args.r0_values.split(",") if x.strip()]
    run_experiment(n_values, r0_tilde_values, args.phi, subset_metric=args.subset_metric)
