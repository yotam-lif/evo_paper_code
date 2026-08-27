r"""Poster Figure 5: cutoff-dependent scrambling in fitted REL607 FGM models.

The figure compares predictive simulations from either pair of REL607
ancestral-DFE fits reported by ``poster_fig3.py``:

* Gaussian (canonical) FGM;
* shared-scale heavy-tailed FGM.

Both simulations use the additive Malthusian-fitness convention

    F(r) = -|r|^2 / 2,
    s_g(r) = F(r + delta_g) - F(r)
           = -r . delta_g - |delta_g|^2 / 2.

For every replicate, an independent 3,500-mutation probe library is held fixed
while the genetic background follows a finite-pool SSWM adaptive walk.  The
probe library is conditioned on ancestral s >= -0.5, matching the observed
range used in the Figure 3 likelihood.  The adaptive mutation pool is not
conditioned.  Probe and adaptive mutations are independent so fixing a
background mutation never removes an assayed probe mutation.

The three Pearson correlations retain 100%, 95%, or 90% of the probe library.
The nested exclusions are defined once from the ancestral effects and are then
held fixed along the walk.

Run from any directory:

    python code_tmp/poster_fig5.py --fit-source without-errors --poster
    python code_tmp/poster_fig5.py --fit-source with-errors

Outputs:

    data/FGM_HEAVY_TAILED/poster_fig5_rel607_<mode>_m3500.npz
    data/FGM_HEAVY_TAILED/poster_fig5_rel607_<mode>_m3500_summary.json
    code_tmp/out_tmp/poster_fig5_<mode>.pdf

With ``--poster``, the selected result is also copied to
``../PhD/Posters/GRC_evo_26/poster_fig5.pdf``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np

# All figures from code_tmp are scratch output; they go to code_tmp/out_tmp,
# never to figs_paper (which holds only the paper figures built by code_figs).
_OUT_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")



# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "FGM_HEAVY_TAILED")
FIG_DIR = _OUT_TMP
POSTER_PDF = (
    "/Users/yotamlifschytz/Desktop/PhD/Posters/"
    "GRC_evo_26/poster_fig5.pdf"
)
FIT_CONFIGS = {
    "with-errors": {
        "fit_path": os.path.join(
            REPO_ROOT, "data", "poster_fig3_fit.json"
        ),
        "cache_path": os.path.join(
            DATA_DIR, "poster_fig5_rel607_with_errors_m3500.npz"
        ),
        "summary_path": os.path.join(
            DATA_DIR,
            "poster_fig5_rel607_with_errors_m3500_summary.json",
        ),
        "out_pdf": os.path.join(
            FIG_DIR, "poster_fig5_with_errors.pdf"
        ),
    },
    "without-errors": {
        "fit_path": os.path.join(
            REPO_ROOT, "data", "poster_fig3_no_errors_fit.json"
        ),
        "cache_path": os.path.join(
            DATA_DIR, "poster_fig5_rel607_no_errors_m3500.npz"
        ),
        "summary_path": os.path.join(
            DATA_DIR,
            "poster_fig5_rel607_no_errors_m3500_summary.json",
        ),
        "out_pdf": os.path.join(
            FIG_DIR, "poster_fig5_no_errors.pdf"
        ),
    },
}


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

PROBE_MUTATIONS = 3500
BACKGROUND_MUTATIONS = 3500
REPLICATES = 500
MAX_WALK_STEPS = 60
MAX_DISPLAY_STEPS = 10
PANEL_DISPLAY_STEPS = (8, 20)
MIN_DISPLAY_REPLICATES = 75
OBSERVED_LOWER_CUT = -0.5
RETAINED_FRACTIONS = (1.00, 0.95, 0.90)
FLOOR_TOLERANCE = 0.05
MASTER_SEED = 260730
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Poster style
# ---------------------------------------------------------------------------

for font_path in (
    "/Library/Fonts/AGaramondPro-Regular.otf",
    "/Library/Fonts/AGaramondPro-Italic.otf",
    "/Library/Fonts/AGaramondPro-Bold.otf",
    "/Library/Fonts/AGaramondPro-BoldItalic.otf",
):
    font_manager.fontManager.addfont(font_path)

mpl.rcParams.update({
    "font.family": "Adobe Garamond Pro",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Adobe Garamond Pro",
    "mathtext.it": "Adobe Garamond Pro:italic",
    "mathtext.bf": "Adobe Garamond Pro:bold",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# The same three colors and ordering used to partition the lower panels of
# poster Figure 1.  Here the nested curves progressively remove the dark and
# then the medium-blue deleterious partitions.
CURVE_COLORS = ("#211b34", "#2572e0", "#abd9dc")
CURVE_LINESTYLES = ("-", "-", "-")
FLOOR_LINE_COLOR = "#686868"


@dataclass(frozen=True)
class ModelConfig:
    """One fitted FGM parameter set, converted to an integer dimension."""

    key: str
    title: str
    n_fit: float
    n: int
    radius: float
    sigma: float
    mu: float | None

    def serializable(self) -> dict[str, float | int | str | None]:
        return {
            "key": self.key,
            "title": self.title,
            "n_fit": self.n_fit,
            "n_simulation": self.n,
            "radius": self.radius,
            "sigma": self.sigma,
            "mu": self.mu,
        }


def load_model_configs(
    fit_source: str,
) -> tuple[ModelConfig, ModelConfig]:
    """Read the selected Figure 3 parameter estimates."""
    configuration = FIT_CONFIGS[fit_source]
    with open(configuration["fit_path"], encoding="utf-8") as handle:
        fit_output = json.load(handle)

    if fit_source == "with-errors":
        canonical = fit_output[
            "canonical_moment_constrained_mle"
        ]["fit"]
        heavy = fit_output[
            "heavy_tailed_log_fitness_free_mu_mle"
        ]["with_gene_specific_errors"]["fit"]
    else:
        canonical = fit_output["canonical_full_mle"]["fit"]
        heavy = fit_output["heavy_tailed_full_mle"]["fit"]

    return (
        ModelConfig(
            key="canonical",
            title="Gaussian (canonical)",
            n_fit=float(canonical["n"]),
            n=int(np.rint(canonical["n"])),
            radius=float(canonical["r"]),
            sigma=float(canonical["sigma"]),
            mu=None,
        ),
        ModelConfig(
            key="heavy_tailed",
            title="Heavy-tailed",
            n_fit=float(heavy["n"]),
            n=int(np.rint(heavy["n"])),
            radius=float(heavy["r"]),
            sigma=float(heavy["sigma"]),
            mu=float(heavy["mu"]),
        ),
    )


def build_metadata(
    models: tuple[ModelConfig, ...],
    fit_source: str,
) -> dict[str, object]:
    """Metadata used both for reproducibility and cache invalidation."""
    return {
        "cache_version": CACHE_VERSION,
        "fit_source": fit_source,
        "fitness_convention": "F(r)=-|r|^2/2",
        "effect_convention": "s=-r.delta-|delta|^2/2",
        "probe_mutations": PROBE_MUTATIONS,
        "background_mutations": BACKGROUND_MUTATIONS,
        "replicates": REPLICATES,
        "max_walk_steps": MAX_WALK_STEPS,
        "max_display_steps": MAX_DISPLAY_STEPS,
        "minimum_display_replicates": MIN_DISPLAY_REPLICATES,
        "observed_lower_cut": OBSERVED_LOWER_CUT,
        "retained_fractions": list(RETAINED_FRACTIONS),
        "master_seed": MASTER_SEED,
        "probe_ascertainment": "conditioned on ancestral s >= lower cut",
        "background_ascertainment": "unconditioned",
        "background_protocol": (
            "independent finite mutation pool; beneficial mutations fixed "
            "without replacement with probability proportional to s"
        ),
        "measurement_noise": "not added to predictive simulations",
        "models": [model.serializable() for model in models],
    }


def draw_mutations(
    rng: np.random.Generator,
    number: int,
    model: ModelConfig,
) -> np.ndarray:
    """Draw Gaussian or shared-scale heavy-tailed mutation vectors."""
    deltas = rng.normal(
        loc=0.0,
        scale=model.sigma,
        size=(number, model.n),
    )
    if model.mu is not None:
        gamma_scale = rng.gamma(
            shape=model.mu,
            scale=1.0,
            size=(number, 1),
        )
        # Underflow is not expected at this finite library size, but retaining
        # a strictly positive denominator keeps the generator well-defined.
        gamma_scale = np.maximum(gamma_scale, np.finfo(float).tiny)
        deltas /= np.sqrt(2.0 * gamma_scale)
    return deltas


def mutation_effects(
    position: np.ndarray,
    deltas: np.ndarray,
    squared_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Malthusian FGM effects for a fixed mutation matrix."""
    if squared_lengths is None:
        with np.errstate(over="ignore", invalid="ignore"):
            squared_lengths = np.einsum(
                "ij,ij->i", deltas, deltas, optimize=True
            )
    with np.errstate(over="ignore", invalid="ignore"):
        return -(deltas @ position) - 0.5 * squared_lengths


def draw_observed_probe_library(
    rng: np.random.Generator,
    model: ModelConfig,
    initial_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample 3,500 probes in the Figure 3 observed-effect range."""
    accepted: list[np.ndarray] = []
    accepted_number = 0
    while accepted_number < PROBE_MUTATIONS:
        missing = PROBE_MUTATIONS - accepted_number
        batch_size = max(4096, 2 * missing)
        batch = draw_mutations(rng, batch_size, model)
        effects = mutation_effects(initial_position, batch)
        keep = np.isfinite(effects) & (effects >= OBSERVED_LOWER_CUT)
        if np.any(keep):
            accepted_batch = batch[keep]
            accepted.append(accepted_batch)
            accepted_number += accepted_batch.shape[0]

    probes = np.concatenate(accepted, axis=0)[:PROBE_MUTATIONS]
    probe_squared_lengths = np.einsum(
        "ij,ij->i", probes, probes, optimize=True
    )
    return probes, probe_squared_lengths


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Numerically stable Pearson correlation for two finite vectors."""
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(
        np.dot(x_centered, x_centered)
        * np.dot(y_centered, y_centered)
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def simulate_replicate(
    model: ModelConfig,
    seed: int,
) -> tuple[np.ndarray, int, float, float, float]:
    """Simulate one probe library and one independent SSWM background walk."""
    rng = np.random.default_rng(seed)
    position = np.zeros(model.n, dtype=float)
    position[0] = model.radius

    probes, probe_q = draw_observed_probe_library(rng, model, position)
    ancestral_effects = mutation_effects(position, probes, probe_q)
    order = np.argsort(ancestral_effects, kind="stable")
    retained_indices = []
    for retained_fraction in RETAINED_FRACTIONS:
        number_removed = int(
            np.floor((1.0 - retained_fraction) * PROBE_MUTATIONS)
        )
        retained_indices.append(order[number_removed:])

    background = draw_mutations(rng, BACKGROUND_MUTATIONS, model)
    with np.errstate(over="ignore", invalid="ignore"):
        background_q = np.einsum(
            "ij,ij->i", background, background, optimize=True
        )
    available = np.ones(BACKGROUND_MUTATIONS, dtype=bool)

    trace = np.full(
        (MAX_WALK_STEPS + 1, len(RETAINED_FRACTIONS)),
        np.nan,
        dtype=float,
    )
    steps_fixed = 0

    for time in range(MAX_WALK_STEPS + 1):
        current_effects = mutation_effects(position, probes, probe_q)
        for cut_index, indices in enumerate(retained_indices):
            trace[time, cut_index] = pearson(
                ancestral_effects[indices],
                current_effects[indices],
            )

        if time == MAX_WALK_STEPS:
            steps_fixed = MAX_WALK_STEPS
            break

        candidate_effects = mutation_effects(
            position, background, background_q
        )
        beneficial = (
            available
            & np.isfinite(candidate_effects)
            & (candidate_effects > 0.0)
        )
        beneficial_indices = np.flatnonzero(beneficial)
        if beneficial_indices.size == 0:
            steps_fixed = time
            break

        weights = candidate_effects[beneficial_indices]
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            steps_fixed = time
            break
        chosen = int(
            rng.choice(
                beneficial_indices,
                p=weights / total_weight,
            )
        )
        position += background[chosen]
        available[chosen] = False
        steps_fixed = time + 1

    return (
        trace,
        steps_fixed,
        float(np.min(ancestral_effects)),
        float(np.max(ancestral_effects)),
        float(np.linalg.norm(position)),
    )


def run_simulations(
    models: tuple[ModelConfig, ...],
) -> dict[str, np.ndarray]:
    """Run all deterministic replicates and return compact arrays."""
    correlations = np.full(
        (
            len(models),
            REPLICATES,
            MAX_WALK_STEPS + 1,
            len(RETAINED_FRACTIONS),
        ),
        np.nan,
        dtype=float,
    )
    walk_lengths = np.zeros((len(models), REPLICATES), dtype=np.int16)
    ancestral_minima = np.zeros((len(models), REPLICATES), dtype=float)
    ancestral_maxima = np.zeros((len(models), REPLICATES), dtype=float)
    endpoint_radii = np.zeros((len(models), REPLICATES), dtype=float)

    model_seed_sequences = np.random.SeedSequence(MASTER_SEED).spawn(
        len(models)
    )
    for model_index, (model, model_seed) in enumerate(
        zip(models, model_seed_sequences)
    ):
        replicate_seeds = model_seed.spawn(REPLICATES)
        print(
            f"Simulating {model.title}: n={model.n} "
            f"(fit {model.n_fit:.4f}), r={model.radius:.6g}, "
            f"sigma={model.sigma:.6g}, mu={model.mu}",
            flush=True,
        )
        for replicate, seed_sequence in enumerate(replicate_seeds):
            seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
            (
                trace,
                length,
                ancestor_minimum,
                ancestor_maximum,
                endpoint_radius,
            ) = simulate_replicate(model, seed)
            correlations[model_index, replicate] = trace
            walk_lengths[model_index, replicate] = length
            ancestral_minima[model_index, replicate] = ancestor_minimum
            ancestral_maxima[model_index, replicate] = ancestor_maximum
            endpoint_radii[model_index, replicate] = endpoint_radius
            if (replicate + 1) % 50 == 0:
                print(
                    f"  {replicate + 1}/{REPLICATES} replicates complete",
                    flush=True,
                )

    return {
        "correlations": correlations,
        "walk_lengths": walk_lengths,
        "ancestral_minima": ancestral_minima,
        "ancestral_maxima": ancestral_maxima,
        "endpoint_radii": endpoint_radii,
    }


def load_or_run(
    models: tuple[ModelConfig, ...],
    metadata: dict[str, object],
    cache_path: str,
) -> dict[str, np.ndarray]:
    """Reuse only a cache produced from the identical simulation settings."""
    metadata_text = json.dumps(metadata, sort_keys=True)
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_metadata = str(cache["metadata"].item())
            if cached_metadata == metadata_text:
                print(
                    f"Loading matching cache: {cache_path}",
                    flush=True,
                )
                return {
                    key: np.asarray(cache[key])
                    for key in (
                        "correlations",
                        "walk_lengths",
                        "ancestral_minima",
                        "ancestral_maxima",
                        "endpoint_radii",
                    )
                }
        print("Existing cache settings differ; recomputing.", flush=True)

    arrays = run_simulations(models)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        metadata=np.array(metadata_text),
        **arrays,
    )
    print(f"Saved simulation cache: {cache_path}", flush=True)
    return arrays


def validate_simulations(
    arrays: dict[str, np.ndarray],
    models: tuple[ModelConfig, ...],
) -> None:
    """Fail loudly on broken nesting, normalization, or insufficient walks."""
    correlations = arrays["correlations"]
    expected_shape = (
        len(models),
        REPLICATES,
        MAX_WALK_STEPS + 1,
        len(RETAINED_FRACTIONS),
    )
    if correlations.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected correlation array {correlations.shape}; "
            f"expected {expected_shape}."
        )
    if not np.allclose(correlations[:, :, 0, :], 1.0, atol=2.0e-12):
        raise RuntimeError("Not every ancestral correlation is one.")
    counts = np.array([
        np.sum(np.isfinite(correlations[model_index, :, last_time, 0]))
        for model_index, last_time in enumerate(PANEL_DISPLAY_STEPS)
    ])
    if np.any(counts < MIN_DISPLAY_REPLICATES):
        detail = ", ".join(
            f"{model.title}: {count}"
            for model, count in zip(models, counts)
        )
        raise RuntimeError(
            f"Too few walks reach the displayed endpoint: {detail}."
        )
    if np.any(arrays["ancestral_minima"] < OBSERVED_LOWER_CUT - 1.0e-12):
        raise RuntimeError("An observed probe escaped the ancestral lower cut.")


def aggregate_curves(
    correlations: np.ndarray,
    last_time: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median, central 68% interval, and contributing replicate count."""
    shown = correlations[:, :last_time + 1, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(shown, axis=0)
        lower = np.nanquantile(shown, 0.16, axis=0)
        upper = np.nanquantile(shown, 0.84, axis=0)
    count = np.sum(np.isfinite(shown[:, :, 0]), axis=0)
    return median, lower, upper, count


def endpoint_floor(
    correlations: np.ndarray,
    walk_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and central 68% interval of per-walk endpoint correlations."""
    endpoint_values = np.asarray([
        correlations[replicate, int(length), :]
        for replicate, length in enumerate(walk_lengths)
    ])
    return (
        np.nanmedian(endpoint_values, axis=0),
        np.nanquantile(endpoint_values, 0.16, axis=0),
        np.nanquantile(endpoint_values, 0.84, axis=0),
    )


def symmetric_interval_half_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Half-width of an interval, used for compact ``value +/- error`` labels."""
    return 0.5 * (upper - lower)


def floor_reaching_time(
    median_curve: np.ndarray,
    floor_values: np.ndarray,
) -> int:
    """First step where every curve is within 5% of its total decay to floor."""
    total_decay = np.maximum(np.abs(1.0 - floor_values), 1.0e-12)
    normalized_distance = (
        np.abs(median_curve - floor_values[np.newaxis, :])
        / total_decay[np.newaxis, :]
    )
    reached = np.flatnonzero(
        np.all(normalized_distance <= FLOOR_TOLERANCE, axis=1)
    )
    if reached.size == 0:
        raise RuntimeError(
            "The plotted range ends before all three curves reach their floors."
        )
    return int(reached[0])


def write_summary(
    arrays: dict[str, np.ndarray],
    models: tuple[ModelConfig, ...],
    metadata: dict[str, object],
    summary_path: str,
) -> dict[str, object]:
    """Write human-readable plotted medians and simulation diagnostics."""
    summary: dict[str, object] = {"metadata": metadata, "models": {}}
    times = [0, 1, 2, 5, MAX_DISPLAY_STEPS]
    for model_index, model in enumerate(models):
        model_correlations = arrays["correlations"][model_index]
        floor_values, floor_lower, floor_upper = endpoint_floor(
            model_correlations,
            arrays["walk_lengths"][model_index],
        )
        floor_errors = symmetric_interval_half_width(
            floor_lower,
            floor_upper,
        )
        median_curve, _, _, _ = aggregate_curves(
            model_correlations,
            PANEL_DISPLAY_STEPS[model_index],
        )
        floor_time = floor_reaching_time(median_curve, floor_values)
        plotted: dict[str, object] = {}
        model_times = sorted(set(times + [PANEL_DISPLAY_STEPS[model_index]]))
        for time in model_times:
            finite = np.isfinite(model_correlations[:, time, 0])
            values = model_correlations[finite, time, :]
            plotted[str(time)] = {
                "replicates": int(np.sum(finite)),
                "median": np.nanmedian(values, axis=0).tolist(),
                "quantile_16": np.nanquantile(values, 0.16, axis=0).tolist(),
                "quantile_84": np.nanquantile(values, 0.84, axis=0).tolist(),
            }
        model_summary = {
            "parameters": model.serializable(),
            "walk_length_quantiles": dict(
                zip(
                    ("minimum", "q16", "median", "q84", "maximum"),
                    np.quantile(
                        arrays["walk_lengths"][model_index],
                        (0.0, 0.16, 0.50, 0.84, 1.0),
                    ).tolist(),
                )
            ),
            "endpoint_radius_median": float(
                np.median(arrays["endpoint_radii"][model_index])
            ),
            "ancestor_effect_minimum_median": float(
                np.median(arrays["ancestral_minima"][model_index])
            ),
            "endpoint_correlation_floor": {
                "median": floor_values.tolist(),
                "quantile_16": floor_lower.tolist(),
                "quantile_84": floor_upper.tolist(),
                "symmetric_68_percent_error": floor_errors.tolist(),
            },
            "floor_tolerance_fraction_of_total_decay": FLOOR_TOLERANCE,
            "floor_reaching_time": floor_time,
            "curves": plotted,
        }
        summary["models"][model.key] = model_summary

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved summary: {summary_path}", flush=True)
    return summary


def print_summary(
    arrays: dict[str, np.ndarray],
    models: tuple[ModelConfig, ...],
) -> None:
    """Print the plotted values for a quick numerical audit."""
    for model_index, model in enumerate(models):
        lengths = arrays["walk_lengths"][model_index]
        floor_values, floor_lower, floor_upper = endpoint_floor(
            arrays["correlations"][model_index],
            lengths,
        )
        floor_errors = symmetric_interval_half_width(
            floor_lower,
            floor_upper,
        )
        median_curve, _, _, _ = aggregate_curves(
            arrays["correlations"][model_index],
            PANEL_DISPLAY_STEPS[model_index],
        )
        floor_time = floor_reaching_time(median_curve, floor_values)
        print(
            f"\n{model.title}: walk length "
            f"median={np.median(lengths):.0f}, "
            f"16-84%=[{np.quantile(lengths, 0.16):.0f}, "
            f"{np.quantile(lengths, 0.84):.0f}]"
        )
        formatted_floor = ", ".join(
            f"{label}={value:.3f}+/-{error:.3f}"
            for label, value, error in zip(
                ("r100", "r95", "r90"),
                floor_values,
                floor_errors,
            )
        )
        print(f"  endpoint floor at t~{floor_time}: {formatted_floor}")
        audit_times = sorted({
            0,
            1,
            2,
            5,
            MAX_DISPLAY_STEPS,
            PANEL_DISPLAY_STEPS[model_index],
        })
        for time in audit_times:
            values = arrays["correlations"][model_index, :, time, :]
            finite = np.isfinite(values[:, 0])
            medians = np.nanmedian(values[finite], axis=0)
            formatted = ", ".join(
                f"{label}={value:.3f}"
                for label, value in zip(
                    ("r100", "r95", "r90"), medians
                )
            )
            print(
                f"  t={time:2d}: N={np.sum(finite):3d}; {formatted}"
            )


def make_figure(
    arrays: dict[str, np.ndarray],
    models: tuple[ModelConfig, ...],
    out_pdf: str,
) -> None:
    """Render the two-panel poster figure and copy it into the poster."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.9),
        sharex=False,
        sharey=False,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.20,
        top=0.86,
        wspace=0.16,
    )

    all_lower = []
    all_upper = []
    for model_index, (model, axis) in enumerate(zip(models, axes)):
        last_time = PANEL_DISPLAY_STEPS[model_index]
        times = np.arange(last_time + 1)
        median, lower, upper, counts = aggregate_curves(
            arrays["correlations"][model_index],
            last_time,
        )
        floor_values, floor_lower, floor_upper = endpoint_floor(
            arrays["correlations"][model_index],
            arrays["walk_lengths"][model_index],
        )
        floor_errors = symmetric_interval_half_width(
            floor_lower,
            floor_upper,
        )
        floor_time = floor_reaching_time(median, floor_values)
        all_lower.append(lower)
        all_upper.append(upper)

        axis.axvline(
            floor_time,
            color=FLOOR_LINE_COLOR,
            linestyle=(0, (1.2, 2.2)),
            linewidth=2.0,
            zorder=2,
        )
        for cut_index, (color, linestyle) in enumerate(
            zip(CURVE_COLORS, CURVE_LINESTYLES)
        ):
            axis.fill_between(
                times,
                lower[:, cut_index],
                upper[:, cut_index],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                times,
                median[:, cut_index],
                color=color,
                linestyle=linestyle,
                linewidth=2.8,
                marker="o",
                markersize=4.2,
                markerfacecolor=color,
                markeredgewidth=0,
                zorder=3 + cut_index,
            )

        axis.set_title(f"{model.title} FGM", pad=9)
        axis.set_xlabel("Fixed background mutations")
        axis.set_xscale("linear")
        axis.set_yscale("linear")
        axis.set_xlim(0.0, last_time)
        tick_values = (
            list(range(0, last_time + 1, 2))
            if last_time <= 10
            else [0, 5, 10, 15, 20]
        )
        axis.xaxis.set_major_locator(FixedLocator(tick_values))
        axis.set_xticklabels([str(tick) for tick in tick_values])
        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=5.5,
            width=1.1,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3.0,
            width=0.9,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.13,
            1.12,
            chr(ord("A") + model_index),
            transform=axis.transAxes,
            fontsize=25,
            fontweight="bold",
            ha="left",
            va="top",
        )
        if int(np.min(counts)) < MIN_DISPLAY_REPLICATES:
            raise RuntimeError("A plotted point has insufficient replicates.")

        legend_handles = [
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=2.8,
                marker="o",
                markersize=4.2,
                markeredgewidth=0,
            )
            for color, linestyle in zip(
                CURVE_COLORS, CURVE_LINESTYLES
            )
        ]
        floor_labels = [
            (
                rf"$r_{{{int(100 * fraction)}\%}}^"
                rf"{{\mathrm{{floor}}}}={value:.3f}"
                rf"\pm{error:.3f}$"
            )
            for fraction, value, error in zip(
                RETAINED_FRACTIONS,
                floor_values,
                floor_errors,
            )
        ]
        legend = axis.legend(
            legend_handles,
            floor_labels,
            loc="lower left",
            bbox_to_anchor=(0.025, 0.025),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.90,
            handlelength=2.5,
            handletextpad=0.6,
            borderpad=0.35,
            labelspacing=0.3,
            fontsize=16.5,
        )
        legend.set_zorder(20)

    panel_a_lower = float(np.nanmin(all_lower[0]))
    panel_a_upper = float(np.nanmax(all_upper[0]))
    panel_a_y_lower = np.floor(20.0 * (panel_a_lower - 0.015)) / 20.0
    panel_a_y_upper = min(1.04, max(1.02, panel_a_upper + 0.02))
    axes[0].set_ylim(panel_a_y_lower, panel_a_y_upper)
    panel_a_ticks = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes[0].yaxis.set_major_locator(FixedLocator(panel_a_ticks))
    axes[0].set_yticklabels(
        ["-0.2", "0", "0.2", "0.4", "0.6", "0.8", "1"]
    )
    axes[0].set_ylabel("Pearson autocorrelation")

    # Panel B remains strictly positive, so give it a tighter independent
    # linear range instead of inheriting panel A's negative lower bound.
    axes[1].set_yscale("linear")
    axes[1].set_ylim(0.0, 1.04)
    panel_b_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes[1].yaxis.set_major_locator(FixedLocator(panel_b_ticks))
    axes[1].set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    os.makedirs(FIG_DIR, exist_ok=True)
    figure.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    print(f"Saved figure: {out_pdf}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate cutoff-dependent scrambling from Figure 3 fits."
        )
    )
    parser.add_argument(
        "--fit-source",
        choices=tuple(FIT_CONFIGS),
        default="without-errors",
        help=(
            "Use parameters inferred with or without measurement-error "
            "convolution (default: without-errors)."
        ),
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Copy the selected simulation figure into the poster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = FIT_CONFIGS[args.fit_source]
    models = load_model_configs(args.fit_source)
    for model in models:
        if model.n < 1:
            raise RuntimeError(f"Invalid rounded dimension for {model.title}.")

    metadata = build_metadata(models, args.fit_source)
    arrays = load_or_run(
        models,
        metadata,
        configuration["cache_path"],
    )
    validate_simulations(arrays, models)
    write_summary(
        arrays,
        models,
        metadata,
        configuration["summary_path"],
    )
    print_summary(arrays, models)
    make_figure(arrays, models, configuration["out_pdf"])
    if args.poster:
        os.makedirs(os.path.dirname(POSTER_PDF), exist_ok=True)
        shutil.copy2(configuration["out_pdf"], POSTER_PDF)
        print(f"Copied selected figure: {POSTER_PDF}", flush=True)


if __name__ == "__main__":
    main()
