#!/usr/bin/env python3
r"""Rank-matched Couce measurement noise on the fitted beta-prime FGM walk.

This is the Couce analogue of ``poster_fig5_limdi_noise.py``.  The mutation
model is the no-error Couce 0K radial beta-prime MLE.  Couce 0K and evolved
per-segment standard errors are assigned to simulated fitness effects by exact
effect rank, independently at both measurement endpoints.  Evolution always
uses latent effects.

The bare command runs 0K -> 2K.  ``--ancestor`` and ``--evolved`` between them
reach any ordered pair of the three sequenced timepoints, which is what TableS2
needs: 2K -> 15K starts from the background the population was actually on at
2K, so it is simulated from the 2K MLE rather than the 0K one.  Name that fit
with ``--fit-dataset couce_2K``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cmn import cmn_exper  # noqa: E402
from code_tmp import poster_fig5 as base  # noqa: E402
from code_tmp import walk_summary as single_noise  # noqa: E402
from code_tmp import poster_fig5_limdi_noise as limdi_noise  # noqa: E402

# All figures from code_tmp are scratch output; they go to code_tmp/out_tmp,
# never to figs_paper (which holds only the paper figures built by code_figs).
_OUT_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")



FIT_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "couce_ancestor_beta_prime_vs_iid_stable_no_errors_fit.json",
)
# The paper's own fit file, which holds one entry per fitted DFE.  The 0K entry here
# and ``FIT_PATH`` above are the same optimum to seven digits, so which one a 0K walk
# reads does not matter; 2K exists only here.
FIG3_FIT_PATH = os.path.join(REPO_ROOT, "data", "fig3_fgm_fits.json")
DATA_DIR = os.path.join(REPO_ROOT, "data", "FGM_HEAVY_TAILED")
FIG_DIR = _OUT_TMP

CACHE_VERSION = 3
MASTER_SEED = 260826
DEFAULT_REPLICATES = 500
DEFAULT_NOISE_REPLICATES = 10
DEFAULT_WORKERS = 4
DEFAULT_DISPLAY_STEPS = 20
# One colour per retained fraction.  ``base.CURVE_COLORS`` has three; a fourth rung
# (the 2% cut TableS2 reports) would otherwise be dropped silently by ``zip``.
CURVE_COLORS = base.CURVE_COLORS + ("#c94f7c",)
# Spell these exactly so 10% of a sample whose size is a multiple of ten does
# not become 9.999...% through floating-point subtraction before ``floor``.
TAIL_EXCLUSIONS = (0.00, 0.05, 0.10)
# See poster_fig5_limdi_noise.EXCLUSION_MODE for what these mean.
EXCLUSION_MODE = "signed"
# The reused Limdi simulation helpers read these module-level values.
limdi_noise.TAIL_EXCLUSIONS = TAIL_EXCLUSIONS
limdi_noise.EXCLUSION_MODE = EXCLUSION_MODE
# Mutations fixed DURING each interval, from ltee_muts as quoted in the main text:
# roughly 8 between 0K and 2K and 22 between 2K and 15K, hence 30 cumulative over
# 0K -> 15K.  Keyed by transition, not by endpoint: the previous ``{"15K": 22}`` gave
# 0K -> 15K the count that belongs to 2K -> 15K.
KNOWN_FIXED_MUTATIONS = {"0K -> 2K": 8, "2K -> 15K": 22, "0K -> 15K": 30}
COUCE_TIMEPOINTS = ("0K", "2K", "15K")
TIMEPOINT_GENERATIONS = {"0K": 0, "2K": 2000, "15K": 15000}


def load_model_config(
    fit_path: str = FIT_PATH,
    fit_dataset: str | None = None,
) -> base.ModelConfig:
    """Load a Couce radial beta-prime fit from either fit-file layout.

    Two layouts are accepted, because the two files predate each other.  A standalone
    fit exposes ``radial_beta_prime_fit`` at the top level; ``fig3_fgm_fits.json``
    holds one entry per fitted DFE under ``datasets``, and ``fit_dataset`` names which.
    Passing ``fit_dataset`` selects the second layout.

    Defaults to the 0K no-error MLE.  Pass an error-aware fit instead only when the
    walk is NOT going to re-add the published per-segment errors, or the measurement
    noise is counted twice.
    """
    with open(fit_path, encoding="utf-8") as handle:
        stored = json.load(handle)
    if fit_dataset is not None:
        entries = stored.get("datasets", {})
        if fit_dataset not in entries:
            raise RuntimeError(
                f"{fit_path} has no dataset {fit_dataset!r}; available: "
                f"{', '.join(sorted(entries))}"
            )
        fit = entries[fit_dataset]["heavy_tailed_full_mle"]["fit"]
    else:
        fit = stored["radial_beta_prime_fit"]
    return base.ModelConfig(
        key="beta_prime",
        title="Beta-prime",
        n_fit=float(fit["n"]),
        n=int(np.rint(fit["n"])),
        radius=float(fit["r"]),
        sigma=float(fit["sigma"]),
        mu=float(fit["mu"]),
    )


def load_noise_profile(evolved: str, ancestor: str = "0K") -> limdi_noise.NoiseProfile:
    """Load matched Couce segments and their fitted standard errors.

    Any ordered pair of sequenced timepoints is allowed, so 2K can be the early side.
    Segments are matched on ``alle`` exactly as TableS2 does.
    """
    for label, value in (("ancestor", ancestor), ("evolved", evolved)):
        if value not in COUCE_TIMEPOINTS:
            raise ValueError(
                f"Couce {label} timepoint must be one of {COUCE_TIMEPOINTS}, got {value!r}")
    if TIMEPOINT_GENERATIONS[evolved] <= TIMEPOINT_GENERATIONS[ancestor]:
        raise ValueError(f"{ancestor} -> {evolved} is not a forward transition")
    ancestor_effect = cmn_exper.load_couce_segment_series(ancestor)
    ancestor_error = cmn_exper.load_couce_segment_errors(ancestor)
    evolved_effect = cmn_exper.load_couce_segment_series(evolved)
    evolved_error = cmn_exper.load_couce_segment_errors(evolved)
    index = ancestor_effect.index.intersection(evolved_effect.index).sort_values()
    a = ancestor_effect.loc[index].to_numpy(float)
    a_error = ancestor_error.loc[index].to_numpy(float)
    b = evolved_effect.loc[index].to_numpy(float)
    b_error = evolved_error.loc[index].to_numpy(float)
    valid = (
        np.isfinite(a)
        & np.isfinite(a_error)
        & np.isfinite(b)
        & np.isfinite(b_error)
        & (a_error > 0.0)
        & (b_error > 0.0)
    )
    if int(np.sum(valid)) < 3:
        raise RuntimeError(
            f"Couce {ancestor} -> {evolved} has fewer than three usable segments")
    return limdi_noise.NoiseProfile(
        ancestor=ancestor,
        evolved=evolved,
        ancestor_effects=a[valid],
        ancestor_errors=a_error[valid],
        evolved_effects=b[valid],
        evolved_errors=b_error[valid],
    )


def empirical_ladder(profile: limdi_noise.NoiseProfile) -> dict[str, float]:
    """Observed Couce r100/r95/r90, ranked on the measured 0K effects."""
    kept_indices = limdi_noise.subset_indices(
        profile.ancestor_effects, TAIL_EXCLUSIONS, EXCLUSION_MODE)
    output = {}
    for excluded, kept in zip(TAIL_EXCLUSIONS, kept_indices):
        output[f"r{int(round(100 * (1.0 - excluded)))}"] = limdi_noise.pearson_1d(
            profile.ancestor_effects[kept], profile.evolved_effects[kept]
        )
    return output


def output_paths(
    profile: limdi_noise.NoiseProfile,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    tag: str = "",
) -> dict[str, str]:
    stem = (
        f"poster_fig5_couce_{profile.ancestor}_to_{profile.evolved}_beta_prime_"
        f"observed_window_rank_noise_k{error_scale:g}_w{replicates}_e{noise_replicates}_"
        f"m{profile.number}{('_' + tag) if tag else ''}"
    )
    return {
        "cache": os.path.join(DATA_DIR, f"{stem}.npz"),
        "summary": os.path.join(DATA_DIR, f"{stem}_summary.json"),
    }


def figure_path(
    ancestor: str,
    evolved: tuple[str, ...],
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    tag: str = "",
) -> str:
    endpoint = "_and_".join(evolved)
    stem = (
        f"poster_fig5_couce_{ancestor}_to_{endpoint}_beta_prime_"
        f"observed_window_rank_noise_k{error_scale:g}_w{replicates}_e{noise_replicates}"
        f"{('_' + tag) if tag else ''}.pdf"
    )
    return os.path.join(FIG_DIR, stem)


def build_metadata(
    model: base.ModelConfig,
    profile: limdi_noise.NoiseProfile,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
    display_steps: int,
    fit_path: str = FIT_PATH,
    fit_dataset: str | None = None,
) -> dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "master_seed": MASTER_SEED,
        "fit_path": os.path.relpath(fit_path, REPO_ROOT),
        "fit_dataset": fit_dataset,
        "model": model.serializable(),
        "profile": profile.serializable(),
        "empirical_correlations": empirical_ladder(profile),
        "known_fixed_background_mutations": KNOWN_FIXED_MUTATIONS.get(
            f"{profile.ancestor} -> {profile.evolved}"
        ),
        "replicates": replicates,
        "noise_replicates_per_walk": noise_replicates,
        "max_walk_steps": max_walk_steps,
        "display_steps": display_steps,
        "retained_fractions": [1.0 - fraction for fraction in TAIL_EXCLUSIONS],
        "tail_exclusions": list(TAIL_EXCLUSIONS),
        "exclusion_mode": EXCLUSION_MODE,
        "probe_mutations": profile.number,
        "background_mutations": base.BACKGROUND_MUTATIONS,
        "probe_ancestral_effect_window": [
            float(np.min(profile.ancestor_effects)),
            float(np.max(profile.ancestor_effects)),
        ],
        "fitness_convention": "F(x)=-|x|^2/2",
        "effect_convention": "s=-x.delta-|delta|^2/2",
        "probe_ascertainment": (
            "latent ancestral s restricted to the exact minimum and maximum of the "
            "matched empirical 0K probe library"
        ),
        "background_ascertainment": "unconditioned",
        "background_protocol": (
            "independent finite mutation pool; beneficial mutations fixed without "
            "replacement with probability proportional to latent s"
        ),
        "measurement_noise": (
            "independent Gaussian with Couce sterr1 assigned by exact effect rank"
        ),
        "ancestor_measurement": "drawn once per observation replicate and fixed along curve",
        "endpoint_measurement": "fresh draw at every walk step",
        "ancestor_error_assignment": "exact empirical 0K-effect rank",
        "endpoint_error_assignment": "exact empirical current-effect rank",
        "observed_subset_definition": "ranked once from noisy 0K measurement and held fixed",
        "error_scale": error_scale,
        "evolution_uses_measurement_noise": False,
    }


def draw_probe_library(
    rng: np.random.Generator,
    model: base.ModelConfig,
    initial_position: np.ndarray,
    profile: limdi_noise.NoiseProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw probes only inside the matched empirical 0K effect window."""
    lower = float(np.min(profile.ancestor_effects))
    upper = float(np.max(profile.ancestor_effects))
    accepted: list[np.ndarray] = []
    accepted_number = 0
    while accepted_number < profile.number:
        missing = profile.number - accepted_number
        batch = base.draw_mutations(rng, max(4096, 2 * missing), model)
        effects = base.mutation_effects(initial_position, batch)
        keep = (
            np.isfinite(effects)
            & (effects >= lower)
            & (effects <= upper)
        )
        if np.any(keep):
            selected = batch[keep]
            accepted.append(selected)
            accepted_number += selected.shape[0]
    probes = np.concatenate(accepted, axis=0)[: profile.number]
    squared_lengths = np.einsum("ij,ij->i", probes, probes, optimize=True)
    return probes, squared_lengths


def simulate_replicate(
    model: base.ModelConfig,
    profile: limdi_noise.NoiseProfile,
    biological_seed: int,
    observation_seed: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """One beta-prime walk with empirically windowed probes and noisy observations."""
    biological_rng = np.random.default_rng(biological_seed)
    observation_rng = np.random.default_rng(observation_seed)

    position = np.zeros(model.n, dtype=float)
    position[0] = model.radius
    probes, probe_q = draw_probe_library(
        biological_rng, model, position, profile
    )
    ancestral_effects = base.mutation_effects(position, probes, probe_q)

    latent_indices = limdi_noise.subset_indices(
        ancestral_effects, TAIL_EXCLUSIONS, EXCLUSION_MODE)
    ancestor_sigma = limdi_noise.assign_errors_by_rank(
        ancestral_effects, profile.ancestor_errors_by_rank
    )
    observed_ancestor = (
        ancestral_effects[np.newaxis, :]
        + error_scale
        * ancestor_sigma[np.newaxis, :]
        * observation_rng.normal(size=(noise_replicates, profile.number))
    )
    observed_indices = limdi_noise.subset_indices(
        observed_ancestor, TAIL_EXCLUSIONS, EXCLUSION_MODE)

    # The empirical window describes the assayed library, not the mutations
    # available to adaptation, so the background pool remains unconditioned.
    background = base.draw_mutations(
        biological_rng, base.BACKGROUND_MUTATIONS, model
    )
    background_q = np.einsum("ij,ij->i", background, background, optimize=True)
    available = np.ones(base.BACKGROUND_MUTATIONS, dtype=bool)

    latent_trace = np.full(
        (max_walk_steps + 1, len(TAIL_EXCLUSIONS)), np.nan, dtype=float
    )
    observed_trace = np.full(
        (noise_replicates, max_walk_steps + 1, len(TAIL_EXCLUSIONS)),
        np.nan,
        dtype=np.float32,
    )
    steps_fixed = 0
    walk_finished = False
    for time in range(max_walk_steps + 1):
        current_effects = base.mutation_effects(position, probes, probe_q)
        evolved_sigma = limdi_noise.assign_errors_by_rank(
            current_effects, profile.evolved_errors_by_rank
        )
        observed_current = (
            current_effects[np.newaxis, :]
            + error_scale
            * evolved_sigma[np.newaxis, :]
            * observation_rng.normal(size=(noise_replicates, profile.number))
        )
        for cut, (latent_kept, observed_kept) in enumerate(
            zip(latent_indices, observed_indices)
        ):
            latent_trace[time, cut] = limdi_noise.pearson_1d(
                ancestral_effects[latent_kept], current_effects[latent_kept]
            )
            observed_trace[:, time, cut] = limdi_noise.pearson_rows_at_indices(
                observed_ancestor, observed_current, observed_kept
            )

        if time == max_walk_steps:
            if not walk_finished:
                steps_fixed = max_walk_steps
            break
        if walk_finished:
            # The genotype no longer changes, but fresh endpoint measurement
            # noise is still drawn above.  This exposes the true steady plateau.
            continue
        candidate_effects = base.mutation_effects(
            position, background, background_q
        )
        beneficial_indices = np.flatnonzero(
            available & np.isfinite(candidate_effects) & (candidate_effects > 0.0)
        )
        if beneficial_indices.size == 0:
            steps_fixed = time
            walk_finished = True
            continue
        weights = candidate_effects[beneficial_indices]
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            steps_fixed = time
            walk_finished = True
            continue
        chosen = int(
            biological_rng.choice(beneficial_indices, p=weights / total_weight)
        )
        position += background[chosen]
        available[chosen] = False
        steps_fixed = time + 1
    return latent_trace, observed_trace, steps_fixed, float(np.linalg.norm(position))


def _simulate_task(arguments):
    return simulate_replicate(*arguments)


def run_simulations(
    model: base.ModelConfig,
    profile: limdi_noise.NoiseProfile,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
    workers: int,
) -> dict[str, np.ndarray]:
    latent = np.full(
        (replicates, max_walk_steps + 1, len(TAIL_EXCLUSIONS)),
        np.nan,
        dtype=float,
    )
    observed = np.full(
        (replicates, noise_replicates, max_walk_steps + 1, len(TAIL_EXCLUSIONS)),
        np.nan,
        dtype=np.float32,
    )
    walk_lengths = np.zeros(replicates, dtype=np.int16)
    endpoint_radii = np.zeros(replicates, dtype=float)

    # A stable per-transition stream.  Keying on the endpoint alone -- which is what
    # this did before 2K became a possible early side -- would have given 0K -> 15K and
    # 2K -> 15K the same draws.  The 0K rows keep their original single-element key so
    # that caches written before the ancestor was a parameter still reproduce exactly;
    # any other early side appends its own generation.
    endpoint_code = TIMEPOINT_GENERATIONS[profile.evolved] // 1000
    key = ([MASTER_SEED, endpoint_code] if profile.ancestor == "0K" else
           [MASTER_SEED, endpoint_code,
            TIMEPOINT_GENERATIONS[profile.ancestor] // 1000])
    sequences = np.random.SeedSequence(key).spawn(replicates)
    tasks = []
    for sequence in sequences:
        biological, observation = sequence.spawn(2)
        tasks.append((
            model,
            profile,
            int(biological.generate_state(1, dtype=np.uint64)[0]),
            int(observation.generate_state(1, dtype=np.uint64)[0]),
            noise_replicates,
            error_scale,
            max_walk_steps,
        ))

    print(
        f"Simulating Couce {profile.ancestor} -> {profile.evolved}: {replicates} walks x "
        f"{noise_replicates} observations, N={profile.number}, n={model.n}, "
        f"workers={workers}",
        flush=True,
    )
    executor = None
    if workers == 1:
        results = map(_simulate_task, tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(_simulate_task, tasks, chunksize=1)
    try:
        for replicate, (latent_trace, observed_trace, length, radius) in enumerate(
            results
        ):
            latent[replicate] = latent_trace
            observed[replicate] = observed_trace
            walk_lengths[replicate] = length
            endpoint_radii[replicate] = radius
            if (replicate + 1) % 25 == 0 or replicate + 1 == replicates:
                print(f"  {replicate + 1}/{replicates} walks complete", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    return {
        "latent_correlations": latent,
        "observed_correlations": observed,
        "walk_lengths": walk_lengths,
        "endpoint_radii": endpoint_radii,
    }


def load_or_run(
    model: base.ModelConfig,
    profile: limdi_noise.NoiseProfile,
    metadata: dict[str, object],
    cache_path: str,
    force: bool,
    workers: int,
) -> dict[str, np.ndarray]:
    metadata_text = json.dumps(metadata, sort_keys=True)
    keys = (
        "latent_correlations",
        "observed_correlations",
        "walk_lengths",
        "endpoint_radii",
    )
    if os.path.exists(cache_path) and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["metadata"].item()) == metadata_text:
                print(f"Loading matching cache: {cache_path}", flush=True)
                return {key: np.asarray(cached[key]) for key in keys}
        print("Existing cache metadata differ; recomputing.", flush=True)
    arrays = run_simulations(
        model=model,
        profile=profile,
        replicates=int(metadata["replicates"]),
        noise_replicates=int(metadata["noise_replicates_per_walk"]),
        error_scale=float(metadata["error_scale"]),
        max_walk_steps=int(metadata["max_walk_steps"]),
        workers=workers,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, metadata=np.array(metadata_text), **arrays)
    print(f"Saved cache: {cache_path}", flush=True)
    return arrays


def make_figure(
    results: dict[str, tuple[dict[str, np.ndarray], limdi_noise.NoiseProfile]],
    model: base.ModelConfig,
    error_scale: float,
    display_steps: int,
    path: str,
) -> None:
    figure, axes = plt.subplots(
        1, len(results), figsize=(6.2 * len(results), 5.0), sharey=True
    )
    if len(results) == 1:
        axes = np.asarray([axes])
    global_minimum = 1.0
    for axis, (transition, (arrays, profile)) in zip(axes, results.items()):
        last_time = min(display_steps, arrays["latent_correlations"].shape[1] - 1)
        times = np.arange(last_time + 1)
        latent = arrays["latent_correlations"][:, : last_time + 1, :]
        observed = arrays["observed_correlations"][:, :, : last_time + 1, :]
        observed_flat = observed.reshape(-1, observed.shape[-2], observed.shape[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            latent_median = np.nanmedian(latent, axis=0)
            observed_median = np.nanmedian(observed_flat, axis=0)
            observed_lower = np.nanquantile(observed_flat, 0.16, axis=0)
            observed_upper = np.nanquantile(observed_flat, 0.84, axis=0)
        global_minimum = min(global_minimum, float(np.nanmin(observed_lower)))
        for cut, color in enumerate(CURVE_COLORS[:len(TAIL_EXCLUSIONS)]):
            axis.fill_between(
                times,
                observed_lower[:, cut],
                observed_upper[:, cut],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axis.plot(times, latent_median[:, cut], color=color, linewidth=2.7)
            axis.plot(
                times,
                observed_median[:, cut],
                color=color,
                linewidth=2.5,
                linestyle=(0, (4.0, 2.4)),
            )
        observed_data = empirical_ladder(profile)
        fixed_mutations = KNOWN_FIXED_MUTATIONS.get(transition)
        if fixed_mutations is not None and fixed_mutations <= last_time:
            axis.axvline(
                fixed_mutations,
                color="#777777",
                linewidth=1.3,
                linestyle=(0, (2.0, 2.5)),
                zorder=1,
            )
            for key, color in zip(tuple(observed_data), CURVE_COLORS):
                axis.scatter(
                    [fixed_mutations],
                    [observed_data[key]],
                    s=48,
                    marker="o",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=7,
                )
            axis.text(
                fixed_mutations + 0.35,
                0.02,
                f"{fixed_mutations} fixed",
                rotation=90,
                ha="left",
                va="bottom",
                fontsize=10,
                color="#666666",
            )
        axis.text(
            0.97,
            0.96,
            "observed data: "
            + ", ".join(f"{key}={value:.3f}" for key, value in observed_data.items()),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color="#444444",
        )
        axis.set_title(
            f"Couce {transition.replace(' -> ', ' $\\rightarrow$ ')} "
            f"($n={model.n}$; fit $n={model.n_fit:.2f}$; $k={error_scale:g}$)"
        )
        axis.set_xlabel("Fixed background mutations")
        axis.set_xlim(0, last_time)
        ticks = list(range(0, last_time + 1, 5))
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=5.5, width=1.1)
    axes[0].set_ylabel("Pearson autocorrelation")
    lower_limit = min(-0.05, np.floor(20.0 * (global_minimum - 0.02)) / 20.0)
    axes[0].set_ylim(lower_limit, 1.04)

    fraction_handles = [
        Line2D([], [], color=color, linewidth=2.7)
        for color in CURVE_COLORS[:len(TAIL_EXCLUSIONS)]
    ]
    semantic_handles = [
        Line2D([], [], color="#555555", linewidth=2.7),
        Line2D([], [], color="#555555", linewidth=2.5, linestyle=(0, (4.0, 2.4))),
    ]
    labels = [
        *(f"r{int(round(100 * (1.0 - fraction)))}" for fraction in TAIL_EXCLUSIONS),
        "latent",
        "rank-matched measurement noise",
    ]
    if any(transition in KNOWN_FIXED_MUTATIONS for transition in results):
        semantic_handles.append(
            Line2D(
                [], [], marker="o", linestyle="none", markersize=7,
                markerfacecolor="#777777", markeredgecolor="white",
            )
        )
        labels.append("observed data at known substitutions")
    figure.legend(
        fraction_handles + semantic_handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3 if len(labels) > 5 else 5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.1,
    )
    figure.suptitle(
        "Couce beta-prime FGM - probes restricted to observed 0K range",
        y=1.13,
        fontsize=19,
    )
    figure.subplots_adjust(
        left=0.085 if len(results) > 1 else 0.14,
        right=0.985,
        bottom=0.16,
        top=0.80,
        wspace=0.16,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Saved figure: {path}", flush=True)


def apply_exclusion_options(args) -> None:
    """Install any requested subset rule, here and in the reused Limdi helpers."""
    global TAIL_EXCLUSIONS, EXCLUSION_MODE
    if args.exclusions is not None:
        exclusions = tuple(float(value) for value in args.exclusions)
        if not all(0.0 <= value < 1.0 for value in exclusions):
            raise ValueError("exclusion fractions must lie in [0, 1)")
        TAIL_EXCLUSIONS = exclusions
    if args.exclusion_mode is not None:
        EXCLUSION_MODE = args.exclusion_mode
    limdi_noise.TAIL_EXCLUSIONS = TAIL_EXCLUSIONS
    limdi_noise.EXCLUSION_MODE = EXCLUSION_MODE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ancestor", choices=COUCE_TIMEPOINTS, default="0K")
    parser.add_argument("--evolved", choices=("2K", "15K", "both"), default="2K")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--noise-replicates", type=int, default=DEFAULT_NOISE_REPLICATES
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=1.0,
        help="multiplier applied to the Couce per-segment sterr1",
    )
    parser.add_argument("--max-walk-steps", type=int, default=base.MAX_WALK_STEPS)
    parser.add_argument("--display-steps", type=int, default=DEFAULT_DISPLAY_STEPS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fit-json",
        default=None,
        help="beta-prime fit to simulate; default is the 0K no-error MLE",
    )
    parser.add_argument(
        "--fit-dataset",
        default=None,
        help=("dataset key inside a fig3_fgm_fits.json-style file, e.g. couce_2K; "
              "implies --fit-json data/fig3_fgm_fits.json when that is not given"),
    )
    parser.add_argument("--exclusions", type=float, nargs="+", default=None)
    parser.add_argument(
        "--exclusion-mode", choices=("signed", "magnitude"), default=None)
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.replicates < 1 or args.noise_replicates < 1 or args.workers < 1:
        raise ValueError("replicate, noise-replicate, and worker counts must be positive")
    if args.error_scale < 0.0:
        raise ValueError("error scale must be non-negative")
    if args.max_walk_steps < 1 or not 1 <= args.display_steps <= args.max_walk_steps:
        raise ValueError("walk steps must be positive and display steps must not exceed them")

    apply_exclusion_options(args)
    if (args.fit_json is not None or args.fit_dataset is not None) and not args.tag:
        raise ValueError("a non-default fit requires --tag so outputs stay separate")
    fit_path = args.fit_json or (FIG3_FIT_PATH if args.fit_dataset else FIT_PATH)
    model = load_model_config(fit_path, args.fit_dataset)
    evolved = ("2K", "15K") if args.evolved == "both" else (args.evolved,)
    evolved = tuple(
        endpoint for endpoint in evolved
        if TIMEPOINT_GENERATIONS[endpoint] > TIMEPOINT_GENERATIONS[args.ancestor])
    if not evolved:
        raise ValueError(f"no forward endpoint from {args.ancestor}")
    results = {}
    for endpoint in evolved:
        profile = load_noise_profile(endpoint, args.ancestor)
        metadata = build_metadata(
            model=model,
            profile=profile,
            replicates=args.replicates,
            noise_replicates=args.noise_replicates,
            error_scale=args.error_scale,
            max_walk_steps=args.max_walk_steps,
            display_steps=args.display_steps,
            fit_path=fit_path,
            fit_dataset=args.fit_dataset,
        )
        paths = output_paths(
            profile,
            args.replicates,
            args.noise_replicates,
            args.error_scale,
            args.tag,
        )
        arrays = load_or_run(
            model=model,
            profile=profile,
            metadata=metadata,
            cache_path=paths["cache"],
            force=args.force,
            workers=args.workers,
        )
        single_noise.validate(
            arrays=arrays,
            profile=profile,
            replicates=args.replicates,
            noise_replicates=args.noise_replicates,
            error_scale=args.error_scale,
            max_walk_steps=args.max_walk_steps,
            cuts=len(TAIL_EXCLUSIONS),
        )
        # The empirical ladder is dataset-specific, so the driver computes it.
        summary = single_noise.write_summary(
            arrays, metadata, empirical_ladder(profile), paths["summary"]
        )
        transition = f"{args.ancestor} -> {endpoint}"
        print(f"\nCouce {transition}")
        single_noise.print_summary(summary)
        results[transition] = (arrays, profile)

    path = figure_path(
        args.ancestor,
        evolved,
        args.replicates,
        args.noise_replicates,
        args.error_scale,
        args.tag,
    )
    make_figure(
        results=results,
        model=model,
        error_scale=args.error_scale,
        display_steps=args.display_steps,
        path=path,
    )


if __name__ == "__main__":
    main()
