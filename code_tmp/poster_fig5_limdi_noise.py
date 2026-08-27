#!/usr/bin/env python3
r"""Rank-matched Limdi measurement noise on fitted FGM adaptive walks.

The first application is REL607 -> Ara+2, but the implementation is parameterized by
``--ancestor`` and ``--evolved`` so the same observation layer can later be run for every Limdi
transition.

For one empirical transition, genes are matched exactly as in Table S1.  Simulated mutations and
empirical genes are sorted by fitness effect, and the published per-gene errors are assigned
rank-for-rank:

* ancestral errors are assigned from the latent ancestral-effect ranks;
* evolved-library errors are reassigned from the current latent-effect ranks at every walk step.

One noisy ancestor measurement is drawn and held fixed along each observed curve.  The 100/95/90
subsets are defined once from that noisy ancestor, while independent evolved measurements are drawn
at every step.  Evolution itself always uses latent FGM effects and is therefore unchanged by the
observation layer.

Example:

    python code_tmp/poster_fig5_limdi_noise.py \
        --ancestor REL607 --evolved Ara+2 --fit-source without-errors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from code_tmp import poster_fig5 as base  # noqa: E402
from cmn.cmn_exper import LIMDI_EVOLVED  # noqa: E402
from code_figs import TableS1_limdi_autocorr as table_s1  # noqa: E402

# All figures from code_tmp are scratch output; they go to code_tmp/out_tmp,
# never to figs_paper (which holds only the paper figures built by code_figs).
_OUT_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tmp")



CACHE_VERSION = 1
MASTER_SEED = 260821
DEFAULT_REPLICATES = 500
DEFAULT_NOISE_REPLICATES = 10
TAIL_EXCLUSIONS = tuple(1.0 - fraction for fraction in base.RETAINED_FRACTIONS)
# How the retained subsets are defined.  "signed" ranks on the signed effect and drops the
# most deleterious fraction, which is the rule every cache written before this option existed
# was built under.  "magnitude" ranks on |effect| and drops the largest-magnitude fraction,
# which is the rule cmn_scatter applies in fig1 and figs S1-S4; a figure that puts measured
# dots on these curves has to use whichever rule the curves were simulated with.
EXCLUSION_MODE = "signed"


def subset_indices(values, exclusions, mode):
    """Kept-index arrays for each exclusion fraction, over the last axis of ``values``.

    Works for a single ancestral trace (1-D) and for a stack of noisy ancestral measurements
    (2-D, one row per noise replicate) alike.
    """
    if mode not in ("signed", "magnitude"):
        raise ValueError(f"unknown exclusion mode {mode!r}")
    values = np.asarray(values, dtype=float)
    number = values.shape[-1]
    ranked = np.argsort(
        values if mode == "signed" else np.abs(values), axis=-1, kind="stable"
    )
    kept = []
    for fraction in exclusions:
        dropped = int(np.floor(fraction * number))
        # signed: the dropped effects are the most deleterious, at the head of the order.
        # magnitude: they are the largest in absolute value, at the tail of it.
        kept.append(ranked[..., dropped:] if mode == "signed"
                    else ranked[..., : number - dropped])
    return kept


@dataclass(frozen=True)
class NoiseProfile:
    """One matched Limdi founder/clone profile with errors ordered by empirical rank."""

    ancestor: str
    evolved: str
    ancestor_effects: np.ndarray
    ancestor_errors: np.ndarray
    evolved_effects: np.ndarray
    evolved_errors: np.ndarray

    @property
    def number(self) -> int:
        return int(self.ancestor_effects.size)

    @property
    def ancestor_errors_by_rank(self) -> np.ndarray:
        order = np.argsort(self.ancestor_effects, kind="stable")
        return self.ancestor_errors[order]

    @property
    def evolved_errors_by_rank(self) -> np.ndarray:
        order = np.argsort(self.evolved_effects, kind="stable")
        return self.evolved_errors[order]

    def serializable(self) -> dict[str, object]:
        return {
            "ancestor": self.ancestor,
            "evolved": self.evolved,
            "transition": f"{self.ancestor} -> {self.evolved}",
            "matched_genes": self.number,
            "ancestor_effect_range": [
                float(np.min(self.ancestor_effects)),
                float(np.max(self.ancestor_effects)),
            ],
            "evolved_effect_range": [
                float(np.min(self.evolved_effects)),
                float(np.max(self.evolved_effects)),
            ],
            "ancestor_error_quantiles": np.quantile(
                self.ancestor_errors, (0.0, 0.16, 0.5, 0.84, 1.0)
            ).tolist(),
            "evolved_error_quantiles": np.quantile(
                self.evolved_errors, (0.0, 0.16, 0.5, 0.84, 1.0)
            ).tolist(),
            "profile_hash": profile_hash(self),
        }


def profile_hash(profile: NoiseProfile) -> str:
    """Stable digest of the empirical effects and errors used by the observation layer."""
    digest = hashlib.sha256()
    for array in (
        profile.ancestor_effects,
        profile.ancestor_errors,
        profile.evolved_effects,
        profile.evolved_errors,
    ):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_noise_profile(ancestor: str, evolved: str) -> NoiseProfile:
    """Load one valid Limdi transition using the table's exact gene matching."""
    if ancestor not in LIMDI_EVOLVED or evolved not in LIMDI_EVOLVED[ancestor]:
        expected = ", ".join(
            f"{founder} -> {clone}"
            for founder, clones in LIMDI_EVOLVED.items()
            for clone in clones
        )
        raise ValueError(
            f"{ancestor} -> {evolved} is not a Limdi transition; expected one of: {expected}"
        )

    a, a_error, b, b_error = (
        np.asarray(value, dtype=float)
        for value in table_s1.limdi_pair(ancestor, evolved)
    )
    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(a_error)
        & np.isfinite(b_error)
        & (a_error > 0.0)
        & (b_error > 0.0)
    )
    if int(np.sum(valid)) < 3:
        raise RuntimeError(f"{ancestor} -> {evolved} has fewer than three usable genes")
    return NoiseProfile(
        ancestor=ancestor,
        evolved=evolved,
        ancestor_effects=a[valid],
        ancestor_errors=a_error[valid],
        evolved_effects=b[valid],
        evolved_errors=b_error[valid],
    )


def safe_label(value: str) -> str:
    """Filesystem-safe population or mode label."""
    value = value.replace("+", "_plus_").replace("-", "_minus_")
    return re.sub(r"[^A-Za-z0-9_.]+", "_", value).strip("_")


def output_paths(
    profile: NoiseProfile,
    fit_source: str,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    tag: str = "",
) -> dict[str, str]:
    stem = (
        f"poster_fig5_limdi_{safe_label(profile.ancestor)}_to_{safe_label(profile.evolved)}_"
        f"{fit_source.replace('-', '_')}_rank_noise_k{error_scale:g}_"
        f"w{replicates}_e{noise_replicates}_m{profile.number}"
        f"{('_' + safe_label(tag)) if tag else ''}"
    )
    return {
        "cache": os.path.join(base.DATA_DIR, f"{stem}.npz"),
        "summary": os.path.join(base.DATA_DIR, f"{stem}_summary.json"),
        "figure": os.path.join(_OUT_TMP, f"{stem}.pdf"),
    }


def heavy_config_from_profile(path: str, dimension: int) -> base.ModelConfig:
    """One heavy-tailed model taken from a profile entry of a stored fit.

    The dimension is not identified in these beta-prime fits, so a fit file
    carries a whole ridge rather than a single answer.  Simulating "the fit"
    means naming the point on that ridge, and taking r, sigma and mu from the
    entry re-optimised at that n -- reusing another n's r and sigma would move
    the DFE far more than changing n does.
    """
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    profile = stored.get("profile_likelihood", {})
    key = str(int(dimension))
    if key not in profile:
        raise RuntimeError(
            f"{path} has no profile entry at n={dimension}; available: "
            f"{', '.join(sorted(profile, key=int))}"
        )
    entry = profile[key]
    return base.ModelConfig(
        key="heavy_tailed",
        title=f"Heavy-tailed (n={int(dimension)})",
        n_fit=float(entry["n"]),
        n=int(round(entry["n"])),
        radius=float(entry["r"]),
        sigma=float(entry["sigma"]),
        mu=float(entry["mu"]),
    )


def load_model_configs(
    fit_source: str,
    ancestor: str,
) -> tuple[base.ModelConfig, base.ModelConfig]:
    """Load the requested ancestor's FGM fits without silently reusing REL607.

    The no-error fit JSON contains independent REL606 and REL607 fits.  The current error-aware
    poster fit contains REL607 only, so requesting an unavailable REL606 error-aware fit is an
    explicit error rather than an accidental REL607 substitution.
    """
    configuration = base.FIT_CONFIGS[fit_source]
    with open(configuration["fit_path"], encoding="utf-8") as handle:
        fit_output = json.load(handle)

    if fit_source == "without-errors":
        populations = fit_output.get("populations", {})
        if ancestor not in populations:
            raise RuntimeError(
                f"{configuration['fit_path']} has no no-error fit for {ancestor}"
            )
        result = populations[ancestor]
        canonical = result["canonical_full_mle"]["fit"]
        heavy = result["heavy_tailed_full_mle"]["fit"]
    else:
        fitted_ancestor = fit_output.get("dataset", {}).get("name")
        if fitted_ancestor != ancestor:
            raise RuntimeError(
                f"The error-aware fit contains {fitted_ancestor}, not {ancestor}; "
                f"fit {ancestor} before running this mode."
            )
        canonical = fit_output["canonical_moment_constrained_mle"]["fit"]
        heavy = fit_output["heavy_tailed_log_fitness_free_mu_mle"][
            "with_gene_specific_errors"
        ]["fit"]

    return (
        base.ModelConfig(
            key="canonical",
            title="Gaussian (canonical)",
            n_fit=float(canonical["n"]),
            n=int(np.rint(canonical["n"])),
            radius=float(canonical["r"]),
            sigma=float(canonical["sigma"]),
            mu=None,
        ),
        base.ModelConfig(
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
    profile: NoiseProfile,
    models: tuple[base.ModelConfig, ...],
    fit_source: str,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
) -> dict[str, object]:
    """Complete cache identity and a human-readable description of the simulation."""
    return {
        "cache_version": CACHE_VERSION,
        "master_seed": MASTER_SEED,
        "fit_source": fit_source,
        "models": [model.serializable() for model in models],
        "profile": profile.serializable(),
        "replicates": replicates,
        "noise_replicates_per_walk": noise_replicates,
        "max_walk_steps": max_walk_steps,
        "retained_fractions": [1.0 - fraction for fraction in TAIL_EXCLUSIONS],
        "tail_exclusions": list(TAIL_EXCLUSIONS),
        "exclusion_mode": EXCLUSION_MODE,
        "probe_mutations": profile.number,
        "background_mutations": base.BACKGROUND_MUTATIONS,
        "observed_lower_cut": base.OBSERVED_LOWER_CUT,
        "probe_ascertainment": "latent ancestral s >= lower cut, as in poster_fig5.py",
        "error_scale": error_scale,
        "measurement_noise": "independent Gaussian with rank-matched Limdi per-gene sigma",
        "ancestor_measurement": "drawn once per noise replicate and fixed along the curve",
        "endpoint_measurement": "fresh draw at each walk step",
        "ancestor_error_assignment": "exact empirical ancestor-effect rank",
        "endpoint_error_assignment": "exact empirical current-effect rank",
        "observed_subset_definition": "ranked once from noisy ancestor and held fixed",
        "evolution_uses_measurement_noise": False,
    }


def draw_probe_library(
    rng: np.random.Generator,
    model: base.ModelConfig,
    initial_position: np.ndarray,
    number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``number`` probes with the same ascertainment as the existing Figure 5."""
    accepted: list[np.ndarray] = []
    accepted_number = 0
    while accepted_number < number:
        missing = number - accepted_number
        batch = base.draw_mutations(rng, max(4096, 2 * missing), model)
        effects = base.mutation_effects(initial_position, batch)
        keep = np.isfinite(effects) & (effects >= base.OBSERVED_LOWER_CUT)
        if np.any(keep):
            accepted_batch = batch[keep]
            accepted.append(accepted_batch)
            accepted_number += accepted_batch.shape[0]

    probes = np.concatenate(accepted, axis=0)[:number]
    squared_lengths = np.einsum("ij,ij->i", probes, probes, optimize=True)
    return probes, squared_lengths


def assign_errors_by_rank(values: np.ndarray, errors_by_rank: np.ndarray) -> np.ndarray:
    """Give each value the error at the identical empirical fitness-effect rank."""
    values = np.asarray(values, dtype=float)
    errors_by_rank = np.asarray(errors_by_rank, dtype=float)
    if values.ndim != 1 or errors_by_rank.shape != values.shape:
        raise ValueError("values and errors_by_rank must be equal-length one-dimensional arrays")
    order = np.argsort(values, kind="stable")
    assigned = np.empty_like(errors_by_rank)
    assigned[order] = errors_by_rank
    return assigned


def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two finite one-dimensional arrays."""
    da = a - np.mean(a)
    db = b - np.mean(b)
    denominator = np.sqrt(np.sum(da * da) * np.sum(db * db))
    return float(np.sum(da * db) / denominator) if denominator > 0.0 else np.nan


def pearson_rows_at_indices(
    a: np.ndarray,
    b: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Row-wise Pearson correlation after selecting row-specific column indices."""
    selected_a = np.take_along_axis(a, indices, axis=1)
    selected_b = np.take_along_axis(b, indices, axis=1)
    da = selected_a - np.mean(selected_a, axis=1, keepdims=True)
    db = selected_b - np.mean(selected_b, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(da * da, axis=1) * np.sum(db * db, axis=1))
    result = np.full(a.shape[0], np.nan, dtype=float)
    np.divide(
        np.sum(da * db, axis=1),
        denominator,
        out=result,
        where=denominator > 0.0,
    )
    return result


def simulate_replicate(
    model: base.ModelConfig,
    profile: NoiseProfile,
    biological_seed: int,
    observation_seed: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """One latent FGM walk and its rank-matched noisy observations."""
    biological_rng = np.random.default_rng(biological_seed)
    observation_rng = np.random.default_rng(observation_seed)

    position = np.zeros(model.n, dtype=float)
    position[0] = model.radius
    probes, probe_q = draw_probe_library(
        biological_rng, model, position, profile.number
    )
    ancestral_effects = base.mutation_effects(position, probes, probe_q)

    latent_indices = subset_indices(ancestral_effects, TAIL_EXCLUSIONS, EXCLUSION_MODE)

    ancestor_sigma = assign_errors_by_rank(
        ancestral_effects, profile.ancestor_errors_by_rank
    )
    observed_ancestor = (
        ancestral_effects[np.newaxis, :]
        + error_scale
        * ancestor_sigma[np.newaxis, :]
        * observation_rng.normal(size=(noise_replicates, profile.number))
    )
    observed_indices = subset_indices(
        observed_ancestor, TAIL_EXCLUSIONS, EXCLUSION_MODE)

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

    for time in range(max_walk_steps + 1):
        current_effects = base.mutation_effects(position, probes, probe_q)
        evolved_sigma = assign_errors_by_rank(
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
            latent_trace[time, cut] = pearson_1d(
                ancestral_effects[latent_kept], current_effects[latent_kept]
            )
            observed_trace[:, time, cut] = pearson_rows_at_indices(
                observed_ancestor, observed_current, observed_kept
            )

        if time == max_walk_steps:
            steps_fixed = max_walk_steps
            break

        candidate_effects = base.mutation_effects(position, background, background_q)
        beneficial = available & np.isfinite(candidate_effects) & (candidate_effects > 0.0)
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
            biological_rng.choice(beneficial_indices, p=weights / total_weight)
        )
        position += background[chosen]
        available[chosen] = False
        steps_fixed = time + 1

    return latent_trace, observed_trace, steps_fixed, float(np.linalg.norm(position))


def run_simulations(
    models: tuple[base.ModelConfig, ...],
    profile: NoiseProfile,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
) -> dict[str, np.ndarray]:
    """Run deterministic seed trees for all model/walk/noise combinations."""
    latent = np.full(
        (len(models), replicates, max_walk_steps + 1, len(TAIL_EXCLUSIONS)),
        np.nan,
        dtype=float,
    )
    observed = np.full(
        (
            len(models),
            replicates,
            noise_replicates,
            max_walk_steps + 1,
            len(TAIL_EXCLUSIONS),
        ),
        np.nan,
        dtype=np.float32,
    )
    walk_lengths = np.zeros((len(models), replicates), dtype=np.int16)
    endpoint_radii = np.zeros((len(models), replicates), dtype=float)

    model_seeds = np.random.SeedSequence(MASTER_SEED).spawn(len(models))
    for model_index, (model, model_seed) in enumerate(zip(models, model_seeds)):
        replicate_seeds = model_seed.spawn(replicates)
        print(
            f"Simulating {profile.ancestor} -> {profile.evolved}, {model.title}: "
            f"{replicates} walks x {noise_replicates} observations; N={profile.number}",
            flush=True,
        )
        for replicate, replicate_seed in enumerate(replicate_seeds):
            biological_sequence, observation_sequence = replicate_seed.spawn(2)
            biological_seed = int(
                biological_sequence.generate_state(1, dtype=np.uint64)[0]
            )
            observation_seed = int(
                observation_sequence.generate_state(1, dtype=np.uint64)[0]
            )
            latent_trace, observed_trace, length, radius = simulate_replicate(
                model=model,
                profile=profile,
                biological_seed=biological_seed,
                observation_seed=observation_seed,
                noise_replicates=noise_replicates,
                error_scale=error_scale,
                max_walk_steps=max_walk_steps,
            )
            latent[model_index, replicate] = latent_trace
            observed[model_index, replicate] = observed_trace
            walk_lengths[model_index, replicate] = length
            endpoint_radii[model_index, replicate] = radius
            if (replicate + 1) % 50 == 0 or replicate + 1 == replicates:
                print(f"  {replicate + 1}/{replicates} walks complete", flush=True)

    return {
        "latent_correlations": latent,
        "observed_correlations": observed,
        "walk_lengths": walk_lengths,
        "endpoint_radii": endpoint_radii,
    }


def load_or_run(
    models: tuple[base.ModelConfig, ...],
    profile: NoiseProfile,
    metadata: dict[str, object],
    cache_path: str,
    force: bool,
) -> dict[str, np.ndarray]:
    metadata_text = json.dumps(metadata, sort_keys=True)
    if os.path.exists(cache_path) and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["metadata"].item()) == metadata_text:
                print(f"Loading matching cache: {cache_path}", flush=True)
                return {
                    key: np.asarray(cached[key])
                    for key in (
                        "latent_correlations",
                        "observed_correlations",
                        "walk_lengths",
                        "endpoint_radii",
                    )
                }
        print("Existing cache metadata differ; recomputing.", flush=True)

    arrays = run_simulations(
        models=models,
        profile=profile,
        replicates=int(metadata["replicates"]),
        noise_replicates=int(metadata["noise_replicates_per_walk"]),
        error_scale=float(metadata["error_scale"]),
        max_walk_steps=int(metadata["max_walk_steps"]),
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        metadata=np.array(metadata_text),
        **arrays,
    )
    print(f"Saved cache: {cache_path}", flush=True)
    return arrays


def validate(
    arrays: dict[str, np.ndarray],
    models: tuple[base.ModelConfig, ...],
    profile: NoiseProfile,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
) -> None:
    """Fail on broken ranks, shapes, latent normalization, or observation behavior."""
    expected_latent = (
        len(models),
        replicates,
        max_walk_steps + 1,
        len(TAIL_EXCLUSIONS),
    )
    expected_observed = (
        len(models),
        replicates,
        noise_replicates,
        max_walk_steps + 1,
        len(TAIL_EXCLUSIONS),
    )
    if arrays["latent_correlations"].shape != expected_latent:
        raise RuntimeError("Unexpected latent-correlation shape")
    if arrays["observed_correlations"].shape != expected_observed:
        raise RuntimeError("Unexpected observed-correlation shape")
    if not np.allclose(
        arrays["latent_correlations"][:, :, 0, :], 1.0, atol=2.0e-12
    ):
        raise RuntimeError("Latent ancestral correlations are not all one")
    if not np.all(np.isfinite(arrays["observed_correlations"][:, :, :, 0, :])):
        raise RuntimeError("A noisy ancestral correlation is non-finite")
    if error_scale == 0.0:
        expanded = arrays["latent_correlations"][:, :, np.newaxis, :, :]
        if not np.allclose(
            arrays["observed_correlations"], expanded, equal_nan=True, atol=2.0e-6
        ):
            raise RuntimeError("Zero-noise observations do not reproduce latent traces")
    elif np.allclose(
        arrays["observed_correlations"][:, :, :, 0, :], 1.0, atol=1.0e-6
    ):
        raise RuntimeError("Nonzero measurement noise left every t=0 correlation at one")

    test_values = np.linspace(-1.0, 1.0, profile.number)[::-1]
    assigned = assign_errors_by_rank(test_values, profile.ancestor_errors_by_rank)
    recovered = assigned[np.argsort(test_values, kind="stable")]
    if not np.array_equal(recovered, profile.ancestor_errors_by_rank):
        raise RuntimeError("Exact rank assignment failed")


def quantiles(values: np.ndarray) -> dict[str, list[float] | float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": np.nanmedian(values, axis=0).tolist(),
        "quantile_16": np.nanquantile(values, 0.16, axis=0).tolist(),
        "quantile_84": np.nanquantile(values, 0.84, axis=0).tolist(),
    }


def write_summary(
    arrays: dict[str, np.ndarray],
    models: tuple[base.ModelConfig, ...],
    metadata: dict[str, object],
    summary_path: str,
) -> dict[str, object]:
    """Write plotted values and the paired latent-to-observed changes."""
    summary: dict[str, object] = {"metadata": metadata, "models": {}}
    for model_index, model in enumerate(models):
        latent = arrays["latent_correlations"][model_index]
        observed = arrays["observed_correlations"][model_index]
        last_display = min(base.PANEL_DISPLAY_STEPS[model_index], latent.shape[1] - 1)
        requested_times = [
            time
            for time in sorted(set((0, 1, 2, 5, 10, last_display)))
            if time < latent.shape[1]
        ]
        curves = {}
        for time in requested_times:
            finite_walks = np.isfinite(latent[:, time, 0])
            latent_values = latent[finite_walks, time, :]
            observed_values = observed[finite_walks, :, time, :]
            paired_delta = observed_values - latent_values[:, np.newaxis, :]
            curves[str(time)] = {
                "walks": int(np.sum(finite_walks)),
                "observations": int(np.sum(finite_walks) * observed.shape[1]),
                "latent": quantiles(latent_values),
                "observed": quantiles(observed_values.reshape(-1, observed.shape[-1])),
                "observed_minus_latent": quantiles(
                    paired_delta.reshape(-1, paired_delta.shape[-1])
                ),
            }

        endpoint_latent = np.asarray(
            [latent[walk, int(length), :] for walk, length in enumerate(
                arrays["walk_lengths"][model_index]
            )]
        )
        endpoint_observed = np.asarray(
            [observed[walk, :, int(length), :] for walk, length in enumerate(
                arrays["walk_lengths"][model_index]
            )]
        )
        summary["models"][model.key] = {
            "parameters": model.serializable(),
            "walk_length_quantiles": np.quantile(
                arrays["walk_lengths"][model_index], (0.0, 0.16, 0.5, 0.84, 1.0)
            ).tolist(),
            "endpoint_latent": quantiles(endpoint_latent),
            "endpoint_observed": quantiles(
                endpoint_observed.reshape(-1, endpoint_observed.shape[-1])
            ),
            "curves": curves,
        }

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved summary: {summary_path}", flush=True)
    return summary


def make_figure(
    arrays: dict[str, np.ndarray],
    models: tuple[base.ModelConfig, ...],
    profile: NoiseProfile,
    error_scale: float,
    figure_path: str,
) -> None:
    """Plot latent and rank-matched observed autocorrelation curves."""
    figure, axes = plt.subplots(1, len(models), figsize=(12.2, 4.9), sharey=True)
    if len(models) == 1:
        axes = np.asarray([axes])

    global_minimum = 1.0
    for model_index, (model, axis) in enumerate(zip(models, axes)):
        last_time = min(
            base.PANEL_DISPLAY_STEPS[model_index],
            arrays["latent_correlations"].shape[2] - 1,
        )
        times = np.arange(last_time + 1)
        latent = arrays["latent_correlations"][model_index, :, : last_time + 1, :]
        observed = arrays["observed_correlations"][
            model_index, :, :, : last_time + 1, :
        ]
        observed_flat = observed.reshape(
            -1, observed.shape[-2], observed.shape[-1]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            latent_median = np.nanmedian(latent, axis=0)
            observed_median = np.nanmedian(observed_flat, axis=0)
            observed_lower = np.nanquantile(observed_flat, 0.16, axis=0)
            observed_upper = np.nanquantile(observed_flat, 0.84, axis=0)
        global_minimum = min(global_minimum, float(np.nanmin(observed_lower)))

        for cut, color in enumerate(base.CURVE_COLORS):
            axis.fill_between(
                times,
                observed_lower[:, cut],
                observed_upper[:, cut],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axis.plot(
                times,
                latent_median[:, cut],
                color=color,
                linewidth=2.7,
                linestyle="-",
            )
            axis.plot(
                times,
                observed_median[:, cut],
                color=color,
                linewidth=2.5,
                linestyle=(0, (4.0, 2.4)),
            )

        axis.set_title(f"{model.title} FGM")
        axis.set_xlabel("Fixed background mutations")
        axis.set_xlim(0, last_time)
        ticks = list(range(0, last_time + 1, 2)) if last_time <= 10 else [0, 5, 10, 15, 20]
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=5.5, width=1.1)

    axes[0].set_ylabel("Pearson autocorrelation")
    lower_limit = min(-0.05, np.floor(20.0 * (global_minimum - 0.02)) / 20.0)
    axes[0].set_ylim(lower_limit, 1.04)

    fraction_handles = [
        Line2D([], [], color=color, linewidth=2.7)
        for color in base.CURVE_COLORS
    ]
    semantic_handles = [
        Line2D([], [], color="#555555", linewidth=2.7, linestyle="-"),
        Line2D(
            [], [], color="#555555", linewidth=2.5, linestyle=(0, (4.0, 2.4))
        ),
    ]
    labels = [
        *(f"r{int(100 * fraction)}" for fraction in base.RETAINED_FRACTIONS),
        "latent",
        "rank-matched measurement noise",
    ]
    figure.legend(
        fraction_handles + semantic_handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.1,
    )
    figure.suptitle(
        f"Limdi {profile.ancestor} $\\rightarrow$ {profile.evolved}; "
        f"exact error-rank matching ($k={error_scale:g}$)",
        y=1.13,
        fontsize=19,
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.82, wspace=0.16)
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    figure.savefig(figure_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Saved figure: {figure_path}", flush=True)


def print_summary(summary: dict[str, object]) -> None:
    labels = tuple(
        f"r{int(round(100 * fraction))}"
        for fraction in summary["metadata"]["retained_fractions"])
    for model_name, result in summary["models"].items():
        print(f"\n{model_name}")
        for time, values in result["curves"].items():
            latent = values["latent"]["median"]
            observed = values["observed"]["median"]
            delta = values["observed_minus_latent"]["median"]
            fields = ", ".join(
                f"{label}: {a:.3f} -> {b:.3f} (delta {d:+.3f})"
                for label, a, b, d in zip(labels, latent, observed, delta)
            )
            print(f"  t={int(time):2d}: {fields}")


def apply_exclusion_options(args) -> None:
    """Install any requested subset rule before anything reads the module-level values."""
    global TAIL_EXCLUSIONS, EXCLUSION_MODE
    if args.exclusions is not None:
        exclusions = tuple(float(value) for value in args.exclusions)
        if not all(0.0 <= value < 1.0 for value in exclusions):
            raise ValueError("exclusion fractions must lie in [0, 1)")
        TAIL_EXCLUSIONS = exclusions
    if args.exclusion_mode is not None:
        EXCLUSION_MODE = args.exclusion_mode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ancestor", default="REL607")
    parser.add_argument("--evolved", default="Ara+2")
    parser.add_argument(
        "--fit-source",
        choices=tuple(base.FIT_CONFIGS),
        default="without-errors",
        help="FGM parameter source; default matches the current poster Figure 5",
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--noise-replicates", type=int, default=DEFAULT_NOISE_REPLICATES
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the published per-gene Limdi errors",
    )
    parser.add_argument("--max-walk-steps", type=int, default=base.MAX_WALK_STEPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--heavy-fit-json",
        default=None,
        help=(
            "stored fit whose profile supplies the heavy-tailed model, instead "
            "of --fit-source; use with --heavy-dimension and --tag"
        ),
    )
    parser.add_argument("--heavy-dimension", type=int, default=None)
    parser.add_argument(
        "--heavy-only",
        action="store_true",
        help="simulate the heavy-tailed model alone, omitting the canonical one",
    )
    parser.add_argument(
        "--display-steps",
        type=int,
        default=None,
        help="walk steps to draw; default uses the per-model poster values",
    )
    parser.add_argument(
        "--exclusions", type=float, nargs="+", default=None,
        help="tail fractions to drop; default is the historic 0 0.05 0.10",
    )
    parser.add_argument(
        "--exclusion-mode", choices=("signed", "magnitude"), default=None,
        help=("signed drops the most deleterious fraction (the historic rule); "
              "magnitude drops the largest |effect| fraction, as in fig1"),
    )
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.replicates < 1 or args.noise_replicates < 1:
        raise ValueError("replicate counts must be positive")
    if args.error_scale < 0.0 or args.max_walk_steps < 1:
        raise ValueError("error scale must be non-negative and max walk steps positive")

    apply_exclusion_options(args)
    profile = load_noise_profile(args.ancestor, args.evolved)
    if args.heavy_fit_json is not None:
        if args.heavy_dimension is None or not args.tag:
            raise ValueError(
                "--heavy-fit-json requires --heavy-dimension and --tag"
            )
        heavy = heavy_config_from_profile(
            args.heavy_fit_json, args.heavy_dimension
        )
        models = (
            (heavy,) if args.heavy_only
            else (load_model_configs(args.fit_source, profile.ancestor)[0], heavy)
        )
    else:
        models = load_model_configs(args.fit_source, profile.ancestor)
        if args.heavy_only:
            models = tuple(
                model for model in models if model.key == "heavy_tailed"
            )
            if not models:
                raise RuntimeError(
                    f"--fit-source {args.fit_source} has no heavy-tailed model"
                )
    if args.display_steps is not None:
        if not 1 <= args.display_steps <= args.max_walk_steps:
            raise ValueError("display steps must lie within the walk length")
        base.PANEL_DISPLAY_STEPS = (args.display_steps,) * len(models)
    metadata = build_metadata(
        profile=profile,
        models=models,
        fit_source=args.fit_source,
        replicates=args.replicates,
        noise_replicates=args.noise_replicates,
        error_scale=args.error_scale,
        max_walk_steps=args.max_walk_steps,
    )
    paths = output_paths(
        profile=profile,
        fit_source=args.fit_source,
        replicates=args.replicates,
        noise_replicates=args.noise_replicates,
        error_scale=args.error_scale,
        tag=args.tag,
    )
    arrays = load_or_run(
        models=models,
        profile=profile,
        metadata=metadata,
        cache_path=paths["cache"],
        force=args.force,
    )
    validate(
        arrays=arrays,
        models=models,
        profile=profile,
        replicates=args.replicates,
        noise_replicates=args.noise_replicates,
        error_scale=args.error_scale,
        max_walk_steps=args.max_walk_steps,
    )
    summary = write_summary(
        arrays=arrays,
        models=models,
        metadata=metadata,
        summary_path=paths["summary"],
    )
    print_summary(summary)
    make_figure(
        arrays=arrays,
        models=models,
        profile=profile,
        error_scale=args.error_scale,
        figure_path=paths["figure"],
    )


if __name__ == "__main__":
    main()
