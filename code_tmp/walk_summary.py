#!/usr/bin/env python3
r"""Validation, summary-writing and printing for a cached adaptive-walk run.

Factored out of the (now removed) Ascensao walk driver, which is where these were first
written.  Nothing here is dataset-specific: the caller passes the simulated arrays and any
profile object exposing ``number`` and ``ancestor_errors_by_rank``, so the Couce and Limdi
drivers share one implementation.

``validate`` is deliberately strict.  A walk cache is expensive to regenerate and easy to
misread -- a transposed axis or an un-normalised t=0 correlation would quietly change every
curve drawn from it -- so shapes, ranks and the t=0 normalisation are all asserted rather
than assumed.
"""

from __future__ import annotations

import json
import os

import numpy as np

from code_tmp.poster_fig5_limdi_noise import assign_errors_by_rank, quantiles


def validate(
    arrays: dict[str, np.ndarray],
    profile,
    replicates: int,
    noise_replicates: int,
    error_scale: float,
    max_walk_steps: int,
    cuts: int,
) -> None:
    """Fail on broken shapes, ranks, normalization, or observation behavior.

    ``cuts`` is the number of retained subsets the caller simulated.  It is a parameter
    rather than a constant of this module: the drivers choose their own ladder -- Couce
    reports a 2% rung that Limdi does not -- and a hardcoded three silently rejected any
    other choice as a corrupt cache.
    """
    expected_latent = (
        replicates,
        max_walk_steps + 1,
        cuts,
    )
    expected_observed = (
        replicates,
        noise_replicates,
        max_walk_steps + 1,
        cuts,
    )
    if arrays["latent_correlations"].shape != expected_latent:
        raise RuntimeError("Unexpected latent-correlation shape")
    if arrays["observed_correlations"].shape != expected_observed:
        raise RuntimeError("Unexpected observed-correlation shape")
    if not np.allclose(
        arrays["latent_correlations"][:, 0, :], 1.0, atol=2.0e-12
    ):
        raise RuntimeError("Latent ancestral correlations are not all one")
    if not np.all(np.isfinite(arrays["observed_correlations"][:, :, 0, :])):
        raise RuntimeError("A noisy ancestral correlation is non-finite")
    if error_scale == 0.0:
        expanded = arrays["latent_correlations"][:, np.newaxis, :, :]
        if not np.allclose(
            arrays["observed_correlations"], expanded, equal_nan=True, atol=2.0e-6
        ):
            raise RuntimeError("Zero-noise observations do not reproduce latent traces")
    elif np.allclose(
        arrays["observed_correlations"][:, :, 0, :], 1.0, atol=1.0e-6
    ):
        raise RuntimeError("Nonzero measurement noise left every t=0 correlation at one")

    test_values = np.linspace(-1.0, 1.0, profile.number)[::-1]
    assigned = assign_errors_by_rank(test_values, profile.ancestor_errors_by_rank)
    recovered = assigned[np.argsort(test_values, kind="stable")]
    if not np.array_equal(recovered, profile.ancestor_errors_by_rank):
        raise RuntimeError("Exact error-rank assignment failed")

def write_summary(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    ladder: dict[str, float],
    summary_path: str,
) -> dict[str, object]:
    """Write all plotted values and paired latent-to-observed changes."""
    latent = arrays["latent_correlations"]
    observed = arrays["observed_correlations"]
    last_display = min(int(metadata["display_steps"]), latent.shape[1] - 1)
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
        [latent[walk, int(length), :] for walk, length in enumerate(arrays["walk_lengths"])]
    )
    endpoint_observed = np.asarray(
        [
            observed[walk, :, int(length), :]
            for walk, length in enumerate(arrays["walk_lengths"])
        ]
    )
    summary = {
        "metadata": metadata,
        "experimental_correlations": ladder,
        "walk_length_quantiles": np.quantile(
            arrays["walk_lengths"], (0.0, 0.16, 0.5, 0.84, 1.0)
        ).tolist(),
        "endpoint_radius_quantiles": np.quantile(
            arrays["endpoint_radii"], (0.0, 0.16, 0.5, 0.84, 1.0)
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

def print_summary(summary: dict[str, object]) -> None:
    labels = tuple(
        f"r{int(round(100 * fraction))}"
        for fraction in summary["metadata"]["retained_fractions"])
    for time, values in summary["curves"].items():
        latent = values["latent"]["median"]
        observed = values["observed"]["median"]
        delta = values["observed_minus_latent"]["median"]
        fields = ", ".join(
            f"{label}: {a:.3f} -> {b:.3f} (delta {d:+.3f})"
            for label, a, b, d in zip(labels, latent, observed, delta)
        )
        print(f"  t={int(time):2d}: {fields}")
