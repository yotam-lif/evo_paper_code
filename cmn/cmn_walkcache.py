#!/usr/bin/env python3
r"""Read simulated autocorrelation ladders out of a cached adaptive-walk run.

The walk drivers in ``code_tmp`` write one ``.npz`` per transition holding the Pearson
autocorrelation of every walk against its own ancestor, on each retained subset, at every
step.  Three consumers need the same few numbers out of those files -- Figure 4 and the
two autocorrelation tables -- so the reading lives here rather than being reimplemented
per caller with its own axis conventions.

    latent_correlations    the model's own effects, no measurement error anywhere
    observed_correlations  the same walk re-measured with rank-matched published errors
    walk_lengths           the step at which each walk ran out of beneficial mutations

TWO CACHE LAYOUTS, AND WHY IT MATTERS.  The Couce driver writes ``(walks, steps, cuts)``;
the Limdi driver carries a leading model axis, ``(models, walks, steps, cuts)``, with a
single model in it.  ``read`` drops that axis so both give the same shape.  Confusing the
two is not a loud failure -- indexing ``[:, 0]`` on the 4-D array silently selects walk 0
rather than model 0 and returns one walk's trace as if it were the median of five hundred
-- so the axis count is checked rather than assumed.

TWO TERMINATION CONVENTIONS, AND WHY THAT MATTERS MORE.  A walk stops when no beneficial
mutation is left.  The Couce driver then holds the genotype at its peak and keeps
recording, so every walk contributes at every step and a fixed-``time`` read is over all
of them.  The Limdi driver writes NaN instead, so past the median walk length a fixed-time
median is taken over a shrinking and increasingly atypical set of survivors.  That is why
``ladder`` offers ``time=None``: it reads each walk at its OWN last step, which is the
plateau every walk actually reached and the only summary that uses all of them.  Use a
fixed time when the transition has a known substitution count small enough to be inside
the walk, and the plateau when it does not -- an LTEE clone at 50K carries of order a
thousand mutations, and no SSWM walk reaches that.
"""

from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np


def locate(directory: str, pattern: str) -> str:
    """The single cache in ``directory`` matching ``pattern``; ambiguity is an error.

    Cache stems embed the matched-probe count, which differs per transition and is not
    worth recomputing at the call site, so callers glob over it.  Two matches means two
    runs of the same transition are on disk and picking either silently would be wrong.
    """
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    if not matches:
        raise FileNotFoundError(f"no walk cache matches {pattern!r} in {directory}")
    if len(matches) > 1:
        listed = "\n  ".join(os.path.basename(match) for match in matches)
        raise RuntimeError(f"{pattern!r} matches {len(matches)} caches:\n  {listed}")
    return matches[0]


def read(path: str) -> dict[str, object]:
    """Load one cache, dropping a single-model leading axis if there is one."""
    with np.load(path, allow_pickle=False) as cached:
        latent = np.asarray(cached["latent_correlations"], dtype=float)
        observed = np.asarray(cached["observed_correlations"], dtype=float)
        lengths = np.asarray(cached["walk_lengths"])
        metadata = json.loads(str(cached["metadata"].item()))
    if latent.ndim == 4:
        if latent.shape[0] != 1:
            raise RuntimeError(
                f"{os.path.basename(path)} holds {latent.shape[0]} models; expected one")
        latent, observed, lengths = latent[0], observed[0], lengths[0]
    if latent.ndim != 3 or observed.ndim != 4:
        raise RuntimeError(f"{os.path.basename(path)} has unexpected array ranks")
    if not np.allclose(latent[:, 0, :], 1.0, atol=1.0e-9):
        raise RuntimeError(f"{os.path.basename(path)}: t=0 latent correlations are not 1")
    return {"latent": latent, "observed": observed, "walk_lengths": lengths,
            "metadata": metadata, "path": path}


def ladder(cache: dict[str, object], time: int | None) -> dict[str, object]:
    """Median latent and noisy autocorrelation per retained subset.

    ``time`` is a step index, or ``None`` to read each walk at its own terminal step.
    The latent median is over walks; the noisy median is over walks x noise replicates
    pooled, which is what Figure 4 plots.
    """
    latent, observed = cache["latent"], cache["observed"]
    lengths = np.asarray(cache["walk_lengths"], dtype=int)
    if time is None:
        rows = np.arange(latent.shape[0])
        latent_values = latent[rows, lengths, :]
        observed_values = observed[rows, :, lengths, :]
    else:
        if not 0 <= time < latent.shape[1]:
            raise ValueError(
                f"t={time} is outside the {latent.shape[1] - 1}-step cache "
                f"{os.path.basename(cache['path'])}")
        latent_values = latent[:, time, :]
        observed_values = observed[:, :, time, :]
    observed_values = observed_values.reshape(-1, observed_values.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "time": time,
            "walks": int(np.isfinite(latent_values[:, 0]).sum()),
            "total_walks": int(latent.shape[0]),
            "retained_fractions": list(cache["metadata"]["retained_fractions"]),
            "exclusion_mode": cache["metadata"]["exclusion_mode"],
            "latent": np.nanmedian(latent_values, axis=0),
            "noisy": np.nanmedian(observed_values, axis=0),
            "walk_length_median": float(np.median(lengths)),
        }


def require_mode(cache: dict[str, object], mode: str, exclusions) -> None:
    """Fail unless the cache was simulated under the subset rule the caller reports.

    A curve and the measured number printed beside it must mean the same thing.  The
    signed rule (drop the most deleterious fraction) and the magnitude rule (drop the
    largest ``|s|`` fraction) give very different ladders on the same data, and nothing
    downstream can tell which a cache holds except this field.
    """
    metadata = cache["metadata"]
    stored_mode = metadata.get("exclusion_mode")
    if stored_mode != mode:
        raise RuntimeError(
            f"{os.path.basename(cache['path'])} was simulated with exclusion_mode="
            f"{stored_mode!r}, not {mode!r}; regenerate it")
    stored = [float(value) for value in metadata["tail_exclusions"]]
    wanted = [float(value) for value in exclusions]
    if not np.allclose(stored, wanted, atol=1.0e-12):
        raise RuntimeError(
            f"{os.path.basename(cache['path'])} holds exclusions {stored}, "
            f"not {wanted}; regenerate it")
