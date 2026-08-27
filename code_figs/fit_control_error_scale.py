#!/usr/bin/env python3
r"""Calibrate the published per-gene errors against zero-change control comparisons.

For each dataset, fit one multiplier ``k`` applied to every published per-gene standard error.
Limdi uses all 14 green/red controls.  Ascensao uses only the six replicate controls in GHI and
MNO; SLR, PQT, and U are deliberately excluded from calibration as problematic experiments.
At a proposed ``k``, each control is simulated under

    Y_A = X + k * sigma_A * Z_A
    Y_E = X + k * sigma_E * Z_E,

where ``X`` is the inverse-variance weighted mean of the two observed control measurements and
``Z_A`` and ``Z_E`` are independent standard-normal draws.  The ancestor-ranked 100/95/90
subsets are rebuilt in every simulation, exactly as in Tables S1 and S2.

One ``k`` is selected per dataset by minimizing the mean squared difference between the median
simulated null correlation and the observed control correlation, giving every control and each
of the 100/95/90 levels equal weight.  The random normal draws are identical for every candidate
``k`` (common random numbers), so changes in the loss reflect ``k`` rather than Monte Carlo
jitter.

This is calibration to the observed controls.  Minimizing the distance of the null correlations
themselves from 1 would have the trivial solution ``k = 0`` and therefore cannot estimate error.

Run:
    .venv/bin/python code_figs/fit_control_error_scale.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cmn.cmn_exper import (  # noqa: E402
    ASENCAO_MONO,
    ASENCAO_MONO_ENVIRONMENTS,
    DATA_DIR,
    LIMDI_PANEL,
)
from code_figs import TableS1_limdi_autocorr as table_s1  # noqa: E402
from code_figs import TableS3_ascensao_autocorr as table_s2  # noqa: E402


TAIL_EXCLUSIONS = (0.00, 0.05, 0.10)
PERCENTAGES = (100, 95, 90)
MASTER_SEED = 260821
OUT_JSON = os.path.join(DATA_DIR, "control_error_scale_fit.json")
ASCENSAO_CALIBRATION_EXPERIMENTS = ("GHI", "MNO")


@dataclass(frozen=True)
class Control:
    """One zero-change comparison after restricting to genes with usable errors."""

    dataset: str
    label: str
    a: np.ndarray
    a_err: np.ndarray
    b: np.ndarray
    b_err: np.ndarray
    shared_effect: np.ndarray
    observed_r: np.ndarray
    seed_a: int
    seed_b: int

    @property
    def n(self) -> int:
        return int(self.a.size)


def _stable_seed(dataset: str, label: str, endpoint: str) -> int:
    payload = f"{MASTER_SEED}:{dataset}:{label}:{endpoint}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between corresponding rows of two two-dimensional arrays."""
    da = a - np.mean(a, axis=1, keepdims=True)
    db = b - np.mean(b, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(da * da, axis=1) * np.sum(db * db, axis=1))
    result = np.full(a.shape[0], np.nan, dtype=float)
    np.divide(np.sum(da * db, axis=1), denominator, out=result, where=denominator > 0.0)
    return result


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - np.mean(a), b - np.mean(b)
    denominator = np.sqrt(np.sum(da * da) * np.sum(db * db))
    return float(np.sum(da * db) / denominator) if denominator > 0.0 else np.nan


def make_control(dataset: str, label: str, pairs) -> Control:
    """Prepare one control and its observed 100/95/90 correlation ladder."""
    a, a_err, b, b_err = (np.asarray(v, dtype=float) for v in pairs)
    valid = (np.isfinite(a) & np.isfinite(b) & np.isfinite(a_err) & np.isfinite(b_err)
             & (a_err > 0.0) & (b_err > 0.0))
    a, a_err, b, b_err = a[valid], a_err[valid], b[valid], b_err[valid]
    if a.size < 3:
        raise ValueError(f"{dataset} {label}: fewer than three valid control genes")

    weight_a, weight_b = 1.0 / a_err ** 2, 1.0 / b_err ** 2
    shared_effect = (weight_a * a + weight_b * b) / (weight_a + weight_b)
    order = np.argsort(a, kind="stable")
    observed = []
    for fraction in TAIL_EXCLUSIONS:
        removed = int(np.floor(fraction * a.size))
        kept = order[removed:]
        observed.append(_pearson_1d(a[kept], b[kept]))

    return Control(
        dataset=dataset,
        label=label,
        a=a,
        a_err=a_err,
        b=b,
        b_err=b_err,
        shared_effect=shared_effect,
        observed_r=np.asarray(observed, dtype=float),
        seed_a=_stable_seed(dataset, label, "A"),
        seed_b=_stable_seed(dataset, label, "E"),
    )


def limdi_controls() -> list[Control]:
    """Green/red technical controls for both ancestors and all twelve evolved libraries."""
    return [
        make_control("Limdi", f"{population} green -> red", table_s1.replicate_pair(population))
        for population in LIMDI_PANEL
    ]


def ascensao_controls() -> list[Control]:
    """Replicate controls for the three strains in the accepted GHI and MNO experiments."""
    controls: list[Control] = []
    for folder, _media, _description, ancestor_letter, evolved in ASENCAO_MONO_ENVIRONMENTS:
        if folder not in ASCENSAO_CALIBRATION_EXPERIMENTS:
            continue
        ancestor_name = ASENCAO_MONO[ancestor_letter][1]
        for ecotype, letter in ((ancestor_name, ancestor_letter),) + evolved:
            replicate_1, replicate_2 = table_s2.reps(letter)[:2]
            label = f"{folder}: {ecotype} rep1 vs rep2"
            controls.append(make_control(
                "Ascensao", label, table_s2.matched(replicate_1, replicate_2)))

    return controls


def simulate_control(control: Control, k: float, simulations: int,
                     batch_size: int) -> np.ndarray:
    """Return median simulated null r at 100/95/90 for one proposed error multiplier."""
    if k < 0.0:
        raise ValueError("k must be non-negative")
    if simulations < 1 or batch_size < 1:
        raise ValueError("simulations and batch_size must be positive")

    rng_a = np.random.default_rng(control.seed_a)
    rng_b = np.random.default_rng(control.seed_b)
    all_r = np.full((simulations, len(TAIL_EXCLUSIONS)), np.nan, dtype=float)
    completed = 0
    while completed < simulations:
        size = min(batch_size, simulations - completed)
        sim_a = (control.shared_effect[None, :]
                 + k * control.a_err[None, :] * rng_a.normal(size=(size, control.n)))
        sim_b = (control.shared_effect[None, :]
                 + k * control.b_err[None, :] * rng_b.normal(size=(size, control.n)))
        order = np.argsort(sim_a, axis=1, kind="stable")
        for column, fraction in enumerate(TAIL_EXCLUSIONS):
            removed = int(np.floor(fraction * control.n))
            kept_order = order[:, removed:]
            kept_a = np.take_along_axis(sim_a, kept_order, axis=1)
            kept_b = np.take_along_axis(sim_b, kept_order, axis=1)
            all_r[completed:completed + size, column] = _pearson_rows(kept_a, kept_b)
        completed += size
    return np.nanmedian(all_r, axis=0)


def evaluate(controls: list[Control], k: float, simulations: int,
             batch_size: int, details: bool = False):
    """Joint raw-correlation MSE, optionally with per-control predictions."""
    rows = []
    squared_residuals = []
    for control in controls:
        predicted = simulate_control(control, k, simulations, batch_size)
        residual = predicted - control.observed_r
        squared_residuals.extend((residual ** 2).tolist())
        if details:
            rows.append({
                "label": control.label,
                "n_genes": control.n,
                "observed": {str(p): float(r) for p, r in zip(PERCENTAGES, control.observed_r)},
                "fitted_null": {str(p): float(r) for p, r in zip(PERCENTAGES, predicted)},
                "residual": {str(p): float(r) for p, r in zip(PERCENTAGES, residual)},
            })
    loss = float(np.mean(squared_residuals))
    return (loss, rows) if details else loss


def fit_dataset(controls: list[Control], fit_simulations: int, report_simulations: int,
                batch_size: int, max_k: float, x_tolerance: float) -> dict:
    """Fit with a cheap deterministic profile, then report using more simulations."""
    objective = lambda k: evaluate(controls, float(k), fit_simulations, batch_size)
    result = minimize_scalar(
        objective,
        bounds=(0.0, max_k),
        method="bounded",
        options={"xatol": x_tolerance, "maxiter": 40},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"error-scale optimization failed: {result.message}")

    # Re-evaluate a small local grid with the full reporting simulation count.  This prevents the
    # final k from being an accidental optimum of the cheaper fitting simulation profile.
    half_width = max(4.0 * x_tolerance, 0.02)
    candidates = np.linspace(
        max(0.0, float(result.x) - half_width),
        min(max_k, float(result.x) + half_width),
        5,
    )
    candidate_losses = [
        evaluate(controls, float(k), report_simulations, batch_size) for k in candidates
    ]
    best_index = int(np.argmin(candidate_losses))
    best_k = float(candidates[best_index])
    final_loss, rows = evaluate(
        controls, best_k, report_simulations, batch_size, details=True)
    uncorrected_loss, uncorrected_rows = evaluate(
        controls, 1.0, report_simulations, batch_size, details=True)

    def level_summary(detail_rows):
        summary = {}
        for percentage in PERCENTAGES:
            residuals = np.asarray([
                row["residual"][str(percentage)] for row in detail_rows
            ])
            summary[str(percentage)] = {
                "mean_residual": float(np.mean(residuals)),
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            }
        return summary

    by_level = level_summary(rows)
    uncorrected_by_level = level_summary(uncorrected_rows)

    return {
        "k": best_k,
        "loss": final_loss,
        "rmse": float(np.sqrt(final_loss)),
        "n_controls": len(controls),
        "n_control_correlations": len(controls) * len(PERCENTAGES),
        "fit_simulations": fit_simulations,
        "report_simulations": report_simulations,
        "optimizer_k": float(result.x),
        "optimizer_loss": float(result.fun),
        "refinement_candidates": [
            {"k": float(k), "loss": float(loss)}
            for k, loss in zip(candidates, candidate_losses)
        ],
        "uncorrected": {
            "k": 1.0,
            "loss": uncorrected_loss,
            "rmse": float(np.sqrt(uncorrected_loss)),
            "by_level": uncorrected_by_level,
            "controls": uncorrected_rows,
        },
        "by_level": by_level,
        "controls": rows,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("both", "limdi", "ascensao"), default="both")
    parser.add_argument("--fit-simulations", type=int, default=200,
                        help="simulations per control during scalar optimization")
    parser.add_argument("--report-simulations", type=int, default=1000,
                        help="simulations per control for local refinement and final reporting")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-k", type=float, default=3.0)
    parser.add_argument("--x-tolerance", type=float, default=0.005)
    parser.add_argument("--out", default=OUT_JSON)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fit_simulations < 1 or args.report_simulations < 1:
        raise ValueError("simulation counts must be positive")
    if args.max_k <= 0.0 or args.x_tolerance <= 0.0:
        raise ValueError("max-k and x-tolerance must be positive")

    datasets = {}
    if args.dataset in ("both", "limdi"):
        datasets["Limdi"] = limdi_controls()
    if args.dataset in ("both", "ascensao"):
        datasets["Ascensao"] = ascensao_controls()

    output = {
        "model": "published per-gene standard errors multiplied by one dataset-wide k",
        "target": "observed zero-change control Pearson correlations",
        "objective": "equal-weight mean squared error over all controls and 100/95/90 levels",
        "master_seed": MASTER_SEED,
        "ascensao_calibration_experiments": list(ASCENSAO_CALIBRATION_EXPERIMENTS),
        "datasets": {},
    }
    for name, controls in datasets.items():
        print(f"Fitting {name}: {len(controls)} controls x 3 correlation levels ...", flush=True)
        output["datasets"][name] = fit_dataset(
            controls=controls,
            fit_simulations=args.fit_simulations,
            report_simulations=args.report_simulations,
            batch_size=args.batch_size,
            max_k=args.max_k,
            x_tolerance=args.x_tolerance,
        )
        fit = output["datasets"][name]
        print(f"  uncorrected k = 1.0000; joint RMSE = {fit['uncorrected']['rmse']:.4f}")
        print(f"  fitted      k = {fit['k']:.4f}; joint RMSE = {fit['rmse']:.4f}")
        for level in PERCENTAGES:
            before = fit["uncorrected"]["by_level"][str(level)]
            diagnostic = fit["by_level"][str(level)]
            print(f"    r_{level}: RMSE {before['rmse']:.4f} -> {diagnostic['rmse']:.4f}; "
                  f"corrected mean(null-observed)={diagnostic['mean_residual']:+.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
