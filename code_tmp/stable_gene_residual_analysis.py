#!/usr/bin/env python3
"""Stable-gene and residual-correlation analysis for the Limdi LTEE panel.

The descriptive stable effect of gene g is its mean fitness effect over the 12 retained
backgrounds (REL606, REL607, and the ten reliable 50K clones).  Directly subtracting that
inclusive mean is reported, but it mechanically induces negative pairwise residual
correlations.  The primary diagnostic therefore excludes the tested pair, partitions the
remaining ten backgrounds into disjoint groups of five, estimates the stable effect
independently on the two sides, and summarizes all C(10, 5) = 252 assignments.

Primary gene set:
  * both technical replicates measured in all 12 retained backgrounds;
  * at least five interior TA sites under the source notebook's coordinate rule;
  * mean fitness effect greater than -0.3 in every retained background.

Outputs are CSV/JSON files intended for an auditable workbook and manuscript review.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from cmn import cmn_exper  # noqa: E402


GOOD_BACKGROUNDS = (
    "REL606",
    "REL607",
    "Ara-1",
    "Ara-3",
    "Ara-4",
    "Ara-5",
    "Ara-6",
    "Ara+1",
    "Ara+2",
    "Ara+3",
    "Ara+5",
    "Ara+6",
)

PAIRS = (
    ("control", "REL606", "REL607"),
    ("evolved", "REL606", "Ara-1"),
    ("evolved", "REL606", "Ara-3"),
    ("evolved", "REL606", "Ara-4"),
    ("evolved", "REL606", "Ara-5"),
    ("evolved", "REL606", "Ara-6"),
    ("evolved", "REL607", "Ara+1"),
    ("evolved", "REL607", "Ara+2"),
    ("evolved", "REL607", "Ara+3"),
    ("evolved", "REL607", "Ara+5"),
    ("evolved", "REL607", "Ara+6"),
)

PRIMARY_CUTOFF = -0.3
BOOTSTRAP_REPS = 2000
GENOMIC_BLOCK_BP = 50_000
SEED = 20260724


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation over finite entries, without a scipy dependency."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    x = x[keep] - np.mean(x[keep])
    y = y[keep] - np.mean(y[keep])
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / den) if den > 0.0 else float("nan")


def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Population covariance over finite entries."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    x = x[keep] - np.mean(x[keep])
    y = y[keep] - np.mean(y[keep])
    return float(np.mean(x * y))


def _linear_residual(y: np.ndarray, predictor: np.ndarray) -> np.ndarray:
    """Residual from y = intercept + loading * predictor."""
    y = np.asarray(y, float)
    predictor = np.asarray(predictor, float)
    xc = predictor - np.mean(predictor)
    yc = y - np.mean(y)
    den = float(np.dot(xc, xc))
    loading = float(np.dot(xc, yc) / den) if den > 0.0 else 0.0
    return yc - loading * xc


def _fasta_sequence(path: Path) -> str:
    parts: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts).upper()


def _interior_ta_counts(meta: pd.DataFrame, fasta_path: Path) -> np.ndarray:
    """Reproduce the Limdi notebook's interior-TA coordinate rule."""
    sequence = _fasta_sequence(fasta_path)
    ta_sites = np.asarray([m.start() for m in re.finditer("TA", sequence)], float)
    counts = np.zeros(len(meta), int)
    starts = meta.iloc[:, 3].to_numpy(float)
    ends = meta.iloc[:, 4].to_numpy(float)
    strands = meta.iloc[:, 5].to_numpy(int)
    for i, (start, end, strand) in enumerate(zip(starts, ends, strands)):
        length = end - start
        if strand == 1:
            inside = (ta_sites > start + length * 0.10) & (ta_sites < end - length * 0.25)
        elif strand == -1:
            inside = (ta_sites < start + length * 0.10) & (ta_sites > end - length * 0.25)
        else:
            inside = np.zeros(ta_sites.size, dtype=bool)
        counts[i] = int(np.sum(inside))
    return counts


def load_panel():
    fitness, error, names = cmn_exper.load_limdi_arrays()
    cols = [cmn_exper.LIMDI_LIBRARIES.index(name) for name in GOOD_BACKGROUNDS]
    replicate = np.asarray(fitness[:, cols, :], float)
    valid = np.all(replicate > cmn_exper.LIMDI_MISSING, axis=2)
    effect = np.mean(replicate, axis=2)
    effect[~valid] = np.nan

    sigma = np.asarray(error[:, cols], float)
    sigma[(sigma <= cmn_exper.LIMDI_MISSING) | ~np.isfinite(sigma)] = np.nan

    meta = pd.read_csv(cmn_exper.LIMDI_META, sep="\t")
    fasta = Path(cmn_exper.DATA_DIR) / "anurag_data" / "Metadata" / "rel606_reference.fasta"
    ta_count = _interior_ta_counts(meta, fasta)
    midpoint = 0.5 * (
        meta.iloc[:, 3].to_numpy(float) + meta.iloc[:, 4].to_numpy(float)
    )
    block_id = np.floor(midpoint / GENOMIC_BLOCK_BP).astype(int)

    assert replicate.shape == (4017, 12, 2)
    assert effect.shape == sigma.shape == (4017, 12)
    assert len(names) == len(meta) == len(ta_count)
    return replicate, effect, sigma, np.asarray(names, object), meta, ta_count, block_id


def eligible_mask(
    effect: np.ndarray,
    ta_count: np.ndarray,
    ia: int | None,
    ib: int | None,
    cutoff: float | None,
    mode: str,
    apply_ta_gate: bool = True,
) -> np.ndarray:
    complete = np.all(np.isfinite(effect), axis=1)
    mask = complete & ((ta_count >= 5) if apply_ta_gate else True)
    if cutoff is None:
        return mask
    if mode == "global":
        return mask & np.all(effect > cutoff, axis=1)
    if mode == "pairwise":
        if ia is None or ib is None:
            raise ValueError("pairwise cutoff requires target indices")
        return mask & (effect[:, ia] > cutoff) & (effect[:, ib] > cutoff)
    raise ValueError(f"unknown eligibility mode: {mode}")


def balanced_splits(reference_indices: tuple[int, ...]):
    half = len(reference_indices) // 2
    reference_set = set(reference_indices)
    return [
        (tuple(group_a), tuple(sorted(reference_set.difference(group_a))))
        for group_a in itertools.combinations(reference_indices, half)
    ]


def split_residual_statistics(
    replicate: np.ndarray,
    effect: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    ia: int,
    ib: int,
):
    references = tuple(i for i in range(effect.shape[1]) if i not in (ia, ib))
    splits = balanced_splits(references)
    y = effect[mask]
    s = sigma[mask]
    rep = replicate[mask]
    ya = y[:, ia]
    yb = y[:, ib]

    rs = []
    covs = []
    corrected = []
    scaled = []
    replicate_reliability_a = []
    replicate_reliability_b = []
    for group_a, group_b in splits:
        ma = np.mean(y[:, group_a], axis=1)
        mb = np.mean(y[:, group_b], axis=1)
        ra = ya - ma
        rb = yb - mb
        r = _pearson(ra, rb)
        rs.append(r)
        covs.append(_covariance(ra, rb))

        err_var_a = s[:, ia] ** 2 + np.sum(s[:, group_a] ** 2, axis=1) / len(group_a) ** 2
        err_var_b = s[:, ib] ** 2 + np.sum(s[:, group_b] ** 2, axis=1) / len(group_b) ** 2
        var_a = float(np.var(ra))
        var_b = float(np.var(rb))
        rel_a = (var_a - float(np.mean(err_var_a))) / var_a if var_a > 0.0 else np.nan
        rel_b = (var_b - float(np.mean(err_var_b))) / var_b if var_b > 0.0 else np.nan
        corrected.append(
            r / math.sqrt(rel_a * rel_b)
            if np.isfinite(r) and rel_a > 0.0 and rel_b > 0.0
            else np.nan
        )

        scaled.append(_pearson(_linear_residual(ya, ma), _linear_residual(yb, mb)))

        ra_green = rep[:, ia, 0] - np.mean(rep[:, group_a, 0], axis=1)
        ra_red = rep[:, ia, 1] - np.mean(rep[:, group_a, 1], axis=1)
        rb_green = rep[:, ib, 0] - np.mean(rep[:, group_b, 0], axis=1)
        rb_red = rep[:, ib, 1] - np.mean(rep[:, group_b, 1], axis=1)
        replicate_reliability_a.append(_pearson(ra_green, ra_red))
        replicate_reliability_b.append(_pearson(rb_green, rb_red))

    return {
        "splits": splits,
        "r": np.asarray(rs, float),
        "cov": np.asarray(covs, float),
        "corrected": np.asarray(corrected, float),
        "scaled": np.asarray(scaled, float),
        "replicate_reliability_a": np.asarray(replicate_reliability_a, float),
        "replicate_reliability_b": np.asarray(replicate_reliability_b, float),
    }


def analyze_pair(
    replicate: np.ndarray,
    effect: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    founder: str,
    target: str,
):
    ia = GOOD_BACKGROUNDS.index(founder)
    ib = GOOD_BACKGROUNDS.index(target)
    y = effect[mask]
    ya = y[:, ia]
    yb = y[:, ib]

    stable_all = np.mean(y, axis=1)
    naive_r = _pearson(ya - stable_all, yb - stable_all)

    references = tuple(i for i in range(effect.shape[1]) if i not in (ia, ib))
    stable_shared = np.mean(y[:, references], axis=1)
    shared_r = _pearson(ya - stable_shared, yb - stable_shared)

    split = split_residual_statistics(replicate, effect, sigma, mask, ia, ib)
    return {
        "n_genes": int(mask.sum()),
        "raw_r": _pearson(ya, yb),
        "naive_all_mean_r": naive_r,
        "shared_leave_pair_out_r": shared_r,
        "disjoint_r_median": float(np.nanmedian(split["r"])),
        "disjoint_r_mean": float(np.nanmean(split["r"])),
        "disjoint_r_split_q025": float(np.nanpercentile(split["r"], 2.5)),
        "disjoint_r_split_q975": float(np.nanpercentile(split["r"], 97.5)),
        "disjoint_r_min": float(np.nanmin(split["r"])),
        "disjoint_r_max": float(np.nanmax(split["r"])),
        "disjoint_r_error_corrected_median": float(np.nanmedian(split["corrected"])),
        "disjoint_scaled_r_median": float(np.nanmedian(split["scaled"])),
        "disjoint_cov_median": float(np.nanmedian(split["cov"])),
        "replicate_reliability_a_median": float(
            np.nanmedian(split["replicate_reliability_a"])
        ),
        "replicate_reliability_b_median": float(
            np.nanmedian(split["replicate_reliability_b"])
        ),
        "n_splits": len(split["splits"]),
        "_split": split,
    }


def _block_bootstrap_correlations(
    effect: np.ndarray,
    mask: np.ndarray,
    block_id: np.ndarray,
    ia: int,
    ib: int,
    split_info: dict,
    rng: np.random.Generator,
    reps: int,
) -> np.ndarray:
    eligible_rows = np.where(mask)[0]
    eligible_blocks = block_id[eligible_rows]
    unique_blocks = np.unique(eligible_blocks)
    local_by_block = {
        block: np.where(eligible_blocks == block)[0] for block in unique_blocks
    }
    y = effect[mask]
    out = np.empty(reps, float)
    splits = split_info["splits"]
    for b in range(reps):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sampled = np.concatenate([local_by_block[x] for x in sampled_blocks])
        group_a, group_b = splits[int(rng.integers(len(splits)))]
        ra = y[sampled, ia] - np.mean(y[np.ix_(sampled, group_a)], axis=1)
        rb = y[sampled, ib] - np.mean(y[np.ix_(sampled, group_b)], axis=1)
        out[b] = _pearson(ra, rb)
    return out


def _paired_block_bootstrap_gap(
    effect: np.ndarray,
    mask: np.ndarray,
    block_id: np.ndarray,
    control_pair: tuple[int, int],
    control_splits: list[tuple[tuple[int, ...], tuple[int, ...]]],
    evolved_pair: tuple[int, int],
    evolved_splits: list[tuple[tuple[int, ...], tuple[int, ...]]],
    rng: np.random.Generator,
    reps: int,
) -> np.ndarray:
    """Control-minus-evolved residual-r bootstrap using the same genomic blocks."""
    eligible_rows = np.where(mask)[0]
    eligible_blocks = block_id[eligible_rows]
    unique_blocks = np.unique(eligible_blocks)
    local_by_block = {
        block: np.where(eligible_blocks == block)[0] for block in unique_blocks
    }
    y = effect[mask]
    out = np.empty(reps, float)
    control_a, control_b = control_pair
    evolved_a, evolved_b = evolved_pair
    for b in range(reps):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sampled = np.concatenate([local_by_block[x] for x in sampled_blocks])

        group_ca, group_cb = control_splits[int(rng.integers(len(control_splits)))]
        control_ra = y[sampled, control_a] - np.mean(
            y[np.ix_(sampled, group_ca)], axis=1
        )
        control_rb = y[sampled, control_b] - np.mean(
            y[np.ix_(sampled, group_cb)], axis=1
        )
        control_r = _pearson(control_ra, control_rb)

        group_ea, group_eb = evolved_splits[int(rng.integers(len(evolved_splits)))]
        evolved_ra = y[sampled, evolved_a] - np.mean(
            y[np.ix_(sampled, group_ea)], axis=1
        )
        evolved_rb = y[sampled, evolved_b] - np.mean(
            y[np.ix_(sampled, group_eb)], axis=1
        )
        evolved_r = _pearson(evolved_ra, evolved_rb)
        out[b] = control_r - evolved_r
    return out


def primary_bootstrap(
    effect: np.ndarray,
    mask: np.ndarray,
    block_id: np.ndarray,
    results: list[dict],
    reps: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    distributions = {}
    for row in results:
        ia = GOOD_BACKGROUNDS.index(row["founder"])
        ib = GOOD_BACKGROUNDS.index(row["target"])
        distributions[row["transition"]] = _block_bootstrap_correlations(
            effect,
            mask,
            block_id,
            ia,
            ib,
            row["_split"],
            rng,
            reps,
        )
        dist = distributions[row["transition"]]
        row["block_boot_ci_low"] = float(np.nanpercentile(dist, 2.5))
        row["block_boot_ci_high"] = float(np.nanpercentile(dist, 97.5))

    control = results[0]
    control_median = control["disjoint_r_median"]
    control_pair = (
        GOOD_BACKGROUNDS.index(control["founder"]),
        GOOD_BACKGROUNDS.index(control["target"]),
    )
    for i, row in enumerate(results):
        if row["comparison_type"] == "control":
            row["control_minus_evolved_gap"] = np.nan
            row["gap_boot_ci_low"] = np.nan
            row["gap_boot_ci_high"] = np.nan
            continue
        evolved_pair = (
            GOOD_BACKGROUNDS.index(row["founder"]),
            GOOD_BACKGROUNDS.index(row["target"]),
        )
        dist = _paired_block_bootstrap_gap(
            effect,
            mask,
            block_id,
            control_pair,
            control["_split"]["splits"],
            evolved_pair,
            row["_split"]["splits"],
            np.random.default_rng(seed + 10_000 + i),
            reps,
        )
        row["control_minus_evolved_gap"] = control_median - row["disjoint_r_median"]
        row["gap_boot_ci_low"] = float(np.nanpercentile(dist, 2.5))
        row["gap_boot_ci_high"] = float(np.nanpercentile(dist, 97.5))


def stable_variance_summary(
    effect: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    label: str,
):
    y = effect[mask]
    s = sigma[mask]
    covariance = np.cov(y, rowvar=False, bias=True)
    offdiag = covariance[~np.eye(covariance.shape[0], dtype=bool)]
    stable_covariance = float(np.mean(offdiag))
    observed_variance = float(np.mean(np.diag(covariance)))
    signal_variances = np.diag(covariance) - np.mean(s ** 2, axis=0)
    signal_variance = float(np.mean(signal_variances))
    eig = np.linalg.eigvalsh(covariance)
    return {
        "gene_set": label,
        "n_genes": int(mask.sum()),
        "variance_of_gene_mean": float(np.var(np.mean(y, axis=1))),
        "mean_observed_background_variance": observed_variance,
        "mean_noise_corrected_background_variance": signal_variance,
        "mean_offdiagonal_covariance": stable_covariance,
        "stable_fraction_observed": stable_covariance / observed_variance,
        "stable_fraction_noise_corrected": stable_covariance / signal_variance,
        "pc1_covariance_fraction": float(eig[-1] / np.sum(eig)),
    }


def _public_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def run_analysis(output_dir: Path, bootstrap_reps: int, seed: int):
    replicate, effect, sigma, names, meta, ta_count, block_id = load_panel()
    complete = eligible_mask(effect, ta_count, None, None, None, "global", True)
    primary_mask = eligible_mask(
        effect, ta_count, None, None, PRIMARY_CUTOFF, "global", True
    )

    primary_results = []
    for comparison_type, founder, target in PAIRS:
        row = {
            "comparison_type": comparison_type,
            "transition": f"{founder} -> {target}",
            "founder": founder,
            "target": target,
        }
        row.update(
            analyze_pair(replicate, effect, sigma, primary_mask, founder, target)
        )
        primary_results.append(row)

    primary_bootstrap(
        effect,
        primary_mask,
        block_id,
        primary_results,
        bootstrap_reps,
        seed,
    )

    sensitivity_specs = (
        ("complete_no_cut", None, "global", True),
        ("global_gt_-0.3", -0.3, "global", True),
        ("pairwise_gt_-0.3", -0.3, "pairwise", True),
        ("global_gt_-0.1", -0.1, "global", True),
        ("pairwise_gt_-0.1", -0.1, "pairwise", True),
        ("global_gt_-0.3_no_5TA", -0.3, "global", False),
    )
    sensitivity_rows = []
    for label, cutoff, mode, apply_ta_gate in sensitivity_specs:
        for comparison_type, founder, target in PAIRS:
            ia = GOOD_BACKGROUNDS.index(founder)
            ib = GOOD_BACKGROUNDS.index(target)
            mask = eligible_mask(
                effect, ta_count, ia, ib, cutoff, mode, apply_ta_gate
            )
            values = analyze_pair(replicate, effect, sigma, mask, founder, target)
            sensitivity_rows.append(
                {
                    "gene_set": label,
                    "cutoff": cutoff,
                    "cutoff_scope": mode,
                    "five_ta_gate": apply_ta_gate,
                    "comparison_type": comparison_type,
                    "transition": f"{founder} -> {target}",
                    "n_genes": values["n_genes"],
                    "raw_r": values["raw_r"],
                    "naive_all_mean_r": values["naive_all_mean_r"],
                    "shared_leave_pair_out_r": values["shared_leave_pair_out_r"],
                    "disjoint_r_median": values["disjoint_r_median"],
                    "disjoint_r_split_q025": values["disjoint_r_split_q025"],
                    "disjoint_r_split_q975": values["disjoint_r_split_q975"],
                    "disjoint_r_error_corrected_median": values[
                        "disjoint_r_error_corrected_median"
                    ],
                    "disjoint_scaled_r_median": values["disjoint_scaled_r_median"],
                }
            )

    variance_specs = (
        ("complete_no_cut", None, True),
        ("global_gt_-0.3", -0.3, True),
        ("global_gt_-0.1", -0.1, True),
        ("global_gt_-0.3_no_5TA", -0.3, False),
    )
    variance_rows = []
    for label, cutoff, apply_ta_gate in variance_specs:
        mask = eligible_mask(
            effect, ta_count, None, None, cutoff, "global", apply_ta_gate
        )
        variance_rows.append(stable_variance_summary(effect, sigma, mask, label))

    stable = np.mean(effect[complete], axis=1)
    gene_rows = np.where(complete)[0]
    gene_table = pd.DataFrame(
        {
            "gene_row": gene_rows,
            "gene_name": names[complete],
            "interior_ta_sites": ta_count[complete],
            "stable_effect_mean_12_backgrounds": stable,
            "across_background_sd": np.std(effect[complete], axis=1),
            "mean_reported_error": np.mean(sigma[complete], axis=1),
            "all_12_gt_-0.3": np.all(effect[complete] > -0.3, axis=1),
            "all_12_gt_-0.1": np.all(effect[complete] > -0.1, axis=1),
        }
    )
    for i, background in enumerate(GOOD_BACKGROUNDS):
        gene_table[f"effect_{background}"] = effect[complete, i]

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_public_row(x) for x in primary_results]).to_csv(
        output_dir / "pair_results.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "sensitivity_results.csv", index=False
    )
    pd.DataFrame(variance_rows).to_csv(
        output_dir / "stable_variance.csv", index=False
    )
    gene_table.to_csv(output_dir / "gene_components.csv", index=False)

    evolved = [x for x in primary_results if x["comparison_type"] == "evolved"]
    summary = {
        "backgrounds": list(GOOD_BACKGROUNDS),
        "excluded_backgrounds": ["Ara-2", "Ara+4"],
        "n_complete_five_ta": int(complete.sum()),
        "n_primary_global_gt_minus_0_3": int(primary_mask.sum()),
        "primary_cutoff": PRIMARY_CUTOFF,
        "primary_cutoff_scope": "all 12 retained backgrounds",
        "control_disjoint_residual_r": primary_results[0]["disjoint_r_median"],
        "mean_evolved_disjoint_residual_r": float(
            np.mean([x["disjoint_r_median"] for x in evolved])
        ),
        "max_evolved_disjoint_residual_r": float(
            np.max([x["disjoint_r_median"] for x in evolved])
        ),
        "min_control_minus_evolved_gap": float(
            np.min([x["control_minus_evolved_gap"] for x in evolved])
        ),
        "min_gap_bootstrap_lower_95": float(
            np.min([x["gap_boot_ci_low"] for x in evolved])
        ),
        "bootstrap_reps": bootstrap_reps,
        "genomic_block_bp": GENOMIC_BLOCK_BP,
        "seed": seed,
        "naive_null_artifact": -1.0 / (len(GOOD_BACKGROUNDS) - 1),
        "shared_loo_null_artifact": 1.0 / (len(GOOD_BACKGROUNDS) - 1),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nPrimary results")
    display_cols = [
        "transition",
        "n_genes",
        "raw_r",
        "naive_all_mean_r",
        "shared_leave_pair_out_r",
        "disjoint_r_median",
        "block_boot_ci_low",
        "block_boot_ci_high",
        "control_minus_evolved_gap",
        "gap_boot_ci_low",
        "gap_boot_ci_high",
    ]
    print(pd.DataFrame([_public_row(x) for x in primary_results])[display_cols].to_string(index=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_analysis(args.output_dir, args.bootstrap_reps, args.seed)


if __name__ == "__main__":
    main()
