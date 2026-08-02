#!/usr/bin/env python3
r"""Table: DFE autocorrelation across evolution for the Ascensao experiments.

The Ascensao analog of the Limdi block of TableS1_autocorr.py.  Each experiment (GHI, MNO, PQT,
SLR) is one ANCESTOR ``R`` and two evolved offspring ``L`` and ``S``; the three arrays are
index-aligned within an experiment (``genes.npy`` is the shared gene identity), so a transition
``R -> L`` or ``R -> S`` is matched by row.  For each we report the Pearson r of the matched
fitness effects -- how well a mutation's effect in the ancestor predicts its effect in the
evolved offspring -- exactly the observable TableS1 reports for Couce and Limdi.

Three things differ from the Limdi/Couce table, and each is a real property of this dataset, not
a choice:

  A DIFFERENT SCALE (units of 1/generation).  Ascensao report s per GENERATION (Fig. 1C of the
  paper); multiply by ~6.64 to get the per-cycle scale.  The deleterious tail is therefore at
  much smaller |s| than in Limdi: the ancestor DFE reaches -0.16 to -0.47 per generation (i.e.
  -1 to -3 per cycle -- a big tail), with 6-15% of genes below -0.02 and 38-124 below -0.05.
  Applying Limdi's -0.3 cut here is a mistake -- it is ~6.6x too large and removes almost
  everything -- so the TableS1 cutoffs are not transferable as-is.  The Pearson r itself is
  scale-free, so the autocorrelation values below are unaffected by the unit; only a tail
  conditioning would need a scale-appropriate cut (~ -0.03 to -0.05 per generation, matching the
  paper's ~1-2% non-neutral scale).  A tail table analogous to Table_tail_autocorr.py IS
  therefore feasible and is a natural follow-up.

  PER-GENE ERRORS.  ``autocorr_corr`` is disattenuated with the authors' published per-gene
  errors (``s std`` from the data release, github.com/joaoascensao/S-L-REL606-BarSeq), loaded by
  cmn_exper.load_asencao_errors and aligned row-for-row to the effects: reliability of a side =
  (V - mean(sigma_i^2))/V, r_corr = r / sqrt(rel_anc rel_evo) -- the same classical correction
  TableS1 applies to Limdi/Couce.  Over the full DFE the per-gene errors are small next to the
  effect spread, so the reliabilities are high (~0.8-0.97) and the correction is modest; it bites
  harder in the deleterious tail (see Table_ascensao_tail_autocorr.py), where errors are larger.

  NO ISOGENIC CONTROL.  The four ancestors are different genotypes (their R arrays correlate only
  0.01-0.66 across experiments), so unlike the Limdi REL606 -> REL607 pair there is no zero-
  evolution control to serve as a noise/decorrelation ceiling.  r is therefore reported on its
  own, with no r/ceiling column.

For context the two offspring of each ancestor (``L`` vs ``S``, both evolved) are also printed:
they correlate about as much as an ancestor does with an offspring, an internal check that the
decorrelation is set by evolution, not by which array is called the ancestor.

    data/Table_ascensao_autocorr.csv
    columns: dataset, experiment, transition, n, autocorr, autocorr_corr, reliability_anc,
             reliability_evo

Run:
    python code_tmp/Table_ascensao_autocorr.py
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_exper  # noqa: E402  (shared experimental-data loaders + structure)
from cmn.cmn_exper import DATA_DIR, ASENCAO_ANCESTOR, ASENCAO_EVOLVED  # noqa: E402

OUT_CSV = os.path.join(DATA_DIR, "Table_ascensao_autocorr.csv")
COLUMNS = ["dataset", "experiment", "transition", "n", "autocorr", "autocorr_corr",
           "reliability_anc", "reliability_evo"]

def pearson(a, b):
    """Pearson r over the entries finite in both a and b, plus the pair count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0.0 or np.std(b[mask]) == 0.0:
        return np.nan, n
    return float(pearsonr(a[mask], b[mask])[0]), n


def reliability(vals, sig):
    """Fraction of a side's variance that is signal, from per-gene errors: (V - mean(sig^2))/V."""
    sig = np.asarray(sig, float)
    sig = sig[np.isfinite(sig)]
    var = float(np.var(np.asarray(vals, float)))
    if sig.size == 0 or var <= 0.0:
        return np.nan
    return (var - float(np.mean(sig ** 2))) / var


def disattenuate(r, anc, evo, sig_anc, sig_evo):
    """r corrected for per-gene measurement noise on both sides (classical attenuation formula)."""
    if not np.isfinite(r):
        return np.nan, np.nan, np.nan
    rel_a, rel_e = reliability(anc, sig_anc), reliability(evo, sig_evo)
    if not (np.isfinite(rel_a) and np.isfinite(rel_e)) or min(rel_a, rel_e) <= 0.0:
        return np.nan, rel_a, rel_e
    return float(r / np.sqrt(rel_a * rel_e)), rel_a, rel_e


def ascensao_pair(exp, offspring):
    """Matched (ancestor, offspring, sigma_anc, sigma_evo) for one experiment, genes finite in both.

    R / L / S effects and their per-gene errors are index-aligned within an experiment
    (genes.npy is the shared identity), so the match is by row.
    """
    a = cmn_exper.load_asencao_array(exp, ASENCAO_ANCESTOR)
    b = cmn_exper.load_asencao_array(exp, offspring)
    sa = cmn_exper.load_asencao_errors(exp, ASENCAO_ANCESTOR)
    sb = cmn_exper.load_asencao_errors(exp, offspring)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], sa[m], sb[m]


def make_row(exp, offspring):
    """One row: autocorrelation of ancestor R vs evolved offspring for one experiment."""
    a, b, sa, sb = ascensao_pair(exp, offspring)
    r, n = pearson(a, b)
    r_corr, rel_a, rel_e = disattenuate(r, a, b, sa, sb)
    return {
        "dataset": "Ascensao",
        "experiment": exp,
        "transition": f"{ASENCAO_ANCESTOR} -> {offspring}",
        "n": n,
        "autocorr": r,
        "autocorr_corr": r_corr,
        "reliability_anc": rel_a,
        "reliability_evo": rel_e,
    }


def build_rows():
    """One row per (experiment, offspring): R -> L and R -> S for GHI, MNO, PQT, SLR."""
    rows = []
    for exp in cmn_exper.asencao_experiments():
        for offspring in ASENCAO_EVOLVED:
            rows.append(make_row(exp, offspring))
    return rows


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["dataset"], row["experiment"], row["transition"], row["n"],
                             f"{row['autocorr']:.4g}", f"{row['autocorr_corr']:.4g}",
                             f"{row['reliability_anc']:.4g}", f"{row['reliability_evo']:.4g}"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    print(f"\nAscensao DFE autocorrelation: ancestor R vs evolved offspring, matched by gene row")
    print("corrected with the authors' published per-gene errors (s std from the data release)\n")
    header = (f"{'experiment':<11}{'transition':<10}{'n':>7}{'autocorr':>10}"
              f"{'corrected':>11}{'rel_anc':>9}{'rel_evo':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['experiment']:<11}{row['transition']:<10}{row['n']:>7}"
              f"{row['autocorr']:>10.3f}{row['autocorr_corr']:>11.3f}"
              f"{row['reliability_anc']:>9.3f}{row['reliability_evo']:>9.3f}")

    r_all = np.array([row["autocorr"] for row in rows])
    rc_all = np.array([row["autocorr_corr"] for row in rows])
    print(f"\n8 transitions: raw r = {r_all.mean():.3f} (range {r_all.min():.3f}..{r_all.max():.3f}), "
          f"corrected = {rc_all.mean():.3f}")

    # Internal check: the two offspring of one ancestor (both evolved) vs each other.
    print("\nsibling check -- L vs S (two offspring of the same ancestor, both evolved):")
    for exp in cmn_exper.asencao_experiments():
        L = cmn_exper.load_asencao_array(exp, "L")
        S = cmn_exper.load_asencao_array(exp, "S")
        m = np.isfinite(L) & np.isfinite(S)
        r, _ = pearson(L[m], S[m])
        print(f"  {exp}: L vs S  r = {r:.3f}  (n = {int(m.sum())})")

    print("\nautocorr        = Pearson r of matched fitness effects, ancestor R vs evolved offspring")
    print("autocorr_corr   = disattenuated with the authors' per-gene errors (s std) -- QUOTE THIS")
    print("reliability_*   = (V - mean(sigma_i^2))/V per side, from the per-gene errors")
    print("NOTE: effects are per-GENERATION (x6.64 for per-cycle); the deleterious tail sits at")
    print("small |s| (~ -0.03..-0.05) -- see Table_ascensao_tail_autocorr.py.")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
