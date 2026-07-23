#!/usr/bin/env python3
r"""Table S1: DFE autocorrelation across consecutive evolutionary transitions.

For every transition ``early -> late`` we ask how well a mutation's fitness effect measured
in the EARLIER background predicts its effect in the LATER one, over the mutations assayed in
both.  That is the DFE autocorrelation -- the Pearson ``r`` of the matched pairs -- and it is
the observable the p-spin / FGM scrambling picture predicts decays with the number of fixed
mutations separating the two genotypes.  Both datasets come from Couce et al., Science 383,
eadd1417 (2024), and both are LOG-fitness selection coefficients, so no conversion is needed.

Rows:

  Couce Ara+2 lineage (data/alex_code) -- 2 consecutive transitions along one lineage,
      plus the whole span.  0K -> 2K and 2K -> 15K, where 0K is the REL607 ancestor, and
      then 0K -> 15K end to end.  The span row is NOT independent of the other two (same
      lineage, same libraries), so it must not be counted as a third data point in any fit;
      it is there because 30 fixed mutations is the only Couce distance in the same decade
      as the Limdi non-mutators, and because 0K -> 2K vs 0K -> 15K is a within-lineage test
      of whether r keeps falling as mutations accumulate.  Per-SEGMENT effects: each
      locus is cut into 5 equal segments and all insertions in a segment are pooled (median 7
      mutants), so a row is a fifth of a gene, not one insertion.  Matched on ``alle`` =
      "<ORF>-<segment 1..5>", which is what the authors' own scripts use.  (The ``site``
      column is only one representative coordinate out of the pooled insertions and differs
      between independently mutagenised libraries for 40% of shared segments, so matching on
      it silently discards those: 7678 pairs instead of 12799.)

  Limdi LTEE TnSeq panel (data/anurag_data) -- 12 independent 0 -> 50K transitions.
      Each evolved 50K clone against its own founder: REL606 -> Ara-N and REL607 -> Ara+N.
      Per-GENE effects (averaged over the TA sites in the gene and over both replicates), so
      these are far less noisy, but also far more coarse-grained, than the Couce per-site
      values.  Genes are matched on the metadata row index of the .npy fitness matrices.

  ZERO-EVOLUTION CONTROL: REL606 -> REL607.
      The two LTEE ancestors are isogenic apart from the araA marker, so their DFEs differ
      only by measurement noise and batch.  Their correlation is therefore an empirical
      ceiling: no transition can exceed it, and r/ceiling is the part of the decorrelation
      that evolution actually has to explain.  Read every Limdi row against this one.  It
      carries ``n_fixed_mut = 0``: the araA marker is a single neutral point mutation, so on
      the scrambling axis this row is the origin.

    A NOTE ON MATCHING (this is what the previous version of this table got wrong).  The
    Limdi pairs used to be matched on the ``Genes`` column of ``dfe_data_pandas.csv``.  That
    column is mislabelled upstream by a pandas index-alignment slip -- see the block comment
    in ``cmn/cmn_exper.py`` -- so matching on it pairs genes by row *position*, which are the
    same real gene for only ~0.3-6.5% of rows.  It reported r ~ 0.008-0.18 for the Limdi
    transitions, i.e. essentially uncorrelated noise, which also contradicted the source
    paper's central finding that the landscape is largely conserved.  Matching on the row
    index of the .npy matrices is exact and gives r ~ 0.85-0.9.

THE NON-LETHAL RESTRICTION.  Every row is restricted to effects ``s > -0.3`` in BOTH
backgrounds.  This is exactly the restriction Couce et al. apply when comparing the effects
of individual mutations across backgrounds ("we restricted this analysis to insertions with
fitness effects s > -0.3 in both the ancestor and evolved strain, as measurements of
extremely deleterious effects have more measurement noise").  Two reasons it is the right
single number here:

  * Without it, r is dominated by the conserved lethal tail rather than by the landscape:
    the 3.7% of Limdi genes with |s| > 0.3 carry 71% of the total DFE variance, and a
    knockout that is lethal in the ancestor is still lethal at 50K.  That inflates r for
    reasons that have nothing to do with scrambling.
  * Without it the two datasets are not commensurate at all.  The cut is a NO-OP for Couce --
    it removes exactly 0 of their 9040, 8495 and 8429 matched pairs (printed as ``cut`` when this
    script runs) -- because knockouts lethal in DM25 are absent from their library rather than
    filtered from the analysis (see the block comment in ``cmn/cmn_exper.py``).  Couce
    therefore only ever measures the non-lethal range, and the Limdi rows must be cut to
    match; the cut costs Limdi 4-8% of its pairs, which is the lethal tail.

Conditioning on both variables truncates the range and so biases r down a little; the control
row absorbs that, since it is conditioned the same way.

EXCLUDED POPULATIONS (``excluded``).  Limdi et al. themselves drop Ara-2 and Ara+4 -- "we
excluded two populations derived from evolved clones from further analyses because their
fitness measurements were unreliable for technical reasons, and therefore not comparable to
the ancestor" -- on grounds stated before any cross-background comparison, so this is their
criterion and not a post-hoc convenience.  Both rows are KEPT here and merely flagged, so the
reader can see what is being set aside.  Both claims were verified directly against the two
technical replicates; see ``LIMDI_EXCLUDED`` for the numbers.  Note that Ara-2 is the case the
disattenuation column cannot help with: its error bars are among the smallest in the panel, so
the correction moves it 0.127 -> 0.137, while its actual problem is a systematic bias that
random-error correction is blind to by construction.

FIXED BACKGROUND MUTATIONS (``n_fixed_mut``).  Mutations fixed DURING that transition -- so
the Couce interval rows are the 8 and 22 mutations fixed within each interval, the span row
is their sum (30), and the Limdi rows are the full 0 -> 50K complement.  This is the distance the
p-spin / FGM picture predicts r decays with.  The Limdi counts split cleanly into the six
LTEE mutator lines (Ara-1/-2/-3/-4, Ara+3, Ara+6: 800-2600) and the six non-mutators
(70-125), which is the standard picture and an internal check on the column.

Measurement noise attenuates r: with per-observation error variance ``eps^2`` on each side,
``r_obs = r_true * sqrt((V_e - eps_e^2)/V_e * (V_l - eps_l^2)/V_l)``.  We report the raw
``autocorr`` and, using the per-observation errors published alongside the effects
(``sterr1`` for Couce, ``errors_genes_inv.npy`` for Limdi), the disattenuated
``autocorr_corr``.  ``autocorr_corr`` IS THE COLUMN TO QUOTE: the two assays have very
different noise levels (Couce reliability ~0.72, Limdi ~0.93 in this range), so their raw
r values are not comparable to each other, and the correction is what puts them on one scale.
Both error estimates were checked and are well calibrated: the Limdi one predicts a
REL606/REL607 ceiling of 0.96 against the 0.955 observed, and the Couce one matches the
scatter of putatively neutral intergenic insertions (MAD-based sd 0.0139 vs rms sterr1
0.0127).  The correction only removes *random* error; a systematic per-library batch shift
is invisible to it, and Couce has no control against which to measure one.

    data/TableS1_autocorr.csv
    columns: dataset, transition, n_fixed_mut, n, autocorr, autocorr_corr

Run:
    python code_figs/TableS1_autocorr.py
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_exper  # noqa: E402  (shared experimental-data loaders + structure)
from cmn.cmn_exper import (  # noqa: E402
    DATA_DIR, COUCE_INTERVALS, COUCE_SPAN, LIMDI_ANCESTORS, LIMDI_EVOLVED,
)

OUT_CSV = os.path.join(DATA_DIR, "TableS1_autocorr.csv")
COLUMNS = ["dataset", "transition", "n_fixed_mut", "n", "autocorr", "autocorr_corr", "excluded"]

# Non-lethal cutoff, applied to BOTH backgrounds of every pair.  -0.3 is the essentiality
# threshold used throughout Couce et al.; see THE NON-LETHAL RESTRICTION in the docstring.
NONLETHAL_CUT = -0.3

# Mutations fixed during each transition, keyed by the (unique) transition label.  Couce:
# per-interval counts along the Ara+2 lineage.  Limdi: the full 0 -> 50K complement of each
# population.  The control is the isogenic pair, i.e. the origin of the scrambling axis.
N_FIXED_MUT = {
    "0K -> 2K": 8,          "2K -> 15K": 22,        # Couce Ara+2 lineage, per interval
    "0K -> 15K": 30,                                # ... and their sum, over the whole span
    "REL606 -> REL607": 0,                          # isogenic control
    "REL606 -> Ara-1": 1100, "REL606 -> Ara-2": 1000, "REL606 -> Ara-3": 800,
    "REL606 -> Ara-4": 1300, "REL606 -> Ara-5": 90,   "REL606 -> Ara-6": 90,
    "REL607 -> Ara+1": 125,  "REL607 -> Ara+2": 70,   "REL607 -> Ara+3": 1800,
    "REL607 -> Ara+4": 70,   "REL607 -> Ara+5": 80,   "REL607 -> Ara+6": 2600,
}

# Populations Limdi et al. themselves exclude, on measurement-quality grounds stated before any
# cross-background comparison -- so dropping them here is their criterion, not ours.  Flagged
# rather than deleted: the rows stay in the CSV, and ``excluded`` marks them.  Both claims were
# checked directly against the two technical replicates in fitness_corrected_genes.npy.
LIMDI_EXCLUDED = {
    # "the within-gene measurement variability for fitness was extremely high, and the
    # correlation between technical replicates was poor".  Verified: replicate-replicate
    # r = 0.69 against 0.83-0.96 for every other population, median |green - red| = 0.017
    # against 0.005-0.009, median published error 0.019 against 0.005-0.009.
    "Ara+4": "poor technical replicates",
    # "a few insertion mutations increased rapidly, outcompeting other mutations ... which made
    # the measurements unreliable and systematically biased".  Verified: one gene reaches
    # s = 0.44 when no other population exceeds 0.125, and 76 genes exceed s = 0.05 against
    # 0-15 elsewhere.  Its replicate agreement is normal (0.88) precisely because both
    # replicates track the same sweep -- the bias is systematic, so neither replicate
    # concordance nor the disattenuation correction can see it (r 0.127 -> only 0.137).
    "Ara-2": "sweeping mutants bias assay",
}


def pearson(a, b):
    """Pearson r over the entries finite in both a and b, plus the pair count."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0.0 or np.std(b[mask]) == 0.0:
        return np.nan, n
    r, _ = pearsonr(a[mask], b[mask])
    return float(r), n


def disattenuate(r, early, late, sig_early, sig_late):
    """Correct ``r`` for measurement noise on both sides (classical attenuation formula).

    The reliability of a side is the fraction of its observed variance that is real signal,
    ``(V - mean(sigma^2)) / V``; ``r_true = r_obs / sqrt(rel_early * rel_late)``.  NaN if
    either side has no usable error estimate or is noise-dominated (reliability <= 0).
    """
    if not np.isfinite(r):
        return np.nan
    rel = []
    for vals, sig in ((early, sig_early), (late, sig_late)):
        if sig is None:
            return np.nan
        sig = np.asarray(sig, dtype=float)
        sig = sig[np.isfinite(sig)]
        var = float(np.var(np.asarray(vals, dtype=float)))
        if sig.size == 0 or var <= 0.0:
            return np.nan
        rel.append((var - float(np.mean(sig ** 2))) / var)
    if min(rel) <= 0.0:
        return np.nan
    return float(r / np.sqrt(rel[0] * rel[1]))


def make_row(dataset, transition, a, b, sig_a, sig_b):
    """One output row for a matched pair of backgrounds, over the non-lethal range."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sig_a, sig_b = np.asarray(sig_a, float), np.asarray(sig_b, float)
    m = (a > NONLETHAL_CUT) & (b > NONLETHAL_CUT)
    r, n = pearson(a[m], b[m])
    return {
        "dataset": dataset,
        "transition": transition,
        "n_fixed_mut": N_FIXED_MUT[transition],
        "n": n,
        "autocorr": r,
        "autocorr_corr": disattenuate(r, a[m], b[m], sig_a[m], sig_b[m]),
        "excluded": LIMDI_EXCLUDED.get(transition.split(" -> ")[1], ""),
        "n_dropped": int((~m).sum()),   # pairs removed by the cut; printed, not written to CSV
    }


# ══════════════════════════════════════════════════════════════════════════════
# Couce et al. -- consecutive timepoints of the Ara+2 lineage, matched on `alle`
# ══════════════════════════════════════════════════════════════════════════════
def couce_rows():
    """The two consecutive Couce intervals, then the whole 0K -> 15K span.

    The span row is not independent of the two intervals -- it is the same lineage measured
    end to end -- but it is the only Couce point at a comparable distance to the Limdi ones,
    and comparing it with 0K -> 2K asks directly whether r keeps falling along one lineage.
    """
    eff = {n: cmn_exper.load_couce_segment_series(n) for n in ("0K", "2K", "15K")}
    err = {n: cmn_exper.load_couce_segment_errors(n) for n in ("0K", "2K", "15K")}
    rows = []
    for early, late in list(COUCE_INTERVALS) + [COUCE_SPAN]:
        joined = pd.concat([eff[early], eff[late]], axis=1, join="inner")
        idx = joined.index
        rows.append(make_row("Couce Ara+2", f"{early} -> {late}",
                             joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy(),
                             err[early].reindex(idx).to_numpy(),
                             err[late].reindex(idx).to_numpy()))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Limdi et al. -- each 50K clone against its own founder, matched on gene row index
# ══════════════════════════════════════════════════════════════════════════════
def limdi_pair(early, late):
    """Matched (effects, effects, sigma, sigma) for two Limdi populations."""
    a_eff, a_sig = cmn_exper.limdi_gene_series(early, errors=True)
    b_eff, b_sig = cmn_exper.limdi_gene_series(late, errors=True)
    idx = a_eff.index.intersection(b_eff.index)
    return (a_eff[idx].to_numpy(), b_eff[idx].to_numpy(),
            a_sig[idx].to_numpy(), b_sig[idx].to_numpy())


def limdi_rows():
    """The isogenic-ancestor control, then one block per evolved LTEE population."""
    anc_a, anc_b = LIMDI_ANCESTORS
    rows = [make_row("Limdi control", f"{anc_a} -> {anc_b}", *limdi_pair(anc_a, anc_b))]
    for anc in LIMDI_ANCESTORS:
        for evo in LIMDI_EVOLVED[anc]:
            rows.append(make_row(f"Limdi {evo}", f"{anc} -> {evo}", *limdi_pair(anc, evo)))
    return rows


def build_rows():
    return couce_rows() + limdi_rows()


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["dataset"], row["transition"], row["n_fixed_mut"], row["n"],
                             f"{row['autocorr']:.4g}", f"{row['autocorr_corr']:.4g}",
                             row["excluded"]])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    # The docstring claims the cut is a no-op for Couce; check it rather than assert it, by
    # counting what it actually removes from each pair (``cut`` column below).
    print(f"\nnon-lethal cut s > {NONLETHAL_CUT}, applied to both sides of every pair")

    ceiling = next(r["autocorr_corr"] for r in rows if r["dataset"] == "Limdi control")
    header = (f"{'dataset':<14}{'transition':<20}{'n_fixed':>9}{'n':>7}{'cut':>6}"
              f"{'autocorr':>11}{'corrected':>11}{'r/ceiling':>11}")
    print(header)
    print("-" * len(header))
    for row in rows:
        ratio = row["autocorr_corr"] / ceiling if row["dataset"].startswith("Limdi") else np.nan
        print(f"{row['dataset']:<14}{row['transition']:<20}{row['n_fixed_mut']:>9}{row['n']:>7}"
              f"{row['n_dropped']:>6}"
              f"{row['autocorr']:>11.3f}{row['autocorr_corr']:>11.3f}{ratio:>11.3f}"
              f"   {row['excluded']}")

    # Does the corrected autocorrelation actually decay with the number of fixed mutations?
    # Reported on the evolved Limdi populations only (the control is the origin, and the two
    # Couce intervals are a different assay), with and without the two the source paper drops.
    evolved = [r for r in rows if r["dataset"].startswith("Limdi") and r["n_fixed_mut"] > 0]
    print()
    for tag, sub in (("all 12", evolved),
                     ("10 retained", [r for r in evolved if not r["excluded"]])):
        x = np.log10([r["n_fixed_mut"] for r in sub])
        y = [r["autocorr_corr"] for r in sub]
        rr, pp = pearsonr(x, y)
        print(f"decay of corrected r with log10(n_fixed_mut), {tag:<12} r = {rr:+.3f}  p = {pp:.3f}")

    print("\ncut            = pairs removed by the non-lethal cut (0 for Couce: they have no tail)")
    print("n_fixed_mut    = mutations fixed DURING that transition (cumulative only for Limdi)")
    print("autocorr       = Pearson r of matched fitness effects, early vs late background")
    print("autocorr_corr  = the same, disattenuated for published measurement error -- QUOTE THIS")
    print("r/ceiling      = fraction of the isogenic-ancestor correlation retained (Limdi only)")
    print("excluded       = non-empty for the two populations Limdi et al. drop as unreliable")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
