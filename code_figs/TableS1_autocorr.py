#!/usr/bin/env python3
r"""Table S1: DFE autocorrelation across the Limdi LTEE TnSeq panel.

For every transition ``early -> late`` we ask how well a mutation's fitness effect measured
in the EARLIER background predicts its effect in the LATER one, over the mutations assayed in
both.  That is the DFE autocorrelation -- the Pearson ``r`` of the matched pairs -- and it is
the observable the p-spin / FGM scrambling picture predicts decays with the number of fixed
mutations separating the two genotypes.  Effects are LOG-fitness selection coefficients, so no
conversion is needed.

Rows, all from the Limdi LTEE TnSeq panel (data/anurag_data).  Per-GENE effects, averaged over
the TA sites in the gene and (except in the two replicate rows) over both technical replicates.
Genes are matched on the metadata row index of the .npy fitness matrices.

  12 EVOLVED transitions -- each 50K clone against its own founder: REL606 -> Ara-N and
      REL607 -> Ara+N, twelve independent 0 -> 50K comparisons.

  TWO CONTROL TIERS, and the difference between them is the point of having both.

    Technical: REL606 green -> red and REL607 green -> red.  The two barcode channels of ONE
      library, unaveraged -- same strain, same mutagenesis, same assay, correlated against each
      other.  Nothing separates the two numbers but measurement noise, so this is the assay's
      own reproducibility and the hardest ceiling there is.  These are the rows the poster
      figure shows as its control panel (``code_tmp/poster_fig1.py``, panel C).

    Isogenic: REL606 -> REL607.  The two LTEE ancestors are isogenic apart from the araA
      marker, but they are separately mutagenised libraries measured as separate samples, so
      this row carries the technical noise PLUS whatever library-to-library and batch effects
      come with comparing two libraries.  That is exactly the situation every evolved row is
      in, which is why THIS is the ceiling f95/f90 divide by: it is the like-for-like zero, and
      the gap between it and the green/red rows is the price of the comparison itself.

  All three control rows carry ``n_fixed_mut = 0`` -- the araA marker is a single neutral point
  mutation -- so on the scrambling axis they are the origin.

    A NOTE ON MATCHING (this is what the previous version of this table got wrong).  The
    Limdi pairs used to be matched on the ``Genes`` column of ``dfe_data_pandas.csv``.  That
    column is mislabelled upstream by a pandas index-alignment slip -- see the block comment
    in ``cmn/cmn_exper.py`` -- so matching on it pairs genes by row *position*, which are the
    same real gene for only ~0.3-6.5% of rows.  It reported r ~ 0.008-0.18 for the Limdi
    transitions, i.e. essentially uncorrelated noise, which also contradicted the source
    paper's central finding that the landscape is largely conserved.  Matching on the row
    index of the .npy matrices is exact and gives r ~ 0.85-0.9.

THE LETHAL TAIL DOMINATES r, which is why the table reports more than one subset.  The 3.7% of
genes with |s| > 0.3 carry 71% of the total DFE variance, and a knockout lethal in the ancestor is
still lethal at 50K, so the full-range r is largely a statement about essential genes rather than
about the landscape.  Earlier versions of this table dealt with that by cutting both sides at
Couce's ``s > -0.3`` essentiality threshold and reporting the disattenuated result as the primary
number.  That column is gone: cutting the LATE side conditions on the outcome being predicted (see
ANCESTOR-DEFINED NESTED SUBSETS below), which is the wrong thing to do to a scrambling
measurement.  The replacement is to rank by the ANCESTOR and remove a fixed fraction, which is
what every column here now does.

An earlier version also carried the Couce et al. Ara+2 lineage (0K -> 2K -> 15K, per-segment
insertion effects) in the same table.  Those rows are gone too.  They were never on the same
scale: knockouts lethal in DM25 are absent from the Couce library altogether rather than filtered
out of the analysis (see the block comment in ``cmn/cmn_exper.py``), so a percentile of the Couce
ancestor lands at s ~ -0.03 where a Limdi percentile lands at s ~ -0.25, and the two could not be
read against each other row by row.  The Couce comparison belongs in its own table.

EXCLUDED POPULATIONS (``excluded``).  Limdi et al. themselves drop Ara-2 and Ara+4 -- "we
excluded two populations derived from evolved clones from further analyses because their
fitness measurements were unreliable for technical reasons, and therefore not comparable to
the ancestor" -- on grounds stated before any cross-background comparison, so this is their
criterion and not a post-hoc convenience.  Both rows are KEPT here and merely flagged, so the
reader can see what is being set aside.  Both claims were verified directly against the two
technical replicates; see ``LIMDI_EXCLUDED`` for the numbers.  Ara-2 is the instructive one: its
error bars are among the smallest in the panel, yet it collapses to r_95 = 0.008 and r_90 =
-0.081 (f90 = -0.15), far below every other row.  Its problem is a systematic bias, not noise, so
no error-based correction could have rescued it -- which is why the flag is the right handling.

FIXED BACKGROUND MUTATIONS (``n_fixed_mut``).  Mutations fixed DURING that transition: the full
0 -> 50K complement for an evolved row, and 0 for all three controls.  This is the distance the
p-spin / FGM picture predicts r decays with.  The counts split cleanly into the six LTEE mutator
lines (Ara-1/-2/-3/-4, Ara+3, Ara+6: 800-2600) and the six non-mutators (70-125), which is the
standard picture and an internal check on the column.  Note the panel spans barely more than one
decade once the mutators are set aside, so the decay fits printed at the end are weak by
construction -- they are a consistency check, not the paper's evidence for decay.

MEASUREMENT NOISE.  Noise attenuates r: with per-observation error variance ``eps^2`` on each
side, ``r_obs = r_true * sqrt((V_e - eps_e^2)/V_e * (V_l - eps_l^2)/V_l)``.  Every number in this
table is the raw ``r_obs``; NOTHING is disattenuated, and that is deliberate.  The correction
needs the reliability of each side, and stripping the deleterious tail strips most of the signal
variance with it: the ancestor reliability falls from ~0.98 at r_100 to ~0.49-0.67 at r_90, which
is exactly where the classical formula stops being trustworthy.  The control rows answer the
question empirically instead, and the two tiers separate the two sources.  The green/red rows are
pure assay noise -- one library against itself -- and they land at r_90 = 0.650 and 0.707.  The
isogenic row adds the library-to-library term and drops to 0.550.  So roughly a fifth of the
control's own decorrelation at r_90 is the cost of comparing two separately mutagenised libraries
rather than measurement noise, and since every evolved row pays that same cost, ``f95``/``f90``
divide by the isogenic row and are already normalised for both.

    data/TableS1_autocorr.csv
    columns: dataset, transition, n_fixed_mut, n_100, r_100, n_95, cut_95, r_95,
             n_90, cut_90, r_90, excluded

ANCESTOR-DEFINED NESTED SUBSETS (the ``r_100 / r_95 / r_90`` columns).  The autocorrelation r
measures how much of the DFE is *preserved*; scrambling is its complement, roughly 1 - r.
Because r is a Pearson correlation it is dominated by the points farthest from the origin -- the
conserved deleterious tail -- so where the tail is cut changes r a great deal, and reporting one
cut hides that.  Each row therefore carries r on three nested subsets, obtained by removing
exactly 0%, 5% and 10% of the most deleterious effects:

  r_100   every matched pair, nothing removed (the lethal tail included)
  r_95    the lowest 5% of ANCESTOR effects removed   (``cut_95`` = largest excluded effect)
  r_90    the lowest 10% of ANCESTOR effects removed  (``cut_90`` likewise)

This is the exclusion rule of the poster figure (``code_tmp/poster_fig1.py``, panels C-D), here
applied to every transition.  Two properties are why it replaced the fixed two-sided cutoffs the
table used to report:

  * The exclusion is defined ONLY from the early/ancestor measurement, never from the late one
    whose correlation is being computed.  A cutoff applied to both sides conditions on the
    outcome: at -0.1 it additionally discards the genes that are near-neutral in the ancestor but
    deleterious at 50K -- for REL607 -> Ara+2 that is 48 genes, 1.6% of the points, carrying 57%
    of the covariance, and dropping them takes r from 0.296 down to 0.223.  Those genes ARE the
    change being measured, so conditioning them away is exactly wrong here.  This is the same
    trap as in the tail table below, mirrored: there two-sided conditioning hides essential
    knockouts that became dispensable, here it hides neutral ones that became costly.
  * Removing a fixed FRACTION by rank keeps the subsets nested (90% inside 95% inside 100%) and
    the sample sizes equal across rows, so r_90 of one population is comparable to r_90 of
    another and of the control, which fixed absolute cutoffs do not guarantee.

Reading down the ladder: the high full-range r is carried by the preserved tail, and stripping it
exposes the scrambling in the near-neutral bulk.  Removing a tenth of the genes takes the ten
RETAINED evolved Limdi rows from r_100 ~ 0.82-0.91 to r_90 ~ 0.13-0.34, while the isogenic
control, cut the same way, only falls 0.955 -> 0.550.  (The two flagged populations start lower
and fall further still -- Ara+4 0.729 -> 0.129 and Ara-2 0.532 -> -0.081 -- which is the
measurement problem the source paper excluded them for, not extra scrambling.)  That control-versus-evolved gap at r_90 is the real bulk
scrambling; the full-range number understates it because the tail dominates it.  Quote the
fraction of the control retained (``f95``, ``f90`` in the printed table), not the bare r: r is
not comparable across ranges, only within one.

    A percentile is not a fixed value of s, so ``cut_95`` and ``cut_90`` are reported for every
    row.  Within this panel they barely move -- the 5% quantile of the ancestor sits at
    s = -0.23 to -0.29 and the 10% quantile at s = -0.07 to -0.11 across all fifteen rows -- so
    the subsets are near-identical in absolute terms as well as in size, and r_90 really is
    row-to-row comparable.  That would NOT hold against a library with a different lethal
    fraction; see the note on the removed Couce rows above.

TAIL DECORRELATION.  The complementary question -- does the conserved tail *also* scramble? --
has its own table, code_tmp/Table_tail_autocorr.py, because it must be conditioned differently:
on the ANCESTOR tail only (genes with ancestral s < -0.3), with the evolved effect left free.
Conditioning on both sides would discard the initially-essential knockouts that became
dispensable after evolution, which are precisely the tail's strongest decorrelation.  Done that
way, the initial tail keeps only ~63% of its noise-corrected predictability relative to the
isogenic ceiling, and ~4.5% of essential knockouts revert to near-neutral against 0% in the
control.  So the tail is *mostly* conserved but not perfectly -- even large knockout costs shift
over 50K generations -- while the near-neutral bulk is where the bulk of the scrambling lives.

Run:
    python code_figs/TableS1_autocorr.py
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
from cmn.cmn_exper import DATA_DIR, LIMDI_ANCESTORS, LIMDI_EVOLVED  # noqa: E402

OUT_CSV = os.path.join(DATA_DIR, "TableS1_autocorr.csv")
# Every r column is the poster-figure ladder: r on nested subsets defined by removing a fraction
# of the ANCESTOR side only (see ANCESTOR-DEFINED NESTED SUBSETS in the docstring).  The earlier
# ``n / autocorr / autocorr_corr`` block -- r at a fixed ``s > -0.3`` cut applied to BOTH sides,
# plus its disattenuated value -- is gone: cutting the late side conditions on the outcome, and
# with no fixed cut left there is no range where disattenuation is trustworthy.
COLUMNS = ["dataset", "transition", "n_fixed_mut", "n_100", "r_100",
           "n_95", "cut_95", "r_95", "n_90", "cut_90", "r_90", "excluded"]

# Fractions of the EARLY (ancestor) side removed, lowest effect first, for the nested-subset
# ladder.  0.00 keeps every matched pair, so the subsets are nested: 90% inside 95% inside 100%.
# Same rule and same fractions as code_tmp/poster_fig1.py, which shows two of these rows.
TAIL_EXCLUSIONS = (0.00, 0.05, 0.10)

# Mutations fixed during each transition, keyed by the (unique) transition label: the full
# 0 -> 50K complement of each population.  All three control rows sit at 0, the origin of the
# scrambling axis -- the araA marker separating REL606 from REL607 is a single neutral point
# mutation, and the green/red rows are one library against itself.
N_FIXED_MUT = {
    "REL606 green -> red": 0, "REL607 green -> red": 0,   # technical (replicate) controls
    "REL606 -> REL607": 0,                                # isogenic (library-to-library) control
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
    # replicates track the same sweep, so replicate concordance cannot see the problem -- but the
    # ladder does: r_95 = 0.008 and r_90 = -0.081, far below every other row in the panel.
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


def ancestor_exclusion_ladder(a, b):
    """``r`` on nested subsets built by removing the lowest ANCESTOR effects.

    For each fraction in ``TAIL_EXCLUSIONS`` the ``floor(frac * n)`` matched pairs with the
    smallest EARLY-background effect are dropped and Pearson r is recomputed on what is left.
    Only ``a`` enters the exclusion -- never ``b``, the side being predicted -- so the subsets do
    not condition on the outcome, and because a fixed fraction is removed by rank they are nested
    and equally sized across rows.  Returns one dict per fraction with the retained pair count,
    the cutoff (the largest EXCLUDED ancestor effect; ``-inf`` when nothing is excluded) and r.
    """
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    order = np.argsort(a, kind="stable")
    ladder = []
    for frac in TAIL_EXCLUSIONS:
        n_removed = int(np.floor(frac * a.size))
        kept = order[n_removed:]
        r, n = pearson(a[kept], b[kept])
        ladder.append({
            "frac": frac,
            "pct": int(round(100 * (1.0 - frac))),
            "n": n,
            "r": r,
            "cut": -np.inf if n_removed == 0 else float(a[order[n_removed - 1]]),
        })
    return ladder


def make_row(dataset, transition, a, b):
    """One output row for a matched pair of backgrounds.

    ``r_100 / r_95 / r_90`` is the poster-figure ladder: r on nested subsets with 0%, 5% and 10%
    of the most deleterious ANCESTOR effects removed, which never conditions on the outcome and
    keeps the subsets comparable across rows.  All three are raw -- see MEASUREMENT NOISE and
    ANCESTOR-DEFINED NESTED SUBSETS in the docstring for why nothing here is disattenuated, and
    read every row against the isogenic control rather than against an absolute scale.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    row = {
        "dataset": dataset,
        "transition": transition,
        "n_fixed_mut": N_FIXED_MUT[transition],
        "excluded": LIMDI_EXCLUDED.get(transition.split(" -> ")[1], ""),
    }
    for step in ancestor_exclusion_ladder(a, b):     # r_100 / r_95 / r_90 and their subset sizes
        row[f"r_{step['pct']}"] = step["r"]
        row[f"n_{step['pct']}"] = step["n"]
        row[f"cut_{step['pct']}"] = step["cut"]
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Limdi et al. -- each 50K clone against its own founder, matched on gene row index
# ══════════════════════════════════════════════════════════════════════════════
def limdi_pair(early, late):
    """Matched (effects, effects) for two Limdi populations, on the genes measured in both."""
    a_eff = cmn_exper.limdi_gene_series(early)
    b_eff = cmn_exper.limdi_gene_series(late)
    idx = a_eff.index.intersection(b_eff.index)
    return a_eff[idx].to_numpy(), b_eff[idx].to_numpy()


def replicate_pair(pop):
    """The Green and Red channels of one library, unaveraged -- the technical control."""
    green, red = cmn_exper.limdi_channel_series(pop)
    return green.to_numpy(), red.to_numpy()


def build_rows():
    """Technical controls, then the isogenic control, then one row per evolved population.

    The order is deliberate: the two green/red rows bound what the assay can do at all, the
    isogenic row bounds what a two-library comparison can do, and only then do the evolved rows
    have a scale to be read against.  ``green`` is the ancestor side of a replicate row, so the
    nested subsets are defined from it exactly as they are from the founder elsewhere -- which
    channel plays that part is arbitrary, and it changes r by less than 0.001.
    """
    rows = [make_row("Limdi rep", f"{pop} green -> red", *replicate_pair(pop))
            for pop in LIMDI_ANCESTORS]
    anc_a, anc_b = LIMDI_ANCESTORS
    rows.append(make_row("Limdi control", f"{anc_a} -> {anc_b}", *limdi_pair(anc_a, anc_b)))
    for anc in LIMDI_ANCESTORS:
        for evo in LIMDI_EVOLVED[anc]:
            rows.append(make_row(f"Limdi {evo}", f"{anc} -> {evo}", *limdi_pair(anc, evo)))
    return rows


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["dataset"], row["transition"], row["n_fixed_mut"],
                             row["n_100"], f"{row['r_100']:.4g}",
                             row["n_95"], f"{row['cut_95']:.4g}", f"{row['r_95']:.4g}",
                             row["n_90"], f"{row['cut_90']:.4g}", f"{row['r_90']:.4g}",
                             row["excluded"]])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    # Every row is now Limdi, so every row gets a ceiling ratio.  The isogenic REL606 -> REL607
    # row is the divisor, not the green/red rows: it is the like-for-like zero, two separately
    # mutagenised libraries compared as separate samples, which is the situation the evolved rows
    # are in.  The green/red rows divided by it therefore come out ABOVE 1, and that excess is the
    # library-to-library cost of the comparison itself, isolated from pure assay noise.
    ceil = next(r for r in rows if r["dataset"] == "Limdi control")

    # ---- the poster-figure ladder: nested subsets defined from the ancestor side only --------
    print("\nnested subsets: the lowest 0% / 5% / 10% of ANCESTOR effects removed, evolved side free")
    print("cut = largest excluded ancestor effect; f95, f90 = fraction of the ISOGENIC control's r")
    print("rows: 2 technical (green vs red, one library) + 1 isogenic (606 vs 607) + 12 evolved")
    header = (f"{'dataset':<15}{'transition':<18}{'n_fix':>6}"
              f"{'n_100':>7}{'r_100':>8}"
              f"{'n_95':>7}{'cut_95':>8}{'r_95':>8}{'f95':>7}"
              f"{'n_90':>7}{'cut_90':>8}{'r_90':>8}{'f90':>7}")
    print(header)
    print("-" * len(header))
    for row in rows:
        f95, f90 = row["r_95"] / ceil["r_95"], row["r_90"] / ceil["r_90"]
        print(f"{row['dataset']:<15}{row['transition']:<18}{row['n_fixed_mut']:>6}"
              f"{row['n_100']:>7}{row['r_100']:>8.3f}"
              f"{row['n_95']:>7}{row['cut_95']:>8.3f}{row['r_95']:>8.3f}{f95:>7.3f}"
              f"{row['n_90']:>7}{row['cut_90']:>8.3f}{row['r_90']:>8.3f}{f90:>7.3f}"
              f"   {row['excluded']}")

    # Does the autocorrelation actually decay with the number of fixed mutations?  Reported on the
    # evolved populations only (all three controls sit at the origin), with and without the two
    # the source paper drops, and at each level of the ladder -- the full range is tail-dominated,
    # so the bulk subsets are the informative ones.
    evolved = [r for r in rows if r["n_fixed_mut"] > 0]
    print()
    for key, label in (("r_100", "r_100"), ("r_95", "r_95"), ("r_90", "r_90")):
        for tag, sub in (("all 12", evolved),
                         ("10 retained", [r for r in evolved if not r["excluded"]])):
            x = np.log10([r["n_fixed_mut"] for r in sub])
            rr, pp = pearsonr(x, [r[key] for r in sub])
            print(f"decay of {label:<6} with log10(n_fixed_mut), {tag:<12} "
                  f"r = {rr:+.3f}  p = {pp:.3f}")

    # TAIL DECORRELATION lives in its own table, code_tmp/Table_tail_autocorr.py.  It must
    # condition on the ANCESTOR tail only (evolved side free), not on both sides: requiring the
    # gene to stay deleterious at 50K discards the initially-essential knockouts that became
    # dispensable, which are the tail's strongest decorrelation.  That table finds the initial
    # tail keeps only ~63% of its (noise-corrected) predictability, with ~4.5% of essential
    # knockouts reverting to neutral against 0% in the isogenic control.

    print("\nr_100          = Pearson r over every matched pair (the lethal tail included)")
    print("r_95 / r_90    = same, after removing the lowest 5% / 10% of ANCESTOR effects only;")
    print("                 the evolved side is never used to define the subset")
    print("cut_95/cut_90  = largest ancestor effect excluded; a PERCENTILE, but within this")
    print("                 panel it barely moves (-0.23..-0.29 and -0.07..-0.11)")
    print("f95 / f90      = r_95, r_90 as a fraction of the ISOGENIC control's -- QUOTE THESE,")
    print("                 since raw r is comparable only within one range.  The two green/red")
    print("                 rows exceed 1 by construction: they carry no library-to-library term")
    print("excluded       = non-empty for the two populations Limdi et al. drop as unreliable")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
