#!/usr/bin/env python3
r"""Table S1: DFE autocorrelation across the Limdi LTEE TnSeq panel.

For every transition ``early -> late`` we ask how well a mutation's fitness effect measured
in the EARLIER background predicts its effect in the LATER one, over the mutations assayed in
both.  That is the DFE autocorrelation -- the Pearson ``r`` of the matched pairs -- and it is
the observable the p-spin / FGM scrambling picture predicts decays with the number of fixed
mutations separating the two genotypes.  Effects are LOG-fitness selection coefficients, so no
conversion is needed.

Rows, all from the Limdi LTEE TnSeq panel (data/anurag_data).  Per-GENE effects, averaged over
the TA sites in the gene and (except in the replicate rows) over both technical replicates.
Genes are matched on the metadata row index of the .npy fitness matrices.

  12 EVOLVED transitions -- each 50K clone against its own founder: REL606 -> Ara-N and
      REL607 -> Ara+N, twelve independent 0 -> 50K comparisons.

  TECHNICAL CONTROLS, ONE PER LIBRARY -- 14 rows, "<pop> green -> red".  The two barcode channels of a
      single library, unaveraged: same strain, same mutagenesis, same assay, correlated against
      each other.  Nothing separates the two numbers but measurement noise, so each is the
      reproducibility of that one measurement and the hardest ceiling there is.  Both ancestors
      have one (these are the rows the poster figure shows as its control panel,
      ``code_tmp/poster_fig1.py`` panel C) and so does every 50K clone, printed directly above
      its own transition.

      The per-clone rows are there because the twelve 50K measurements are NOT equally good, and
      a single shared ceiling hides that.  Their r_90 runs 0.650 to 0.893.  The printed block
      ``evolved r_90 against its OWN 50K technical control`` puts each transition next to the
      reproducibility of the very measurement it depends on: an evolved row can only be read as
      scrambling where its own control is high.  Two rows are worth reading that way.  Ara-2
      reproduces BETTER than almost anything in the panel (control r_90 = 0.848) yet its
      transition is -0.081, which is the cleanest possible demonstration that its problem is
      systematic bias and not noise -- exactly what Limdi et al. excluded it for.  Ara+6, a
      mutator with 2600 fixed mutations, has a fine control (0.833) and the lowest genuine
      transition (0.132), so there the collapse IS the landscape.

  All 14 control rows carry ``n_fixed_mut = 0`` because a green -> red pair is one library
  against itself, so on the scrambling axis they are the origin.  REL606 -> REL607 is deliberately
  not included as a control because the araA marker may itself affect fitness.

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
-0.081, far below every other row.  Its problem is a systematic bias, not noise, so
no error-based correction could have rescued it -- which is why the flag is the right handling.

FIXED BACKGROUND MUTATIONS (``n_fixed_mut``).  Mutations fixed DURING that transition: the full
0 -> 50K complement for an evolved row, and 0 for every technical control.  This is the distance the
p-spin / FGM picture predicts r decays with.  The counts split cleanly into the six LTEE mutator
lines (Ara-1/-2/-3/-4, Ara+3, Ara+6: 800-2600) and the six non-mutators (70-125), which is the
standard picture and an internal check on the column.  Note the panel spans barely more than one
decade once the mutators are set aside, so the decay fits printed at the end are weak by
construction -- they are a consistency check, not the paper's evidence for decay.

MEASUREMENT NOISE.  Noise attenuates r: with per-observation error variance ``eps^2`` on each
side, ``r_obs = r_true * sqrt((V_e - eps_e^2)/V_e * (V_l - eps_l^2)/V_l)``.  Every number in this
table's ``r_100/r_95/r_90`` columns is the raw ``r_obs``; NOTHING is disattenuated, and that is
deliberate.  The adjacent ``r_*_null`` columns instead give a forward null expectation under
``Y_A = X + e_A`` and ``Y_E = X + e_E``: one numerically identical true effect ``X`` per gene,
independent mean-zero Gaussian errors, and the published per-gene standard errors on both sides.
For each gene, ``X`` is fitted as the inverse-variance weighted mean of the two observed effects.
The null columns are the median of 1,000 simulations that add fresh errors to BOTH endpoints and
then rebuild the ancestor-ranked 100/95/90 subsets inside each simulation.  This is an expected raw
correlation under no scrambling, not an estimate of a corrected correlation.  The correction
needs the reliability of each side, and stripping the deleterious tail strips most of the signal
variance with it: the ancestor reliability falls from ~0.98 at r_100 to ~0.49-0.67 at r_90, which
is exactly where the classical formula stops being trustworthy.  The 14 green/red controls answer
the within-library question empirically: they land at r_90 = 0.650 to 0.893, with the two ancestors
at 0.650 and 0.707.  The per-clone green/red row is the diagnostic that says whether a given evolved
row is worth interpreting at all.

THE WEIGHTED COLUMN, ``r_100_w``.  The one place the published per-gene errors are used.  Exactly
the pairs of ``r_100`` -- no gene filtered, by effect size or by error -- but each weighted by
w = 1/(sigma_early^2 + sigma_late^2) from ``errors_genes_inv.npy``.  This is NOT disattenuation
(see above, which still stands): it does not try to divide the noise out, it changes which genes
dominate the sum, demoting the badly measured ones instead of trusting them equally.  The same
column is in TableS2, where it reconciles this repo's r_100 with the rho = 0.99 the Ascensao paper
reports for a replicate pair in its Fig. 1E.

What it does here is the same in direction and sharper in the two places that matter.  The
green/red controls barely move (0.939-0.981 raw against 0.953-0.982 weighted) because for those
rows noise is the ONLY thing in play and it is already small, while every evolved row falls
(Ara-1 0.869 -> 0.780, Ara+6 0.820 -> 0.659), because weighting demotes the badly-measured
deleterious genes whose leverage was holding r_100 up.  The two flagged libraries are where it
speaks loudest.  Ara-2's technical control goes 0.879 -> 0.982, the best in the panel, while its
transition goes 0.532 -> 0.179: with the noisy genes demoted, that library reproduces essentially
perfectly and still fails to predict itself across 50K generations, which is systematic bias
stated about as plainly as the data can state it.  Ara+4, excluded for poor technical replicates,
likewise recovers to 0.958.

Read ``r_100_w`` against the controls in this table only.  The weights differ row to row, so it is
even less comparable across rows than r_100; the unweighted ladder remains the table's primary
result.

    data/TableS1_couce_autocorr.csv
    columns: dataset, transition, kind, n_fixed_mut, n_100, r_100, r_100_null, r_100_w,
             n_95, cut_95, r_95, r_95_null, n_90, cut_90, r_90, r_90_null, excluded

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
RETAINED evolved Limdi rows from r_100 ~ 0.82-0.91 to r_90 ~ 0.13-0.34.  The two flagged
populations start lower and fall further still -- Ara+4 0.729 -> 0.129 and Ara-2 0.532 -> -0.081
-- which is the measurement problem the source paper excluded them for, not extra scrambling.
Compare r only within the same retained fraction; r is not comparable across ranges.

    A percentile is not a fixed value of s, so ``cut_95`` and ``cut_90`` are reported for every
    row.  Across the twelve EVOLVED rows they barely move -- the 5% quantile of the ancestor sits
    at s = -0.233 to -0.286 and the 10% quantile at s = -0.071 to -0.098 -- which is unsurprising,
    since all twelve rank on one of only two founders.  So those subsets are near-identical in
    absolute terms as well as in size, and r_90 really is row-to-row comparable among them.

    The green/red CONTROL rows are the exception and must not be read that way.  Each ranks on its
    own library's green channel, and a single unaveraged channel has both more noise and a
    library-specific tail, so their cuts spread over s = -0.072 to -0.402 (5%) and -0.029 to
    -0.282 (10%).  Ara-2 sits at the shallow end and Ara+4 at the deep end.  Compare a control
    with its own evolved row, which is what the per-strain pairing is for, and do not compare two
    controls' raw r to each other.  The same caution applies to any library with a different
    lethal fraction; see the note on the removed Couce rows above.

TAIL DECORRELATION.  The complementary question -- does the conserved tail *also* scramble? --
has its own table, code_tmp/Table_tail_autocorr.py, because it must be conditioned differently:
on the ANCESTOR tail only (genes with ancestral s < -0.3), with the evolved effect left free.
Conditioning on both sides would discard the initially-essential knockouts that became
dispensable after evolution, which are precisely the tail's strongest decorrelation.  Done that
way, ~4.5% of essential knockouts revert to near-neutral.  So the tail is *mostly* conserved but not perfectly -- even large knockout costs shift
over 50K generations -- while the near-neutral bulk is where the bulk of the scrambling lives.

Run:
    python code_figs/TableS1_couce_autocorr.py
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np
try:
    from scipy.stats import pearsonr as scipy_pearsonr
except ImportError:  # The table itself needs only NumPy; p-values below are optional diagnostics.
    scipy_pearsonr = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_exper  # noqa: E402  (shared experimental-data loaders + structure)
from cmn.cmn_exper import (  # noqa: E402
    DATA_DIR, LIMDI_ANCESTORS, LIMDI_EVOLVED, LIMDI_PANEL,
)

OUT_CSV = os.path.join(DATA_DIR, "TableS1_couce_autocorr.csv")
# Every r column is the poster-figure ladder: r on nested subsets defined by removing a fraction
# of the ANCESTOR side only (see ANCESTOR-DEFINED NESTED SUBSETS in the docstring).  The earlier
# ``n / autocorr / autocorr_corr`` block -- r at a fixed ``s > -0.3`` cut applied to BOTH sides,
# plus its disattenuated value -- is gone: cutting the late side conditions on the outcome, and
# with no fixed cut left there is no range where disattenuation is trustworthy.
COLUMNS = ["dataset", "transition", "kind", "n_fixed_mut",
           "n_100", "r_100", "r_100_null", "r_100_w",
           "n_95", "cut_95", "r_95", "r_95_null",
           "n_90", "cut_90", "r_90", "r_90_null", "excluded"]

# Fractions of the EARLY (ancestor) side removed, lowest effect first, for the nested-subset
# ladder.  0.00 keeps every matched pair, so the subsets are nested: 90% inside 95% inside 100%.
# Same rule and same fractions as code_tmp/poster_fig1.py, which shows two of these rows.
TAIL_EXCLUSIONS = (0.00, 0.05, 0.10)

# Forward, two-ended noise-only null.  A fixed seed plus a transition-specific stable hash makes
# every row bit-for-bit reproducible while keeping its random stream independent of row order.
NULL_SIMULATIONS = 1000
NULL_MASTER_SEED = 260820

# Mutations fixed during each transition, keyed by the (unique) transition label: the full
# 0 -> 50K complement of each population.  Every control row sits at 0, the origin of the
# scrambling axis -- the araA marker separating REL606 from REL607 is a single neutral point
# mutation, and a green -> red row is one library against itself.
N_FIXED_MUT = {
    "REL606 -> Ara-1": 1100, "REL606 -> Ara-2": 1000, "REL606 -> Ara-3": 800,
    "REL606 -> Ara-4": 1300, "REL606 -> Ara-5": 90,   "REL606 -> Ara-6": 90,
    "REL607 -> Ara+1": 125,  "REL607 -> Ara+2": 70,   "REL607 -> Ara+3": 1800,
    "REL607 -> Ara+4": 70,   "REL607 -> Ara+5": 80,   "REL607 -> Ara+6": 2600,
}
# One green -> red technical control per library -- the two ancestors AND all twelve 50K clones.
# Zero evolution separates the two channels of a single measurement, whichever library it is.
N_FIXED_MUT.update({f"{pop} green -> red": 0 for pop in LIMDI_PANEL})

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
    x, y = a[mask], b[mask]
    dx, dy = x - np.mean(x), y - np.mean(y)
    denominator = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    return (float(np.sum(dx * dy) / denominator) if denominator > 0.0 else np.nan), n


def _null_seed(label):
    """Stable uint64 seed for one row label; unlike Python's ``hash``, stable across processes."""
    payload = f"{NULL_MASTER_SEED}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def noise_only_null_ladder(a, a_err, b, b_err, seed):
    r"""Median r from a forward null with independent Gaussian noise on BOTH endpoints.

    Under no scrambling each gene has one shared effect ``X_i``.  Its fitted value is the
    inverse-variance weighted mean of the two observed measurements.  Each of
    ``NULL_SIMULATIONS`` simulations draws a new ancestor measurement around ``X_i`` using that
    gene's ancestor error and a new late measurement using its late error.  The 100/95/90 subsets
    are then rebuilt from the SIMULATED ancestor, so the null includes noise in the endpoint used
    for selection as well as noise in the endpoint being predicted.

    Returns the median simulated r and retained count for each fraction in ``TAIL_EXCLUSIONS``.
    """
    a, a_err, b, b_err = (np.asarray(v, dtype=float) for v in (a, a_err, b, b_err))
    mask = (np.isfinite(a) & np.isfinite(b) & np.isfinite(a_err) & np.isfinite(b_err)
            & (a_err > 0.0) & (b_err > 0.0))
    n = int(mask.sum())
    if n < 3:
        return [(np.nan, 0) for _ in TAIL_EXCLUSIONS]
    a, a_err, b, b_err = a[mask], a_err[mask], b[mask], b_err[mask]
    weight_a, weight_b = 1.0 / a_err ** 2, 1.0 / b_err ** 2
    shared_effect = (weight_a * a + weight_b * b) / (weight_a + weight_b)

    rng = np.random.default_rng(seed)
    simulated_r = np.full((NULL_SIMULATIONS, len(TAIL_EXCLUSIONS)), np.nan, dtype=float)
    retained_counts = []
    for frac in TAIL_EXCLUSIONS:
        retained_counts.append(n - int(np.floor(frac * n)))

    for simulation in range(NULL_SIMULATIONS):
        sim_a = shared_effect + rng.normal(size=n) * a_err
        sim_b = shared_effect + rng.normal(size=n) * b_err
        order = np.argsort(sim_a, kind="stable")
        for column, frac in enumerate(TAIL_EXCLUSIONS):
            n_removed = int(np.floor(frac * n))
            kept = order[n_removed:]
            simulated_r[simulation, column], _ = pearson(sim_a[kept], sim_b[kept])

    medians = np.nanmedian(simulated_r, axis=0)
    return [(float(medians[i]), retained_counts[i]) for i in range(len(TAIL_EXCLUSIONS))]


def inverse_variance_pearson(a, a_err, b, b_err):
    """Pearson r with every gene weighted by w = 1/(sigma_early^2 + sigma_late^2).

    Same pairs as ``r_100`` -- nothing is filtered, by effect size or by error -- only counted
    differently.  ``sigma`` is the panel's published per-gene measurement error
    (``errors_genes_inv.npy``).  A pair is dropped only where a sigma is missing or non-positive,
    which is one gene in REL607 and nowhere else in the 14 libraries; the count used is returned
    so the caller can flag any shortfall.  Duplicated from TableS2_ascensao_autocorr.py rather
    than imported: table scripts in this repo do not import one another.
    """
    m = (np.isfinite(a) & np.isfinite(b) & np.isfinite(a_err) & np.isfinite(b_err)
         & (a_err > 0) & (b_err > 0))
    n = int(m.sum())
    if n < 3:
        return np.nan, n
    x, y = a[m], b[m]
    w = 1.0 / (a_err[m] ** 2 + b_err[m] ** 2)
    w = w / w.sum()
    dx, dy = x - (w * x).sum(), y - (w * y).sum()
    denom = np.sqrt((w * dx * dx).sum() * (w * dy * dy).sum())
    return (float((w * dx * dy).sum() / denom) if denom > 0 else np.nan), n


def ancestor_exclusion_ladder(a, a_err, b, b_err, null_seed):
    """``r`` on nested subsets built by removing the lowest ANCESTOR effects.

    For each fraction in ``TAIL_EXCLUSIONS`` the ``floor(frac * n)`` matched pairs with the
    smallest EARLY-background effect are dropped and Pearson r is recomputed on what is left.
    Only ``a`` enters the exclusion -- never ``b``, the side being predicted -- so the subsets do
    not condition on the outcome, and because a fixed fraction is removed by rank they are nested
    and equally sized across rows.  Returns one dict per fraction with the retained pair count,
    the cutoff (the largest EXCLUDED ancestor effect; ``-inf`` when nothing is excluded) and r.
    """
    a, a_err, b, b_err = (np.asarray(v, dtype=float) for v in (a, a_err, b, b_err))
    m = np.isfinite(a) & np.isfinite(b)
    a, a_err, b, b_err = a[m], a_err[m], b[m], b_err[m]
    order = np.argsort(a, kind="stable")
    null_results = noise_only_null_ladder(a, a_err, b, b_err, null_seed)
    ladder = []
    for null_column, frac in enumerate(TAIL_EXCLUSIONS):
        n_removed = int(np.floor(frac * a.size))
        kept = order[n_removed:]
        r, n = pearson(a[kept], b[kept])
        r_null, n_null = null_results[null_column]
        ladder.append({
            "frac": frac,
            "pct": int(round(100 * (1.0 - frac))),
            "n": n,
            "r": r,
            "r_null": r_null,
            "n_null": n_null,
            "cut": -np.inf if n_removed == 0 else float(a[order[n_removed - 1]]),
        })
    return ladder


def make_row(dataset, transition, pairs, population, kind):
    """One output row for a matched pair of backgrounds.

    ``population`` is the library the row is ABOUT -- the late background of a transition, or the
    single library of a green -> red control.  It is passed explicitly rather than parsed back out
    of ``transition`` so that a control row carries the same quality flag as its evolved row: for
    "Ara+4 green -> red" the old string-splitting would have looked up "red" and found nothing,
    silently un-flagging the one row that is the direct evidence for the exclusion.

    ``r_100 / r_95 / r_90`` is the poster-figure ladder: r on nested subsets with 0%, 5% and 10%
    of the most deleterious ANCESTOR effects removed, which never conditions on the outcome and
    keeps the subsets comparable across rows.  All three are raw -- see MEASUREMENT NOISE and
    ANCESTOR-DEFINED NESTED SUBSETS in the docstring for why nothing here is disattenuated.
    """
    a, a_err, b, b_err = (np.asarray(v, float) for v in pairs)
    r_w, n_w = inverse_variance_pearson(a, a_err, b, b_err)
    row = {
        "dataset": dataset,
        "transition": transition,
        "kind": kind,
        "n_fixed_mut": N_FIXED_MUT[transition],
        "excluded": LIMDI_EXCLUDED.get(population, ""),
        "r_100_w": r_w,
        "n_100_w": n_w,
    }
    for step in ancestor_exclusion_ladder(a, a_err, b, b_err, _null_seed(transition)):
        row[f"r_{step['pct']}"] = step["r"]
        row[f"r_{step['pct']}_null"] = step["r_null"]
        row[f"n_{step['pct']}"] = step["n"]
        row[f"n_{step['pct']}_null"] = step["n_null"]
        row[f"cut_{step['pct']}"] = step["cut"]
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Limdi et al. -- each 50K clone against its own founder, matched on gene row index
# ══════════════════════════════════════════════════════════════════════════════
def limdi_pair(early, late):
    """Matched ``(a, a_err, b, b_err)`` for two Limdi populations, on the genes measured in both."""
    a_eff, a_err = cmn_exper.limdi_gene_series(early, errors=True)
    b_eff, b_err = cmn_exper.limdi_gene_series(late, errors=True)
    idx = a_eff.index.intersection(b_eff.index)
    return (a_eff[idx].to_numpy(), a_err[idx].to_numpy(),
            b_eff[idx].to_numpy(), b_err[idx].to_numpy())


def replicate_pair(pop):
    """The Green and Red channels of one library, unaveraged -- the technical control.

    The panel publishes ONE sigma per (gene, library): the inverse-variance weighted SEM over the
    per-TA-site estimates of BOTH channels.  There is no per-channel sigma to be had, so the same
    array is handed to both sides, which is the statement that the two channels are equally
    precise.  That is the only assumption available and it costs nothing here: if each channel has
    variance 2*sigma^2, then 1/(sigma_green^2 + sigma_red^2) = 1/(4*sigma^2), proportional to
    1/sigma^2 -- and a weighted Pearson is invariant to an overall scaling of the weights, so the
    weighting is exactly what it would be with the true per-channel errors.
    """
    green, red = cmn_exper.limdi_channel_series(pop)
    _, sigma = cmn_exper.limdi_gene_series(pop, errors=True)
    # ``sigma`` is the standard error of the Green/Red average.  With equally precise channels,
    # an individual channel has standard error ``sqrt(2) * sigma``.  Absolute scaling did not
    # affect r_100_w, but it is essential for the noise-only null expectation.
    channel_sigma = np.sqrt(2.0) * sigma[green.index].to_numpy()
    return green.to_numpy(), channel_sigma, red.to_numpy(), channel_sigma


def replicate_row(pop):
    """The green -> red technical control row for one library."""
    return make_row("Limdi rep", f"{pop} green -> red", replicate_pair(pop),
                    population=pop, kind="control")


def build_rows():
    """The panel-level controls, then each evolved population preceded by its OWN control.

    The order is deliberate.  First the two ancestors' green -> red rows.  Then, for each 50K
    population, its own green -> red control at
    50K immediately before its transition, so the evolved r is read against the reproducibility of
    that very measurement rather than against a panel average.  That adjacency is the point: the
    twelve 50K clones do NOT measure equally well, and a single shared ceiling hides it.

    ``green`` is the early side of a replicate row, so the nested subsets are defined from it
    exactly as they are from the founder elsewhere -- which channel plays that part is arbitrary,
    and it changes r by less than 0.001.
    """
    rows = [replicate_row(pop) for pop in LIMDI_ANCESTORS]
    for anc in LIMDI_ANCESTORS:
        for evo in LIMDI_EVOLVED[anc]:
            rows.append(replicate_row(evo))
            rows.append(make_row(f"Limdi {evo}", f"{anc} -> {evo}", limdi_pair(anc, evo),
                                 population=evo, kind="evolved"))
    return rows


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["dataset"], row["transition"], row["kind"], row["n_fixed_mut"],
                             row["n_100"], f"{row['r_100']:.4g}",
                             f"{row['r_100_null']:.4g}",
                             f"{row['r_100_w']:.4g}",
                             row["n_95"], f"{row['cut_95']:.4g}", f"{row['r_95']:.4g}",
                             f"{row['r_95_null']:.4g}",
                             row["n_90"], f"{row['cut_90']:.4g}", f"{row['r_90']:.4g}",
                             f"{row['r_90_null']:.4g}",
                             row["excluded"]])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    # ---- the poster-figure ladder: nested subsets defined from the ancestor side only --------
    print("\nnested subsets: the lowest 0% / 5% / 10% of ANCESTOR effects removed, evolved side free")
    print("cut = largest excluded ancestor effect")
    print("rows: 14 green/red technical controls (one per library)")
    print("      + 12 evolved, each printed directly under its own 50K green/red control")
    header = (f"{'dataset':<15}{'transition':<20}{'kind':<9}{'n_fix':>6}"
              f"{'n_100':>7}{'r_100':>8}{'null':>8}{'r_100_w':>9}"
              f"{'n_95':>7}{'cut_95':>8}{'r_95':>8}{'null':>8}"
              f"{'n_90':>7}{'cut_90':>8}{'r_90':>8}{'null':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['dataset']:<15}{row['transition']:<20}{row['kind']:<9}"
              f"{row['n_fixed_mut']:>6}"
              f"{row['n_100']:>7}{row['r_100']:>8.3f}{row['r_100_null']:>8.3f}"
              f"{row['r_100_w']:>9.3f}"
              f"{row['n_95']:>7}{row['cut_95']:>8.3f}{row['r_95']:>8.3f}"
              f"{row['r_95_null']:>8.3f}"
              f"{row['n_90']:>7}{row['cut_90']:>8.3f}{row['r_90']:>8.3f}"
              f"{row['r_90_null']:>8.3f}"
              f"   {row['excluded']}")

    short = [r for r in rows if r["n_100_w"] < r["n_100"]]
    if short:
        print("\nr_100_w dropped pairs with a missing/non-positive published sigma:")
        for r in short:
            print(f"  {r['transition']}: {r['n_100'] - r['n_100_w']} of {r['n_100']}")

    # Does the autocorrelation actually decay with the number of fixed mutations?  Reported on the
    # evolved populations only (all controls sit at the origin), with and without the two
    # the source paper drops, and at each level of the ladder -- the full range is tail-dominated,
    # so the bulk subsets are the informative ones.
    evolved = [r for r in rows if r["n_fixed_mut"] > 0]
    print()
    for key, label in (("r_100", "r_100"), ("r_100_w", "r_100_w"),
                       ("r_95", "r_95"), ("r_90", "r_90")):
        for tag, sub in (("all 12", evolved),
                         ("10 retained", [r for r in evolved if not r["excluded"]])):
            x = np.log10([r["n_fixed_mut"] for r in sub])
            if scipy_pearsonr is None:
                rr, _ = pearson(x, [r[key] for r in sub])
                pp = np.nan
            else:
                rr, pp = scipy_pearsonr(x, [r[key] for r in sub])
            print(f"decay of {label:<8} with log10(n_fixed_mut), {tag:<12} "
                  f"r = {rr:+.3f}  p = {pp:.3f}")

    # With a control per strain, the obvious question is whether an evolved row is low because the
    # landscape scrambled or because that clone simply measured badly.  Print them side by side:
    # ``own`` is the strain's green -> red r_90, ``anc`` its founder's, and ``r/own`` the evolved
    # r_90 over its own control.  A row can only be trusted as scrambling where ``own`` is high.
    print("\nevolved r_90 against its OWN 50K technical control (green vs red of that clone):")
    hdr = (f"  {'population':<10}{'own_r90':>9}{'anc_r90':>9}{'evo_r90':>9}{'r/own':>8}"
           f"   excluded")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    own = {r["transition"].split(" ")[0]: r for r in rows if r["kind"] == "control"}
    for row in rows:
        if row["kind"] != "evolved":
            continue
        pop = row["transition"].split(" -> ")[1]
        anc = row["transition"].split(" -> ")[0]
        ratio = row["r_90"] / own[pop]["r_90"] if own[pop]["r_90"] > 0 else np.nan
        print(f"  {pop:<10}{own[pop]['r_90']:>9.3f}{own[anc]['r_90']:>9.3f}"
              f"{row['r_90']:>9.3f}{ratio:>8.3f}   {row['excluded']}")

    # TAIL DECORRELATION lives in its own table, code_tmp/Table_tail_autocorr.py.  It must
    # condition on the ANCESTOR tail only (evolved side free), not on both sides: requiring the
    # gene to stay deleterious at 50K discards the initially-essential knockouts that became
    # dispensable, which are the tail's strongest decorrelation.

    print("\nr_100          = Pearson r over every matched pair (the lethal tail included)")
    print(f"r_*_null       = median of {NULL_SIMULATIONS} forward simulations under Y_A = X + e_A and")
    print("                 Y_E = X + e_E, using independent published per-gene errors on BOTH sides")
    print("                 and rebuilding the ancestor-ranked subset inside every simulation")
    print("r_100_w        = the same pairs as r_100 -- NO genes filtered out -- but weighted by")
    print("                 w = 1/(sigma_early^2 + sigma_late^2) from the published per-gene")
    print("                 errors.  Not disattenuation: it demotes badly measured genes")
    print("                 rather than dividing the noise out.  Controls hold, evolved rows")
    print("                 fall; Ara-2 goes 0.879 -> 0.982 control against 0.532 -> 0.179")
    print("r_95 / r_90    = same, after removing the lowest 5% / 10% of ANCESTOR effects only;")
    print("                 the evolved side is never used to define the subset")
    print("cut_95/cut_90  = largest ancestor effect excluded; a PERCENTILE, but within this")
    print("                 panel it barely moves (-0.23..-0.29 and -0.07..-0.11)")
    print("kind           = control (one library against itself) or evolved")
    print("own_r90        = that clone's OWN green/red control at 50K -- the reproducibility of")
    print("                 the very measurement the evolved row depends on")
    print("excluded       = non-empty for the two populations Limdi et al. drop as unreliable")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
