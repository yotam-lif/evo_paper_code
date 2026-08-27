#!/usr/bin/env python3
r"""Table S2: DFE autocorrelation across the Ascensao monoculture panel.

The Ascensao analog of TableS1.  For every transition ``early -> late`` we ask how well a gene
knockout's fitness effect measured in the EARLIER background predicts its effect in the LATER one,
over the genes assayed in both, and report the Pearson ``r`` of the matched pairs on three nested
subsets defined from the early side alone.  Data: Ascensao et al., "Quantifying the Adaptive
Potential of a Nascent Bacterial Community" (github.com/joaoascensao/S-L-REL606-BarSeq).

WHAT THE ROWS ARE.  Every MONOCULTURE experiment in the release -- one barcoded knockout library
growing alone -- which is four environments assayed with all three genotypes, plus a standalone
REL606 repeat:

    E_SLR  "Mono"      DM25,           1:100 every 24 h,       6.64 gen/day
    E_MNO  "1:10 dil"  DM27.8,         1:10  every 24 h,       3.32 gen/day
    E_PQT  "Glu exp"   DM25,           1:100 every 4.5-8 h,    ~4-6.7 gen/transfer
    E_GHI  "Ac exp"    DM2000-acetate, variable every 24 h,    ~2.5-6 gen/day
    E_U    "Mono 2"    DM25,           1:100 every 24 h, 8 days, four replicates

EVERY ONE OF THESE IS A SERIAL DILUTION -- nothing here is a chemostat, and the ``description``
column should be read as (dilution factor)/(transfer interval) plus what that does to the growth
cycle.  Only two knobs are ever turned, and the second column below is the consequence:

  E_SLR  1:100/24h  full cycle   the LTEE regime exactly.  log2(100) = 6.6439 gen/day, glucose
                                 exhausted within hours, the remaining ~19 h in stationary phase.
  E_U    1:100/24h  full cycle   the same regime again, 8 days and four replicates instead of two.
  E_MNO  1:10/24h   long stat    still a full batch cycle, but only log2(10) = 3.3219 doublings are
                                 needed to refill, so the growth phase is short and stationary
                                 phase correspondingly long -- which the authors state as the
                                 point of the experiment.  DM27.8 is used because 0.9 x 27.8 = 25,
                                 so the glucose AFTER a 1:10 transfer equals DM25's.
  E_PQT  1:100/5-8h exp only     same medium and same 1:100 dilution as E_SLR; the culture is
                                 simply harvested at the end of exponential phase (measured at
                                 5.25 h for S and L, 8.25 h for REL606) instead of at 24 h, which
                                 deletes stationary phase.  Generations per transfer from CFU
                                 counts.  Only REL606's first two transfers passed QC, so T rests
                                 on 3 timepoints where P and Q have 5 -- it is the noisiest
                                 ancestor in the panel, and its control ceiling shows it.
  E_GHI  var/24h    exp only     acetate at 80x the carbon of DM25, transferred daily by a VARIABLE
                                 volume: OD is read and the culture set to an OD_0 that lands it at
                                 OD ~ 0.6 a day later, still mid-exponential.  Generations =
                                 log2(OD_f/OD_0), so they differ by strain and replicate -- on
                                 acetate REL606 grows at ~0.08/h against L 0.12/h and S 0.18/h.

So "exp only" never means continuous culture.  It means diluted often enough (E_PQT) or into
enough carbon (E_GHI) that the cells never run out and never leave exponential phase.  Source:
Ascensao et al. 2023, Nat Commun 14:248, Methods "BarSeq experiments"; the per-cycle generation
counts in the ``E_*_meta.csv`` files agree with it exactly (6.64385619 = log2 100 for E_SLR and
E_U, 3.3219 = log2 10 for E_MNO, measured and strain-specific for E_GHI and E_PQT).

The genotypes are the LTEE Ara-2 ancestor REL606 and the two ecotypes S and L that diversified
from it, so ``REL606 -> S`` and ``REL606 -> L`` are the ancestor-to-evolved transitions, eight in
all.  The CO-CULTURE experiments (C, D, F, V, W, X, Y -- one library against a wild-type majority,
or at ecological equilibrium) are deliberately excluded: there the selection coefficients are
measured against a different and changing background, so they are not the same observable.

    NOTE, since an earlier version of the Ascensao table got this wrong.  The four experiment
    folders are four ENVIRONMENTS, not four different ancestors.  ``E_GHI_meta.csv`` and its
    siblings show the same ``TnPool R`` library in each, and BarSeq_meta.csv describes I, O, T and
    R all as "REL606 monoculture".  The four ancestor arrays are therefore one genotype measured
    four ways, and the 0.01-0.66 correlation between them is G x E, not genotype difference.  Those
    cross-environment comparisons are printed at the end as a reference scale; they are NOT
    controls and are not rows of the table.

A PER-EXPERIMENT CONTROL (this is what TableS1 could not do).  TableS1 has ONE shared control for
the whole Limdi panel, because there is only one isogenic ancestor pair.  Here every experiment
carries its own.  The release fits each biological replicate separately (``s_inference.py``) and
then re-fits jointly over both replicates' barcode counts to produce the combined value
(``s_inference_combo.py``, which is what our .npy arrays hold).  So for every strain in every
experiment, replicate 1 against replicate 2 is a zero-evolution control measured in exactly the
conditions of that experiment: same genotype, same library, same medium, same flask regime,
nothing between the two numbers but assay noise.

That matters more here than it would in Limdi, because the assay quality varies enormously between
experiments -- the control's own r_90 runs from 0.13 (L in DM25) to 0.84 (REL606 at 1:10 dilution).
A single shared ceiling would be meaningless across that range.  ``f95``/``f90`` therefore divide
each row by the ANCESTOR's control IN ITS OWN EXPERIMENT: the ancestor is the side the nested
subsets are defined from, and it is common to both transitions of an environment.  Each evolved
strain's own control is a row too, so its reproducibility is visible rather than assumed.

MEASUREMENT DEPTH.  Every transition uses the authors' COMBINED two-replicate fit on BOTH sides --
the same values as the .npy arrays the rest of this repo uses, and never a single replicate.  Every
control is necessarily two SINGLE replicates, because that is the only form a within-experiment
control can take.  This is not a column: it follows from the comparison, which is either
"X rep1 vs rep2" (a control, replicate depth) or "REL606 -> X" (a transition, combined depth).

That asymmetry biases ``f95``/``f90`` in a known direction, and it is worth being explicit about
it.  A combined fit is less noisy than one replicate -- by Spearman-Brown its reliability is
2r/(1+r) against a single replicate's r -- so each transition is divided by a ceiling noisier than
itself, which makes f GENEROUS: it overstates how much of the DFE survived.  The transitions
collapse to f90 ~ 0 regardless, so the finding is safe against a bias that works in its favour.
This was checked directly rather than assumed: recomputing the transitions at single-replicate
depth, averaged over all cross-replicate pairings so they are exactly noise-matched to the
controls, moves r_90 by at most ~0.02 (e.g. REL606 -> S in DM25 goes -0.067 to -0.044, in glucose
-0.048 to -0.051) and changes no conclusion.  Those rows are not carried in the table; only
combined-to-combined transitions are.

GENE MATCHING.  Genes are matched on ``gene_ID`` (ECB_#####) through a pandas index intersection --
never on row position, and never on gene_symbol.  This is not a formality: the gene sets genuinely
differ between strains, with 3021 genes for REL606 in DM25 against 2361 for L in the same
experiment, and they differ again between a replicate fit and the combined fit (the combined
requires the gene to pass in both replicates).  The cache builder verifies before writing that
gene_ID is unique and non-null in all 41 tables and that the gene_ID -> gene_symbol map has zero
conflicts across them, and value-checks every combined table against the existing .npy arrays
(all match to max|diff| = 0).  Every row here prints the pair count it actually used.

THE LADDER, AND WHAT ``cut`` MEANS.  Each row reports r three times, on nested subsets built by
throwing away the most deleterious genes as measured in the EARLY background only:

  r_100   every matched gene, nothing removed
  r_95    the lowest 5% of early-side effects removed
  r_90    the lowest 10% of early-side effects removed

The removal is by RANK, so the subsets are nested and equally sized across rows.  ``cut_95`` and
``cut_90`` translate that rank back onto the effect scale: each is the LARGEST (least negative)
early-side ``s`` that the cut threw away, so the subset kept is everything above it.  Reading the
E_SLR ancestor control, ``cut_95 = -0.0326`` means dropping the worst 5% amounted to discarding
every gene with s <= -0.0326 per generation and keeping the rest.  There is no ``cut_100`` because
nothing is removed at that rung.  The column exists so the knife's position is visible in physical
units, since a percentile is not a fixed value of s and moves from row to row.

THE WEIGHTED COLUMN, ``r_100_w``.  Exactly the pairs of ``r_100`` -- no gene is filtered out, by
fitness effect or by anything else -- but each weighted by w = 1/(sigma_early^2 + sigma_late^2),
using the published per-gene ``s std``.  That is the authors' own weight and formula:
``analyses/corr_bw_envs/WeightedCorr.py`` takes weighted means and a weighted covariance, and
``corr_bootstrap.py`` builds w as ``1/(stdA**2 + stdB**2)`` after merging on gene_ID.  The only
pairs it can lose are those with a missing or non-positive published sigma, which is 2 genes in
one replicate of E_U and nowhere else in the 41 tables; any shortfall is printed.

It is here because the paper's Fig. 1E reports rho = 0.99 for the E_SLR REL606 replicate pair,
against 0.747 for our r_100, and the gap deserves to be visible rather than explained away.  It is
not a disagreement: that panel's caption says the correlation is weighted AND that every knockout
with sigma_s > 0.3% was excluded, which is 70% of the genes.  Decomposing on that pair:

    unweighted, all 3013 genes ........ 0.747   <- r_100, what this table reports
    weighted,   all 3013 genes ........ 0.961   <- r_100_w, what this column adds
    unweighted, sigma_s < 0.3% (897) .. 0.958
    weighted,   sigma_s < 0.3% (897) .. 0.987   <- the full Fig. 1E recipe

Only the weight is adopted here; the noise filter is not, because filtering on sigma in BOTH
backgrounds partly conditions on the late side (see below), which is the thing the ladder exists
to avoid.

Adopting the weight sharpens the table rather than softening it.  Ceilings rise (E_SLR REL606
0.747 -> 0.961, E_PQT 0.795 -> 0.970) while transitions fall (REL606 -> S 0.314 -> 0.056 in DM25,
0.432 -> -0.003 in glucose exponential phase), so control and transition separate further.  The
mechanism is worth stating because it is the same one the ladder exploits: sigma tracks |s|
(Spearman +0.41 to +0.47 across these strains), since a knockout whose barcodes crash is measured
badly, so inverse-variance weighting demotes the deleterious tail -- for E_SLR REL606 a
sigma < 0.3% cut keeps half of all genes but only 7% of the lowest-decile tail.  For a CONTROL,
noise is the only thing separating the two numbers, so demoting it drives r towards 1.  For a
TRANSITION it also strips the high-leverage tail that was holding r_100 up, leaving the bulk,
which is uncorrelated.  Weighted and rank-based subsetting reach the same place by different doors.

Read ``r_100_w`` against the control in its own block only: the weights differ from row to row, so
it is even less comparable across rows than r_100 is.  There is deliberately no weighted f-column
-- f95/f90 stay defined on the unweighted ladder, which is the table's primary result.

Removing a rank rather than a fixed value is what makes this table commensurate with TableS1.
Effects here are per GENERATION (multiply by ~6.64 for per-cycle), roughly an order of magnitude
smaller than Limdi's per-cycle values -- Limdi's percentiles land near s = -0.25 where these land
near s = -0.03 -- but Pearson r is scale-free and a rank is scale-free, so the two ladders mean the
same thing despite the units.  An absolute cutoff would not transfer at all.

NOISE-ONLY NULL.  Each experimental ``r_100/r_95/r_90`` is followed by ``r_*_null``, the expected
raw correlation under ``Y_A = X + e_A`` and ``Y_E = X + e_E``: one numerically identical true
effect ``X`` per gene, independent mean-zero Gaussian errors, and the published gene-specific
``s_std`` on both sides.  For each gene, ``X`` is fitted as the inverse-variance weighted mean of
the two observed effects.  The null columns are the median of 1,000 simulations that add fresh
errors to BOTH endpoints and then rebuild the ancestor-ranked 100/95/90 subsets inside each
simulation.  This is a forward expectation under no scrambling, not a disattenuated version of
the experimental correlation.

    data/TableS3_ascensao_autocorr.csv
    columns: experiment, media, description, comparison, n_100, r_100, r_100_null, r_100_w,
             n_95, cut_95, r_95, r_95_null, n_90, cut_90, r_90, r_90_null, f95, f90

``media`` and ``description`` are split because they are independent, and DM25 is why: it is the
medium of THREE different experiments here -- E_SLR's 24 h cycle, E_PQT's 5-8 h cycle and E_U's
8-day repeat -- so the medium alone would collapse three regimes into one label, and the regime
alone would not say that E_GHI's carbon source is different.

Requires the cache built by ``data/data_ascensao_2/build_monoculture_from_repo.py``.

Run:
    python code_figs/TableS3_ascensao_autocorr.py
"""
import argparse
import csv
import hashlib
import itertools
import os
import sys

import numpy as np
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_exper  # noqa: E402  (shared experimental-data loaders + structure)
from cmn.cmn_exper import (  # noqa: E402
    DATA_DIR, ASENCAO_MONO, ASENCAO_MONO_ENVIRONMENTS,
)

OUT_CSV = os.path.join(DATA_DIR, "TableS3_ascensao_autocorr.csv")
COLUMNS = ["experiment", "media", "description", "comparison",
           "n_100", "r_100", "r_100_null", "r_100_w",
           "n_95", "cut_95", "r_95", "r_95_null",
           "n_90", "cut_90", "r_90", "r_90_null", "f95", "f90"]

# Fractions of the EARLY (ancestor) side removed, lowest effect first, for the nested-subset
# ladder.  Same rule and same fractions as TableS1_limdi_autocorr.py, so the two tables are read the
# same way.  Deliberately duplicated rather than imported: figure/table scripts in this repo do
# not import one another (shared loaders live in cmn/).
TAIL_EXCLUSIONS = (0.00, 0.05, 0.10)

# Forward, two-ended noise-only null.  A fixed seed plus a row-specific stable hash makes every
# value bit-for-bit reproducible while keeping random streams independent of row order.
NULL_SIMULATIONS = 1000
NULL_MASTER_SEED = 260820


def pearson(a, b):
    """Pearson r over the entries finite in both a and b, plus the pair count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0.0 or np.std(b[mask]) == 0.0:
        return np.nan, n
    return float(pearsonr(a[mask], b[mask])[0]), n


def _null_seed(label):
    """Stable uint64 seed for one row label; unlike Python's ``hash``, stable across processes."""
    payload = f"{NULL_MASTER_SEED}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def noise_only_null_ladder(a, a_err, b, b_err, seed):
    r"""Median r from a forward null with independent Gaussian noise on BOTH endpoints.

    Under no scrambling each gene has one shared effect ``X_i``.  Its fitted value is the
    inverse-variance weighted mean of the two observed measurements.  Each of
    ``NULL_SIMULATIONS`` simulations draws a new early measurement around ``X_i`` using that
    gene's early-side error and a new late measurement using its late-side error.  The 100/95/90
    subsets are rebuilt from the SIMULATED early side, so the null includes noise in the endpoint
    used for selection as well as noise in the endpoint being predicted.

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
    retained_counts = [n - int(np.floor(frac * n)) for frac in TAIL_EXCLUSIONS]
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


def matched(early, late):
    """Values of two strains over the genes measured in BOTH, matched by gene_ID.

    Each argument is the ``(s, s_std)`` pair the loader returns.  The intersection is the whole
    point: the two strains' gene sets differ by hundreds of genes, so anything positional would
    silently pair different genes.  Returns ``(a, a_err, b, b_err)`` in a common, sorted gene
    order; the ladder uses the values alone and only ``r_100_w`` touches the errors.
    """
    (a, a_err), (b, b_err) = early, late
    idx = a.index.intersection(b.index).sort_values()
    return (a.loc[idx].to_numpy(float), a_err.loc[idx].to_numpy(float),
            b.loc[idx].to_numpy(float), b_err.loc[idx].to_numpy(float))


def inverse_variance_pearson(a, a_err, b, b_err):
    """Pearson r with every gene weighted by w = 1/(sigma_a^2 + sigma_b^2).

    The authors' own definition: ``analyses/corr_bw_envs/WeightedCorr.py`` computes weighted means
    and a weighted covariance, and ``corr_bootstrap.py`` builds the weights as
    ``1/(stdA**2 + stdB**2)`` after merging on gene_ID -- exactly what is done here.  The genes are
    NOT filtered, by fitness effect or by error: every pair that enters ``r_100`` enters this too,
    it is only counted differently.  A pair is dropped only if a published sigma is missing or
    non-positive, which happens for 2 genes in one replicate of E_U and nowhere else in the 41
    tables; the count actually used is returned so the caller can flag any shortfall.
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


def ancestor_exclusion_ladder(a, a_err, b, b_err, null_seed=None):
    """``r`` on nested subsets built by removing the lowest EARLY-side effects.

    For each fraction in ``TAIL_EXCLUSIONS`` the ``floor(frac * n)`` matched pairs with the
    smallest early effect are dropped and r is recomputed on the rest.  Only ``a`` enters the
    exclusion -- never ``b``, the side being predicted -- so the subsets never condition on the
    outcome, and because a fixed fraction is removed by rank they are nested and equally sized
    across rows regardless of how different the two experiments' effect scales are.
    """
    a, a_err, b, b_err = (np.asarray(v, dtype=float) for v in (a, a_err, b, b_err))
    m = np.isfinite(a) & np.isfinite(b)
    a, a_err, b, b_err = a[m], a_err[m], b[m], b_err[m]
    order = np.argsort(a, kind="stable")
    if null_seed is None:
        null_results = [(np.nan, 0) for _ in TAIL_EXCLUSIONS]
    else:
        null_results = noise_only_null_ladder(a, a_err, b, b_err, null_seed)
    ladder = []
    for null_column, frac in enumerate(TAIL_EXCLUSIONS):
        n_removed = int(np.floor(frac * a.size))
        kept = order[n_removed:]
        r, n = pearson(a[kept], b[kept])
        r_null, n_null = null_results[null_column]
        ladder.append({
            "pct": int(round(100 * (1.0 - frac))),
            "n": n,
            "r": r,
            "r_null": r_null,
            "n_null": n_null,
            "cut": -np.inf if n_removed == 0 else float(a[order[n_removed - 1]]),
        })
    return ladder


def reps(letter):
    """Every individual replicate fit of one strain, as a list of gene_ID-indexed (s, s_std)."""
    return [cmn_exper.asencao_mono_series(letter, rep=r, errors=True)
            for r in range(1, ASENCAO_MONO[letter][4] + 1)]


def make_row(experiment, media, description, comparison, kind, pairs):
    """Assemble one output row from the matched ``(a, a_err, b, b_err)`` arrays.

    ``kind`` ("control" / "evolved") is kept on the row for the ceiling logic and the printed
    grouping but is NOT a column: the comparison string already says which a row is
    ("X rep1 vs rep2" against "REL606 -> S").  Measurement depth is likewise not a column, since
    it follows from the same distinction -- every transition is a combined-vs-combined fit and
    every control is replicate-vs-replicate.  See MEASUREMENT DEPTH in the docstring.
    """
    a, a_err, b, b_err = pairs
    ladder = ancestor_exclusion_ladder(
        a, a_err, b, b_err, _null_seed(f"{experiment}:{comparison}"))
    r_w, n_w = inverse_variance_pearson(a, a_err, b, b_err)
    row = {"experiment": experiment, "media": media, "description": description,
           "comparison": comparison, "kind": kind, "r_100_w": r_w, "n_100_w": n_w}
    for step in ladder:
        row[f"r_{step['pct']}"] = step["r"]
        row[f"r_{step['pct']}_null"] = step["r_null"]
        row[f"n_{step['pct']}"] = step["n"]
        row[f"n_{step['pct']}_null"] = step["n_null"]
        row[f"cut_{step['pct']}"] = step["cut"]
    return row


def build_rows():
    """Per environment: the three replicate controls, then the two combined-fit transitions.

    The control rows come first in each block because they set the scale everything after them is
    read against.  E_U has no evolved partner, so it contributes a control only.
    """
    rows = []
    for folder, media, desc, anc_letter, evolved in ASENCAO_MONO_ENVIRONMENTS:
        anc_eco = ASENCAO_MONO[anc_letter][1]

        # Controls: replicate 1 vs replicate 2 of each strain, in this experiment.
        for eco, letter in ((anc_eco, anc_letter),) + evolved:
            r1, r2 = reps(letter)[0], reps(letter)[1]
            rows.append(make_row(folder, media, desc, f"{eco} rep1 vs rep2", "control",
                                 matched(r1, r2)))

        # Transitions: the authors' COMBINED fit on both sides, never a single replicate.
        anc_comb = cmn_exper.asencao_mono_series(anc_letter, errors=True)
        for eco, letter in evolved:
            rows.append(make_row(
                folder, media, desc, f"{anc_eco} -> {eco}", "evolved",
                matched(anc_comb, cmn_exper.asencao_mono_series(letter, errors=True))))

    # The standalone REL606 repeat: controls only, and every disjoint replicate pair it allows.
    _, eco, media, desc, n_rep = ASENCAO_MONO["U"]
    u = reps("U")
    for i in range(0, n_rep - 1, 2):
        rows.append(make_row("U", media, desc, f"{eco} rep{i+1} vs rep{i+2}", "control",
                             matched(u[i], u[i + 1])))
    return rows


def attach_ceilings(rows):
    """Add f95/f90: each row over the ANCESTOR's control in the SAME experiment, same depth.

    Per-experiment by construction -- there is no shared ceiling in this panel and there should
    not be one, since assay quality varies several-fold between experiments.  A control row
    divided by itself is 1.000 by definition and is left in as the visual anchor.
    """
    ceilings = {}
    for row in rows:
        if row["kind"] == "control" and row["comparison"].startswith("REL606 rep1"):
            ceilings[row["experiment"]] = row
    for row in rows:
        ceil = ceilings.get(row["experiment"])
        for level in (95, 90):
            row[f"f{level}"] = (row[f"r_{level}"] / ceil[f"r_{level}"]
                                if ceil is not None else np.nan)
    return rows


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["experiment"], row["media"], row["description"],
                             row["comparison"], row["n_100"], f"{row['r_100']:.4g}",
                             f"{row['r_100_null']:.4g}",
                             f"{row['r_100_w']:.4g}",
                             row["n_95"], f"{row['cut_95']:.4g}", f"{row['r_95']:.4g}",
                             f"{row['r_95_null']:.4g}",
                             row["n_90"], f"{row['cut_90']:.4g}", f"{row['r_90']:.4g}",
                             f"{row['r_90_null']:.4g}",
                             f"{row['f95']:.4g}", f"{row['f90']:.4g}"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = attach_ceilings(build_rows())
    write_table(rows, args.out)

    print("\nnested subsets: the lowest 0% / 5% / 10% of EARLY-side effects removed, late side free")
    print("controls are replicate 1 vs replicate 2 of one strain in ONE experiment -- zero")
    print("evolution, so f95/f90 divide by the ANCESTOR's control in that same experiment")
    print("effects are per GENERATION (x6.64 for per-cycle); genes matched on gene_ID")
    header = (f"{'exp':<6}{'media':<16}{'description':<25}{'comparison':<20}"
              f"{'n_100':>7}{'r_100':>8}{'null':>8}{'r_100_w':>9}"
              f"{'n_95':>7}{'cut_95':>9}{'r_95':>8}{'null':>8}{'f95':>8}"
              f"{'n_90':>7}{'cut_90':>9}{'r_90':>8}{'null':>8}{'f90':>8}")
    print()
    print(header)
    print("-" * len(header))
    last = None
    for row in rows:
        if last is not None and row["experiment"] != last:
            print()
        last = row["experiment"]
        print(f"{row['experiment']:<6}{row['media']:<16}{row['description']:<25}"
              f"{row['comparison']:<20}"
              f"{row['n_100']:>7}{row['r_100']:>8.3f}{row['r_100_null']:>8.3f}"
              f"{row['r_100_w']:>9.3f}"
              f"{row['n_95']:>7}{row['cut_95']:>9.4f}{row['r_95']:>8.3f}"
              f"{row['r_95_null']:>8.3f}{row['f95']:>8.3f}"
              f"{row['n_90']:>7}{row['cut_90']:>9.4f}{row['r_90']:>8.3f}"
              f"{row['r_90_null']:>8.3f}{row['f90']:>8.3f}")

    short = [r for r in rows if r["n_100_w"] < r["n_100"]]
    if short:
        print("\nr_100_w dropped pairs with a missing/non-positive published sigma:")
        for r in short:
            print(f"  {r['experiment']} {r['comparison']}: "
                  f"{r['n_100'] - r['n_100_w']} of {r['n_100']}")

    # The same REL606 library across environments: a reference scale, NOT a control.  This is what
    # an earlier version of the table mistook for four different ancestors.
    print("\nreference scale -- the SAME REL606 library across environments (G x E, not a control):")
    anc = [(f"E_{folder} ({media})", letter)
           for folder, media, _desc, letter, _ in ASENCAO_MONO_ENVIRONMENTS]
    for (c1, l1), (c2, l2) in itertools.combinations(anc, 2):
        pairs = matched(cmn_exper.asencao_mono_series(l1, errors=True),
                        cmn_exper.asencao_mono_series(l2, errors=True))
        lad = ancestor_exclusion_ladder(*pairs)
        print(f"  {c1:<24} vs {c2:<24} n={lad[0]['n']:>5} "
              f"r_100={lad[0]['r']:+.3f}  r_100_w={inverse_variance_pearson(*pairs)[0]:+.3f}  "
              f"r_95={lad[1]['r']:+.3f}  r_90={lad[2]['r']:+.3f}")

    # E_SLR's REL606 and E_U are the same genotype in the same medium, measured as two separate
    # experiments -- the one true zero-evolution pair in the panel at COMBINED depth.
    pairs = matched(cmn_exper.asencao_mono_series("R", errors=True),
                    cmn_exper.asencao_mono_series("U", errors=True))
    lad = ancestor_exclusion_ladder(*pairs)
    print(f"\ncombined-depth zero-evolution check -- E_SLR REL606 vs E_U REL606 (same medium, "
          f"separate experiment):\n  n={lad[0]['n']}  r_100={lad[0]['r']:+.3f}  "
          f"r_100_w={inverse_variance_pearson(*pairs)[0]:+.3f}  "
          f"r_95={lad[1]['r']:+.3f}  r_90={lad[2]['r']:+.3f}")

    print("\nr_100          = Pearson r over every matched pair")
    print(f"r_*_null       = median of {NULL_SIMULATIONS} forward simulations under Y_A = X + e_A and")
    print("                 Y_E = X + e_E, using the published per-gene errors on both sides;")
    print("                 the 100/95/90 subsets are rebuilt from simulated Y_A each time")
    print("r_100_w        = the same pairs as r_100 -- NO genes filtered out -- but weighted by")
    print("                 w = 1/(sigma_early^2 + sigma_late^2), the authors' own weight")
    print("                 (analyses/corr_bw_envs/). Compare Fig 1E, which ALSO drops every")
    print("                 gene with sigma_s > 0.3%; that filter is not applied here")
    print("r_95 / r_90    = same, after removing the lowest 5% / 10% of EARLY-side effects only;")
    print("                 the late side is never used to define the subset")
    print("cut_95/cut_90  = the rank cut put back on the effect scale: the LARGEST (least")
    print("                 negative) early-side s that was thrown away, so the kept subset is")
    print("                 everything above it.  Per generation.  No cut_100: nothing is cut")
    print("description    = (dilution factor)/(transfer interval), and what that does to the")
    print("                 growth cycle.  ALL FIVE experiments are serial dilutions -- none is")
    print("                 a chemostat.  'full cycle' = carbon runs out and the culture sits in")
    print("                 stationary phase (the LTEE regime); 'long stat' = the same but only")
    print("                 3.32 doublings refill it, so stationary phase is longer still;")
    print("                 'exp only' = transferred before the carbon runs out, so the cells")
    print("                 never leave exponential phase -- E_PQT by transferring every 5-8 h,")
    print("                 E_GHI by diluting daily into 80x the carbon (acetate)")
    print("comparison     = 'X rep1 vs rep2' is a control (one strain against itself, replicate")
    print("                 depth); 'REL606 -> X' is a transition (combined fits on both sides)")
    print("f95 / f90      = r as a fraction of the ANCESTOR's control in the SAME experiment")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
