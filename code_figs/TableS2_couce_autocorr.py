#!/usr/bin/env python3
r"""Table S2: DFE autocorrelation along the Couce Ara+2 lineage.

The Couce analog of TableS1.  For every transition ``early -> late`` we ask how well a
knockout's fitness effect measured in the EARLIER background predicts its effect in the LATER
one, over the segments assayed in both, and report the Pearson ``r`` of the matched pairs on
four nested subsets defined from the early side alone.  Data: Couce et al., three sequenced
timepoints of the LTEE Ara+2 lineage (data/data_couce), per-SEGMENT transposon-insertion
effects.  Effects are log-fitness selection coefficients, so no conversion is needed.

WHAT THE ROWS ARE.

  3 TRANSITIONS -- the whole lineage, every ordered pair of the three timepoints:

        0K -> 2K     2,000 generations
        0K -> 15K   15,000 generations
        2K -> 15K   13,000 generations

      0K is the REL607 ancestor.  The three libraries were mutagenised INDEPENDENTLY, so a
      matched pair is two separate experiments on two genomes and the pairing is exact only
      because it is keyed on ``alle`` = "<ORF>-<segment 1..5>", the sub-genic unit the authors'
      own scripts match on.  ``site`` is one representative coordinate out of the ``abn``
      insertions pooled into a segment and differs between timepoints for 40% of shared
      segments, so matching on it would be wrong; see the block comment in ``cmn/cmn_exper.py``.

  3 FIT-VARIANT ROWS, one per timepoint -- "<timepoint> fit1 -> fit2".

      READ THESE AS AN UPPER BOUND ON THE CEILING, NOT AS A TECHNICAL CONTROL.  The release
      publishes no independent replicate of a Couce timepoint.  What it does publish is three
      fits of the same five read-count timepoints per segment -- ``fitted`` / ``fitted1`` /
      ``fitted2`` -- and these rows correlate the second against the first.

      ``fitted`` is the plain log-frequency slope (regressing log(count / library total) on the
      timepoint index reproduces it at r = 0.9999, scale 0.150 = 1/6.644, which identifies the
      units as per-generation at log2(100) generations per daily 1:100 transfer).  ``fitted1``
      -- the one the authors use and the one every figure and table in this repo is built on --
      and ``fitted2`` are undocumented variants of that same regression, and no subset of the
      five timepoints reproduces either better than the full set does.

      Because both read the SAME counts their errors are strongly correlated, and the numbers
      say so: if they were independent, ``fitted1 - fitted2`` would have spread
      sqrt(sterr1^2 + sterr2^2), and it is 0.42 / 0.45 / 0.37 times that at 0K / 2K / 15K.  The
      genuinely independent Limdi green/red channels score 1.5-1.8 on the same test.  So these
      rows understate the assay's noise and their r (0.97-0.98 at full range) is a ceiling the
      true reproducibility cannot exceed -- useful because every transition sits far below it,
      but not a substitute for a replicate.  The ``r_*_null`` columns assume independent errors
      on the two sides, so on these three rows the null is a LOWER bound too; compare them only
      to each other.

    THE COUCE DFE HAS NO LETHAL TAIL, which is why this table exists separately from TableS1
    rather than as extra rows in it.  Across all three timepoints exactly one segment of 38,882
    falls below s = -0.3, and it never enters a matched pair.  The knockouts that are lethal in
    DM25 are absent from the library rather than filtered out of the analysis.  A percentile of
    the Couce ancestor therefore lands at s ~ -0.03 where the same percentile of a Limdi
    ancestor lands at s ~ -0.25, and on shared genes of the same strain s_Limdi ~ 1.4 s_Couce,
    so no row here is commensurate with a row there.  Compare within this table only.

ANCESTOR-DEFINED NESTED SUBSETS (the ``r_100 / r_98 / r_95 / r_90`` columns).  Pearson r is
dominated by the points farthest from the origin, so where the subset is cut changes it a great
deal and reporting one cut hides that.  Each row therefore carries r on four nested subsets,
obtained by removing exactly 0%, 2%, 5% and 10% of the LARGEST-MAGNITUDE early effects:

  r_100   every matched pair, nothing removed
  r_98    the largest 2% of |EARLY effect| removed  (``cut_98`` = the |s| threshold)
  r_95    the largest 5% removed                    (``cut_95`` likewise)
  r_90    the largest 10% removed                   (``cut_90`` likewise)

    RANKED ON |s|, NOT ON s.  This table used to rank on the signed effect and drop the most
    deleterious fraction, which left a one-sided subset: the whole beneficial tail kept, only the
    deleterious one trimmed.  Ranking on |s| drops the largest effects of EITHER sign, so the
    retained set is symmetric about zero and each rung is a statement about the near-neutral bulk.
    It is the rule ``cmn/cmn_scatter.py`` applies in fig1 and figs S3-S4 and the rule the walk
    caches behind the simulated columns are generated under, so a measured number, its scatter
    panel and its simulated counterpart all mean the same thing.

The 2% rung is the one fig1 F and figs S3-S4 report, and it is here because the Couce DFE is
compact: its 5% cut already sits inside the bulk, so by r_95 the ladder is no longer removing a
tail at all.  That is also why r_95 and r_90 fall so much faster here than in TableS1 -- they are
eating signal, not tail.

Two properties are why removing a fraction of the EARLY side replaced fixed two-sided cutoffs:

  * The exclusion is defined ONLY from the early measurement, never from the late one whose
    correlation is being computed.  A cutoff applied to both sides conditions on the outcome,
    discarding the segments that are near-neutral early and deleterious late -- which are part
    of the change being measured.
  * Removing a fixed FRACTION by rank keeps the subsets nested (90% inside 95% inside 98%
    inside 100%) and the sample sizes comparable across rows.

MEASUREMENT NOISE.  Every ``r_100/r_98/r_95/r_90`` is the raw observed r; NOTHING is
disattenuated.  The adjacent ``r_*_null`` columns give a forward null expectation under
``Y_E = X + e_E`` and ``Y_L = X + e_L``: one numerically identical true effect ``X`` per
segment, independent mean-zero Gaussian errors, and the published per-segment ``sterr1`` on
both sides.  ``X`` is the inverse-variance weighted mean of the two observed effects.  The null
columns are the median of 1,000 simulations that add fresh errors to BOTH endpoints and rebuild
the early-ranked subsets inside each simulation.  This is an expected raw correlation under no
scrambling, not an estimate of a corrected correlation.

SIMULATED COLUMNS (``r_100_sim_*`` and ``r_98_sim_*``), transition rows only.  What the fitted
adaptive walk predicts for the two rungs the paper reports, from the caches written by
``code_tmp/poster_fig5_couce_noise.py`` under the SAME |s| subset rule as the measured columns.
500 SSWM walks in a heavy-tailed (radial beta-prime) FGM, with a probe library the size of the
matched segment set drawn inside the observed early effect window.  ``_sim_latent`` is the
model's own effects with no measurement error; ``_sim_noisy`` re-measures them with the
published per-segment errors assigned by effect rank, and is the like-for-like comparison.

    EACH TRANSITION IS SIMULATED FROM ITS OWN EARLY BACKGROUND.  0K -> 2K and 0K -> 15K start
    from the 0K MLE; 2K -> 15K starts from the 2K MLE (``couce_2K`` in
    ``data/fig3_fgm_fits.json``), because by 2K the population is on a different part of the
    landscape -- the fitted radius has fallen from r = 0.54 to r = 0.43 -- and starting that
    walk at 0K would ask the model to re-traverse 2,000 generations it has already covered.
    2K is the only background in either dataset that is fitted as well as being an endpoint,
    which makes that pair the one direct check that the fitted radius falls as fitness rises.

WHERE ALONG THE WALK THEY ARE READ (``sim_t``).  At the transition's own fixed-mutation count,
``n_fixed_mut`` below.  This is what separates these rows from TableS1's, whose 0 -> 50K
transitions carry 70 to 2,600 fixed mutations and are far past anything an SSWM walk reaches, so
that table has to read its plateau instead.  Reading at the matching step is available here
because the Couce intervals are 8 to 30 mutations, the same order as the walks themselves.

    It is the same order, not comfortably inside it.  ``sim_walk_len``, the median walk length,
    is reported for every row precisely so this is visible: 0K -> 15K is read at t = 30 against a
    median walk length of 27, so over half the walks have already reached their peak by then and
    that row is closer to a plateau reading than to a mid-walk one.  0K -> 2K at t = 8 is
    genuinely mid-walk.  This costs nothing in validity -- the Couce driver holds a walk at its
    peak once it runs out of beneficial mutations, so all 500 contribute at every step and no row
    is a median over a thinning set of survivors -- but it does mean the three rows are not
    equally far along their walks, and a flat comparison across them would hide that.

FIXED BACKGROUND MUTATIONS (``n_fixed_mut``).  Mutations fixed DURING the interval: roughly 8
between 0K and 2K and 22 between 2K and 15K~\cite{ltee_muts}, as quoted in the main text, hence
30 cumulative over 0K -> 15K.  Earlier versions of this table left the column out on the grounds
that it would need Ara+2 clone sequencing that is not in this repo.  That is true of deriving it
here, but not of the number: it is published, the paper already quotes it, and the simulated
columns have to be read somewhere.  The three fit-variant rows carry 0, as they must -- two fits
of one library are separated by no evolution at all.  ``delta_gen`` is kept alongside it.

THE WEIGHTED COLUMN, ``r_100_w``.  Exactly the pairs of ``r_100`` -- no segment filtered, by
effect size or by error -- but each weighted by w = 1/(sterr_early^2 + sterr_late^2).  This is
NOT disattenuation: it does not divide the noise out, it changes which segments dominate the
sum, demoting the badly measured ones instead of trusting them equally.

    data/TableS2_couce_autocorr.csv
    columns: dataset, transition, kind, delta_gen, n_fixed_mut,
             n_100, r_100, r_100_null, r_100_sim_latent, r_100_sim_noisy, r_100_w,
             n_98, cut_98, r_98, r_98_null, r_98_sim_latent, r_98_sim_noisy,
             n_95, cut_95, r_95, r_95_null, n_90, cut_90, r_90, r_90_null,
             sim_t, sim_walks, sim_walk_len

Run:
    python code_figs/TableS2_couce_autocorr.py
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
from cmn import cmn_exper, cmn_walkcache  # noqa: E402  (shared loaders + walk-cache reader)
from cmn.cmn_exper import DATA_DIR  # noqa: E402

OUT_CSV = os.path.join(DATA_DIR, "TableS2_couce_autocorr.csv")
COLUMNS = ["dataset", "transition", "kind", "delta_gen", "n_fixed_mut",
           "n_100", "r_100", "r_100_null", "r_100_sim_latent", "r_100_sim_noisy",
           "r_100_w",
           "n_98", "cut_98", "r_98", "r_98_null", "r_98_sim_latent", "r_98_sim_noisy",
           "n_95", "cut_95", "r_95", "r_95_null",
           "n_90", "cut_90", "r_90", "r_90_null",
           "sim_t", "sim_walks", "sim_walk_len"]

WALK_DIR = os.path.join(DATA_DIR, "FGM_HEAVY_TAILED")
# Tag of the walk runs written for this table; distinct from the caches Figure 4 reads.
WALK_TAG = "tbl"

# Fractions of the EARLY side removed, LARGEST |effect| first.  0.00 keeps every matched pair,
# so the subsets are nested.  The 2% rung is the one fig1 F and figs S3-S4 report; TableS1's
# 5%/10% rungs are kept for comparability of shape, not of value -- see the docstring on why
# they eat signal rather than tail in this dataset.
TAIL_EXCLUSIONS = (0.00, 0.02, 0.05, 0.10)
# Rank on |s| and drop the largest, not on s dropping the most deleterious.  See RANKED ON |s|
# in the docstring.
EXCLUSION_MODE = "magnitude"
# The rungs that carry a simulated counterpart: the two the paper reports for this dataset.
SIMULATED_PCTS = (100, 98)


def kept_indices(values, fraction):
    """Indices surviving a ``fraction`` cut of the largest-|value| entries."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values), kind="stable")
    return order[: values.size - int(np.floor(fraction * values.size))]


def magnitude_cut(values, fraction):
    """The |value| threshold at or above which entries were dropped; inf when none were."""
    values = np.asarray(values, dtype=float)
    removed = int(np.floor(fraction * values.size))
    if removed == 0:
        return np.inf
    order = np.argsort(np.abs(values), kind="stable")
    return float(np.abs(values[order[values.size - removed]]))

# Forward, two-ended noise-only null.  A fixed seed plus a transition-specific stable hash makes
# every row bit-for-bit reproducible while keeping its random stream independent of row order.
NULL_SIMULATIONS = 1000
NULL_MASTER_SEED = 260820

# Sequenced timepoints of the Ara+2 lineage, and the elapsed generations of each ordered pair.
COUCE_TIMEPOINTS = ("0K", "2K", "15K")
GENERATIONS = {"0K": 0, "2K": 2000, "15K": 15000}
COUCE_TRANSITIONS = (("0K", "2K"), ("0K", "15K"), ("2K", "15K"))
# Mutations fixed DURING each interval, from ltee_muts as quoted in the main text: roughly 8
# over 0K -> 2K and 22 over 2K -> 15K, hence 30 cumulative.  Not derivable in this repo -- the
# local LTEE table carries the 50K timepoint only -- so these are literals, exactly as TableS1's
# N_FIXED_MUT is.  This is where the simulated columns are read.
N_FIXED_MUT = {("0K", "2K"): 8, ("0K", "15K"): 30, ("2K", "15K"): 22}
# Which fit each transition's walk starts from.  A walk out of 2K has to start on the landscape
# the population was on at 2K, not the one it left at 0K.
FIT_DATASETS = {"0K": "couce_0K", "2K": "couce_2K"}


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

    Under no scrambling each segment has one shared effect ``X_i``, fitted as the
    inverse-variance weighted mean of the two observed measurements.  Each of
    ``NULL_SIMULATIONS`` simulations draws a new early measurement around ``X_i`` using that
    segment's early error and a new late measurement using its late error.  The subsets are then
    rebuilt from the SIMULATED early side -- by |s|, exactly as the measured ladder does -- so the
    null includes noise in the endpoint used for selection as well as noise in the endpoint being
    predicted.

    Duplicated from TableS1_limdi_autocorr.py rather than imported: table scripts in this repo
    do not import one another.  The independence assumption is exact for the three transitions
    (independently mutagenised libraries) and violated for the three fit-variant rows, where the
    two sides share their read counts -- there the null is a lower bound.
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
        for column, frac in enumerate(TAIL_EXCLUSIONS):
            kept = kept_indices(sim_a, frac)
            simulated_r[simulation, column], _ = pearson(sim_a[kept], sim_b[kept])

    medians = np.nanmedian(simulated_r, axis=0)
    return [(float(medians[i]), retained_counts[i]) for i in range(len(TAIL_EXCLUSIONS))]


def inverse_variance_pearson(a, a_err, b, b_err):
    """Pearson r with every segment weighted by w = 1/(sterr_early^2 + sterr_late^2).

    Same pairs as ``r_100`` -- nothing is filtered, by effect size or by error -- only counted
    differently.  A pair is dropped only where a sterr is missing or non-positive, which does
    not happen anywhere in the three Couce timepoints; the count used is returned so the caller
    can flag any shortfall.
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


def early_exclusion_ladder(a, a_err, b, b_err, null_seed):
    """``r`` on nested subsets built by removing the largest-|EARLY effect| pairs.

    For each fraction in ``TAIL_EXCLUSIONS`` the ``floor(frac * n)`` matched pairs with the
    largest ``|early effect|`` are dropped and Pearson r is recomputed on what is left.  Only
    ``a`` enters the exclusion -- never ``b``, the side being predicted -- so the subsets do not
    condition on the outcome, and because a fixed fraction is removed by rank they are nested
    and comparably sized across rows.  ``cut`` is the ``|s|`` threshold at or above which pairs
    were dropped, and ``inf`` when nothing was.
    """
    a, a_err, b, b_err = (np.asarray(v, dtype=float) for v in (a, a_err, b, b_err))
    m = np.isfinite(a) & np.isfinite(b)
    a, a_err, b, b_err = a[m], a_err[m], b[m], b_err[m]
    null_results = noise_only_null_ladder(a, a_err, b, b_err, null_seed)
    ladder = []
    for null_column, frac in enumerate(TAIL_EXCLUSIONS):
        kept = kept_indices(a, frac)
        r, n = pearson(a[kept], b[kept])
        r_null, n_null = null_results[null_column]
        ladder.append({
            "frac": frac,
            "pct": int(round(100 * (1.0 - frac))),
            "n": n,
            "r": r,
            "r_null": r_null,
            "n_null": n_null,
            "cut": magnitude_cut(a, frac),
        })
    return ladder


def simulated_ladder(early, late):
    """Simulated latent and noisy r per subset, read at this interval's fixed-mutation count.

    Returns ``None`` when the walk cache is absent, so the table still builds where the walks
    have not been run; a cache simulated under a different subset rule is a hard error instead,
    since reporting it beside an |s|-rule measurement would not look wrong anywhere.
    """
    pattern = (f"poster_fig5_couce_{early}_to_{late}_beta_prime_observed_window_"
               f"rank_noise_k1_w*_e*_m*_{WALK_TAG}.npz")
    try:
        path = cmn_walkcache.locate(WALK_DIR, pattern)
    except FileNotFoundError:
        return None
    cache = cmn_walkcache.read(path)
    cmn_walkcache.require_mode(cache, EXCLUSION_MODE, TAIL_EXCLUSIONS)
    expected = FIT_DATASETS[early]
    stored = cache["metadata"].get("fit_dataset")
    # 0K walks predate the ``fit_dataset`` key and carry the standalone 0K fit file, which is
    # the same optimum; anything else must name the fit it started from, or a 2K -> 15K walk
    # accidentally started at 0K would be reported as if it had started at 2K.
    if stored != expected and not (early == "0K" and stored is None):
        raise RuntimeError(
            f"{os.path.basename(path)} was simulated from fit_dataset={stored!r}, "
            f"not {expected!r}")
    return cmn_walkcache.ladder(cache, time=N_FIXED_MUT[(early, late)])


def make_row(dataset, transition, pairs, kind, delta_gen, interval=None):
    """One output row for a matched pair of measurements.

    ``interval`` is the ``(early, late)`` timepoint pair for a transition row and ``None`` for
    a fit-variant row, which has no evolution to simulate.
    """
    a, a_err, b, b_err = (np.asarray(v, float) for v in pairs)
    r_w, n_w = inverse_variance_pearson(a, a_err, b, b_err)
    row = {
        "dataset": dataset,
        "transition": transition,
        "kind": kind,
        "delta_gen": delta_gen,
        "n_fixed_mut": N_FIXED_MUT[interval] if interval else 0,
        "r_100_w": r_w,
        "n_100_w": n_w,
        "sim_t": "",
        "sim_walks": "",
        "sim_walk_len": "",
    }
    steps = early_exclusion_ladder(a, a_err, b, b_err, _null_seed(transition))
    for step in steps:
        row[f"r_{step['pct']}"] = step["r"]
        row[f"r_{step['pct']}_null"] = step["r_null"]
        row[f"n_{step['pct']}"] = step["n"]
        row[f"n_{step['pct']}_null"] = step["n_null"]
        row[f"cut_{step['pct']}"] = step["cut"]
    for pct in SIMULATED_PCTS:
        row[f"r_{pct}_sim_latent"] = np.nan
        row[f"r_{pct}_sim_noisy"] = np.nan

    simulated = simulated_ladder(*interval) if interval else None
    if simulated is not None:
        by_pct = {step["pct"]: index for index, step in enumerate(steps)}
        if len(simulated["latent"]) != len(steps):
            raise RuntimeError(f"{transition}: cache has {len(simulated['latent'])} subsets, "
                               f"table has {len(steps)}")
        for pct in SIMULATED_PCTS:
            column = by_pct[pct]
            row[f"r_{pct}_sim_latent"] = float(simulated["latent"][column])
            row[f"r_{pct}_sim_noisy"] = float(simulated["noisy"][column])
        row["sim_t"] = simulated["time"]
        row["sim_walks"] = simulated["total_walks"]
        row["sim_walk_len"] = simulated["walk_length_median"]
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Couce et al. -- three timepoints of the Ara+2 lineage, matched on ``alle``
# ══════════════════════════════════════════════════════════════════════════════
def timepoint_pair(early, late, fit=1):
    """Matched ``(a, a_err, b, b_err)`` for two timepoints, on the segments assayed in both."""
    a_eff = cmn_exper.load_couce_segment_series(early, fit=fit)
    a_err = cmn_exper.load_couce_segment_errors(early, fit=fit)
    b_eff = cmn_exper.load_couce_segment_series(late, fit=fit)
    b_err = cmn_exper.load_couce_segment_errors(late, fit=fit)
    idx = a_eff.index.intersection(b_eff.index)
    return (a_eff[idx].to_numpy(float), a_err[idx].to_numpy(float),
            b_eff[idx].to_numpy(float), b_err[idx].to_numpy(float))


def fit_variant_pair(timepoint):
    """The two published fits of one timepoint, ``fitted1`` against ``fitted2``.

    Same segments and same index for both -- they are columns of one cleaned frame.  This is
    NOT an independent replicate pair; see the FIT-VARIANT ROWS block in the docstring for the
    diagnostic that shows their errors are shared, and for how far to trust the number.
    """
    a_eff = cmn_exper.load_couce_segment_series(timepoint, fit=1)
    a_err = cmn_exper.load_couce_segment_errors(timepoint, fit=1)
    b_eff = cmn_exper.load_couce_segment_series(timepoint, fit=2)
    b_err = cmn_exper.load_couce_segment_errors(timepoint, fit=2)
    return (a_eff.to_numpy(float), a_err.to_numpy(float),
            b_eff.to_numpy(float), b_err.to_numpy(float))


def shared_error_ratio(timepoint):
    """``sd(fitted1 - fitted2) / median sqrt(sterr1^2 + sterr2^2)`` for one timepoint.

    1 if the two fits were independent measurements of the same truth.  Well below 1 means
    their errors are shared, which is what disqualifies the fit-variant rows as controls.
    """
    a, a_err, b, b_err = fit_variant_pair(timepoint)
    expected = np.sqrt(a_err ** 2 + b_err ** 2)
    return float(np.std(a - b) / np.median(expected))


def build_rows():
    """A fit-variant row per timepoint, then the three transitions in increasing span."""
    rows = [make_row(f"Couce {tp}", f"{tp} fit1 -> fit2", fit_variant_pair(tp),
                     kind="fit variant", delta_gen=0)
            for tp in COUCE_TIMEPOINTS]
    for early, late in COUCE_TRANSITIONS:
        rows.append(make_row(f"Couce {early}->{late}", f"{early} -> {late}",
                             timepoint_pair(early, late), kind="evolved",
                             delta_gen=GENERATIONS[late] - GENERATIONS[early],
                             interval=(early, late)))
    return rows


def _number(value):
    """Format a float for the CSV, leaving an absent simulated value as an empty cell."""
    return "" if value is None or not np.isfinite(value) else f"{value:.4g}"


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            record = [row["dataset"], row["transition"], row["kind"], row["delta_gen"],
                      row["n_fixed_mut"],
                      row["n_100"], _number(row["r_100"]), _number(row["r_100_null"]),
                      _number(row["r_100_sim_latent"]), _number(row["r_100_sim_noisy"]),
                      _number(row["r_100_w"])]
            for pct in (98, 95, 90):
                record += [row[f"n_{pct}"], _number(row[f"cut_{pct}"]),
                           _number(row[f"r_{pct}"]), _number(row[f"r_{pct}_null"])]
                if pct in SIMULATED_PCTS:
                    record += [_number(row[f"r_{pct}_sim_latent"]),
                               _number(row[f"r_{pct}_sim_noisy"])]
            record += [row["sim_t"], row["sim_walks"], row["sim_walk_len"]]
            writer.writerow(record)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    print("\nnested subsets: the largest 0% / 2% / 5% / 10% of |EARLY effect| removed, "
          "late side free")
    print("cut = the |s| threshold at or above which early segments were dropped")
    print("sim = the fitted adaptive walk at this interval's fixed-mutation count; "
          "latent, then noisy")
    print("rows: 3 fit-variant rows (fitted1 vs fitted2 of one timepoint -- an UPPER BOUND on the")
    print("      assay ceiling, not a replicate control), then the 3 transitions")

    def cell(value, width=8):
        return f"{value:>{width}.3f}" if np.isfinite(value) else f"{'-':>{width}}"

    header = (f"{'dataset':<16}{'transition':<18}{'kind':<13}{'dgen':>7}{'nfix':>6}"
              f"{'n_100':>7}{'r_100':>8}{'null':>8}{'sim':>8}{'simN':>8}{'r_100_w':>9}"
              f"{'n_98':>7}{'cut_98':>8}{'r_98':>8}{'null':>8}{'sim':>8}{'simN':>8}"
              f"{'n_95':>7}{'cut_95':>8}{'r_95':>8}{'null':>8}"
              f"{'n_90':>7}{'cut_90':>8}{'r_90':>8}{'null':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (f"{row['dataset']:<16}{row['transition']:<18}{row['kind']:<13}"
                f"{row['delta_gen']:>7}{row['n_fixed_mut']:>6}"
                f"{row['n_100']:>7}{cell(row['r_100'])}{cell(row['r_100_null'])}"
                f"{cell(row['r_100_sim_latent'])}{cell(row['r_100_sim_noisy'])}"
                f"{cell(row['r_100_w'], 9)}")
        for pct in (98, 95, 90):
            line += (f"{row[f'n_{pct}']:>7}{cell(row[f'cut_{pct}'])}{cell(row[f'r_{pct}'])}"
                     f"{cell(row[f'r_{pct}_null'])}")
            if pct in SIMULATED_PCTS:
                line += (f"{cell(row[f'r_{pct}_sim_latent'])}"
                         f"{cell(row[f'r_{pct}_sim_noisy'])}")
        print(line)

    without_walks = [r["transition"] for r in rows
                     if r["kind"] == "evolved" and not np.isfinite(r["r_100_sim_latent"])]
    if without_walks:
        print("\nNo walk cache found for: " + ", ".join(without_walks))
        print("  run code_tmp/poster_fig5_couce_noise.py with "
              f"--exclusions {' '.join(str(f) for f in TAIL_EXCLUSIONS)} "
              f"--exclusion-mode {EXCLUSION_MODE} --tag {WALK_TAG}")

    simulated = [r for r in rows
                 if r["kind"] == "evolved" and np.isfinite(r["r_100_sim_noisy"])]
    if simulated:
        print("\nmeasured against the simulated NOISY value at n_fixed_mut "
              "(the like-for-like comparison):")
        hdr = (f"  {'transition':<14}{'t':>4}" + "".join(
            f"{f'r_{pct}':>9}{'sim':>9}{'diff':>9}" for pct in SIMULATED_PCTS))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for row in simulated:
            line = f"  {row['transition']:<14}{row['sim_t']:>4}"
            for pct in SIMULATED_PCTS:
                measured, sim = row[f"r_{pct}"], row[f"r_{pct}_sim_noisy"]
                line += f"{measured:>9.3f}{sim:>9.3f}{measured - sim:>+9.3f}"
            print(line)

    short = [r for r in rows if r["n_100_w"] < r["n_100"]]
    if short:
        print("\nr_100_w dropped pairs with a missing/non-positive sterr:")
        for r in short:
            print(f"  {r['transition']}: {r['n_100'] - r['n_100_w']} of {r['n_100']}")

    # The diagnostic that decides how the fit-variant rows may be read.  1.0 is what an
    # independent replicate pair scores; the Limdi green/red channels give 1.5-1.8.
    print("\nfit-variant independence check, sd(fitted1 - fitted2) / median sqrt(sterr1^2+sterr2^2):")
    for tp in COUCE_TIMEPOINTS:
        print(f"  {tp:>4}: {shared_error_ratio(tp):.3f}")
    print("  (1.0 = independent.  All well below, so the two fits share their read counts and")
    print("   their r overstates the reproducibility of a Couce timepoint.)")

    # Does the autocorrelation decay with the span?  Three points is not a fit, but the ordering
    # is the check that matters: 0->2K should sit above 0->15K at every rung.
    evolved = [r for r in rows if r["kind"] == "evolved"]
    print("\ndecay with elapsed generations (3 transitions -- an ordering check, not a fit):")
    for key in ("r_100", "r_100_w", "r_98", "r_95", "r_90"):
        values = "  ".join(f"{r['transition']}={r[key]:+.3f}" for r in evolved)
        if scipy_pearsonr is None:
            rr, _ = pearson([r["delta_gen"] for r in evolved], [r[key] for r in evolved])
        else:
            rr = scipy_pearsonr([r["delta_gen"] for r in evolved],
                                [r[key] for r in evolved]).statistic
        print(f"  {key:<8} {values}   r(delta_gen) = {rr:+.3f}")

    print("\nr_100          = Pearson r over every matched pair")
    print(f"r_*_null       = median of {NULL_SIMULATIONS} forward simulations under Y_E = X + e_E and")
    print("                 Y_L = X + e_L, using the published per-segment sterr on BOTH sides")
    print("                 and rebuilding the early-ranked subset inside every simulation.")
    print("                 Assumes independent errors, so on the fit-variant rows it is a")
    print("                 lower bound and must not be read against the transitions")
    print("r_100_w        = the same pairs as r_100 -- NO segments filtered out -- but weighted")
    print("                 by w = 1/(sterr_early^2 + sterr_late^2).  Not disattenuation")
    print("r_*_sim_*      = the fitted heavy-tailed FGM adaptive walk on the same subsets, read")
    print("                 at n_fixed_mut steps; _latent has no measurement error and _noisy")
    print("                 re-measures it with rank-matched published per-segment errors.")
    print("                 0K rows start from the 0K MLE, 2K -> 15K from the 2K MLE")
    print("r_98/95/90     = same, after removing the largest 2% / 5% / 10% of |EARLY effect|;")
    print("                 the late side is never used to define the subset.  The Couce DFE is")
    print("                 compact, so by r_95 the cut is inside the bulk and eating signal")
    print("cut_*          = the |s| threshold at or above which early segments were dropped")
    print("kind           = 'fit variant' (two fits of one timepoint) or 'evolved' (a transition)")
    print("delta_gen      = elapsed generations")
    print("n_fixed_mut    = mutations fixed during the interval, from ltee_muts (8 / 22 / 30)")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
