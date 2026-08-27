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
dominated by the points farthest from the origin, so where the tail is cut changes it a great
deal and reporting one cut hides that.  Each row therefore carries r on four nested subsets,
obtained by removing exactly 0%, 2%, 5% and 10% of the most deleterious EARLY effects:

  r_100   every matched pair, nothing removed
  r_98    the lowest 2% of EARLY effects removed   (``cut_98`` = largest excluded effect)
  r_95    the lowest 5% removed                    (``cut_95`` likewise)
  r_90    the lowest 10% removed                   (``cut_90`` likewise)

The 2% rung is the one fig1 F and figs S3-S4 report, and it is here because the Couce tail is
shallow: its 5th percentile already sits inside the bulk (cut_95 ~ -0.03 against cut_98 ~
-0.05), so by r_95 the ladder is no longer removing a tail at all.  That is also why r_95 and
r_90 fall so much faster here than in TableS1 -- they are eating signal, not tail.

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

THE WEIGHTED COLUMN, ``r_100_w``.  Exactly the pairs of ``r_100`` -- no segment filtered, by
effect size or by error -- but each weighted by w = 1/(sterr_early^2 + sterr_late^2).  This is
NOT disattenuation: it does not divide the noise out, it changes which segments dominate the
sum, demoting the badly measured ones instead of trusting them equally.

GENERATIONS, NOT FIXED MUTATIONS (``delta_gen``).  TableS1 carries the number of mutations fixed
during each transition, which is what the p-spin / FGM picture predicts r decays with.  No
equivalent column is given here: it would need the Ara+2 clone sequencing at 2K and 15K, which
is not in this repo, and inventing it would be worse than leaving it out.  ``delta_gen`` is the
exact elapsed generations instead.  Ara+2 is a non-mutator, so its fixed-mutation complement
over these intervals is small -- which is the point the main text makes about the timescale.

    data/TableS2_couce_autocorr.csv
    columns: dataset, transition, kind, delta_gen, n_100, r_100, r_100_null, r_100_w,
             n_98, cut_98, r_98, r_98_null, n_95, cut_95, r_95, r_95_null,
             n_90, cut_90, r_90, r_90_null

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
from cmn import cmn_exper  # noqa: E402  (shared experimental-data loaders + structure)
from cmn.cmn_exper import DATA_DIR  # noqa: E402

OUT_CSV = os.path.join(DATA_DIR, "TableS2_couce_autocorr.csv")
COLUMNS = ["dataset", "transition", "kind", "delta_gen",
           "n_100", "r_100", "r_100_null", "r_100_w",
           "n_98", "cut_98", "r_98", "r_98_null",
           "n_95", "cut_95", "r_95", "r_95_null",
           "n_90", "cut_90", "r_90", "r_90_null"]

# Fractions of the EARLY side removed, lowest effect first.  0.00 keeps every matched pair, so
# the subsets are nested.  The 2% rung is the one fig1 F and figs S3-S4 report; TableS1's
# 5%/10% rungs are kept for comparability of shape, not of value -- see the docstring on why
# they eat signal rather than tail in this dataset.
TAIL_EXCLUSIONS = (0.00, 0.02, 0.05, 0.10)

# Forward, two-ended noise-only null.  A fixed seed plus a transition-specific stable hash makes
# every row bit-for-bit reproducible while keeping its random stream independent of row order.
NULL_SIMULATIONS = 1000
NULL_MASTER_SEED = 260820

# Sequenced timepoints of the Ara+2 lineage, and the elapsed generations of each ordered pair.
COUCE_TIMEPOINTS = ("0K", "2K", "15K")
GENERATIONS = {"0K": 0, "2K": 2000, "15K": 15000}
COUCE_TRANSITIONS = (("0K", "2K"), ("0K", "15K"), ("2K", "15K"))


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
    rebuilt from the SIMULATED early side, so the null includes noise in the endpoint used for
    selection as well as noise in the endpoint being predicted.

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
        order = np.argsort(sim_a, kind="stable")
        for column, frac in enumerate(TAIL_EXCLUSIONS):
            kept = order[int(np.floor(frac * n)):]
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
    """``r`` on nested subsets built by removing the lowest EARLY effects.

    For each fraction in ``TAIL_EXCLUSIONS`` the ``floor(frac * n)`` matched pairs with the
    smallest early effect are dropped and Pearson r is recomputed on what is left.  Only ``a``
    enters the exclusion -- never ``b``, the side being predicted -- so the subsets do not
    condition on the outcome, and because a fixed fraction is removed by rank they are nested
    and comparably sized across rows.
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


def make_row(dataset, transition, pairs, kind, delta_gen):
    """One output row for a matched pair of measurements."""
    a, a_err, b, b_err = (np.asarray(v, float) for v in pairs)
    r_w, n_w = inverse_variance_pearson(a, a_err, b, b_err)
    row = {
        "dataset": dataset,
        "transition": transition,
        "kind": kind,
        "delta_gen": delta_gen,
        "r_100_w": r_w,
        "n_100_w": n_w,
    }
    for step in early_exclusion_ladder(a, a_err, b, b_err, _null_seed(transition)):
        row[f"r_{step['pct']}"] = step["r"]
        row[f"r_{step['pct']}_null"] = step["r_null"]
        row[f"n_{step['pct']}"] = step["n"]
        row[f"n_{step['pct']}_null"] = step["n_null"]
        row[f"cut_{step['pct']}"] = step["cut"]
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
                             delta_gen=GENERATIONS[late] - GENERATIONS[early]))
    return rows


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            record = [row["dataset"], row["transition"], row["kind"], row["delta_gen"],
                      row["n_100"], f"{row['r_100']:.4g}", f"{row['r_100_null']:.4g}",
                      f"{row['r_100_w']:.4g}"]
            for pct in (98, 95, 90):
                record += [row[f"n_{pct}"], f"{row[f'cut_{pct}']:.4g}",
                           f"{row[f'r_{pct}']:.4g}", f"{row[f'r_{pct}_null']:.4g}"]
            writer.writerow(record)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    print("\nnested subsets: the lowest 0% / 2% / 5% / 10% of EARLY effects removed, late side free")
    print("cut = largest excluded early effect")
    print("rows: 3 fit-variant rows (fitted1 vs fitted2 of one timepoint -- an UPPER BOUND on the")
    print("      assay ceiling, not a replicate control), then the 3 transitions")
    header = (f"{'dataset':<16}{'transition':<18}{'kind':<13}{'dgen':>7}"
              f"{'n_100':>7}{'r_100':>8}{'null':>8}{'r_100_w':>9}"
              f"{'n_98':>7}{'cut_98':>8}{'r_98':>8}{'null':>8}"
              f"{'n_95':>7}{'cut_95':>8}{'r_95':>8}{'null':>8}"
              f"{'n_90':>7}{'cut_90':>8}{'r_90':>8}{'null':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (f"{row['dataset']:<16}{row['transition']:<18}{row['kind']:<13}"
                f"{row['delta_gen']:>7}"
                f"{row['n_100']:>7}{row['r_100']:>8.3f}{row['r_100_null']:>8.3f}"
                f"{row['r_100_w']:>9.3f}")
        for pct in (98, 95, 90):
            line += (f"{row[f'n_{pct}']:>7}{row[f'cut_{pct}']:>8.3f}"
                     f"{row[f'r_{pct}']:>8.3f}{row[f'r_{pct}_null']:>8.3f}")
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
    print("r_98/95/90     = same, after removing the lowest 2% / 5% / 10% of EARLY effects only;")
    print("                 the late side is never used to define the subset.  The Couce tail is")
    print("                 shallow, so by r_95 the cut is inside the bulk and eating signal")
    print("cut_*          = largest early effect excluded")
    print("kind           = 'fit variant' (two fits of one timepoint) or 'evolved' (a transition)")
    print("delta_gen      = elapsed generations; fixed-mutation counts are deliberately absent")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
