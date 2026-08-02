#!/usr/bin/env python3
r"""Table: autocorrelation of the INITIALLY-DELETERIOUS tail across evolution, noise-corrected.

Companion to TableS1_autocorr.py.  TableS1 removes the deleterious tail (keeps ``s > -0.3`` on
both sides) to measure how the near-neutral landscape scrambles.  This table does the opposite
and asks whether the tail itself scrambles: of the knockouts that start strongly deleterious in
the ANCESTOR, how well does the ancestral effect predict the effect in the evolved clone?

THE CONDITIONING IS ON THE ANCESTOR ONLY (this is the whole point).  A gene enters a row iff its
effect in the ancestor is below ``TAIL_CUT``; its effect in the evolved clone is left completely
free.  Conditioning on *both* sides -- keeping only genes still deleterious at 50K -- would throw
away exactly the events that carry the tail's decorrelation: genes essential in the ancestor that
become dispensable after evolution (an essential -> non-essential switch, e.g. a knockout lethal
in REL606 whose pathway is bypassed by a fixed mutation).  Those land near ``s = 0`` in the
evolved clone and are excluded by a two-sided cut, which then reports the tail as artificially
conserved.  So we fix the initial set and watch where it goes.

  tail_autocorr        Pearson r between the ancestor effect and the evolved effect over the
                       genes with ancestral ``s < TAIL_CUT``.
  tail_autocorr_corr   the same, disattenuated for measurement noise on both sides (classical
                       attenuation formula; reliability = (V - mean sigma^2)/V per side).
  frac_reverted        fraction of the initial tail that climbs to ``s > REVERT_CUT`` in the
                       evolved clone -- the initially-essential knockouts that became ~neutral.

ZERO-EVOLUTION CONTROL: REL606 -> REL607.  Read every evolved row against this one.  Two things
make the tail control indispensable here:

  * Range restriction.  Conditioning on the ancestor being in a narrow deep band (-0.75..-0.3)
    shrinks its variance, which attenuates r for everyone.  Disattenuation removes *noise* but
    not this, so even the isogenic control does not reach 1 after correction (it lands ~0.79).
    The honestly-corrected quantity is therefore ``tail_corr_vs_ceiling`` = tail_autocorr_corr /
    (control's tail_autocorr_corr): the fraction of the tail's predictability that survives once
    both noise and the range-restriction floor are taken out.  1.0 = as conserved as an isogenic
    pair; below 1 = the tail genuinely decorrelated.
  * Selection on ancestor noise.  Selecting genes by their *measured* ancestral effect pulls in
    some whose true effect is milder (regression to the mean), which would masquerade as
    reversion.  The control is selected the exact same way, so its ``frac_reverted`` (which comes
    out ~0) calibrates that artifact directly -- any reversion above the control's is real.

Unlike the near-neutral bulk, the tail is where the noise correction actually behaves: reliability
here is ~0.8 on both sides (the effects are large relative to their ~0.02 errors) and nothing
disattenuates past 1.  That is why this table quotes a corrected number, whereas TableS1 refuses
to disattenuate below -0.3.

Couce et al. are not in this table: their library has essentially no lethal tail (one row of
~38000 below -0.3; see cmn/cmn_exper.py), so there is no initial tail to condition on.  This is a
Limdi-only analysis.  EXCLUDED populations (Ara-2, Ara+4) are kept and flagged, as in TableS1.

    data/Table_tail_autocorr.csv
    columns: dataset, transition, n_fixed_mut, n_tail, tail_autocorr, tail_autocorr_corr,
             tail_corr_vs_ceiling, frac_reverted, excluded

Run:
    python code_tmp/Table_tail_autocorr.py
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

OUT_CSV = os.path.join(DATA_DIR, "Table_tail_autocorr.csv")
COLUMNS = ["dataset", "transition", "n_fixed_mut", "n_tail", "tail_autocorr",
           "tail_autocorr_corr", "tail_corr_vs_ceiling", "frac_reverted", "excluded"]

# Ancestral effect below which a knockout counts as the deleterious "tail" -- the -0.3
# essentiality threshold TableS1 uses, applied here to the ANCESTOR only.
TAIL_CUT = -0.3
# Evolved effect above which an initially-deleterious gene counts as having "reverted" to
# ~neutral (the essential -> non-essential switch).  The control's frac_reverted ~ 0 anchors it.
REVERT_CUT = -0.1

# Mutations fixed during each 0 -> 50K transition (as in TableS1_autocorr.py; Limdi rows only).
# Re-declared here rather than imported -- figure/table scripts do not import one another.
N_FIXED_MUT = {
    "REL606 -> REL607": 0,                          # isogenic control (araA marker only)
    "REL606 -> Ara-1": 1100, "REL606 -> Ara-2": 1000, "REL606 -> Ara-3": 800,
    "REL606 -> Ara-4": 1300, "REL606 -> Ara-5": 90,   "REL606 -> Ara-6": 90,
    "REL607 -> Ara+1": 125,  "REL607 -> Ara+2": 70,   "REL607 -> Ara+3": 1800,
    "REL607 -> Ara+4": 70,   "REL607 -> Ara+5": 80,   "REL607 -> Ara+6": 2600,
}

# Populations Limdi et al. themselves exclude on measurement-quality grounds; kept and flagged.
LIMDI_EXCLUDED = {"Ara+4": "poor technical replicates", "Ara-2": "sweeping mutants bias assay"}


def pearson(a, b):
    """Pearson r over the entries finite in both a and b, plus the pair count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0.0 or np.std(b[mask]) == 0.0:
        return np.nan, n
    return float(pearsonr(a[mask], b[mask])[0]), n


def disattenuate(r, early, late, sig_early, sig_late):
    """Correct ``r`` for measurement noise on both sides (classical attenuation formula).

    reliability of a side = (V - mean(sigma^2)) / V; r_true = r_obs / sqrt(rel_e * rel_l).
    NaN if either side has no usable error or is noise-dominated (reliability <= 0).
    """
    if not np.isfinite(r):
        return np.nan
    rel = []
    for vals, sig in ((early, sig_early), (late, sig_late)):
        sig = np.asarray(sig, float)
        sig = sig[np.isfinite(sig)]
        var = float(np.var(np.asarray(vals, float)))
        if sig.size == 0 or var <= 0.0:
            return np.nan
        rel.append((var - float(np.mean(sig ** 2))) / var)
    if min(rel) <= 0.0:
        return np.nan
    return float(r / np.sqrt(rel[0] * rel[1]))


def limdi_pair(early, late):
    """Matched (effects, effects, sigma, sigma) for two Limdi populations, on the shared genes."""
    a_eff, a_sig = cmn_exper.limdi_gene_series(early, errors=True)
    b_eff, b_sig = cmn_exper.limdi_gene_series(late, errors=True)
    idx = a_eff.index.intersection(b_eff.index)
    return (a_eff[idx].to_numpy(float), b_eff[idx].to_numpy(float),
            a_sig[idx].to_numpy(float), b_sig[idx].to_numpy(float))


def tail_row(dataset, early, late):
    """One row: tail autocorrelation of ``early -> late``, conditioned on the ancestor tail."""
    a, b, sig_a, sig_b = limdi_pair(early, late)
    m = a < TAIL_CUT                                   # condition on the ANCESTOR only
    r, n = pearson(a[m], b[m])
    transition = f"{early} -> {late}"
    return {
        "dataset": dataset,
        "transition": transition,
        "n_fixed_mut": N_FIXED_MUT[transition],
        "n_tail": n,
        "tail_autocorr": r,
        "tail_autocorr_corr": disattenuate(r, a[m], b[m], sig_a[m], sig_b[m]),
        "frac_reverted": float(np.mean(b[m] > REVERT_CUT)) if n else np.nan,
        "excluded": LIMDI_EXCLUDED.get(late, ""),
    }


def build_rows():
    """Control first (it is the ceiling), then one row per evolved LTEE population."""
    anc_a, anc_b = LIMDI_ANCESTORS
    rows = [tail_row("Limdi control", anc_a, anc_b)]
    for anc in LIMDI_ANCESTORS:
        for evo in LIMDI_EVOLVED[anc]:
            rows.append(tail_row(f"Limdi {evo}", anc, evo))
    # tail_corr_vs_ceiling = corrected r as a fraction of the isogenic control's corrected r.
    ceiling = rows[0]["tail_autocorr_corr"]
    for row in rows:
        row["tail_corr_vs_ceiling"] = row["tail_autocorr_corr"] / ceiling
    return rows, ceiling


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["dataset"], row["transition"], row["n_fixed_mut"], row["n_tail"],
                             f"{row['tail_autocorr']:.4g}", f"{row['tail_autocorr_corr']:.4g}",
                             f"{row['tail_corr_vs_ceiling']:.4g}",
                             f"{row['frac_reverted']:.4g}", row["excluded"]])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows, ceiling = build_rows()
    write_table(rows, args.out)

    print(f"\nINITIAL-TAIL autocorrelation: condition on ancestor s < {TAIL_CUT}, evolved side free")
    print(f"'reverted' = initially-deleterious gene with evolved s > {REVERT_CUT} (became ~neutral)\n")
    header = (f"{'dataset':<14}{'transition':<20}{'n_fixed':>8}{'n_tail':>7}"
              f"{'tail_r':>9}{'corrected':>10}{'vs_ceiling':>11}{'reverted':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['dataset']:<14}{row['transition']:<20}{row['n_fixed_mut']:>8}{row['n_tail']:>7}"
              f"{row['tail_autocorr']:>9.3f}{row['tail_autocorr_corr']:>10.3f}"
              f"{row['tail_corr_vs_ceiling']:>11.3f}{row['frac_reverted']:>10.3f}"
              f"   {row['excluded']}")

    # Summary over the ten retained clones, and whether the tail scrambles more with distance.
    # n_fixed_mut > 0 drops the control (its dataset also starts with "Limdi ").
    retained = [r for r in rows if r["n_fixed_mut"] > 0 and not r["excluded"]]
    corr = np.array([r["tail_autocorr_corr"] for r in retained])
    rev = np.array([r["frac_reverted"] for r in retained])
    print(f"\ncontrol (ceiling): corrected r = {ceiling:.3f}, reverted = {rows[0]['frac_reverted']:.3f}")
    print(f"10 retained clones: corrected r = {corr.mean():.3f} "
          f"(= {corr.mean()/ceiling:.0%} of ceiling), reverted = {rev.mean():.3f} "
          f"(vs {rows[0]['frac_reverted']:.3f} for zero evolution)")
    x = np.log10([r["n_fixed_mut"] for r in retained])
    rr, pp = pearsonr(x, corr)
    print(f"tail corrected r vs log10(n_fixed_mut), 10 retained:  r = {rr:+.3f}  p = {pp:.3f}")

    print("\ntail_autocorr        = Pearson r, ancestor effect vs evolved effect, over the initial tail")
    print("tail_autocorr_corr   = the same, disattenuated for measurement noise -- QUOTE THIS")
    print("tail_corr_vs_ceiling = corrected r / control's corrected r (1 = as conserved as isogenic)")
    print("frac_reverted        = fraction of the initial tail that became ~neutral in the evolved clone")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
