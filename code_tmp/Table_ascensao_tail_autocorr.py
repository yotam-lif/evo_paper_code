#!/usr/bin/env python3
r"""Table: autocorrelation of the initially-deleterious tail for the Ascensao experiments.

The Ascensao analog of Table_tail_autocorr.py (which does this for Limdi).  Of the knockouts that
start deleterious in the ANCESTOR R, how well does the ancestral effect predict the effect in the
evolved offspring L / S?  As established for Limdi, the conditioning is on the ANCESTOR only
(ancestral s below the cut), with the offspring effect left free, so the essential -> ~neutral
switches -- a gene deleterious in the ancestor that becomes dispensable after evolution -- are
kept rather than discarded.

SCALE.  Ascensao report s per GENERATION (paper Fig. 1C; x6.64 for per-cycle), so the deleterious
tail sits at much smaller |s| than in the Limdi knockout DFE.  The Limdi/Couce essentiality cut
of -0.3 is ~6.6x too large here and would empty the tail.  We therefore condition at two
scale-appropriate cuts, reported side by side:

    -0.03   ~2x the paper's ~1-2% non-neutral scale; 49-217 tail genes per transition
    -0.05   ~ the Limdi -0.3 cut rescaled by 1/6.64 (= -0.045); 16-105 tail genes, a deeper tail

  tail_autocorr_<c>     Pearson r, ancestor R vs offspring, over genes with ancestral s < c
  tail_autocorr_corr_<c>  the same, disattenuated for measurement noise (see NOISE below)
  frac_reverted_<c>     fraction of that initial tail with offspring s > REVERT_CUT = -0.01,
                        i.e. initially-deleterious knockouts that became ~neutral after evolution

NOISE.  The correction uses the authors' PER-GENE measurement errors (``s std`` from the data
release, github.com/joaoascensao/S-L-REL606-BarSeq), aligned to the effect arrays by
cmn_exper.load_asencao_errors: reliability of a side = (V - mean(sigma_i^2))/V over the tail
subset, r_corr = r / sqrt(rel_anc rel_evo) -- the same classical correction TableS1 applies to
Limdi/Couce.  This matters most exactly here: strongly deleterious knockouts are the noisiest
genes (their per-gene errors run several-fold above the near-neutral bulk), so the reliability in
the tail is genuinely below 1 and the corrected tail r is meaningfully above the raw r.

NO CONTROL.  The four experiments have different ancestors (their R arrays correlate 0.01-0.66),
so unlike Limdi's REL606 -> REL607 there is no isogenic zero-evolution pair, hence no r/ceiling
column -- the raw and corrected tail r are reported on their own.

    data/Table_ascensao_tail_autocorr.csv

Run:
    python code_tmp/Table_ascensao_tail_autocorr.py
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

OUT_CSV = os.path.join(DATA_DIR, "Table_ascensao_tail_autocorr.csv")

# Ancestor-tail cuts, per GENERATION (see SCALE in the docstring).  Tags label the CSV columns.
TAIL_CUTS = (-0.03, -0.05)
CUT_TAG = {-0.03: "03", -0.05: "05"}
# Offspring effect above which an initially-deleterious knockout counts as reverted to ~neutral,
# on the per-generation scale (the paper's non-neutral scale is ~1-2%).
REVERT_CUT = -0.01

COLUMNS = ["dataset", "experiment", "transition"]
for _c in TAIL_CUTS:
    _t = CUT_TAG[_c]
    COLUMNS += [f"n_tail_{_t}", f"tail_autocorr_{_t}", f"tail_autocorr_corr_{_t}",
                f"frac_reverted_{_t}"]


def pearson(a, b):
    """Pearson r over the entries finite in both a and b, plus the pair count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 5 or np.std(a[mask]) == 0.0 or np.std(b[mask]) == 0.0:
        return np.nan, n
    return float(pearsonr(a[mask], b[mask])[0]), n


def disattenuate(r, anc, evo, sig_anc, sig_evo):
    """r corrected for PER-GENE measurement noise on both sides (classical attenuation formula).

    reliability of a side = (V - mean(sigma_i^2))/V; r_true = r_obs / sqrt(rel_anc rel_evo).
    """
    if not np.isfinite(r):
        return np.nan
    rels = []
    for vals, sig in ((anc, sig_anc), (evo, sig_evo)):
        sig = np.asarray(sig, float)
        sig = sig[np.isfinite(sig)]
        var = float(np.var(np.asarray(vals, float)))
        if sig.size == 0 or var <= 0.0:
            return np.nan
        rels.append((var - float(np.mean(sig ** 2))) / var)
    if min(rels) <= 0.0:
        return np.nan
    return float(r / np.sqrt(rels[0] * rels[1]))


def ascensao_pair(exp, offspring):
    """Matched (ancestor R, offspring, sigma_anc, sigma_evo) for one experiment, genes finite in both.

    R / L / S effects and their per-gene errors are index-aligned within an experiment, so the
    match is by gene row -- the same convention as Table_ascensao_autocorr.py.
    """
    a = cmn_exper.load_asencao_array(exp, ASENCAO_ANCESTOR)
    b = cmn_exper.load_asencao_array(exp, offspring)
    sa = cmn_exper.load_asencao_errors(exp, ASENCAO_ANCESTOR)
    sb = cmn_exper.load_asencao_errors(exp, offspring)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], sa[m], sb[m]


def make_row(exp, offspring):
    """Tail autocorrelation of R -> offspring at every cut, conditioned on the ancestor tail."""
    a, b, sa, sb = ascensao_pair(exp, offspring)
    row = {"dataset": "Ascensao", "experiment": exp,
           "transition": f"{ASENCAO_ANCESTOR} -> {offspring}"}
    for cut in TAIL_CUTS:
        t = CUT_TAG[cut]
        m = a < cut                                     # condition on the ANCESTOR only
        r, n = pearson(a[m], b[m])
        row[f"n_tail_{t}"] = n
        row[f"tail_autocorr_{t}"] = r
        row[f"tail_autocorr_corr_{t}"] = disattenuate(r, a[m], b[m], sa[m], sb[m])
        row[f"frac_reverted_{t}"] = float(np.mean(b[m] > REVERT_CUT)) if n else np.nan
    return row


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
            out = [row["dataset"], row["experiment"], row["transition"]]
            for cut in TAIL_CUTS:
                t = CUT_TAG[cut]
                out += [row[f"n_tail_{t}"], f"{row[f'tail_autocorr_{t}']:.4g}",
                        f"{row[f'tail_autocorr_corr_{t}']:.4g}",
                        f"{row[f'frac_reverted_{t}']:.4g}"]
            writer.writerow(out)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out)

    print("\nAscensao INITIAL-TAIL autocorrelation: condition on ancestor R below the cut (per")
    print(f"generation), offspring free.  'reverted' = offspring s > {REVERT_CUT} (became ~neutral).")
    print("corrected with the authors' per-gene errors (s std from the data release).\n")
    head = f"{'experiment':<11}{'transition':<10}"
    for cut in TAIL_CUTS:
        head += f"| {('s<'+str(cut)):>8}{'n':>5}{'r':>7}{'corr':>7}{'revert':>7} "
    print(head)
    print("-" * len(head))
    for row in rows:
        line = f"{row['experiment']:<11}{row['transition']:<10}"
        for cut in TAIL_CUTS:
            t = CUT_TAG[cut]
            line += (f"| {'':>8}{row[f'n_tail_{t}']:>5}{row[f'tail_autocorr_{t}']:>7.3f}"
                     f"{row[f'tail_autocorr_corr_{t}']:>7.3f}{row[f'frac_reverted_{t}']:>7.2f} ")
        print(line)

    for cut in TAIL_CUTS:
        t = CUT_TAG[cut]
        r = np.array([row[f"tail_autocorr_{t}"] for row in rows])
        rc = np.array([row[f"tail_autocorr_corr_{t}"] for row in rows])
        rev = np.array([row[f"frac_reverted_{t}"] for row in rows])
        print(f"\ns < {cut}:  raw r = {np.nanmean(r):.3f} (range {np.nanmin(r):.3f}..{np.nanmax(r):.3f}), "
              f"corrected = {np.nanmean(rc):.3f}, mean reverted = {np.nanmean(rev):.2f}")

    print("\nautocorr        = Pearson r, ancestor R vs offspring, over the initial (ancestor) tail")
    print("corr            = disattenuated with the authors' per-gene errors (s std)")
    print(f"reverted        = fraction of the initial tail with offspring s > {REVERT_CUT} (became ~neutral)")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
