#!/usr/bin/env python3
r"""Table S2: two independent p-spin ``p/N`` estimates from experimental DFEs.

For the Ascensao, Couce and Limdi data we cross two independent p-spin observables
against each other for every consecutive transition (early ancestor -> late evolved):

1. **Autocorrelation decay.**  How well a mutation's fitness effect in an earlier
   background predicts its effect in a later background, measured by the Pearson
   correlation ``r`` over the mutations finite in *both* states.  In the p-spin model the
   DFE autocorrelation between two genotypes separated by ``t`` fixed mutations decays as
   ``r = (1 - 2 t / N)^(p-1) ~= exp(-2 p t / N)``, i.e. with characteristic timescale
   ``tau = N / (2 p)``.  Inverting the small-``t`` form gives the first estimate

       (p/N)_corr = -ln(r) / (2 t).

2. **DFE mean.**  In the p-spin model (fitness normalized so the ancestor sits at 1) the
   mean of the whole DFE is ``mean_DFE = -2 p / N`` -- the deterministic drift of the
   distribution.  Reading it off the ancestor (early) genotype's DFE gives the second,
   fully independent estimate

       (p/N)_mean = -mean_DFE / 2.

If the p-spin picture holds the two agree, so we also report their ratio
``(p/N)_corr / (p/N)_mean`` and, substituting ``N = 4000`` (the ~4000-gene E. coli genome),
the implied interaction order ``p = (p/N) * N`` for each.

Data loading + cleaning conventions live in ``cmn/cmn_exper.py`` (shared with TableS1_means
and, via ``cmn/cmn_fgm_exper.py``, the FGM fit).

Ascensao (per experiment GHI / MNO / PQT / SLR): the R (ancestor), L and S (evolved) arrays
are index-aligned, so two transitions per experiment: R -> L and R -> S (t = 150 each).

Couce (Ara+2 lineage): mutations are matched across strains on the ``site`` column with the
stricter cleaning (drop NaN/duplicate ``fitted1``, ``abn > 1``, finite ``> -100``).  Two
consecutive intervals: 0K -> 2K (t = 8) and 2K -> 15K (t = 22).

Limdi (TnSeq gene-knockout DFEs): each evolved population is matched to its LTEE ancestor
(REL606 -> each Ara-N, REL607 -> each Ara+N) on the ``Genes`` column, with replicate markers
pooled and duplicate genes aggregated by mean.  ``t`` per evolved population is given below.

    data/TableS2_pspin.csv   columns: dataset, transition, pearson_r, n_fixed, n,
        p_over_N_corr, mean_dfe, p_over_N_mean, p_corr_N4000, p_mean_N4000, ratio_corr_mean

Run:
    python code_figs/TableS2_pspin_exper_fit.py
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
    DATA_DIR, COUCE_INTERVALS, ASENCAO_ANCESTOR, ASENCAO_EVOLVED,
    LIMDI_ANCESTORS, LIMDI_EVOLVED,
)

OUT_PN = os.path.join(DATA_DIR, "TableS2_pspin.csv")

# Genome size used to turn a p/N estimate into an interaction order p (~E. coli genes).
N_GENOME = 4000

# Number of fixed mutations (t) per transition.
ASENCAO_NFIX = 150
COUCE_NFIX = {("0K", "2K"): 8, ("2K", "15K"): 22}
# Number of fixed mutations (t) per evolved Limdi population.
LIMDI_NFIX = {
    "Ara+1": 125, "Ara+2": 70, "Ara+3": 1800, "Ara+4": 70, "Ara+5": 80, "Ara+6": 2600,
    "Ara-1": 1100, "Ara-2": 1000, "Ara-3": 800, "Ara-4": 1300, "Ara-5": 90, "Ara-6": 90,
}


def series_mean(a):
    """Mean over the finite entries of an array/series (NaN if none)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else np.nan


def p_over_N_corr(r, t):
    """p/N from p-spin DFE-autocorrelation decay: p/N = -ln(r) / (2 t), timescale N/2p."""
    if not np.isfinite(r) or r <= 0 or t <= 0:
        return np.nan
    return -np.log(r) / (2.0 * t)


def p_over_N_mean(mean_dfe):
    """p/N from the DFE mean: mean_DFE = -2 p / N  =>  p/N = -mean_DFE / 2."""
    if not np.isfinite(mean_dfe):
        return np.nan
    return -mean_dfe / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Matched-DFE correlation (data loaded via cmn_exper)
# ══════════════════════════════════════════════════════════════════════════════
def pearson(a, b):
    """Pearson r over the entries finite in both a and b (NaN if < 3 points)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan, n
    r, _ = pearsonr(a[mask], b[mask])
    return float(r), n


def load_asencao_rows():
    """Yield (dataset, transition, t, r, n, mean_early) for each Ascensao evolved background."""
    rows = []
    for exp in cmn_exper.asencao_experiments():
        anc = cmn_exper.load_asencao_array(exp, ASENCAO_ANCESTOR)
        if anc is None:
            continue
        mean_early = series_mean(anc)
        for evo in ASENCAO_EVOLVED:
            evolved = cmn_exper.load_asencao_array(exp, evo)
            if evolved is None:
                continue
            r, n = pearson(anc, evolved)
            rows.append((f"Asc {exp}", f"{ASENCAO_ANCESTOR} -> {evo}", ASENCAO_NFIX,
                         r, n, mean_early))
    return rows


def load_couce_rows():
    """Yield (dataset, transition, t, r, n, mean_early) for each consecutive Couce interval."""
    strains = {name: cmn_exper.load_couce_site_series(name)
               for name in ("0K", "2K", "15K")}
    rows = []
    for early, late in COUCE_INTERVALS:
        joined = pd.concat([strains[early], strains[late]], axis=1, join="inner")
        a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
        r, n = pearson(a, b)
        rows.append(("Couce Ara+2", f"{early} -> {late}", COUCE_NFIX[(early, late)],
                     r, n, series_mean(strains[early].to_numpy())))
    return rows


def load_limdi_rows():
    """Yield (dataset, transition, t, r, n, mean_early) for each Limdi ancestor -> evolved pair."""
    tab = cmn_exper.load_limdi_frame()
    rows = []
    for anc in LIMDI_ANCESTORS:
        anc_series = cmn_exper.limdi_gene_series(tab, anc)
        mean_early = series_mean(anc_series.to_numpy())
        for evo in LIMDI_EVOLVED[anc]:
            evo_series = cmn_exper.limdi_gene_series(tab, evo)
            joined = pd.concat([anc_series, evo_series], axis=1, join="inner")
            a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
            r, n = pearson(a, b)
            rows.append((f"Limdi {evo}", f"{anc} -> {evo}", LIMDI_NFIX[evo],
                         r, n, mean_early))
    return rows


def build_rows():
    return load_asencao_rows() + load_couce_rows() + load_limdi_rows()


# ══════════════════════════════════════════════════════════════════════════════
# Derived p/N columns
# ══════════════════════════════════════════════════════════════════════════════
def derive(row):
    """(dataset, transition, t, r, n, mean_early) -> full dict of reported quantities."""
    dataset, transition, t, r, n, mean_early = row
    pn_corr = p_over_N_corr(r, t)
    pn_mean = p_over_N_mean(mean_early)
    ratio = (pn_corr / pn_mean if (np.isfinite(pn_corr) and np.isfinite(pn_mean)
                                   and pn_mean != 0) else np.nan)
    return {
        "dataset": dataset,
        "transition": transition,
        "pearson_r": r,
        "n_fixed": t,
        "n": n,
        "p_over_N_corr": pn_corr,
        "mean_dfe": mean_early,
        "p_over_N_mean": pn_mean,
        "p_corr_N4000": pn_corr * N_GENOME,
        "p_mean_N4000": pn_mean * N_GENOME,
        "ratio_corr_mean": ratio,
    }


COLUMNS = ["dataset", "transition", "pearson_r", "n_fixed", "n", "p_over_N_corr",
           "mean_dfe", "p_over_N_mean", "p_corr_N4000", "p_mean_N4000", "ratio_corr_mean"]


def write_table(records, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for rec in records:
            writer.writerow([
                rec["dataset"], rec["transition"],
                f"{rec['pearson_r']:.4g}", rec["n_fixed"], rec["n"],
                f"{rec['p_over_N_corr']:.4g}", f"{rec['mean_dfe']:.4g}",
                f"{rec['p_over_N_mean']:.4g}", f"{rec['p_corr_N4000']:.4g}",
                f"{rec['p_mean_N4000']:.4g}", f"{rec['ratio_corr_mean']:.4g}",
            ])


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_PN)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    records = [derive(row) for row in build_rows()]
    write_table(records, args.out)

    header = (f"{'dataset':<14}{'transition':<16}{'r':>8}{'t':>7}{'n':>7}"
              f"{'pN_corr':>10}{'mean_dfe':>11}{'pN_mean':>10}"
              f"{'p_corr':>9}{'p_mean':>9}{'ratio':>8}")
    print(header)
    print("-" * len(header))
    for rec in records:
        print(f"{rec['dataset']:<14}{rec['transition']:<16}{rec['pearson_r']:>8.3f}"
              f"{rec['n_fixed']:>7}{rec['n']:>7}{rec['p_over_N_corr']:>10.4g}"
              f"{rec['mean_dfe']:>11.4g}{rec['p_over_N_mean']:>10.4g}"
              f"{rec['p_corr_N4000']:>9.3g}{rec['p_mean_N4000']:>9.3g}"
              f"{rec['ratio_corr_mean']:>8.3g}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
