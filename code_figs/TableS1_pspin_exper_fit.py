#!/usr/bin/env python3
r"""Table S1: p-spin p/N estimate from consecutive-DFE Pearson correlations.

For the Ascensao, Couce and Limdi data we measure how well a mutation's fitness effect in
an earlier (ancestral) background predicts its effect in a later (evolved) background, using
the Pearson correlation r over the mutations finite in *both* states.  Cleaning conventions
match ``code_figs/figS5_fgm_exper.py``.

p-spin p/N estimate
-------------------
In the p-spin model the correlation between the DFEs of two genotypes separated by ``t``
fixed mutations decays as ``r = (1 - 2 t / N)^(p-1) ~= exp(-2 p t / N)``.  Inverting the
small-t approximation gives

    p / N = -ln(r) / (2 t)

with ``t`` the number of fixed mutations along the transition.

Ascensao (per experiment GHI / MNO / PQT / SLR): the R (ancestor), L and S (evolved) arrays
are index-aligned, so two transitions per experiment: R -> L and R -> S (t = 150 each).

Couce (Ara+2 lineage): mutations are matched across strains on the ``site`` column with the
stricter cleaning (drop NaN/duplicate ``fitted1``, ``abn > 1``, finite ``> -100``).  Two
consecutive intervals: 0K -> 2K (t = 8) and 2K -> 15K (t = 22).

Limdi (TnSeq gene-knockout DFEs): each evolved population is matched to its LTEE ancestor
(REL606 -> each Ara-N, REL607 -> each Ara+N) on the ``Genes`` column, with replicate markers
pooled and duplicate genes aggregated by mean.  ``t`` per evolved population is given below.

    data/TableS1_pearson_consecutive.csv   columns: dataset, transition, n_fixed, pearson_r,
                                                     log_pearson_r, p_over_N, n

Run:
    python code_figs/TableS1_pspin_exper_fit.py
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
DATA_DIR = os.path.join(REPO_DIR, "data")
ASENCAO_DIR = os.path.join(DATA_DIR, "asencao_dfe_arrays")
COUCE_DIR = os.path.join(DATA_DIR, "alex_code")
LIMDI_CSV = os.path.join(
    DATA_DIR, "anurag_data", "Analysis", "Part_3_TnSeq_analysis",
    "Processed_data_for_plotting", "dfe_data_pandas.csv",
)

OUT_PEARSON = os.path.join(DATA_DIR, "TableS1_pearson_consecutive.csv")

# Ascensao DFE arrays.
ARRAY_ORDER = ("L", "R", "S")

# Couce DFE tables (same cleaning convention as figS5_fgm_exper.py). 0K == REL607 ancestor.
COUCE_FILES = {
    "0K": "Rfitted_fil.txt",
    "2K": "2Kfitted_fil.txt",
    "15K": "15Kfitted_fil.txt",
}

# Pearson: ancestor R -> each evolved Ascensao background; consecutive Couce intervals.
ASENCAO_ANCESTOR = "R"
ASENCAO_EVOLVED = ("L", "S")
COUCE_INTERVALS = (("0K", "2K"), ("2K", "15K"))

# Number of fixed mutations (t) per transition.
ASENCAO_NFIX = 150
COUCE_NFIX = {("0K", "2K"): 8, ("2K", "15K"): 22}

# Limdi et al. TnSeq DFE (gene-knockout fitness effects across the LTEE panel).
# Two LTEE ancestors, each the founder of six evolved populations of matching Ara phenotype.
# Replicates (Green/Red fluorescent markers) are pooled; genes are aggregated by mean.
LIMDI_ANCESTORS = ("REL606", "REL607")
LIMDI_EVOLVED = {
    "REL606": tuple(f"Ara-{i}" for i in range(1, 7)),  # REL606 is the Ara- founder.
    "REL607": tuple(f"Ara+{i}" for i in range(1, 7)),  # REL607 is the Ara+ founder.
}
# Number of fixed mutations (t) per evolved Limdi population.
LIMDI_NFIX = {
    "Ara+1": 125, "Ara+2": 70, "Ara+3": 1800, "Ara+4": 70, "Ara+5": 80, "Ara+6": 2600,
    "Ara-1": 1100, "Ara-2": 1000, "Ara-3": 800, "Ara-4": 1300, "Ara-5": 90, "Ara-6": 90,
}


def p_over_N(r, t):
    """p/N estimate from p-spin DFE-correlation decay: p/N = -ln(r) / (2 t)."""
    if not np.isfinite(r) or r <= 0 or t <= 0:
        return np.nan
    return -np.log(r) / (2.0 * t)


# ══════════════════════════════════════════════════════════════════════════════
# Pearson correlation of matched fitness effects across consecutive DFEs
# ══════════════════════════════════════════════════════════════════════════════
def load_limdi_frame():
    """Load the Limdi DFE table, keeping only finite fitness estimates."""
    tab = pd.read_csv(LIMDI_CSV)
    tab = tab[np.isfinite(tab["Fitness estimate"])]
    return tab


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


def load_asencao_pearson_rows():
    """Yield (dataset, transition, n_fixed, r, n) for each Ascensao evolved background."""
    rows = []
    for exp in sorted(os.listdir(ASENCAO_DIR)):
        sub = os.path.join(ASENCAO_DIR, exp)
        if not os.path.isdir(sub):
            continue
        anc_path = os.path.join(sub, f"{ASENCAO_ANCESTOR}.npy")
        if not os.path.exists(anc_path):
            continue
        anc = np.load(anc_path).astype(float)
        for evo in ASENCAO_EVOLVED:
            evo_path = os.path.join(sub, f"{evo}.npy")
            if not os.path.exists(evo_path):
                continue
            evolved = np.load(evo_path).astype(float)
            r, n = pearson(anc, evolved)
            rows.append((f"Asc {exp}", f"{ASENCAO_ANCESTOR} -> {evo}", ASENCAO_NFIX, r, n))
    return rows


def load_couce_strain(fname):
    """Load one Couce strain, cleaned as in figS5_fgm_exper.py, indexed by mutation site."""
    path = os.path.join(COUCE_DIR, fname)
    tab = pd.read_csv(path, sep="\t").dropna(subset=["fitted1"])
    tab = tab.drop_duplicates(subset=["fitted1"])
    tab = tab[tab["abn"] > 1]
    tab = tab[np.isfinite(tab["fitted1"]) & (tab["fitted1"] > -100.0)]
    # One fitness effect per mutation site so the cross-timepoint merge is unambiguous.
    tab = tab.drop_duplicates(subset=["site"])
    return tab.set_index("site")["fitted1"]


def load_couce_pearson_rows():
    """Yield (dataset, transition, n_fixed, r, n) for each consecutive Couce interval."""
    strains = {name: load_couce_strain(fname) for name, fname in COUCE_FILES.items()}
    rows = []
    for early, late in COUCE_INTERVALS:
        joined = pd.concat([strains[early], strains[late]], axis=1, join="inner")
        a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
        r, n = pearson(a, b)
        rows.append(("Couce Ara+2", f"{early} -> {late}", COUCE_NFIX[(early, late)], r, n))
    return rows


def load_limdi_gene_series(tab, pop):
    """Mean fitness effect per gene for one Limdi population (replicates pooled)."""
    sub = tab[tab["Population"] == pop]
    return sub.groupby("Genes")["Fitness estimate"].mean()


def load_limdi_pearson_rows():
    """Yield (dataset, transition, n_fixed, r, n) for each Limdi ancestor -> evolved pair."""
    tab = load_limdi_frame()
    rows = []
    for anc in LIMDI_ANCESTORS:
        anc_series = load_limdi_gene_series(tab, anc)
        for evo in LIMDI_EVOLVED[anc]:
            evo_series = load_limdi_gene_series(tab, evo)
            joined = pd.concat([anc_series, evo_series], axis=1, join="inner")
            a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
            r, n = pearson(a, b)
            rows.append((f"Limdi {evo}", f"{anc} -> {evo}", LIMDI_NFIX[evo], r, n))
    return rows


def build_pearson_rows():
    return (load_asencao_pearson_rows() + load_couce_pearson_rows()
            + load_limdi_pearson_rows())


def write_pearson_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "transition", "n_fixed", "pearson_r",
                         "log_pearson_r", "p_over_N", "n"])
        for dataset, transition, t, r, n in rows:
            writer.writerow([dataset, transition, t, f"{r:.6g}",
                             f"{float(np.log(r)):.6g}", f"{p_over_N(r, t):.8g}", n])


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-pearson", default=OUT_PEARSON)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    pearson_rows = build_pearson_rows()
    write_pearson_table(pearson_rows, args.out_pearson)

    print(f"{'dataset':<14}{'transition':<14}{'n_fixed':>9}{'pearson_r':>12}"
          f"{'log_pearson_r':>16}{'p/N':>14}{'n':>8}")
    print("-" * 87)
    for dataset, transition, t, r, n in pearson_rows:
        print(f"{dataset:<14}{transition:<14}{t:>9}{r:>12.4f}{float(np.log(r)):>16.4f}"
              f"{p_over_N(r, t):>14.6g}{n:>8}")
    print(f"\nSaved {args.out_pearson}")


if __name__ == "__main__":
    main()
