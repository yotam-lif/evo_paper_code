#!/usr/bin/env python3
r"""Table S1: p-spin p/N fits from experimental DFE means + consecutive-DFE Pearson.

Combines the two Table S1 analyses for the Ascensao, Couce and Limdi data into a single
generator.  Cleaning conventions match ``code_figs/figS5_fgm_exper.py``.

Part A -- p-spin p/N fit (raw DFE means)
----------------------------------------
For each data set the p-spin mean relation

    mean(DFE) = -2 p / N   =>   p / N = -mean(DFE) / 2

is evaluated on the raw (untrimmed) finite DFE values.  Limdi contributes one row per
LTEE population (two ancestors + twelve evolved), pooling the Green/Red replicate markers.
Two CSV tables:

    data/TableS1_pspin_pN.csv       columns: dataset, p/N
    data/TableS1_pspin_pl_ph.csv    columns: dataset, p_l, p_h

where ``p_l = (p/N) * 4000`` and ``p_h = (p/N) * 4500`` are the p estimates for the two
bounding p-spin sizes N = 4000 and N = 4500.

Part B -- Pearson correlation across consecutive DFEs
-----------------------------------------------------
For every mutation measured in two consecutive backgrounds/timepoints we ask how well its
fitness effect in the earlier (ancestral) state predicts its effect in the later (evolved)
state.  Pearson r is computed over the mutations finite in *both* states.

Ascensao (per experiment GHI / MNO / PQT / SLR): the R (ancestor), L and S (evolved) arrays
are index-aligned, so two transitions per experiment: R -> L and R -> S.

Couce (Ara+2 lineage): mutations are matched across strains on the ``site`` column with the
stricter p/N-fit cleaning (drop NaN/duplicate ``fitted1``, ``abn > 1``, finite ``> -100``).
Two consecutive intervals: 0K -> 2K and 2K -> 15K.

Limdi (TnSeq gene-knockout DFEs): each evolved population is matched to its LTEE ancestor
(REL606 -> each Ara-N, REL607 -> each Ara+N) on the ``Genes`` column, with replicate markers
pooled and duplicate genes aggregated by mean.  Twelve ancestor -> evolved transitions.

    data/TableS1_pearson_consecutive.csv   columns: dataset, transition, pearson_r,
                                                     log_pearson_r, n

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

OUT_PN = os.path.join(DATA_DIR, "TableS1_pspin_pN.csv")
OUT_PL_PH = os.path.join(DATA_DIR, "TableS1_pspin_pl_ph.csv")
OUT_PEARSON = os.path.join(DATA_DIR, "TableS1_pearson_consecutive.csv")

# Bounding p-spin sizes used to turn p/N into p_l and p_h.
N_LOW = 4000.0
N_HIGH = 4500.0

# Ascensao DFE arrays.
ARRAY_ORDER = ("L", "R", "S")

# Couce DFE tables (same cleaning convention as figS5_fgm_exper.py). 0K == REL607 ancestor.
COUCE_FILES = {
    "0K": "Rfitted_fil.txt",
    "2K": "2Kfitted_fil.txt",
    "15K": "15Kfitted_fil.txt",
}
COUCE_DISPLAY = {
    "0K": "Couce REL607 / 0K",
    "2K": "Couce Ara+2 2K",
    "15K": "Couce Ara+2 15K",
}

# Pearson: ancestor R -> each evolved Ascensao background; consecutive Couce intervals.
ASENCAO_ANCESTOR = "R"
ASENCAO_EVOLVED = ("L", "S")
COUCE_INTERVALS = (("0K", "2K"), ("2K", "15K"))

# Limdi et al. TnSeq DFE (gene-knockout fitness effects across the LTEE panel).
# Two LTEE ancestors, each the founder of six evolved populations of matching Ara phenotype.
# Replicates (Green/Red fluorescent markers) are pooled; genes are aggregated by mean.
LIMDI_ANCESTORS = ("REL606", "REL607")
LIMDI_EVOLVED = {
    "REL606": tuple(f"Ara-{i}" for i in range(1, 7)),  # REL606 is the Ara- founder.
    "REL607": tuple(f"Ara+{i}" for i in range(1, 7)),  # REL607 is the Ara+ founder.
}
# Population display order: ancestors first, then their evolved descendants.
LIMDI_POP_ORDER = list(LIMDI_ANCESTORS) + [
    pop for anc in LIMDI_ANCESTORS for pop in LIMDI_EVOLVED[anc]
]


# ══════════════════════════════════════════════════════════════════════════════
# Part A: p-spin p/N fit from raw DFE means
# ══════════════════════════════════════════════════════════════════════════════
def load_asencao_means():
    """Yield (dataset_name, raw DFE mean) for each Ascensao experiment/background."""
    rows = []
    for exp in sorted(os.listdir(ASENCAO_DIR)):
        sub = os.path.join(ASENCAO_DIR, exp)
        if not os.path.isdir(sub):
            continue
        for label in ARRAY_ORDER:
            path = os.path.join(sub, f"{label}.npy")
            if not os.path.exists(path):
                continue
            raw = np.load(path).astype(float)
            raw = raw[np.isfinite(raw)]
            if raw.size == 0:
                continue
            rows.append((f"Asc {exp} {label}", float(np.mean(raw))))
    return rows


def load_couce_means():
    """Yield (dataset_name, raw DFE mean) for each Couce strain."""
    rows = []
    for label, fname in COUCE_FILES.items():
        path = os.path.join(COUCE_DIR, fname)
        tab = pd.read_csv(path, sep="\t").dropna(subset=["fitted1"])
        tab = tab.drop_duplicates(subset=["fitted1"])
        tab = tab[tab["abn"] > 1]
        v = tab["fitted1"].to_numpy(float)
        v = v[np.isfinite(v) & (v > -100.0)]
        if v.size == 0:
            continue
        rows.append((COUCE_DISPLAY[label], float(np.mean(v))))
    return rows


def load_limdi_frame():
    """Load the Limdi DFE table, keeping only finite fitness estimates."""
    tab = pd.read_csv(LIMDI_CSV)
    tab = tab[np.isfinite(tab["Fitness estimate"])]
    return tab


def load_limdi_means():
    """Yield (dataset_name, raw DFE mean) per Limdi population (replicates pooled)."""
    tab = load_limdi_frame()
    means = tab.groupby("Population")["Fitness estimate"].mean()
    rows = []
    for pop in LIMDI_POP_ORDER:
        if pop not in means.index:
            continue
        rows.append((f"Limdi {pop}", float(means[pop])))
    return rows


def build_pn_rows():
    """Return list of (dataset, p_over_N) across all experiments."""
    means = load_asencao_means() + load_couce_means() + load_limdi_means()
    return [(name, -0.5 * mean) for name, mean in means]


def write_pn_tables(rows, out_pn, out_pl_ph, n_low=N_LOW, n_high=N_HIGH):
    os.makedirs(os.path.dirname(os.path.abspath(out_pn)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_pl_ph)), exist_ok=True)

    with open(out_pn, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "p/N"])
        for name, p_over_N in rows:
            writer.writerow([name, f"{p_over_N:.8g}"])

    with open(out_pl_ph, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "p_l", "p_h"])
        for name, p_over_N in rows:
            writer.writerow([name, f"{p_over_N * n_low:.8g}", f"{p_over_N * n_high:.8g}"])


# ══════════════════════════════════════════════════════════════════════════════
# Part B: Pearson correlation of matched fitness effects across consecutive DFEs
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


def load_asencao_pearson_rows():
    """Yield (dataset, transition, r, n) for each Ascensao experiment and evolved background."""
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
            rows.append((f"Asc {exp}", f"{ASENCAO_ANCESTOR} -> {evo}", r, n))
    return rows


def load_couce_strain(fname):
    """Load one Couce strain, cleaned as in the p/N fit, indexed by mutation site."""
    path = os.path.join(COUCE_DIR, fname)
    tab = pd.read_csv(path, sep="\t").dropna(subset=["fitted1"])
    tab = tab.drop_duplicates(subset=["fitted1"])
    tab = tab[tab["abn"] > 1]
    tab = tab[np.isfinite(tab["fitted1"]) & (tab["fitted1"] > -100.0)]
    # One fitness effect per mutation site so the cross-timepoint merge is unambiguous.
    tab = tab.drop_duplicates(subset=["site"])
    return tab.set_index("site")["fitted1"]


def load_couce_pearson_rows():
    """Yield (dataset, transition, r, n) for each consecutive Couce interval."""
    strains = {name: load_couce_strain(fname) for name, fname in COUCE_FILES.items()}
    rows = []
    for early, late in COUCE_INTERVALS:
        joined = pd.concat([strains[early], strains[late]], axis=1, join="inner")
        a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
        r, n = pearson(a, b)
        rows.append(("Couce Ara+2", f"{early} -> {late}", r, n))
    return rows


def load_limdi_gene_series(tab, pop):
    """Mean fitness effect per gene for one Limdi population (replicates pooled)."""
    sub = tab[tab["Population"] == pop]
    return sub.groupby("Genes")["Fitness estimate"].mean()


def load_limdi_pearson_rows():
    """Yield (dataset, transition, r, n) for each Limdi ancestor -> evolved pair."""
    tab = load_limdi_frame()
    rows = []
    for anc in LIMDI_ANCESTORS:
        anc_series = load_limdi_gene_series(tab, anc)
        for evo in LIMDI_EVOLVED[anc]:
            evo_series = load_limdi_gene_series(tab, evo)
            joined = pd.concat([anc_series, evo_series], axis=1, join="inner")
            a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
            r, n = pearson(a, b)
            rows.append((f"Limdi {evo}", f"{anc} -> {evo}", r, n))
    return rows


def build_pearson_rows():
    return (load_asencao_pearson_rows() + load_couce_pearson_rows()
            + load_limdi_pearson_rows())


def write_pearson_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "transition", "pearson_r", "log_pearson_r", "n"])
        for dataset, transition, r, n in rows:
            writer.writerow([dataset, transition, f"{r:.6g}", f"{float(np.log(r)):.6g}", n])


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-pn", default=OUT_PN)
    parser.add_argument("--out-pl-ph", default=OUT_PL_PH)
    parser.add_argument("--out-pearson", default=OUT_PEARSON)
    parser.add_argument("--N-low", type=float, default=N_LOW, help="N used for p_l.")
    parser.add_argument("--N-high", type=float, default=N_HIGH, help="N used for p_h.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Part A: p/N fit.
    pn_rows = build_pn_rows()
    write_pn_tables(pn_rows, args.out_pn, args.out_pl_ph, args.N_low, args.N_high)

    print(f"{'dataset':<20}{'p/N':>14}{'p_l':>14}{'p_h':>14}")
    print("-" * 62)
    for name, p_over_N in pn_rows:
        print(f"{name:<20}{p_over_N:>14.8g}{p_over_N * args.N_low:>14.8g}"
              f"{p_over_N * args.N_high:>14.8g}")
    print(f"\nSaved {args.out_pn}")
    print(f"Saved {args.out_pl_ph}")

    # Part B: Pearson across consecutive DFEs.
    pearson_rows = build_pearson_rows()
    write_pearson_table(pearson_rows, args.out_pearson)

    print(f"\n{'dataset':<14}{'transition':<14}{'pearson_r':>12}{'log_pearson_r':>16}{'n':>8}")
    print("-" * 64)
    for dataset, transition, r, n in pearson_rows:
        print(f"{dataset:<14}{transition:<14}{r:>12.4f}{float(np.log(r)):>16.4f}{n:>8}")
    print(f"\nSaved {args.out_pearson}")


if __name__ == "__main__":
    main()
