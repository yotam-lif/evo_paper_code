#!/usr/bin/env python3
r"""Table S1: p-spin p/N fits from experimental DFE means.

Combines the per-experiment mean-DFE analyses for the Ascensao and Couce data
into a single Table S1 generator.  Cleaning conventions match
``code_figs/figS5_exper_bayes.py``.

For each data set the p-spin mean relation

    mean(DFE) = -2 p / N   =>   p / N = -mean(DFE) / 2

is evaluated on the raw (untrimmed) finite DFE values.  Two CSV tables are
written to the data directory:

    data/TableS1_pspin_pN.csv       columns: dataset, p/N
    data/TableS1_pspin_pl_ph.csv    columns: dataset, p_l, p_h

where ``p_l = (p/N) * 4000`` and ``p_h = (p/N) * 4500`` are the p estimates for
the two bounding p-spin sizes N = 4000 and N = 4500.

Run:
    python code_figs/TableS1_pspin_exper_fit.py
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
ASENCAO_DIR = os.path.join(DATA_DIR, "asencao_dfe_arrays")
COUCE_DIR = os.path.join(DATA_DIR, "alex_code")

OUT_PN = os.path.join(DATA_DIR, "TableS1_pspin_pN.csv")
OUT_PL_PH = os.path.join(DATA_DIR, "TableS1_pspin_pl_ph.csv")

# Bounding p-spin sizes used to turn p/N into p_l and p_h.
N_LOW = 4000.0
N_HIGH = 4500.0

# Ascensao DFE arrays.
ARRAY_ORDER = ("L", "R", "S")

# Couce DFE tables (same cleaning convention as figS5_exper_bayes.py).
COUCE_FILES = {
    "REL607": "Rfitted_fil.txt",
    "2K": "2Kfitted_fil.txt",
    "15K": "15Kfitted_fil.txt",
}
COUCE_DISPLAY = {
    "REL607": "Couce REL607 / 0K",
    "2K": "Couce Ara+2 2K",
    "15K": "Couce Ara+2 15K",
}


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


def build_rows():
    """Return list of (dataset, p_over_N) across all experiments."""
    means = load_asencao_means() + load_couce_means()
    return [(name, -0.5 * mean) for name, mean in means]


def write_tables(rows, out_pn, out_pl_ph, n_low=N_LOW, n_high=N_HIGH):
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


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-pn", default=OUT_PN)
    parser.add_argument("--out-pl-ph", default=OUT_PL_PH)
    parser.add_argument("--N-low", type=float, default=N_LOW, help="N used for p_l.")
    parser.add_argument("--N-high", type=float, default=N_HIGH, help="N used for p_h.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_tables(rows, args.out_pn, args.out_pl_ph, args.N_low, args.N_high)

    print(f"{'dataset':<20}{'p/N':>14}{'p_l':>14}{'p_h':>14}")
    print("-" * 62)
    for name, p_over_N in rows:
        print(f"{name:<20}{p_over_N:>14.8g}{p_over_N * args.N_low:>14.8g}{p_over_N * args.N_high:>14.8g}")
    print(f"\nSaved {args.out_pn}")
    print(f"Saved {args.out_pl_ph}")


if __name__ == "__main__":
    main()
