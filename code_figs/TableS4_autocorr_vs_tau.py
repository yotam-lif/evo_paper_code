#!/usr/bin/env python3
r"""Table S4: FGM-predicted vs observed DFE-autocorrelation decay for both adaptive arms.

For every consecutive transition in the Ara+ (REL607-descended) and Ara- (REL606-descended)
lineages we cross the FGM scrambling timescale ``tau`` -- fit to the *ancestor* DFE's shape
(from ``data/TableS3_fgm_fit_params.csv``) -- against the DFE autocorrelation actually
*observed* between the two genotypes (the Pearson ``r`` from ``data/TableS2_pspin.csv``).

The p-spin / FGM scrambling picture predicts the DFE autocorrelation between two genotypes
separated by ``t`` fixed mutations decays as

    r(t) = exp(-t / tau)        <=>        ln r(t) = -t / tau,

so the ancestor's FGM ``tau`` (from the DFE *shape*) together with the number of fixed
mutations ``t`` give an EXPECTED autocorrelation ``exp(-t/tau)`` (log form ``-t/tau``) that we
compare against the REAL, measured Pearson ``r`` (its log ``ln r``). Equivalently we invert the
measured ``r`` into an observed decay timescale ``tau_obs = -t / ln r`` and read it against the
FGM ``tau``.

Datasets -- both arms:
    Ara+ lineage (REL607 and its descendants):
        * Couce Ara+ timepoints:  0K (== the REL607 ancestor) -> 2K -> 15K.
        * Limdi LTEE:             REL607 (ancestor) -> each evolved Ara+N.
    Ara- lineage (REL606 and its descendants):
        * Limdi LTEE:             REL606 (ancestor) -> each evolved Ara-N.

``tau`` is always the FGM tau of the ANCESTOR DFE -- Couce 0K and Limdi REL607 (plus Couce 2K
for the 2K->15K interval) for the Ara+ arm, and Limdi REL606 for the Ara- arm -- the landscape
timescale where the adaptive walk starts. It is read straight from
``data/TableS3_fgm_fit_params.csv``; ``r`` and ``t`` (== n_fixed) come from
``data/TableS2_pspin.csv``.

    data/TableS4_autocorr_vs_tau.csv
    columns: arm, lineage, transition, t, r_real, ln_r_real, tau_anc, tau_obs,
             ln_r_exp, r_exp, ratio_anc_obs

where, per transition:
    ln_r_exp      = -t / tau_anc            expected log-autocorrelation
    r_exp         = exp(-t / tau_anc)       expected autocorrelation
    tau_obs       = -t / ln r_real          observed decay timescale implied by the data
    ratio_anc_obs = tau_anc / tau_obs       <1 <=> FGM (shape) tau decays faster than observed

Run:
    python code_figs/TableS4_autocorr_vs_tau.py
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

DATA_DIR = os.path.join(REPO_DIR, "data")
TABLE_S2 = os.path.join(DATA_DIR, "TableS2_pspin.csv")           # Pearson r + t (n_fixed)
TABLE_S3 = os.path.join(DATA_DIR, "TableS3_fgm_fit_params.csv")  # FGM tau per DFE

OUT_CSV = os.path.join(DATA_DIR, "TableS4_autocorr_vs_tau.csv")

COLUMNS = ["arm", "lineage", "transition", "t", "r_real", "ln_r_real",
           "tau_anc", "tau_obs", "ln_r_exp", "r_exp", "ratio_anc_obs"]


def lineage_arm(dataset):
    """Adaptive arm ('Ara+' / 'Ara-') for a TableS2_pspin.csv row, or None to skip it.

    Ara+ (REL607-descended): Couce Ara+2 timepoints and Limdi Ara+N.
    Ara- (REL606-descended): Limdi Ara-N.
    """
    if dataset == "Couce Ara+2" or dataset.startswith("Limdi Ara+"):
        return "Ara+"
    if dataset.startswith("Limdi Ara-"):
        return "Ara-"
    return None


def tau_key(endpoint):
    """Map a transition's ANCESTOR endpoint to its TableS3 ``dataset`` name.

    Couce timepoints are written "0K"/"2K" in the transition but "Couce 0K"/"Couce 2K" in
    TableS3; the Limdi ancestors ("REL606"/"REL607") already match TableS3 verbatim.
    """
    return f"Couce {endpoint}" if endpoint in ("0K", "2K", "15K") else endpoint


def load_tau_lookup():
    """``{dataset: tau}`` (the FGM shape-tau per DFE) from TableS3."""
    df = pd.read_csv(TABLE_S3)
    return {str(row["dataset"]): float(row["tau"]) for _, row in df.iterrows()}


def load_transitions():
    """Ara+ and Ara- lineage transitions from TableS2_pspin.csv, ancestor resolved to a tau key.

    Ara+ rows are returned first, then Ara-; within each arm the TableS2 file order is kept.
    """
    df = pd.read_csv(TABLE_S2)
    rows = []
    for _, r in df.iterrows():
        arm = lineage_arm(str(r["dataset"]))
        if arm is None:
            continue
        transition = str(r["transition"])   # "0K -> 2K" | "REL607 -> Ara+1" | "REL606 -> Ara-1"
        anc_ep = transition.split("->")[0].strip()
        rows.append({
            "arm": arm,
            "lineage": "Couce" if str(r["dataset"]).startswith("Couce") else "Limdi",
            "transition": transition,
            "anc_key": tau_key(anc_ep),
            "t": int(r["n_fixed"]),
            "r_real": float(r["pearson_r"]),
        })
    rows.sort(key=lambda row: 0 if row["arm"] == "Ara+" else 1)   # Ara+ block, then Ara-
    return rows


def build_rows():
    """Cross the ancestor's FGM tau (shape) against the observed Pearson autocorrelation."""
    tau_lut = load_tau_lookup()
    rows = []
    for tr in load_transitions():
        t, r = tr["t"], tr["r_real"]
        tau_anc = tau_lut.get(tr["anc_key"], float("nan"))

        ln_r_real = np.log(r) if (np.isfinite(r) and r > 0.0) else float("nan")
        # Observed decay timescale implied by the measured autocorrelation: r = exp(-t/tau_obs).
        tau_obs = (-t / ln_r_real) if (np.isfinite(ln_r_real) and ln_r_real != 0.0) \
            else float("nan")
        # Expected autocorrelation from the ancestor's FGM (shape) tau + the observed t.
        ln_r_exp = (-t / tau_anc) if (np.isfinite(tau_anc) and tau_anc != 0.0) else float("nan")
        r_exp = float(np.exp(ln_r_exp)) if np.isfinite(ln_r_exp) else float("nan")
        ratio = (tau_anc / tau_obs) if (np.isfinite(tau_anc) and np.isfinite(tau_obs)
                                        and tau_obs != 0.0) else float("nan")

        rows.append({
            "arm": tr["arm"],
            "lineage": tr["lineage"],
            "transition": tr["transition"],
            "t": t,
            "r_real": r,
            "ln_r_real": ln_r_real,
            "tau_anc": tau_anc,
            "tau_obs": tau_obs,
            "ln_r_exp": ln_r_exp,
            "r_exp": r_exp,
            "ratio_anc_obs": ratio,
        })
    return rows


def _fmt(key, val):
    """CSV cell formatting: strings/ints verbatim, floats to 6 significant figures."""
    if key in ("arm", "lineage", "transition", "t"):
        return val
    return f"{float(val):.6g}"


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_fmt(k, row[k]) for k in COLUMNS])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", default=OUT_CSV)
    return parser.parse_args(argv)


def _g(x, w, p):
    """Right-justified {:g}-ish cell that prints 'nan' cleanly."""
    return f"{'nan':>{w}}" if not np.isfinite(x) else f"{x:>{w}.{p}g}"


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out_csv)

    header = (f"{'arm':<6}{'transition':<16}{'t':>6}{'r_real':>9}{'ln r':>9}"
              f"{'tau_anc':>9}{'tau_obs':>9}{'exp(-t/ta)':>12}{'-t/ta':>9}{'ta/obs':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['arm']:<6}{row['transition']:<16}{row['t']:>6}"
              f"{_g(row['r_real'], 9, 3)}{_g(row['ln_r_real'], 9, 3)}"
              f"{_g(row['tau_anc'], 9, 3)}{_g(row['tau_obs'], 9, 3)}"
              f"{_g(row['r_exp'], 12, 3)}{_g(row['ln_r_exp'], 9, 3)}"
              f"{_g(row['ratio_anc_obs'], 9, 3)}")
    print("\nexpected: r_exp = exp(-t/tau_anc)  vs  real: r_real (Pearson autocorr).")
    print("tau_obs = -t/ln(r_real) is the decay timescale the data implies;")
    print("ratio ta/obs < 1 => FGM shape-tau predicts FASTER decay than observed.")
    print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()
