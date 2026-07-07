#!/usr/bin/env python3
r"""Table companion to figS5_fgm_exper: FGM (moment-locked sigma-profile) fit values.

For each DFE (Couce 0K/2K, every Ascensao background L/R/S for GHI/MNO/PQT -- SLR is
excluded -- and Limdi REL606/REL607) this writes the inferred FGM parameters as a flat CSV:
the three fitted parameters -- phenotypic dimension ``n``, mutation-step s.d. ``sigma`` and
distance-to-optimum ``r`` -- each with a bootstrap 95% CI (2.5 / 97.5 percentiles), plus the
derived s0 = r^2/2 scale, the combined timescale ``tau`` (with CI), and three diagnostics:

    n_e        effective dimension 2 E^2/V from the sample moments alone -- the value n
               would take at the optimum (s0=0, sigma=sigma_max); a moments-only reference.
    floor_frac fraction of bootstrap resamples pinned at the small-sigma/high-n floor, where
               the DFE is too symmetric for FGM to locate r -> r is unidentified.
    identified True iff sample skew < 0 AND floor_frac <= FLOOR_FRAC_FLAG (0.20); the FGM
               log-fitness DFE is always negatively skewed, so a False flags a non-FGM DFE.

s0 = r^2/2 is the maximum beneficial effect (the one-sided support edge s <= r^2/2).

The FGM fit + bootstrap is computed here with figS5's own routines (no figure is produced).
The DFE set is broader than the one figS5 fits for the figure (which keeps only the Ascensao
R's), so figS5's cached JSON is not reused. A CSV is written:

    data/TableS2_fgm_fit_params.csv

Columns: dataset, n, n_lo, n_hi, sigma, sigma_lo, sigma_hi, r, r_lo, r_hi, s0,
         tau, tau_lo, tau_hi, n_e, floor_frac, identified

Run:
    python code_figs/TableS2_fgm_fit_params.py
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
for _p in (REPO_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figS5_fgm_exper as fgm  # noqa: E402  (importing does not build the figure)

DATA_DIR = os.path.join(REPO_DIR, "data")
OUT_CSV = os.path.join(DATA_DIR, "TableS2_fgm_fit_params.csv")

COLUMNS = ["dataset", "n", "n_lo", "n_hi", "sigma", "sigma_lo", "sigma_hi",
           "r", "r_lo", "r_hi", "s0", "tau", "tau_lo", "tau_hi",
           "n_e", "floor_frac", "identified"]


def _row_from_result(name, res):
    """Flatten one figS5 per-DFE result dict into a TableS2 row (dict keyed by COLUMNS)."""
    mp, boot = res["map"], res["boot"]
    tau = fgm._tau(mp["n"], mp["sigma"], mp["r"])
    return {
        "dataset": name,
        "n": mp["n"],
        "n_lo": boot["n"][0],
        "n_hi": boot["n"][2],
        "sigma": mp["sigma"],
        "sigma_lo": boot["sigma"][0],
        "sigma_hi": boot["sigma"][2],
        "r": mp["r"],
        "r_lo": boot["r"][0],
        "r_hi": boot["r"][2],
        "s0": mp["s0"],
        "tau": tau,
        "tau_lo": boot["tau"][0],
        "tau_hi": boot["tau"][2],
        "n_e": res["n_e"],
        "floor_frac": res["floor_frac"],
        "identified": bool(res["identified"]),
    }


# Ascensao experiments to exclude from the table entirely.
ASENCAO_EXCLUDE = ("SLR",)


def ancestor_dfes():
    """The DFEs to fit: Couce 0K/2K, every Ascensao L/R/S (minus ASENCAO_EXCLUDE), REL606/607."""
    couce = dict(fgm.load_couce())                    # 0K, 2K, 15K
    asc = dict(fgm.load_asencao())                    # keys like "Asc GHI L"/"R"/"S"
    limdi = fgm.load_limdi()
    specs = [("Couce 0K", couce["0K"]), ("Couce 2K", couce["2K"])]
    specs += [(k, v) for k, v in sorted(asc.items())
              if not any(k.startswith(f"Asc {x} ") for x in ASENCAO_EXCLUDE)]
    specs += [("REL606", limdi["REL606"]), ("REL607", limdi["REL607"])]
    return specs


def build_rows():
    """Compute the figS5 sigma-profile fit + bootstrap for each DFE (no figure)."""
    specs = ancestor_dfes()
    rows = []
    for name, eff in specs:
        f = fgm.sigma_profile(eff, full=False)
        boot, floor_frac = fgm.bootstrap_sigma_profile(eff)
        sk = float(skew(eff))
        identified = bool((sk < 0.0) and (floor_frac <= fgm.FLOOR_FRAC_FLAG))
        res = {
            "n_e": f["n_e"],
            "map": {"sigma": f["sigma"], "n": f["n"], "s0": f["s0"], "r": f["r"]},
            "boot": boot, "floor_frac": floor_frac, "identified": identified,
        }
        rows.append(_row_from_result(name, res))
    return rows


def _fmt(key, val):
    """CSV cell formatting: ints/strings verbatim, floats to 6 significant figures."""
    if key in ("dataset", "identified"):
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


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    write_table(rows, args.out_csv)

    def ci(v, lo, hi, fmt):
        return f"{v:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

    print(f"\n{'dataset':<11}{'n [95% CI]':>22}{'sigma [95% CI]':>26}"
          f"{'r [95% CI]':>26}  id?")
    print("-" * 90)
    for row in rows:
        print(f"{row['dataset']:<11}"
              f"{ci(row['n'], row['n_lo'], row['n_hi'], '.1f'):>22}"
              f"{ci(row['sigma'], row['sigma_lo'], row['sigma_hi'], '.3f'):>26}"
              f"{ci(row['r'], row['r_lo'], row['r_hi'], '.3f'):>26}"
              f"  {'yes' if row['identified'] else 'NO'}")
    print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()
