r"""Sensitivity of the full Cauchy-FGM MLE to the lower fitness-effect cutoff.

The main fit in ``figS6_cauchy_fgm_full_mle.py`` conditions on s >= -0.4.  Because that
cutoff is approximate, this script refits the two quantitatively usable Limdi ancestral
DFEs over cutoffs -0.25 through -0.45.  The conditional-likelihood normalization is
recomputed at every cutoff and parameter point.

Output:
    data/cauchy_fgm_cutoff_sensitivity.csv
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cmn.cmn_fgm_exper import load_limdi  # noqa: E402
from code_tmp.figS6_cauchy_fgm_full_mle import (  # noqa: E402
    DX,
    JSON_PATH,
    MEAS_ERR,
    OPT_BOUNDS,
    UnbinnedNoisyLikelihood,
)

OUTPUT = os.path.join(REPO_DIR, "data", "cauchy_fgm_cutoff_sensitivity.csv")
CUTS = (-0.25, -0.30, -0.35, -0.40, -0.45)


def main():
    with open(JSON_PATH) as handle:
        main_fit = json.load(handle)["per_dfe"]
    raw = load_limdi(populations=["REL606", "REL607"], trim=(0.0, 0.0))
    bounds = [(np.log(lo), np.log(hi)) for lo, hi in OPT_BOUNDS.values()]
    rows = []

    for name, effects in raw.items():
        fit = main_fit[name]
        start = np.log([fit["n"], fit["C_n_sigma2"], fit["A_r_sigma"]])
        generic_starts = (
            np.log([8.0, 0.002, 0.006]),
            np.log([20.0, 0.002, 0.006]),
        )
        for cut in CUTS:
            retained = effects[effects >= cut]
            likelihood = UnbinnedNoisyLikelihood(
                retained, eps=MEAS_ERR, dx=DX, lower_cut=cut
            )
            candidates = [
                minimize(
                    likelihood.objective,
                    initial,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 1000},
                )
                for initial in (start, *generic_starts)
            ]
            candidates.extend([
                minimize(
                    likelihood.objective,
                    initial,
                    method="Powell",
                    bounds=bounds,
                    options={"xtol": 1e-7, "ftol": 1e-9, "maxiter": 1000},
                )
                for initial in (start, generic_starts[0])
            ])
            best = min(candidates, key=lambda result: result.fun)
            n, r, sigma, C, A = likelihood.unpack(best.x)
            start = best.x
            fine = UnbinnedNoisyLikelihood(
                retained, eps=MEAS_ERR, dx=0.5 * DX, lower_cut=cut
            )
            fine_loglik = fine.loglik(n=n, r=r, sigma=sigma)
            rows.append({
                "dataset": name,
                "lower_cut": cut,
                "N_retained": int(retained.size),
                "n": float(n),
                "r": float(r),
                "sigma": float(sigma),
                "n_sigma2": float(C),
                "r_sigma": float(A),
                "loglik_conditional": float(-best.fun),
                "fine_grid_loglik": float(fine_loglik),
                "fine_minus_fit_loglik": float(fine_loglik + best.fun),
                "success": bool(best.success),
            })

    with open(OUTPUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")
    for row in rows:
        print(
            f"{row['dataset']:<7} cut={row['lower_cut']:.2f} N={row['N_retained']:4d} "
            f"n={row['n']:.3f} r={row['r']:.4f} sigma={row['sigma']:.5f}"
        )


if __name__ == "__main__":
    main()
