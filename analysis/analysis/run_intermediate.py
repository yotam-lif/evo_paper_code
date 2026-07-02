"""Dense reference grid through the intermediate regime (d/N ~ 0.05 - 0.4).

Produces meas_dense_{tag}.npz with the same packed layout as the main runs,
but with references pinned densely in d_H(t_ref, sigma_f).
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import analysis_lib as al  # noqa: E402

RESULTS = os.path.join(HERE, "results")

DENSE_D_FRACS = (0.05, 0.0625, 0.075, 0.09, 0.11, 0.13, 0.155, 0.18,
                 0.21, 0.25, 0.29, 0.33, 0.37, 0.41)

JOBS = [
    ("N1000_P2_pure_repeats10.pkl", "N1000_P2"),
    ("N300_P3_pure_repeats10.pkl", "N300_P3"),
]

if __name__ == "__main__":
    for fname, tag in JOBS:
        out = os.path.join(RESULTS, f"meas_dense_{tag}.npz")
        if os.path.exists(out):
            print(f"-- {tag} dense already done", flush=True)
            continue
        t0 = time.time()
        print(f">> dense {tag}", flush=True)
        res = al.process_dataset(os.path.join(ROOT, fname),
                                 fracs=(0.0,),
                                 d_target_fracs=DENSE_D_FRACS,
                                 d_target_abs=())
        packed = al.pack_results(res)
        np.savez_compressed(out, **packed)
        print(f"<< {tag} dense done in {time.time()-t0:.1f}s", flush=True)
