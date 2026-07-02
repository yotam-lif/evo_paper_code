"""Run the full measurement pipeline on the stored p-spin datasets."""

import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import analysis_lib as al  # noqa: E402

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

FULL_FRACS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SWEEP_FRACS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)

MAIN = [
    ("N1000_P2_pure_repeats10.pkl", FULL_FRACS),
    ("N300_P3_pure_repeats10.pkl", FULL_FRACS),
]
SWEEP = [
    ("N100_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N200_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N300_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N400_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N500_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N1500_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N2000_P2_pure_repeats10.pkl", SWEEP_FRACS),
    ("N100_P3_pure_repeats10.pkl", SWEEP_FRACS),
    ("N200_P3_pure_repeats10.pkl", SWEEP_FRACS),
    ("N400_P3_pure_repeats10.pkl", SWEEP_FRACS),
    ("N500_P3_pure_repeats10.pkl", SWEEP_FRACS),
    ("N400_P3_mixed_repeats10.pkl", SWEEP_FRACS),
]


def snapshot_dump(pkl_path, out_path, rep_idx=0, n_snap=8):
    """Save raw spectra snapshots of one repeat for scatter/illustration plots."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rec = data[rep_idx]
    rw = al.replay_walk(rec)
    scal = al.walk_scalars(rw)
    T = rw["T"]
    ts = sorted(set(np.linspace(0, T, n_snap).astype(int)))
    np.savez_compressed(
        out_path,
        S=rw["S"][ts],
        parity=rw["parity"][ts].astype(np.uint8),
        S_final=rw["S"][-1],
        ts=np.array(ts),
        d_f=scal["d_f"],
        u0=scal["u0"],
        N=rw["N"], T=T,
    )


def main(which):
    jobs = {"main": MAIN, "sweep": SWEEP}[which]
    for fname, fracs in jobs:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            print(f"!! missing {fname}", flush=True)
            continue
        tag = fname.replace("_repeats10.pkl", "")
        out = os.path.join(RESULTS, f"meas_{tag}.npz")
        if os.path.exists(out):
            print(f"-- {tag} already done", flush=True)
            continue
        t0 = time.time()
        print(f">> {tag}", flush=True)
        res = al.process_dataset(path, fracs=fracs)
        packed = al.pack_results(res)
        np.savez_compressed(out, **packed)
        print(f"<< {tag} done in {time.time()-t0:.1f}s -> {out}", flush=True)
        if fname.startswith(("N1000_P2", "N300_P3")):
            snap_out = os.path.join(RESULTS, f"snap_{tag}.npz")
            snapshot_dump(path, snap_out)
            print(f"   snapshots -> {snap_out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
