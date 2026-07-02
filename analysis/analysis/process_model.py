"""The mean-field process model (EBBM-type master equation + SSWM), tested
as an ALL-TIMES theory of the EMD -- and shown to fail in the basin.

Per accepted move (derived from the couplings, exact to O(1/N)):
    pick k with prob ~ lambda_k^+            (SSWM)
    lambda_i += -(a/N) (lambda_i + lambda_k) + eta_i   for i != k,
                a = 2(p-1),  Var(eta) = 4 p (p-1) / N
    lambda_k -> -lambda_k
Initial: lambda ~ N(0, p). The -(a/N)lambda_k piece is a COMMON shift
(invisible to the shift-invariant EMD); the -(a/N)lambda_i piece is the
kernel contraction; kicks are fresh (iid). The model contains no
configuration geometry, hence no pinning to a terminal state.

Comparison protocol: match references to the p-spin data by the number of
surviving raisers n_+ (the only meaningful clock a geometry-free model has
near its end). Runs are capped at 1.6 N moves; many do not terminate at all.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RES = os.path.join(HERE, "results")


def run_process(N, p, rng, cap):
    a = 2.0 * (p - 1)
    sig = np.sqrt(4.0 * p * (p - 1) / N)
    lam = rng.standard_normal(N) * np.sqrt(p)
    traj = np.empty((cap + 1, N), np.float32)
    traj[0] = lam
    T = cap
    for t in range(cap):
        pos = lam > 0
        npos = int(pos.sum())
        if npos == 0:
            T = t
            break
        w = np.where(pos, lam, 0.0)
        k = rng.choice(N, p=w / w.sum())
        lk = lam[k]
        lam += -(a / N) * (lam + lk) + rng.standard_normal(N) * sig
        lam[k] = -lk
        traj[t + 1] = lam
    return traj[:T + 1], T < cap


def emd_curve(traj, t_ref, every=2):
    lam_ref = traj[t_ref].astype(float)
    M = lam_ref > 0
    if M.sum() < 12:
        return None, None
    ref = wasserstein_distance(lam_ref, lam_ref[M])
    ts = np.arange(t_ref, len(traj), every)
    vals = np.array([wasserstein_distance(traj[t].astype(float),
                                          traj[t].astype(float)[M]) for t in ts]) / ref
    return ts - t_ref, vals


def data_npos_at_depth(tag, d_lo, d_hi):
    z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
    vals = []
    for r in range(len(z["T"])):
        for j in range(z["ref_t"].shape[1]):
            tr, d1 = int(z["ref_t"][r, j]), z["ref_d_f"][r, j]
            if tr < 0 or not (d_lo <= d1 <= d_hi):
                continue
            vals.append(z["scal_n_pos"][r, tr])
    return float(np.nanmean(vals)) if vals else np.nan


def main():
    out = {}
    for N, p, n_rep, tag, d_targets in [
            (1000, 2, 40, "N1000_P2", (62, 125)),
            (300, 3, 100, "N300_P3", (20, 37))]:
        rng = np.random.default_rng(11)
        cap = int(1.6 * N)
        npos_targets = {d: data_npos_at_depth(tag, d - d // 8, d + d // 8)
                        for d in d_targets}
        print(f"\n=== process model N={N}, p={p} ===")
        print(" data n_+ at depth:", {d: round(v, 1) for d, v in npos_targets.items()})
        Ts, terminated, bmin = [], 0, []
        acc = {d: [] for d in d_targets}
        acc[0] = []   # t_ref = 0
        for rep in range(n_rep):
            traj, ok = run_process(N, p, rng, cap)
            Ts.append(len(traj) - 1)
            terminated += ok
            npos_t = (traj > 0).sum(axis=1)
            bmin.append(npos_t.min())
            dt, vals = emd_curve(traj, 0)
            if vals is not None:
                acc[0].append((dt, vals))
            for d in d_targets:
                tgt = npos_targets[d]
                idx = np.nonzero(npos_t <= tgt)[0]
                if idx.size == 0:
                    continue
                t_ref = int(idx[0])
                if len(traj) - t_ref < 12:
                    continue
                dt, vals = emd_curve(traj, t_ref)
                if vals is not None:
                    acc[d].append((dt, vals))
        Ts = np.array(Ts)
        print(f" terminated within {cap} moves: {terminated}/{n_rep}; "
              f"min n_+ reached: median {np.median(bmin):.0f}")
        res = {}
        for d in [0] + list(d_targets):
            rows = acc[d]
            if not rows:
                print(f" ref d~{d}: no usable references")
                continue
            Lm = min(len(v) for _, v in rows)
            dtg = rows[0][0][:Lm]
            avg = np.nanmean([v[:Lm] for _, v in rows], axis=0)
            below = np.nonzero(avg <= 0.5)[0]
            th = np.nan
            if below.size and below[0] > 0:
                i = below[0]
                th = np.interp(0.5, [avg[i], avg[i - 1]], [dtg[i], dtg[i - 1]])
            floor = float(np.nanmean(avg[-max(2, Lm // 8):]))
            lbl = "t_ref=0" if d == 0 else f"n+~{npos_targets[d]:.0f} (data d={d})"
            print(f" ref {lbl:24s}: n={len(rows):3d}  t_half={th:6.1f}  "
                  f"late value={floor:.2f}")
            res[str(d)] = dict(n=len(rows), t_half=float(th), late=floor)
        out[f"N{N}_P{p}"] = dict(T_over_N=float(Ts.mean() / N),
                                 terminated=int(terminated), n_rep=n_rep, res=res)
    with open(os.path.join(RES, "process_model.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
