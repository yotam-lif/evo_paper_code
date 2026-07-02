"""Which transient-correlation law is actually derived, and how well does
each do per depth?

V1  tangential (what fig 8d used):
    [ (d1+d2-u12) - 2 d1 d2/N ] / [ 2 sqrt(d1 d2 (1-d1/N)(1-d2/N)) ]
V2  kernel-difference (three-line derivation, general p):
    a_i(t) = Delta_i(t) - Delta_i^f;  with the annealed kernel rhoK,
    Corr(a1,a2) = [rhoK(u12) + 1 - rhoK(d1) - rhoK(d2)]
                  / (2 sqrt[(1-rhoK(d1))(1-rhoK(d2))]).
    For p = 2 this equals the exact chord-counting formula
    (d1+d2-u12)/(2 sqrt(d1 d2)).
V3  same with the parity/selection-corrected pool law
    rho_p(u) = rhoK(u) (1-4u/N)_+  in all three slots.
V4  (p=2 only, exact ground truth): diagonalize J, w(t) = r(t) - r_f,
    rho = w1' L^2 w2 / sqrt((w1' L^2 w1)(w2' L^2 w2)).
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import analysis_lib as al  # noqa: E402

RES = os.path.join(HERE, "results")


def rhoK(u, N, p):
    return al.rho_unflip_theory(np.asarray(u, float), N, [p])


def v_tangential(d1, d2, u12, N):
    num = (d1 + d2 - u12) - 2 * d1 * d2 / N
    den = 2 * np.sqrt(np.maximum(d1 * d2 * (1 - d1 / N) * (1 - d2 / N), 1e-12))
    return num / den


def v_kernel_diff(d1, d2, u12, N, p, pool=False):
    if pool:
        f = lambda u: rhoK(u, N, p) * np.maximum(1 - 4 * np.asarray(u, float) / N, 0)
    else:
        f = lambda u: rhoK(u, N, p)
    r1, r2, ru = f(d1), f(d2), f(u12)
    return (ru + 1 - r1 - r2) / (2 * np.sqrt(np.maximum((1 - r1) * (1 - r2), 1e-12)))


def per_depth_test(tag, p, N):
    z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
    Ts = z["T"]
    edges = [(0.05, 0.08), (0.10, 0.14), (0.17, 0.23), (0.27, 0.34), (0.36, 0.45)]
    print(f"\n=== {tag}: pointwise max|dev| over rho_trans in [0.25,0.97] ===")
    print(f"{'d/N':>10} {'V1 tang':>8} {'V2 kdiff':>9} {'V3 pool':>8}")
    for lo, hi in edges:
        rows = {k: [] for k in ("tr", "v1", "v2", "v3")}
        for r in range(len(Ts)):
            T = int(Ts[r])
            d_f_r = z["scal_d_f"][r]
            for j in range(z["ref_t"].shape[1]):
                tr_, d1 = int(z["ref_t"][r, j]), float(z["ref_d_f"][r, j])
                if tr_ < 0 or tr_ >= T - 4 or not (lo <= d1 / N < hi):
                    continue
                u12 = z["pair_u"][r, j, tr_:T + 1].astype(float)
                d2 = d_f_r[tr_:T + 1].astype(float)
                rows["tr"].append(z["pair_rho_trans"][r, j, tr_:T + 1])
                rows["v1"].append(v_tangential(d1, d2, u12, N))
                rows["v2"].append(v_kernel_diff(d1, d2, u12, N, p))
                rows["v3"].append(v_kernel_diff(d1, d2, u12, N, p, pool=True))
        if not rows["tr"]:
            continue
        Lm = min(len(a) for a in rows["tr"])
        avg = {k: np.nanmean([a[:Lm] for a in v], axis=0) for k, v in rows.items()}
        m = (avg["tr"] > 0.25) & (avg["tr"] < 0.97)
        devs = {k: np.nanmax(np.abs(avg[k][m] - avg["tr"][m])) if m.any() else np.nan
                for k in ("v1", "v2", "v3")}
        print(f"[{lo:.2f},{hi:.2f}) {devs['v1']:8.3f} {devs['v2']:9.3f} {devs['v3']:8.3f}")


def exact_p2_test(n_reps=3):
    """V4: exact Lambda^2-weighted law for p=2 via diagonalization."""
    import helper
    with open(os.path.join(ROOT, "N1000_P2_pure_repeats10.pkl"), "rb") as f:
        data = pickle.load(f)
    N = 1000
    print("\n=== p=2 exact (Lambda^2-weighted) vs data, per depth ===")
    print(f"{'rep':>4} {'d/N':>6} {'maxdev V4':>10} {'maxdev V2':>10}")
    for rep in range(n_reps):
        rec = data[rep]
        # dense J matrix and eigensystem
        sec = rec["J"]["sectors"][0]
        i0, i1 = sec["spin_indices"]
        Jm = np.zeros((N, N))
        Jm[i0.astype(int), i1.astype(int)] = sec["couplings"]
        Jm = Jm + Jm.T
        lam, V = np.linalg.eigh(Jm)         # E = 1/2 sum lam_k r_k^2, r = V^T sigma
        rw = al.replay_walk(rec)
        scal = al.walk_scalars(rw)
        T = rw["T"]
        sig = np.array([np.where(rw["parity"][t], -rec["init_sigma"], rec["init_sigma"])
                        for t in range(T + 1)], dtype=np.int8)
        r_t = sig @ V                        # (T+1, N) coordinates
        r_f = r_t[-1]
        w = r_t - r_f[None, :]
        lam2 = lam ** 2
        S = rw["S"].astype(np.float64)
        a = S - S[-1][None, :]               # measured transient
        for d_target in (60, 120, 200, 310, 400):
            idx = np.nonzero(scal["d_f"] >= d_target)[0]
            if idx.size == 0:
                continue
            t1 = int(idx[-1])
            if T - t1 < 20:
                continue
            d1 = scal["d_f"][t1]
            ts = np.arange(t1, T + 1)
            w1 = w[t1]
            num = (w[ts] * lam2[None, :]) @ w1
            den = np.sqrt(np.maximum((w[ts] ** 2 @ lam2) * (w1 ** 2 @ lam2), 1e-12))
            v4 = num / den
            # measured rho_trans for the same reference
            a1 = a[t1] - a[t1].mean()
            meas = np.empty(len(ts))
            for k, t2 in enumerate(ts):
                a2 = a[t2] - a[t2].mean()
                dd = np.sqrt((a1 @ a1) * (a2 @ a2))
                meas[k] = (a1 @ a2) / dd if dd > 0 else np.nan
            u12 = np.count_nonzero(rw["parity"][ts] ^ rw["parity"][t1][None, :], axis=1)
            d2 = scal["d_f"][ts].astype(float)
            v2 = v_kernel_diff(float(d1), d2, u12.astype(float), N, 2)
            m = (meas > 0.25) & (meas < 0.97)
            if not m.any():
                continue
            dev4 = np.nanmax(np.abs(v4[m] - meas[m]))
            dev2 = np.nanmax(np.abs(v2[m] - meas[m]))
            print(f"{rep:4d} {d1/N:6.2f} {dev4:10.3f} {dev2:10.3f}")


if __name__ == "__main__":
    per_depth_test("N1000_P2", 2, 1000)
    per_depth_test("N300_P3", 3, 300)
    exact_p2_test()
