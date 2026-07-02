"""Control experiments isolating what drives early-time scrambling.

1. Neutral walk (v = 0): flip uniformly random sites on the SAME landscapes,
   ignoring the spectrum entirely. No drift toward anything, no terminal max.
   Prediction: identical kernel law rho_unflip = K_{p-1}(u)/C, and
   rho_pool = q^p with NO selection-bias correction (flips unbiased).

2. Uniform-acceptance greedy walk: accept uniformly among energy-raising
   flips (weighted=False). Different acceptance rule => possibly different
   drift v; kernel law and its slope should be unchanged.

3. Landscape-free Gaussian surrogate: N iid unit Gaussians evolving with the
   exact one-flip kernel factor c = 1 - 2(p-1)/(N-1) per accepted move, the
   flipped spin's value negated (Delta_i -> -Delta_i), selection SSWM
   (prob ~ Delta_+) or uniform. Contains no landscape, no sigma_f, no
   geometry, no v -- only the kernel and the flip bookkeeping. If it
   reproduces the p-spin early-time curves, early scrambling contains no
   radial physics at all.
"""

import os
import pickle
import sys
import time

import numpy as np
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import analysis_lib as al  # noqa: E402
import helper  # noqa: E402

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# 1 + 2: walks on the real landscapes
# ---------------------------------------------------------------------------

def run_alt_walks(pkl_path, tag, mode, n_steps_frac=0.65, seed=12345):
    """mode: 'neutral' (random flips) or 'uniform' (unweighted greedy)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rng = np.random.default_rng(seed)
    reps = []
    for r, rec in enumerate(data):
        N = rec["J"]["N"]
        if mode == "neutral":
            T = int(n_steps_frac * N)
            seq = rng.integers(0, N, size=T)
            rw = al.replay_walk(rec, flip_seq=seq)
        elif mode == "uniform":
            np.random.seed(seed + r)
            seq = helper.relax_pspin(rec["init_sigma"], rec["J"], weighted=False)
            rw = al.replay_walk(rec, flip_seq=seq)
            assert al.validate_replay(rw)
        else:
            raise ValueError(mode)
        scal = al.walk_scalars(rw)
        refs = [0, int(0.25 * rw["T"]), int(0.5 * rw["T"])]
        per_ref = []
        for t_ref in refs:
            m = al.measure_from_ref(rw, t_ref, with_transient=(mode == "uniform"))
            m["t_ref"] = t_ref
            m["d_f_ref"] = int(scal["d_f"][t_ref])
            per_ref.append(m)
        reps.append(dict(scal=scal, refs=refs, per_ref=per_ref))
        print(f"  [{mode}] repeat {r}: T={rw['T']}, d0={scal['d_f'][0]}", flush=True)
        del rw
    res = dict(N=data[0]["J"]["N"],
               orders=[s["order"] for s in data[0]["J"]["sectors"]], reps=reps)
    packed = al.pack_results(res)
    np.savez_compressed(os.path.join(RESULTS, f"ctrl_{mode}_{tag}.npz"), **packed)


# ---------------------------------------------------------------------------
# 3: kernel surrogate
# ---------------------------------------------------------------------------

def surrogate(N, p, T, n_rep, rule="sswm", seed=7, emd_every=2, emd_reps=None):
    """Gaussian kernel surrogate. Returns averaged observables vs t."""
    rng = np.random.default_rng(seed)
    c = 1.0 - 2.0 * (p - 1) / (N - 1.0)
    cc = np.sqrt(max(0.0, 1.0 - c * c))
    if emd_reps is None:
        emd_reps = n_rep
    nt = T + 1
    acc = {k: np.zeros(nt) for k in
           ("rho_pool", "rho_unflip", "rho_flip", "u")}
    cnt = {k: np.zeros(nt) for k in acc}
    emd = np.zeros(nt)
    emd_cnt = np.zeros(nt)

    for rep in range(n_rep):
        x = rng.standard_normal(N)
        x0 = x.copy()
        M = x0 > 0
        par = np.zeros(N, bool)
        do_emd = rep < emd_reps
        if do_emd:
            e0 = wasserstein_distance(x0, x0[M])
        for t in range(nt):
            # measure
            u = int(par.sum())
            acc["u"][t] += u; cnt["u"][t] += 1
            v = al._pearson(x0, x)
            acc["rho_pool"][t] += v; cnt["rho_pool"][t] += 1
            unflip = ~par
            if unflip.sum() > 12:
                acc["rho_unflip"][t] += al._pearson(x0[unflip], x[unflip])
                cnt["rho_unflip"][t] += 1
            if par.sum() > 12:
                acc["rho_flip"][t] += al._pearson(x0[par], x[par])
                cnt["rho_flip"][t] += 1
            if do_emd and (t % emd_every == 0):
                emd[t] += wasserstein_distance(x, x[M]) / e0
                emd_cnt[t] += 1
            if t == T:
                break
            # step: select flip
            pos = x > 0
            if not np.any(pos):
                break
            if rule == "sswm":
                w = np.where(pos, x, 0.0)
                i = rng.choice(N, p=w / w.sum())
            else:  # uniform among raisers
                idx = np.flatnonzero(pos)
                i = idx[rng.integers(idx.size)]
            x[i] = -x[i]
            par[i] = ~par[i]
            # kernel decay of all OTHER spins
            noise = rng.standard_normal(N)
            keep = x[i]
            x *= c
            x += cc * noise
            x[i] = keep
        # NOTE: x[i] of the flipped spin is restored (its own field is
        # unaffected by its own flip); it decays on subsequent flips of others.
    out = {k: acc[k] / np.maximum(cnt[k], 1) for k in acc}
    out["emd_pos"] = np.where(emd_cnt > 0, emd / np.maximum(emd_cnt, 1), np.nan)
    out["t"] = np.arange(nt)
    out["q"] = 1.0 - 2.0 * out["u"] / N
    return out


def main():
    t0 = time.time()
    for pkl, tag in [("N1000_P2_pure_repeats10.pkl", "N1000_P2"),
                     ("N300_P3_pure_repeats10.pkl", "N300_P3")]:
        path = os.path.join(ROOT, pkl)
        for mode in ("neutral", "uniform"):
            out = os.path.join(RESULTS, f"ctrl_{mode}_{tag}.npz")
            if os.path.exists(out):
                print(f"-- {mode} {tag} done already", flush=True)
                continue
            print(f">> {mode} walks on {tag}", flush=True)
            run_alt_walks(path, tag, mode)
            print(f"   done ({time.time()-t0:.0f}s)", flush=True)

    for (N, p, T, tag) in [(1000, 2, 400, "N1000_P2"), (300, 3, 120, "N300_P3")]:
        for rule in ("sswm", "uniform"):
            out = os.path.join(RESULTS, f"surr_{rule}_{tag}.npz")
            if os.path.exists(out):
                continue
            print(f">> surrogate {rule} {tag}", flush=True)
            res = surrogate(N, p, T, n_rep=400, rule=rule,
                            emd_reps=200, emd_every=2)
            np.savez_compressed(out, **res)
            print(f"   done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
