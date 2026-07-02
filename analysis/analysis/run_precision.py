"""Pin the early-time EMD constant kappa precisely.

Theory being tested. Exact identity: with fixed subset membership, the
normalized subset-vs-full EMD equals the normalized W1 separation of the two
complementary sub-distributions (raisers vs lowerers of the reference).
For jointly-Gaussian spectra with per-spin correlation rho and NO flips,
stochastic dominance gives EMD_norm(t) = rho(t) EXACTLY.
Flips perturb this at O(f), f = u/N:  EMD_norm = q^(p-1) (1 - kappa*u/N).
Candidate kappa for SSWM (size-biased flips, mirror bookkeeping): pi.
Candidate for uniform/neutral (beta=1): 2.
This script fits kappa from (a) the p-spin data, (b) high-precision
surrogates, (c) the uniform/neutral controls, plus the pool-law analog
(expect 2*beta = 4 for SSWM, 2 for uniform) as a methodology cross-check.
"""
import sys, os
import numpy as np
sys.path.insert(0, "analysis")
import analysis_lib as al
from run_controls import surrogate

def fit_kappa(u, y, N, p, y_min=0.82):
    """Least-squares kappa in y = (1-2u/(N-1))^(p-1) * (1 - kappa u/N), over y>=y_min."""
    u = np.asarray(u, float); y = np.asarray(y, float)
    ok = np.isfinite(y) & (y >= y_min) & (u > 0)
    base = al.rho_unflip_theory(u[ok], N, [p])
    # y/base = 1 - kappa u/N  -> linear regression through (u/N, 1 - y/base)
    x = u[ok]/N
    r = 1 - y[ok]/base
    return float(np.dot(x, r)/np.dot(x, x))

results = {}
for tag, p, N in [("N1000_P2", 2, 1000), ("N300_P3", 3, 300)]:
    z = np.load(f"analysis/results/meas_{tag}_pure.npz")
    Tm = z["T"].min()
    u = np.nanmean(z["pair_u"][:,0,:Tm+1],axis=0)
    emd = np.nanmean(z["pair_emd_pos"][:,0,:Tm+1],axis=0)
    pool = np.nanmean(z["pair_rho_pool"][:,0,:Tm+1],axis=0)
    # bootstrap kappa over repeats
    R = z["pair_emd_pos"].shape[0]
    rng = np.random.default_rng(0)
    ks = []
    for _ in range(400):
        idx = rng.integers(0,R,R)
        e = np.nanmean(z["pair_emd_pos"][idx,0,:Tm+1],axis=0)
        uu = np.nanmean(z["pair_u"][idx,0,:Tm+1],axis=0)
        ks.append(fit_kappa(uu,e,N,p))
    k_data = fit_kappa(u,emd,N,p); k_err = np.std(ks)
    k_pool = fit_kappa(u,pool,N,p)
    print(f"{tag} data: kappa_EMD = {k_data:.3f} +- {k_err:.3f} | kappa_pool = {k_pool:.3f} (expect 2beta=4)")
    results[tag] = dict(k_data=k_data, k_err=k_err, k_pool=k_pool)

    # high-precision surrogate, both rules
    for rule, nrep in [("sswm", 3000), ("uniform", 3000)]:
        Tt = max(50, int(0.12*N))
        s = surrogate(N, p, Tt, n_rep=nrep, rule=rule, seed=42, emd_every=1, emd_reps=nrep)
        m = np.isfinite(s["emd_pos"])
        k_s = fit_kappa(s["u"][m], s["emd_pos"][m], N, p)
        k_sp = fit_kappa(s["u"], s["rho_pool"], N, p)
        print(f"   surrogate[{rule}] n={nrep}: kappa_EMD = {k_s:.3f} | kappa_pool = {k_sp:.3f}")
        results[f"{tag}_{rule}"] = dict(k=k_s, k_pool=k_sp)

    # controls (real landscapes, other dynamics)
    for mode in ["uniform","neutral"]:
        zc = np.load(f"analysis/results/ctrl_{mode}_{tag}.npz")
        Tmc = zc["T"].min()
        uu = np.nanmean(zc["pair_u"][:,0,:Tmc+1],axis=0)
        ee = np.nanmean(zc["pair_emd_pos"][:,0,:Tmc+1],axis=0)
        pp = np.nanmean(zc["pair_rho_pool"][:,0,:Tmc+1],axis=0)
        print(f"   ctrl[{mode}]: kappa_EMD = {fit_kappa(uu,ee,N,p):.3f} | kappa_pool = {fit_kappa(uu,pp,N,p):.3f}")

print("\ncandidates: SSWM kappa: pi=3.142 (mirror bookkeeping) vs 2+pi/2=3.571 (old) | beta=1: kappa=2")
import json
json.dump(results, open("analysis/results/kappa.json","w"), indent=1)
