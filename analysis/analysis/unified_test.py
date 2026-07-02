"""Test of the endpoint-conditioned (pinned-kernel) unified EMD law.

Effective pairwise separation transfer (established at t_ref = 0):
    rho_hat(u) = [K_{p-1}(u;N-1)/C] * exp(-pi u / N)
Conditioning on the terminal spectrum (Gaussian partial regression on the
triple (t_ref, t, T) with mutual distances u12, d1 = u(t_ref,T), d2 = u(t,T)):
    W(dt) = b1(dt) + Phi * b2(dt)
    b1 = [rho(u12) - rho(d1) rho(d2)] / [1 - rho(d1)^2]     (live part)
    b2 = [rho(d2) - rho(u12) rho(d1)] / [1 - rho(d1)^2]     (floor loading)
Phi = raiser-lowerer separation carried by the frozen spectrum, measured
independently as the EMD at the end of the walk (not fitted).

Limits: far field -> rho(u12); deep basin -> amplitude x angle exactly
(all per-flip constants cancel in the ratio). No interpolation anywhere.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_lib as al  # noqa: E402
from intermediate_analysis import t_half  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
MAIN = [("N1000_P2", 2, 1000), ("N300_P3", 3, 300)]


def rho_hat(u, N, p):
    return al.rho_unflip_theory(np.asarray(u, float), N, [p]) * np.exp(-np.pi * np.asarray(u, float) / N)


def collect(z, N, p):
    Ts = z["T"]
    refs = []
    for r in range(len(Ts)):
        T = int(Ts[r])
        d_f_r = z["scal_d_f"][r]
        for j in range(z["ref_t"].shape[1]):
            tr, d1 = int(z["ref_t"][r, j]), float(z["ref_d_f"][r, j])
            if tr < 0 or tr >= T - 4:
                continue
            emd = z["pair_emd_pos"][r, j, tr:T + 1]
            if not np.isfinite(emd[1:]).any():
                continue
            u12 = z["pair_u"][r, j, tr:T + 1].astype(float)
            d2 = d_f_r[tr:T + 1].astype(float)
            r1 = rho_hat(d1, N, p)
            r2 = rho_hat(d2, N, p)
            ru = rho_hat(u12, N, p)
            den = 1 - r1 ** 2
            b1 = (ru - r1 * r2) / den
            b2 = (r2 - ru * r1) / den
            fin = emd[np.isfinite(emd)]
            phi = float(np.nanmean(fin[-max(2, len(fin) // 10):])) if len(fin) > 4 else np.nan
            refs.append(dict(rep=r, d1=d1, dN=d1 / N, emd=emd, b1=b1, b2=b2, phi=phi))
    return refs


def bin_stats(sel, use_phi="bin"):
    Lm = min(len(x["emd"]) for x in sel)
    dt = np.arange(Lm)
    emd = np.nanmean([x["emd"][:Lm] for x in sel], axis=0)
    b1 = np.nanmean([x["b1"][:Lm] for x in sel], axis=0)
    b2 = np.nanmean([x["b2"][:Lm] for x in sel], axis=0)
    phi = float(np.nanmean([x["phi"] for x in sel]))
    pred = b1 + phi * b2
    pred0 = b1  # live part only (Phi = 0)
    return dt, emd, pred, pred0, phi


def main():
    rng = np.random.default_rng(0)
    edges = {2: [0.04, 0.055, 0.07, 0.085, 0.10, 0.12, 0.145, 0.17, 0.20,
                 0.235, 0.27, 0.31, 0.35, 0.44],
             3: [0.055, 0.08, 0.10, 0.12, 0.145, 0.17, 0.20, 0.235, 0.27, 0.34]}
    out = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.0))
    figr, axesr = plt.subplots(1, 2, figsize=(11, 4.3))
    for row, (tag, p, N) in enumerate(MAIN):
        z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
        refs = collect(z, N, p)
        rows = []
        for lo, hi in zip(edges[p][:-1], edges[p][1:]):
            sel = [x for x in refs if lo <= x["dN"] < hi]
            if len(sel) < 4:
                continue
            dt, emd, pred, pred0, phi = bin_stats(sel)
            th_d, th_p = t_half(dt, emd), t_half(dt, pred)
            # curve-level max abs deviation over W in [0.15, 1]
            m = np.isfinite(emd) & np.isfinite(pred) & (emd > 0.15)
            dev = float(np.nanmax(np.abs(emd[m] - pred[m]))) if m.any() else np.nan
            # bootstrap data half-time
            reps = sorted({x["rep"] for x in sel})
            bs = []
            for _ in range(300):
                pick = set(rng.choice(reps, len(reps)))
                ss = [x for x in sel if x["rep"] in pick]
                if len(ss) < 3:
                    continue
                d2t, e2, *_ = bin_stats(ss)
                tt = t_half(d2t, e2)
                if np.isfinite(tt):
                    bs.append(tt)
            rows.append(dict(dN=float(np.mean([x["dN"] for x in sel])), n=len(sel),
                             phi=phi, t_data=th_d, t_data_err=float(np.std(bs)) if bs else np.nan,
                             t_pred=th_p, dev=dev,
                             curves=(dt, emd, pred, pred0)))
        out[tag] = [{k: v for k, v in r.items() if k != "curves"} for r in rows]

        # example curves at three depths spanning the regimes
        want = [0.33, 0.15, 0.07] if p == 2 else [0.30, 0.15, 0.09]
        for col, wt in enumerate(want):
            r = min(rows, key=lambda r: abs(r["dN"] - wt))
            dt, emd, pred, pred0 = r["curves"]
            ax = axes[row, col]
            ax.plot(dt, emd, color="#6a3fb5", lw=2.2, label="EMD data (bin avg)")
            ax.plot(dt, pred, "--", color="#d62728", lw=2,
                    label="pinned-kernel law (with measured $\\Phi$)")
            ax.plot(dt, pred0, ":", color="#d62728", lw=1.4, alpha=0.8,
                    label="live part only ($\\Phi=0$)")
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            xm = 3.5 * (r["t_data"] if np.isfinite(r["t_data"]) else 40)
            ax.set_xlim(0, min(xm, dt[-1]))
            ax.set_ylim(0, 1.02)
            ax.set_xlabel(r"$\Delta t$")
            if col == 0:
                ax.set_ylabel("normalized EMD")
            ax.set_title(f"p={p}: $d/N={r['dN']:.2f}$, $\\Phi={r['phi']:.2f}$", fontsize=10)
            ax.legend(fontsize=8)

        ax = axesr[row]
        dN = np.array([r["dN"] for r in rows])
        ratio = np.array([r["t_data"] / r["t_pred"] for r in rows])
        err = np.array([r["t_data_err"] / r["t_pred"] for r in rows])
        ax.errorbar(dN, ratio, yerr=err, fmt="o", ms=7, color="#d62728", capsize=3,
                    label="pinned-kernel law (no free params)")
        ax.axhline(1, color="k", lw=1)
        ax.axhspan(0.85, 1.15, color="k", alpha=0.07)
        ax.set_xscale("log")
        ax.set_ylim(0.4, 2.0)
        ax.set_xlabel(r"$d_H(t_{\rm ref},\sigma_f)/N$")
        ax.set_ylabel(r"$t_{1/2}^{\rm data}/t_{1/2}^{\rm law}$")
        ax.set_title(f"p={p}, N={N}", fontsize=10)
        ax.legend(fontsize=8.5)

    fig.tight_layout(); fig.savefig(os.path.join(FIG, "u1_unified.png"), dpi=160)
    figr.tight_layout(); figr.savefig(os.path.join(FIG, "u2_unified_ratio.png"), dpi=160)
    with open(os.path.join(RES, "unified.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    for tag, p, N in MAIN:
        print(f"\n=== {tag}: pinned-kernel unified law ===")
        print(f"{'d/N':>6} {'n':>3} {'Phi':>5} {'t_data':>7} {'err':>5} {'t_law':>6} {'ratio':>6} {'maxdev':>7}")
        for r in out[tag]:
            print(f"{r['dN']:6.3f} {r['n']:3d} {r['phi']:5.2f} {r['t_data']:7.1f} "
                  f"{r['t_data_err']:5.1f} {r['t_pred']:6.1f} "
                  f"{r['t_data']/r['t_pred']:6.2f} {r['dev']:7.3f}")


if __name__ == "__main__":
    main()
