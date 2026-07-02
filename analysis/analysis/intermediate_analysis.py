"""Quantitative test of the crossover description in the intermediate regime.

For every dense reference (binned by depth d/N) we compare the measured EMD
half-time t_1/2 (first crossing of W~ = 1/2; protocol-free) against:

  K       kernel/flip curve      q_u^{p-1} (1 - pi u/N), with the MEASURED u(dt)
  B       basin curve            sqrt(d2/d1) * (d1+d2-u12)/(2 sqrt(d1 d2)),
                                 from the measured walk geometry
  product K*B  (curve-level rate addition)
  harmonic     1/(1/tK + 1/tB)   (scalar-level rate addition)
  min          min(tK, tB)       (sharp crossover)

Outputs: figures/i1_intermediate.png, i2_ratios.png, results/intermediate.json
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

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

MAIN = [("N1000_P2", 2, 1000), ("N300_P3", 3, 300)]
C_DATA, C_K, C_B, C_PROD = "#6a3fb5", "#1f77b4", "#2ca02c", "#d62728"


def t_half(dt, y):
    """First crossing of y = 0.5 (linear interpolation); NaN if never."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    dt, y = np.asarray(dt, float)[ok], y[ok]
    below = np.nonzero(y <= 0.5)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1, y0, y1 = dt[i - 1], dt[i], y[i - 1], y[i]
    return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)


def t_half_kernel(N, p, c=1.0):
    """Half-time of the closed-form kernel/flip law with mutual speed c:
    solve (K-ratio(u))*(1 - pi u/N) = 1/2, then t = u_half / c."""
    u = np.linspace(0, N / np.pi * 0.999, 4000)
    y = al.rho_unflip_theory(u, N, [p]) * (1 - np.pi * u / N)
    u_half = np.interp(-0.5, -y, u)   # y decreasing
    return u_half / c


def collect_refs(z, N, p):
    """Per (rep, ref): forward curves of emd, K, B; ref depth; v_rem; c."""
    Ts = z["T"]
    out = []
    for r in range(len(Ts)):
        T = int(Ts[r])
        d_f_r = z["scal_d_f"][r]
        for j in range(z["ref_t"].shape[1]):
            tr, d1 = int(z["ref_t"][r, j]), float(z["ref_d_f"][r, j])
            if tr < 0 or tr >= T - 4 or d1 <= 0:
                continue
            emd = z["pair_emd_pos"][r, j, tr:T + 1]
            if not np.isfinite(emd[1:]).any():
                continue
            u12 = z["pair_u"][r, j, tr:T + 1].astype(float)
            d2 = d_f_r[tr:T + 1].astype(float)
            kern = al.rho_unflip_theory(u12, N, [p]) * np.maximum(1 - np.pi * u12 / N, 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                angle = (d1 + d2 - u12) / (2 * np.sqrt(d1 * d2))
                basin = np.sqrt(np.maximum(d2, 0) / d1) * angle
            basin = np.clip(basin, 0, None)
            L = min(int(d1 / 2) + 2, len(u12) - 1)
            c = np.polyfit(np.arange(L), u12[:L], 1)[0] if L > 2 else np.nan
            out.append(dict(rep=r, d1=d1, dN=d1 / N, emd=emd, K=kern, B=basin,
                            v_rem=d1 / (T - tr), c=c))
    return out


def bin_refs(refs, N, edges):
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [x for x in refs if lo <= x["dN"] < hi]
        if len(sel) < 4:
            continue
        bins.append(sel)
    return bins


def bin_curves(sel):
    Lm = min(len(x["emd"]) for x in sel)
    dt = np.arange(Lm)
    avg = lambda key: np.nanmean([x[key][:Lm] for x in sel], axis=0)
    return dt, avg("emd"), avg("K"), avg("B")


def analyze(tag, p, N, edges, rng):
    z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
    refs = collect_refs(z, N, p)
    rows = []
    for sel in bin_refs(refs, N, edges):
        dt, emd, K, B = bin_curves(sel)
        c_bin = float(np.nanmean([x["c"] for x in sel]))
        # kernel half-time from the closed-form law with the measured mutual
        # speed (the curve-based version is truncated by the walk end at
        # small d; where both exist they agree to a few %)
        tK = t_half_kernel(N, p, c=c_bin if np.isfinite(c_bin) else 1.0)
        tB = t_half(dt, B)
        row = dict(
            dN=float(np.mean([x["dN"] for x in sel])),
            n=len(sel),
            v_rem=float(np.mean([x["v_rem"] for x in sel])),
            c=c_bin,
            t_data=t_half(dt, emd),
            t_K=tK, t_B=tB,
            t_K_curve=t_half(dt, K),
            t_prod=t_half(dt, K * B),
            t_harm=1 / (1 / tK + 1 / tB) if np.isfinite(tK) and np.isfinite(tB) else np.nan,
            t_min=np.nanmin([tK, tB]),
        )
        # bootstrap the data half-time over repeats
        reps = sorted({x["rep"] for x in sel})
        bs = []
        for _ in range(400):
            pick = set(rng.choice(reps, len(reps)))
            ss = [x for x in sel if x["rep"] in pick]
            if len(ss) < 3:
                continue
            d2t, e2, _, _ = bin_curves(ss)
            th = t_half(d2t, e2)
            if np.isfinite(th):
                bs.append(th)
        row["t_data_err"] = float(np.std(bs)) if bs else np.nan
        row["curves"] = (dt, emd, K, B)
        rows.append(row)
    return rows


def main():
    rng = np.random.default_rng(0)
    edges2 = [0.04, 0.055, 0.07, 0.085, 0.10, 0.12, 0.145, 0.17, 0.20, 0.235,
              0.27, 0.31, 0.35, 0.44]
    edges3 = [0.055, 0.08, 0.10, 0.12, 0.145, 0.17, 0.20, 0.235, 0.27, 0.34]
    results = {}
    fig1, axes1 = plt.subplots(2, 3, figsize=(13, 8.0))
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.3))
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})

    for row_i, ((tag, p, N), edges) in enumerate(zip(MAIN, [edges2, edges3])):
        rows = analyze(tag, p, N, edges, rng)
        results[tag] = [{k: v for k, v in r.items() if k != "curves"} for r in rows]

        # --- example curves at two intermediate depths -----------------
        want = [0.20, 0.11]
        for col, wtarget in enumerate(want):
            r = min(rows, key=lambda r: abs(r["dN"] - wtarget))
            dt, emd, K, B = r["curves"]
            ax = axes1[row_i, col]
            ax.plot(dt, emd, color=C_DATA, lw=2.2, label="EMD data (bin avg)")
            ax.plot(dt, K, "--", color=C_K, label="kernel curve K")
            ax.plot(dt, B, "--", color=C_B, label="basin curve B")
            ax.plot(dt, K * B, "-", color=C_PROD, alpha=0.8, label=r"product $K\times B$")
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.set_xlim(0, min(3.2 * (r["t_data"] if np.isfinite(r["t_data"]) else 50), dt[-1]))
            ax.set_ylim(0, 1.02)
            ax.set_xlabel(r"$\Delta t$ (accepted moves)")
            if col == 0:
                ax.set_ylabel(r"normalized EMD")
            ax.set_title(f"p={p}: reference at $d/N={r['dN']:.2f}$  "
                         f"($n$={r['n']}, $v_{{\\rm rem}}$={r['v_rem']:.2f})", fontsize=10)
            ax.legend(fontsize=8)

        # --- t_half vs depth -------------------------------------------
        ax = axes1[row_i, 2]
        dN = np.array([r["dN"] for r in rows])
        ax.errorbar(dN, [r["t_data"] for r in rows], yerr=[r["t_data_err"] for r in rows],
                    fmt="o", ms=7, color=C_DATA, capsize=3, label=r"$t_{1/2}$ data", zorder=5)
        ax.plot(dN, [r["t_K"] for r in rows], "--", color=C_K, label="kernel only")
        ax.plot(dN, [r["t_B"] for r in rows], "--", color=C_B, label="basin only")
        ax.plot(dN, [r["t_harm"] for r in rows], "-", color=C_PROD, lw=2.4, alpha=0.85,
                label="rates added (harmonic)")
        ax.plot(dN, [r["t_min"] for r in rows], ":", color="k", alpha=0.7,
                label=r"sharp: $\min(t_K,t_B)$")
        dstar = N * (1 + np.mean([r["v_rem"] for r in rows])) / (2 * (2 * (p - 1) + np.pi))
        ax.axvline(dstar / N, color="#666", lw=1, ls="-.")
        ax.text(dstar / N, ax.get_ylim()[0] + 0.04 * np.diff(ax.get_ylim())[0],
                r" $d^*$", color="#444", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel(r"$d_H(t_{\rm ref},\sigma_f)/N$")
        ax.set_ylabel(r"$t_{1/2}$ (accepted moves)")
        ax.set_title(f"p={p}, N={N}: EMD half-time through the crossover", fontsize=10)
        ax.legend(fontsize=8)

        # --- ratio panel -------------------------------------------------
        ax = axes2[row_i]
        for key, colr, lab, mk in [("t_harm", C_PROD, "rates added (harmonic)", "o"),
                                   ("t_prod", "#8c564b", r"product $K\times B$", "s"),
                                   ("t_K", C_K, "kernel only", "^"),
                                   ("t_B", C_B, "basin only", "v"),
                                   ("t_min", "#7f7f7f", r"$\min(t_K,t_B)$", "x")]:
            ratio = np.array([r["t_data"] / r[key] for r in rows])
            err = np.array([r["t_data_err"] / r[key] for r in rows])
            ax.errorbar(dN * (1 + 0.012 * "oshvx".find(mk[0] if mk != "x" else "x")), ratio,
                        yerr=err, fmt=mk, ms=6, color=colr, capsize=2, label=lab, alpha=0.85)
        ax.axhline(1, color="k", lw=1)
        ax.axhspan(0.85, 1.15, color="k", alpha=0.07)
        ax.set_xscale("log")
        ax.set_ylim(0.4, 2.6)
        ax.set_xlabel(r"$d_H(t_{\rm ref},\sigma_f)/N$")
        ax.set_ylabel(r"$t_{1/2}^{\rm data} / t_{1/2}^{\rm prediction}$")
        ax.set_title(f"p={p}: accuracy of each description (band: $\\pm15\\%$)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")

    fig1.tight_layout()
    fig1.savefig(os.path.join(FIG, "i1_intermediate.png"), dpi=160)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG, "i2_ratios.png"), dpi=160)

    with open(os.path.join(RES, "intermediate.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)

    for tag, p, N in MAIN:
        print(f"\n=== {tag} ===")
        print(f"{'d/N':>6} {'n':>3} {'v_rem':>6} {'t_data':>7} {'err':>5} "
              f"{'K':>6} {'B':>6} {'harm':>6} {'prod':>6} {'min':>6} "
              f"{'data/harm':>9} {'data/prod':>9}")
        for r in results[tag]:
            print(f"{r['dN']:6.3f} {r['n']:3d} {r['v_rem']:6.2f} "
                  f"{r['t_data']:7.1f} {r['t_data_err']:5.1f} "
                  f"{r['t_K']:6.1f} {r['t_B']:6.1f} {r['t_harm']:6.1f} "
                  f"{r['t_prod']:6.1f} {r['t_min']:6.1f} "
                  f"{r['t_data']/r['t_harm']:9.2f} {r['t_data']/r['t_prod']:9.2f}")


if __name__ == "__main__":
    main()
