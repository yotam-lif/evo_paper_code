"""Position-only (predictive) form of the transient law.

Substitute the typical trajectory into the kernel-difference law:
    u12(dt) = c dt          (mutual speed, c = 0.95 +- 0.03 measured, ~1)
    d2(dt)  = d - v_rem dt  (remaining drift)
Then the initial decay rate has an EXACT v-cancellation:
    rate(d) = c |rho_p'(0)| / [2 (1 - rho_p(d))] = c*Lambda / [2N(1-rho_p(d))],
    Lambda = 2(p-1) + 4,
i.e. the timescale depends only on the current depth d, not on the drift.
Interpretation: (remaining transient variance at depth d) / (universal
per-move covariance loss).

This script: (a) builds the d-only prediction curve and its t_half for a
grid of depths (showing v-insensitivity numerically), (b) compares with the
measured t_half of rho_trans per depth bin, (c) calibrates the observable
proxy n_+ (current number of beneficial moves) against d.
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


def rho_p(u, N, p):
    u = np.asarray(u, float)
    return al.rho_unflip_theory(u, N, [p]) * np.maximum(1 - 4 * u / N, 0)


def d_only_curve(d, N, p, v, c=0.95, L=None):
    if L is None:
        L = int(3.5 * d / max(v, 1e-9)) + 200
    dt = np.arange(L)
    d2 = np.maximum(d - v * dt, 0.0)
    num = rho_p(c * dt, N, p) + 1 - rho_p(np.array([d]), N, p)[0] - rho_p(d2, N, p)
    den = 2 * np.sqrt(np.maximum((1 - rho_p(np.array([d]), N, p)[0]) * (1 - rho_p(d2, N, p)), 1e-12))
    return dt, num / den


def t_half(dt, y):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    dt, y = np.asarray(dt, float)[ok], y[ok]
    below = np.nonzero(y <= 0.5)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return dt[i - 1] + (0.5 - y[i - 1]) * (dt[i] - dt[i - 1]) / (y[i] - y[i - 1])


def measured_points(tag, p, N):
    """Measured t_half of rho_trans + v_rem + n_+ per depth bin."""
    z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
    Ts = z["T"]
    edges = [(0.05, 0.08), (0.10, 0.14), (0.17, 0.23), (0.27, 0.34), (0.36, 0.45)]
    out = []
    for lo, hi in edges:
        rows, ds, vr, npos = [], [], [], []
        for r in range(len(Ts)):
            T = int(Ts[r])
            for j in range(z["ref_t"].shape[1]):
                tr_, d1 = int(z["ref_t"][r, j]), float(z["ref_d_f"][r, j])
                if tr_ < 0 or tr_ >= T - 4 or not (lo <= d1 / N < hi):
                    continue
                rows.append(z["pair_rho_trans"][r, j, tr_:T + 1])
                ds.append(d1)
                vr.append(d1 / (T - tr_))
                npos.append(z["scal_n_pos"][r, tr_])
        if not rows:
            continue
        Lm = min(len(a) for a in rows)
        avg = np.nanmean([a[:Lm] for a in rows], axis=0)
        out.append(dict(d=float(np.mean(ds)), n=len(rows),
                        v_rem=float(np.mean(vr)), n_pos=float(np.nanmean(npos)),
                        t_half=float(t_half(np.arange(Lm), avg))))
    return out


def main():
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for col, (tag, p, N, vglob) in enumerate([("N1000_P2", 2, 1000, 0.615),
                                              ("N300_P3", 3, 300, 0.745)]):
        pts = measured_points(tag, p, N)
        Lam = 2 * (p - 1) + 4
        dgrid = np.geomspace(0.03 * N, 0.45 * N, 60)
        tang = 2 * N * (1 - rho_p(dgrid, N, p)) / (0.95 * Lam)
        th_v1, th_v2 = [], []
        vr_of_d = np.interp(dgrid, [x["d"] for x in pts], [x["v_rem"] for x in pts])
        for d, vr in zip(dgrid, vr_of_d):
            dt, y = d_only_curve(d, N, p, v=vr)
            th_v1.append(t_half(dt, y))
            dt, y = d_only_curve(d, N, p, v=vglob)
            th_v2.append(t_half(dt, y))
        ax = axes[col]
        ax.plot(dgrid / N, tang, "--", color="#999",
                label=r"tangent: $2N(1-\rho_p(d))/c\Lambda$")
        ax.plot(dgrid / N, th_v1, "-", color="#d62728", lw=2.2,
                label=r"$t_{1/2}$ of the $d$-only law ($v_{\rm rem}$ measured)")
        ax.plot(dgrid / N, th_v2, ":", color="#d62728", lw=2.2,
                label=fr"same with constant $v={vglob}$")
        ax.plot([x["d"] / N for x in pts], [x["t_half"] for x in pts], "o",
                ms=9, color="#6a3fb5", label=r"measured $t_{1/2}$ of $\rho_{\rm trans}$")
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.set_xticks([0.05, 0.1, 0.2, 0.4])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"current depth $d/N$")
        ax.set_ylabel(r"$t_{1/2}$ (accepted moves)")
        ax.set_title(f"p={p}, N={N}: position-only timescale", fontsize=10.5)
        ax.legend(fontsize=8, loc="upper left")
        results[tag] = dict(points=pts,
                            grid=dict(dN=(dgrid / N).tolist(), tangent=tang.tolist(),
                                      t_half_vmeas=th_v1, t_half_vglob=th_v2))
    # n_+ calibration panel
    ax = axes[2]
    for tag, p, N, colr in [("N1000_P2", 2, 1000, "#1f77b4"), ("N300_P3", 3, 300, "#d62728")]:
        z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
        dvals, nvals = [], []
        for r in range(len(z["T"])):
            T = int(z["T"][r])
            d_f = z["scal_d_f"][r, :T + 1]
            npos = z["scal_n_pos"][r, :T + 1]
            dvals.append(d_f / N)
            nvals.append(npos / N)
        dva = np.concatenate(dvals); nva = np.concatenate(nvals)
        bins = np.geomspace(0.02, 0.45, 18)
        idx = np.digitize(dva, bins)
        xs, ys = [], []
        for b in range(1, len(bins)):
            m = idx == b
            if m.sum() > 20:
                xs.append(np.mean(dva[m])); ys.append(np.mean(nva[m]))
        ax.plot(xs, ys, "o-", ms=5, color=colr, label=f"p={p}")
        results[tag]["n_pos_calibration"] = dict(dN=xs, nposN=ys)
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.4])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$d/N$")
    ax.set_ylabel(r"$n_+/N$ (observable now)")
    ax.set_title("read your depth off the spectrum:\n$n_+$ vs $d$ calibration", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d5_position_only.png"), dpi=160)
    with open(os.path.join(RES, "position_only.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)

    for tag in results:
        print(f"\n=== {tag}: measured vs d-only prediction (t_half) ===")
        pts = results[tag]["points"]
        g = results[tag]["grid"]
        for x in pts:
            pred = np.interp(x["d"] / (1000 if "P2" in tag else 300),
                             g["dN"], g["t_half_vmeas"])
            print(f" d/N={x['d']/(1000 if 'P2' in tag else 300):.3f} n={x['n']:3d} "
                  f"n+={x['n_pos']:6.1f} t_half={x['t_half']:6.1f} "
                  f"pred={pred:6.1f} ratio={x['t_half']/pred:5.2f}")


if __name__ == "__main__":
    main()
