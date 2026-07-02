"""Figure: the exact displacement-geometry law for the transient correlation
holds at ALL depths (u3), completing the all-times analysis."""

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


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for col, (tag, p, N) in enumerate([("N1000_P2", 2, 1000), ("N300_P3", 3, 300)]):
        z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
        Ts = z["T"]
        edges = [(0.05, 0.08), (0.10, 0.14), (0.17, 0.23), (0.27, 0.34), (0.36, 0.45)]
        colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(edges)))
        ax = axes[col]
        for (lo, hi), colr in zip(edges, colors):
            rows_tr, rows_pred, ds = [], [], []
            for r in range(len(Ts)):
                T = int(Ts[r])
                d_f_r = z["scal_d_f"][r]
                for j in range(z["ref_t"].shape[1]):
                    tr_, d1 = int(z["ref_t"][r, j]), float(z["ref_d_f"][r, j])
                    if tr_ < 0 or tr_ >= T - 4 or not (lo <= d1 / N < hi):
                        continue
                    u12 = z["pair_u"][r, j, tr_:T + 1].astype(float)
                    d2 = d_f_r[tr_:T + 1].astype(float)
                    # kernel-difference law with the verified pool two-point
                    # function rho_p(u) = q^(p-1) (1 - 4u/N)_+
                    def rp(u):
                        return (al.rho_unflip_theory(u, N, [p])
                                * np.maximum(1 - 4 * np.asarray(u, float) / N, 0))
                    r1, r2, ru = rp(np.array([d1]))[0], rp(d2), rp(u12)
                    pred = (ru + 1 - r1 - r2) / (2 * np.sqrt(
                        np.maximum((1 - r1) * (1 - r2), 1e-12)))
                    rows_pred.append(pred)
                    rows_tr.append(z["pair_rho_trans"][r, j, tr_:T + 1])
                    ds.append(d1)
            if not rows_tr:
                continue
            Lm = min(len(a) for a in rows_tr)
            tr_a = np.nanmean([a[:Lm] for a in rows_tr], axis=0)
            pr_a = np.nanmean([a[:Lm] for a in rows_pred], axis=0)
            x = np.arange(Lm)
            d_m = np.mean(ds)
            # rescale lag by the geometric half-time for a common axis
            ok = np.isfinite(pr_a)
            th = np.interp(-0.5, -pr_a[ok], x[ok]) if pr_a[ok].min() < 0.5 else x[ok][-1]
            ax.plot(x / th, tr_a, color=colr, lw=1.8,
                    label=fr"$d/N\approx{d_m/N:.2f}$")
            ax.plot(x / th, pr_a, "--", color=colr, lw=1.4, alpha=0.9)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"$\Delta t\,/\,t_{1/2}^{\rm geometry}$")
        ax.set_ylabel(r"transient correlation $\rho_{\rm trans}$")
        ax.set_title(f"p={p}, N={N}: the kernel-difference law at every depth\n"
                     "(solid: data; dashed: derived law, no parameters)",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "u3_alltimes_geometry.png"), dpi=160)
    print("u3 written")


if __name__ == "__main__":
    main()
