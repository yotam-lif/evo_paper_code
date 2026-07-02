"""Produce all figures + a stats.json of headline numbers for the report."""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_lib as al  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9.5,
    "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "axes.grid": True, "grid.alpha": 0.25,
})

C_DATA, C_TH, C_SURR, C_NAIVE1, C_NAIVE2 = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#bcbd22"
STATS = {}

MAIN = [("N1000_P2_pure", 2, 0.615), ("N300_P3_pure", 3, 0.745)]


def load(tag):
    return np.load(os.path.join(RES, f"meas_{tag}.npz"))


def ref0_avg(z, key, Tcut=None):
    Ts = z["T"]
    Tmin = Ts.min() if Tcut is None else min(Ts.min(), Tcut)
    return np.nanmean(z[f"pair_{key}"][:, 0, :Tmin + 1], axis=0)


def boot_slope(z, key, y_hi, y_lo, n_boot=400, ref=0, seed=0):
    """Bootstrap (over repeats) the initial slope of an averaged pair curve."""
    rng = np.random.default_rng(seed)
    Ts = z["T"]; R = len(Ts); Tmin = Ts.min()
    ts = np.arange(Tmin + 1)
    rows = z[f"pair_{key}"][:, ref, :Tmin + 1]
    base = al.fit_initial_slope(ts, np.nanmean(rows, axis=0), y_hi, y_lo)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, R, R)
        s = al.fit_initial_slope(ts, np.nanmean(rows[idx], axis=0), y_hi, y_lo)
        if np.isfinite(s):
            bs.append(s)
    return base, np.std(bs)


# ===========================================================================
# Fig 1: phenomenology
# ===========================================================================

def fig_phenomenology():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for row, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Ts = z["T"]; R = len(Ts)
        Tm = Ts.min()
        t = np.arange(Tm + 1)
        d_f = np.nanmean(z["scal_d_f"][:, :Tm + 1], axis=0)
        u0 = np.nanmean(z["scal_u0"][:, :Tm + 1], axis=0)
        ax = axes[row, 0]
        ax.plot(t, d_f / N, color=C_DATA, label=r"$d_H(\sigma(t),\sigma_f)/N$")
        ax.plot(t, u0 / N, color="#ff7f0e", label=r"$u(0,t)/N=d_H(\sigma(0),\sigma(t))/N$")
        ax.plot(t, (d_f[0] - v * t) / N, "--", color=C_DATA, alpha=0.8,
                label=fr"slope $-v$, $v={v}$")
        ax.plot(t, np.minimum(t, u0[-1] * 1.05) / N, "--", color="#ff7f0e", alpha=0.8,
                label="slope $+1$ (unit speed)")
        ax.set_xlabel("t (accepted flips)")
        ax.set_title(f"p={p}, N={N}: radial drift vs displacement")
        ax.legend(loc="center right")
        ax.set_ylim(0, 0.75)

        ax = axes[row, 1]
        R2 = 4 * d_f * (N - d_f) / N
        ax.plot(t, R2 / N, color=C_DATA, label=r"$R^2(t)/N$ (data)")
        dlin = np.maximum(d_f[0] - v * t, 0)
        ax.plot(t, 4 * dlin * (N - dlin) / N / N, "--", color=C_TH,
                label=r"parabola from linear $d_H(t)$")
        ax.plot(t, 1 - 2 * v * t / N, ":", color=C_NAIVE1,
                label="linear-in-t (Model X analog)")
        ax.set_xlabel("t")
        ax.set_title(r"shell radius: $R^2$ is a parabola, not linear")
        ax.legend()
        ax.set_ylim(0, 1.05)

        ax = axes[row, 2]
        w = np.nanmean(z["scal_spec_std"][:, :Tm + 1], axis=0)
        npos = np.nanmean(z["scal_n_pos"][:, :Tm + 1], axis=0)
        ax.plot(t / Tm, w / w[0], color=C_DATA, label="spectrum width (norm.)")
        ax.plot(t / Tm, npos / npos[0], color="#9467bd", label=r"$n_{+}(t)/n_{+}(0)$")
        ax.set_xlabel("t / T")
        ax.set_title("width stays O(1); raisers deplete")
        ax.set_ylim(0, 1.1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_phenomenology.png"))
    plt.close(fig)


# ===========================================================================
# Fig 2: early-time money plot
# ===========================================================================

def fig_early():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Tm = z["T"].min()
        t = np.arange(Tm + 1)
        u = ref0_avg(z, "u")
        r_unf = ref0_avg(z, "rho_unflip")
        r_pool = ref0_avg(z, "rho_pool")
        emd = ref0_avg(z, "emd_pos")
        th_unf = al.rho_unflip_theory(u, N, [p])
        th_pool = th_unf * (1 - 4 * u / N)

        s = np.load(os.path.join(RES, f"surr_sswm_N{N}_P{p}.npz"))
        L = min(Tm, len(s["t"]) - 1)

        d0 = np.nanmean(load(tag)["scal_d_f"][:, 0])
        ax = axes[col]
        ax.plot(t, r_unf, color=C_DATA, label=r"$\rho_{\rm unflipped}(0,t)$ data")
        ax.plot(t, th_unf, "--", color=C_TH, label=r"kernel: $\xi'(q_{0t})/\xi'(1)=q_{0t}^{\,p-1}$")
        ax.plot(t, r_pool, color="#17becf", label=r"$\rho_{\rm pool}(0,t)$ data")
        ax.plot(t, th_pool, "--", color="#e377c2",
                label=r"kernel+flips: $q^{\,p-1}(1-4u/N)$")
        ax.plot(t, emd, color="#9467bd", label="EMD subset-vs-full (norm.)")
        ax.plot(s["t"][:L + 1], s["rho_pool"][:L + 1], ".", ms=2.5, color=C_SURR,
                label="landscape-free surrogate (pool)")
        m = np.isfinite(s["emd_pos"][:L + 1])
        ax.plot(s["t"][:L + 1][m], s["emd_pos"][:L + 1][m], ".", ms=2.5, color="#8c564b",
                label="surrogate (EMD)")
        # naive radial transfers
        qn = np.maximum(1 - 2 * v * t / N, 0)
        ax.plot(t, qn ** (p - 1), ":", color=C_NAIVE1, lw=2,
                label=fr"naive radial: $(1-2v t/N)^{{p-1}}$, $v={v}$")
        ax.plot(t, np.exp(-v * t / d0), ":", color=C_NAIVE2, lw=2,
                label=r"naive radial: $e^{-vt/d_0}$ (time-to-max)")
        ax.set_xlabel("t (accepted flips)")
        ax.set_ylabel("normalized measure")
        ax.set_xlim(0, min(int(0.55 * N / (p - 1) * 1.6), Tm))
        ax.set_ylim(0, 1.02)
        ax.set_title(f"p={p}, N={N}: early-time scrambling, ref $t_{{\\rm ref}}=0$")
        ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_early_money.png"))
    plt.close(fig)


# ===========================================================================
# Fig 3: collapse in mutual overlap; non-collapse in radial coordinate
# ===========================================================================

def fig_collapse():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Ts = z["T"]; R = len(Ts)
        nref = z["ref_t"].shape[1]
        cmap = cm.viridis
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for r in range(R):
            T = Ts[r]
            d_f_r = z["scal_d_f"][r]
            for j in range(nref):
                tr = z["ref_t"][r, j]
                if tr < 0:
                    continue
                frac = tr / T
                if frac > 0.75:
                    continue
                q_mut = z["pair_q"][r, j, :T + 1]
                rho = z["pair_rho_unflip"][r, j, :T + 1]
                q_f = 1 - 2 * d_f_r[:T + 1] / N
                colr = cmap(frac / 0.8)
                sl = slice(tr, T + 1, 3)   # forward direction, subsample
                ax_top.plot(q_mut[sl], rho[sl], ".", ms=1.4, color=colr, alpha=0.5)
                ax_bot.plot(q_f[sl], rho[sl], ".", ms=1.4, color=colr, alpha=0.5)
        qg = np.linspace(0, 1, 200)
        ug = (1 - qg) * N / 2
        ax_top.plot(qg, al.rho_unflip_theory(ug, N, [p]), "-", color=C_TH, lw=2,
                    label=r"$\xi'(q)/\xi'(1)$ (annealed kernel)")
        ax_top.set_xlabel(r"mutual overlap $q(t_{\rm ref},t)$")
        ax_top.set_ylabel(r"$\rho_{\rm unflipped}$")
        ax_top.set_title(f"p={p}: collapse in MUTUAL overlap")
        ax_top.legend(loc="upper left")
        ax_top.set_xlim(0, 1); ax_top.set_ylim(-0.15, 1.02)
        ax_bot.set_xlabel(r"radial coordinate $q_f(t)=1-2d_H(t)/N$ (overlap with $\sigma_f$)")
        ax_bot.set_ylabel(r"$\rho_{\rm unflipped}$")
        ax_bot.set_title(f"p={p}: NO collapse in the radial coordinate")
        ax_bot.set_xlim(0, 1); ax_bot.set_ylim(-0.15, 1.02)
        sm = cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 0.8))
        for a in (ax_top, ax_bot):
            cb = fig.colorbar(sm, ax=a, pad=0.01)
            cb.set_label(r"$t_{\rm ref}/T$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_collapse.png"))
    plt.close(fig)


# ===========================================================================
# Fig 4: controls -- v varies, slope doesn't
# ===========================================================================

def fig_controls():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, (tag, p, v_sswm) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        short = f"N{N}_P{p}"
        entries = []
        # SSWM data
        s_unf, e_unf = boot_slope(z, "rho_unflip", 0.995, 0.90)
        s_pool, e_pool = boot_slope(z, "rho_pool", 0.995, 0.90)
        entries.append(("SSWM\n(greedy $\\propto\\Delta$)", v_sswm, s_unf, e_unf, s_pool, e_pool))
        for mode, label in [("uniform", "uniform greedy"), ("neutral", "neutral walk")]:
            zc = np.load(os.path.join(RES, f"ctrl_{mode}_{short}.npz"))
            if mode == "uniform":
                vs = []
                for r in range(len(zc["T"])):
                    T = zc["T"][r]; b = int(0.65 * T)
                    vs.append(-np.polyfit(np.arange(b + 1), zc["scal_d_f"][r, :b + 1], 1)[0])
                vv = np.mean(vs)
            else:
                vv = 0.0
            su, eu = boot_slope(zc, "rho_unflip", 0.995, 0.90)
            sp, ep = boot_slope(zc, "rho_pool", 0.995, 0.90)
            entries.append((label + f"\n$v={vv:.2f}$", vv, su, eu, sp, ep))
        ax = axes[col]
        x = np.arange(len(entries))
        unf = [e[2] * N / 2 for e in entries]
        unf_e = [e[3] * N / 2 for e in entries]
        pool = [e[4] * N / 2 for e in entries]
        pool_e = [e[5] * N / 2 for e in entries]
        ax.errorbar(x - 0.08, unf, yerr=unf_e, fmt="o", color=C_DATA, ms=7,
                    label=r"$\rho_{\rm unflipped}$: slope$\times N/2$")
        ax.errorbar(x + 0.08, pool, yerr=pool_e, fmt="s", color="#17becf", ms=7,
                    label=r"$\rho_{\rm pool}$: slope$\times N/2$")
        vs = [e[1] for e in entries]
        ax.plot(x, [v * (p - 1) for v in vs], "x", color=C_NAIVE1, ms=9, mew=2,
                label=r"naive radial $v\,(p-1)$")
        ax.axhline(p - 1, color=C_DATA, ls="--", alpha=0.6)
        ax.text(2.35, p - 1, r"$p-1$", color=C_DATA, va="bottom")
        ax.axhline(p + 1, color="#17becf", ls="--", alpha=0.6)
        ax.text(2.35, p + 1, r"$p+1$ (SSWM $\beta$=2)", color="#17becf", va="bottom")
        ax.axhline(p, color="#17becf", ls=":", alpha=0.6)
        ax.text(2.35, p, r"$p$ ($\beta$=1)", color="#17becf", va="bottom")
        ax.set_xticks(x, [e[0] for e in entries])
        ax.set_ylabel(r"initial decay rate $\times N/2$")
        ax.set_title(f"p={p}, N={N}: drift varies, kernel slope does not")
        ax.set_ylim(0, p + 2.2)
        ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_controls.png"))
    plt.close(fig)


# ===========================================================================
# Fig 5: slope scaling across N
# ===========================================================================

def fig_scaling():
    import glob
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, p in [(0, 2), (1, 3)]:
        ax = axes[col]
        rows = []
        for f in sorted(glob.glob(os.path.join(RES, f"meas_N*_P{p}_pure.npz"))):
            z = np.load(f)
            N = int(z["N"])
            su, eu = boot_slope(z, "rho_unflip", 0.995, 0.90)
            sp, ep = boot_slope(z, "rho_pool", 0.995, 0.90)
            se, ee = boot_slope(z, "emd_pos", 0.995, 0.90)
            # matched-protocol theory
            u = ref0_avg(z, "u")
            th = al.rho_unflip_theory(u, N, [p])
            ts = np.arange(len(u))
            s_th = al.fit_initial_slope(ts, th, 0.995, 0.90)
            # drift
            vs = []
            for r in range(len(z["T"])):
                T = z["T"][r]; b = int(0.65 * T)
                vs.append(-np.polyfit(np.arange(b + 1), z["scal_d_f"][r, :b + 1], 1)[0])
            rows.append((N, su, eu, sp, ep, se, ee, s_th, np.mean(vs)))
        rows.sort()
        Ns = np.array([r[0] for r in rows], float)
        def col_(i): return np.array([r[i] for r in rows])
        ax.errorbar(Ns * 0.97, col_(1) * Ns / 2, yerr=col_(2) * Ns / 2, fmt="o",
                    color=C_DATA, label=r"$\rho_{\rm unflipped}$")
        ax.errorbar(Ns, col_(3) * Ns / 2, yerr=col_(4) * Ns / 2, fmt="s",
                    color="#17becf", label=r"$\rho_{\rm pool}$")
        ax.errorbar(Ns * 1.03, col_(5) * Ns / 2, yerr=col_(6) * Ns / 2, fmt="^",
                    color="#9467bd", label="EMD")
        ax.plot(Ns, col_(7) * Ns / 2, "_", color=C_TH, ms=12, mew=2,
                label="kernel theory (matched fit)")
        ax.axhline(p - 1, ls="--", color=C_DATA, alpha=0.6)
        ax.axhline(p + 1, ls="--", color="#17becf", alpha=0.6)
        ax.axhline(p + np.pi / 4, ls="--", color="#9467bd", alpha=0.6)
        vN = col_(8)
        ax.plot(Ns, vN * (p - 1), "x", color=C_NAIVE1, ms=8, mew=2,
                label=r"naive radial $v(p-1)$ (per-N $v$)")
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(r"initial rate $\times N/2$")
        ax.set_title(f"p={p}: rates scale as 1/N with kernel constants")
        ax.set_ylim(0, p + 2.4)
        ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_scaling.png"))
    plt.close(fig)


# ===========================================================================
# Fig 6 + 7: late time
# ===========================================================================

def _late_curves(z, N, d_lo, d_hi):
    Ts = z["T"]; R = len(Ts)
    rows = {"tr": [], "emd": [], "pred": [], "predA": []}
    vrems, cs, drefs = [], [], []
    for r in range(R):
        T = Ts[r]; d_f_r = z["scal_d_f"][r]
        for j in range(z["ref_t"].shape[1]):
            tr_, dfr = z["ref_t"][r, j], z["ref_d_f"][r, j]
            if tr_ < 0 or not (d_lo <= dfr <= d_hi) or tr_ >= T:
                continue
            u12 = z["pair_u"][r, j, tr_:T]
            d1, d2 = float(dfr), d_f_r[tr_:T]
            with np.errstate(invalid="ignore", divide="ignore"):
                uhat = (d1 + d2 - u12) / (2 * np.sqrt(d1 * d2))
            rows["pred"].append(uhat)
            rows["predA"].append(np.sqrt(np.maximum(d2, 0) / d1) * uhat)
            rows["tr"].append(z["pair_rho_trans"][r, j, tr_:T])
            rows["emd"].append(z["pair_emd_pos"][r, j, tr_:T])
            vrems.append(d1 / (T - tr_))
            L = min(int(d1 / 2) + 2, len(u12) - 1)
            if L > 2:
                cs.append(np.polyfit(np.arange(L), u12[:L], 1)[0])
            drefs.append(d1)
    if not rows["tr"]:
        return None
    Lm = min(len(a) for a in rows["tr"])
    out = {k: np.nanmean([a[:Lm] for a in v], axis=0) for k, v in rows.items()}
    out["dt"] = np.arange(Lm)
    out["v_rem"] = np.mean(vrems)
    out["c"] = np.mean(cs) if cs else np.nan
    out["d_ref"] = np.mean(drefs)
    out["n"] = len(rows["tr"])
    return out


def fig_late():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    tau_rows = {2: [], 3: []}
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        targets = [(N // 32 - 3, N // 32 + 4), (N // 16 - 4, N // 16 + 6),
                   (N // 8 - 8, N // 8 + 8)]
        ax_tr, ax_emd = axes[0, col], axes[1, col]
        colors = ["#1f77b4", "#2ca02c", "#9467bd"]
        for (d_lo, d_hi), colr in zip(targets, colors):
            cur = _late_curves(z, N, d_lo, d_hi)
            if cur is None:
                continue
            d = cur["d_ref"]
            x = cur["dt"] / (2 * d)
            ax_tr.plot(x, cur["tr"], color=colr, label=fr"$d_{{\rm ref}}\approx{d:.0f}$")
            ax_tr.plot(x, cur["pred"], "--", color=colr, alpha=0.9)
            ax_emd.plot(x, cur["emd"], color=colr, label=fr"$d_{{\rm ref}}\approx{d:.0f}$")
            ax_emd.plot(x, cur["predA"], "--", color=colr, alpha=0.9)
            # collect taus (shallow window)
            for key, store in [("tr", "tau_tr"), ("emd", "tau_emd"),
                               ("pred", "tau_pred"), ("predA", "tau_predA")]:
                s = al.fit_initial_slope(cur["dt"], cur[key], 0.97, 0.55)
                cur[store] = 1 / s if s and np.isfinite(s) else np.nan
            tau_rows[p].append(cur)
        xg = np.linspace(0, 1.6, 100)
        ax_tr.plot(xg, np.exp(-xg), ":", color="k", alpha=0.7,
                   label=r"$e^{-\Delta t/2d}$  ($\tau=R^2/2$)")
        ax_tr.set_xlabel(r"$\Delta t / 2d_{\rm ref}$")
        ax_tr.set_ylabel(r"$\rho_{\rm trans}$ (Pearson of $\Delta-\Delta^{\rm final}$)")
        ax_tr.set_title(f"p={p}: transient correlation vs exact shell geometry (dashed)")
        ax_tr.legend()
        ax_tr.set_xlim(0, 1.6); ax_tr.set_ylim(0, 1.02)
        ax_emd.plot(xg, np.exp(-xg * (1 + 0.87)), ":", color="k", alpha=0.7,
                    label=r"$e^{-(1+v)\Delta t/2d}$")
        ax_emd.set_xlabel(r"$\Delta t / 2d_{\rm ref}$")
        ax_emd.set_ylabel("EMD subset-vs-full (norm.)")
        ax_emd.set_title(f"p={p}: EMD decays faster: amplitude $\\times$ angle (dashed)")
        ax_emd.legend()
        ax_emd.set_xlim(0, 1.6); ax_emd.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_late.png"))
    plt.close(fig)

    # tau vs d plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for col, (tag, p, v) in enumerate(MAIN):
        ax = axes[col]
        rows = tau_rows[p]
        ds = np.array([c["d_ref"] for c in rows])
        ax.plot(ds, [c["tau_tr"] for c in rows], "o", color=C_DATA, ms=8,
                label=r"$\tau$ of $\rho_{\rm trans}$ (data)")
        ax.plot(ds, [c["tau_pred"] for c in rows], "_", color=C_TH, ms=14, mew=2.5,
                label="exact geometry formula")
        ax.plot(ds, [c["tau_emd"] for c in rows], "^", color="#9467bd", ms=8,
                label=r"$\tau$ of EMD (data)")
        ax.plot(ds, [c["tau_predA"] for c in rows], "x", color="#8c564b", ms=9, mew=2,
                label="amplitude $\\times$ angle")
        dg = np.linspace(ds.min() * 0.7, ds.max() * 1.3, 50)
        ax.plot(dg, 2 * dg, "--", color="k", alpha=0.6, label=r"$\tau=2d=R^2/2$")
        vr = np.mean([c["v_rem"] for c in rows])
        ax.plot(dg, 2 * dg / (1 + vr), ":", color="k", alpha=0.6,
                label=fr"$\tau=2d/(1+v_{{\rm rem}})$, $v_{{\rm rem}}={vr:.2f}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$d_{\rm ref}=d_H(t_{\rm ref},\sigma_f)$")
        ax.set_ylabel(r"initial decay time $\tau$ (accepted moves)")
        ax.set_title(f"p={p}: late-time timescales")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_tau_late.png"))
    plt.close(fig)
    return tau_rows


# ===========================================================================
# Fig 8: crossover -- local rate along the whole walk
# ===========================================================================

def fig_crossover():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Ts = z["T"]; R = len(Ts)
        nref = z["ref_t"].shape[1]
        # group refs by d_f(ref) bins
        bins = {}
        for r in range(R):
            T = Ts[r]
            for j in range(nref):
                tr, dfr = z["ref_t"][r, j], z["ref_d_f"][r, j]
                if tr < 0 or tr >= T - 4:
                    continue
                key = int(np.floor(np.log2(max(dfr, 2))))
                dt = np.arange(T + 1 - tr)
                bins.setdefault(key, {"emd": [], "pool": [], "d": [], "vr": []})
                bins[key]["emd"].append(z["pair_emd_pos"][r, j, tr:T + 1])
                bins[key]["pool"].append(z["pair_rho_pool"][r, j, tr:T + 1])
                bins[key]["d"].append(dfr)
                bins[key]["vr"].append(dfr / (T - tr))
        ax = axes[col]
        ds, r_emd, r_pool, vrs = [], [], [], []
        for key, b in sorted(bins.items()):
            Lm = min(len(a) for a in b["emd"])
            if Lm < 6:
                continue
            emd = np.nanmean([a[:Lm] for a in b["emd"]], axis=0)
            pool = np.nanmean([a[:Lm] for a in b["pool"]], axis=0)
            dt = np.arange(Lm)
            se = al.fit_initial_slope(dt, emd, 0.97, 0.55)
            sp = al.fit_initial_slope(dt, pool, 0.97, 0.55)
            ds.append(np.mean(b["d"]))
            vrs.append(np.mean(b["vr"]))
            r_emd.append(se); r_pool.append(sp)
        ds = np.array(ds); r_emd = np.array(r_emd, float); r_pool = np.array(r_pool, float)
        ax.plot(ds / N, r_emd * N, "o-", color="#9467bd", label="EMD local rate")
        ax.plot(ds / N, r_pool * N, "s-", color="#17becf", label=r"$\rho_{\rm pool}$ local rate")
        dg = np.geomspace(max(ds.min() * 0.8, 2), N / 2, 120)
        kappa_emd = 2 * p + np.pi / 2
        ax.axhline(kappa_emd, ls="--", color="#9467bd", alpha=0.7,
                   label=fr"kernel branch: $2p+\pi/2$")
        ax.axhline(2 * p + 2, ls="--", color="#17becf", alpha=0.7,
                   label=r"kernel branch: $2p+2$")
        vr = 0.87
        ax.plot(dg / N, N * (1 + vr) / (2 * dg), ":", color="k",
                label=r"shell branch: $(1+v_{\rm rem})/2d$")
        ax.plot(dg / N, kappa_emd + N * (1 + vr) / (2 * dg), "-", color="#9467bd",
                alpha=0.35, lw=3, label="sum of branches (EMD)")
        ax.set_xscale("log"); ax.set_yscale("log")
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$d_H(t_{\rm ref},\sigma_f)/N$  (radial position of the reference)")
        ax.set_ylabel(r"initial decay rate $\times N$")
        ax.set_title(f"p={p}, N={N}: early plateau $\\to$ late shell branch")
        ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_crossover.png"))
    plt.close(fig)


# ===========================================================================
# Fig 9: symmetry + raising/lowering subsets
# ===========================================================================

def fig_symmetry():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Ts = z["T"]; R = len(Ts)
        # mid ref ~0.4T
        j = int(np.argmin(np.abs(z["ref_t"][0] / max(Ts[0], 1) - 0.4)))
        fwd_u, bwd_u, fwd_e, bwd_e, fwd_en, bwd_en = [], [], [], [], [], []
        smax = 1000
        for r in range(R):
            T = Ts[r]; tr = z["ref_t"][r, j]
            s_ = min(tr, T - tr)
            smax = min(smax, s_)
        for r in range(R):
            T = Ts[r]; tr = z["ref_t"][r, j]
            row_u = z["pair_rho_unflip"][r, j, :T + 1]
            row_e = z["pair_emd_pos"][r, j, :T + 1]
            row_en = z["pair_emd_neg"][r, j, :T + 1]
            fwd_u.append(row_u[tr:tr + smax]); bwd_u.append(row_u[tr::-1][:smax])
            fwd_e.append(row_e[tr:tr + smax]); bwd_e.append(row_e[tr::-1][:smax])
            fwd_en.append(row_en[tr:tr + smax]); bwd_en.append(row_en[tr::-1][:smax])
        s = np.arange(smax)
        ax = axes[col]
        ax.plot(s, np.nanmean(fwd_u, axis=0), color=C_DATA, label=r"$\rho_{\rm unflip}$ forward")
        ax.plot(s, np.nanmean(bwd_u, axis=0), "--", color=C_DATA, label=r"$\rho_{\rm unflip}$ backward")
        ax.plot(s, np.nanmean(fwd_e, axis=0), color="#9467bd",
                label="EMD forward (raisers $\\equiv$ lowerers)")
        ax.plot(s, np.nanmean(bwd_e, axis=0), "--", color="#9467bd", label="EMD backward")
        # identity check: normalized EMD of raisers == of lowerers (exact)
        dev = np.nanmax(np.abs(np.nanmean(fwd_e, axis=0) - np.nanmean(fwd_en, axis=0)))
        ax.text(0.03, 0.06, f"raiser/lowerer EMD identical (max dev {dev:.1e})",
                transform=ax.transAxes, fontsize=8)
        ax.set_xlabel(r"$|t-t_{\rm ref}|$ (accepted moves)")
        ax.set_ylabel("normalized measure")
        ax.set_title(f"p={p}: symmetry in walk direction, ref at $0.4\\,T$")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_symmetry.png"))
    plt.close(fig)


# ===========================================================================
# Fig 10: mixed model -- general kernel and additive floor
# ===========================================================================

def fig_mixed():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    z = np.load(os.path.join(RES, "meas_N400_P3_mixed.npz"))
    N = int(z["N"]); Tm = z["T"].min()
    u = ref0_avg(z, "u"); q = 1 - 2 * u / N
    r_unf = ref0_avg(z, "rho_unflip")
    ax.plot(q, r_unf, color=C_DATA, label="mixed 1+2+3-spin, N=400 (data)")
    qg = np.linspace(0, 1, 200); ug = (1 - qg) * N / 2
    ax.plot(qg, al.rho_unflip_theory(ug, N, [1, 2, 3]), "--", color=C_TH,
            label=r"$\xi'(q)/\xi'(1)=(1+2q+3q^2)/6$")
    zp = np.load(os.path.join(RES, "meas_N400_P3_pure.npz"))
    up = ref0_avg(zp, "u"); qp_ = 1 - 2 * up / 400
    ax.plot(qp_, ref0_avg(zp, "rho_unflip"), color="#17becf", alpha=0.8,
            label="pure 3-spin, N=400 (data)")
    ax.plot(qg, qg ** 2, ":", color="#17becf", label=r"$q^2$")
    ax.axhline(1 / 6, color="k", ls="--", alpha=0.6)
    ax.text(0.30, 1 / 6 - 0.045, "additive floor 1/6 (p=1 sector never scrambles)",
            fontsize=8)
    ax.set_xlabel(r"mutual overlap $q(0,t)$")
    ax.set_ylabel(r"$\rho_{\rm unflipped}(0,t)$")
    ax.set_title("general kernel law; additive part does not scramble")
    ax.legend()
    ax.set_xlim(0.25, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_mixed.png"))
    plt.close(fig)


# ===========================================================================
# Fig 11: scatter illustration
# ===========================================================================

def fig_scatter():
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
    for row, (tag, p, v) in enumerate(MAIN):
        z = np.load(os.path.join(RES, f"snap_{tag}.npz"))
        S, ts, N = z["S"], z["ts"], int(z["N"])
        par = z["parity"].astype(bool)
        M = S[0] > 0
        for k, ax in enumerate(axes[row]):
            idx = [0, 1, 2, 4][k] if len(ts) > 4 else k
            if idx >= len(ts):
                ax.axis("off"); continue
            x, y = S[0], S[idx]
            fl = par[idx] ^ par[0]
            ax.plot(x[~M & ~fl], y[~M & ~fl], ".", ms=1.5, color="#bbbbbb", alpha=0.6)
            ax.plot(x[M & ~fl], y[M & ~fl], ".", ms=2.0, color="#d62728", alpha=0.7,
                    label="raisers at $t_{ref}$ (unflipped)")
            ax.plot(x[fl], y[fl], ".", ms=2.0, color="#1f77b4", alpha=0.7,
                    label="flipped spins")
            rho = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"p={p}, t={ts[idx]}  ($\\rho$={rho:.2f})", fontsize=9)
            ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
            if k == 0:
                ax.set_ylabel(r"$\Delta_i(t)$")
            ax.set_xlabel(r"$\Delta_i(0)$")
            if row == 0 and k == 3:
                ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig11_scatter.png"))
    plt.close(fig)


# ===========================================================================
# Stats for report
# ===========================================================================

def collect_stats(tau_rows):
    import glob
    stats = {"main": {}, "sweep": [], "late": {}, "controls": {}}
    for tag, p, v in MAIN:
        z = load(tag)
        N = int(z["N"])
        entry = {"N": N, "p": p}
        vs = []
        for r in range(len(z["T"])):
            T = z["T"][r]; b = int(0.65 * T)
            vs.append(-np.polyfit(np.arange(b + 1), z["scal_d_f"][r, :b + 1], 1)[0])
        entry["v"] = float(np.mean(vs)); entry["v_err"] = float(np.std(vs) / np.sqrt(len(vs)))
        for key, name in [("rho_unflip", "unflip"), ("rho_pool", "pool"), ("emd_pos", "emd")]:
            s, e = boot_slope(z, key, 0.995, 0.90)
            entry[f"slope_{name}"] = float(s * N / 2)
            entry[f"slope_{name}_err"] = float(e * N / 2)
        u = ref0_avg(z, "u")
        th = al.rho_unflip_theory(u, N, [p])
        ts = np.arange(len(u))
        entry["slope_kernel_matched"] = float(
            al.fit_initial_slope(ts, th, 0.995, 0.90) * N / 2)
        th_pool = th * (1 - 4 * u / N)
        entry["slope_poolth_matched"] = float(
            al.fit_initial_slope(ts, th_pool, 0.995, 0.90) * N / 2)
        entry["d0_over_N"] = float(np.nanmean(z["scal_d_f"][:, 0]) / N)
        stats["main"][tag] = entry

    for f in sorted(glob.glob(os.path.join(RES, "meas_*.npz"))):
        tag = os.path.basename(f)[5:-4]
        z = np.load(f)
        N = int(z["N"]); orders = list(int(o) for o in z["orders"])
        su, eu = boot_slope(z, "rho_unflip", 0.995, 0.90)
        sp, ep = boot_slope(z, "rho_pool", 0.995, 0.90)
        se, ee = boot_slope(z, "emd_pos", 0.995, 0.90)
        vs = []
        for r in range(len(z["T"])):
            T = z["T"][r]; b = int(0.65 * T)
            vs.append(-np.polyfit(np.arange(b + 1), z["scal_d_f"][r, :b + 1], 1)[0])
        stats["sweep"].append(dict(
            tag=tag, N=N, orders=orders, v=float(np.mean(vs)),
            unflip=float(su * N / 2) if np.isfinite(su) else None,
            unflip_err=float(eu * N / 2) if np.isfinite(eu) else None,
            pool=float(sp * N / 2) if np.isfinite(sp) else None,
            emd=float(se * N / 2) if np.isfinite(se) else None,
            T_mean=float(z["T"].mean()),
            d0_mean=float(np.nanmean(z["scal_d_f"][:, 0]))))

    for p, rows in tau_rows.items():
        stats["late"][f"p{p}"] = [
            dict(d_ref=float(c["d_ref"]), n=int(c["n"]), v_rem=float(c["v_rem"]),
                 c=float(c["c"]), tau_tr=float(c["tau_tr"]),
                 tau_pred=float(c["tau_pred"]), tau_emd=float(c["tau_emd"]),
                 tau_predA=float(c["tau_predA"])) for c in rows]

    for tag, p, v in MAIN:
        N = int(load(tag)["N"])
        short = f"N{N}_P{p}"
        for mode in ("neutral", "uniform"):
            zc = np.load(os.path.join(RES, f"ctrl_{mode}_{short}.npz"))
            su, eu = boot_slope(zc, "rho_unflip", 0.995, 0.90)
            sp, ep = boot_slope(zc, "rho_pool", 0.995, 0.90)
            if mode == "uniform":
                vs = []
                for r in range(len(zc["T"])):
                    T = zc["T"][r]; b = int(0.65 * T)
                    vs.append(-np.polyfit(np.arange(b + 1), zc["scal_d_f"][r, :b + 1], 1)[0])
                vv = float(np.mean(vs))
            else:
                vv = 0.0
            stats["controls"][f"{mode}_{short}"] = dict(
                v=vv, unflip=float(su * N / 2), pool=float(sp * N / 2))
    with open(os.path.join(RES, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    return stats


if __name__ == "__main__":
    fig_phenomenology(); print("fig1 done", flush=True)
    fig_early(); print("fig2 done", flush=True)
    fig_collapse(); print("fig3 done", flush=True)
    fig_controls(); print("fig4 done", flush=True)
    fig_scaling(); print("fig5 done", flush=True)
    tau_rows = fig_late(); print("fig6+7 done", flush=True)
    fig_crossover(); print("fig8 done", flush=True)
    fig_symmetry(); print("fig9 done", flush=True)
    fig_mixed(); print("fig10 done", flush=True)
    fig_scatter(); print("fig11 done", flush=True)
    stats = collect_stats(tau_rows)
    print(json.dumps(stats["main"], indent=1))
