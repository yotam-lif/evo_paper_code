"""EMD-centric figure set for the rewritten report, plus stats_emd.json."""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_lib as al  # noqa: E402
from make_figures import load, ref0_avg, boot_slope, _late_curves, MAIN, RES, FIG  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.titlesize": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 8.5,
    "lines.linewidth": 1.7, "axes.grid": True, "grid.alpha": 0.25,
})

C_DATA = "#6a3fb5"      # EMD data (violet)
C_TH = "#d62728"        # kernel+flip law
C_KERNEL = "#1f77b4"    # zero-flip kernel
C_SURR = "#2ca02c"      # surrogate
C_N1, C_N2 = "#7f7f7f", "#bcbd22"
KAPPA = np.pi
STATS = {}


# ===========================================================================
# F1: what the EMD sees (scatter + EMD curve)
# ===========================================================================

def f1_observable():
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.35])
    for row, (tag, p, v) in enumerate(MAIN):
        z = np.load(os.path.join(RES, f"snap_{tag}.npz"))
        S, ts, N = z["S"], z["ts"], int(z["N"])
        M = S[0] > 0
        w1_ref = wasserstein_distance(S[0], S[0][M])
        snap_idx = [0, 1, 3] if p == 2 else [0, 1, 3]
        emd_snap = [wasserstein_distance(S[k], S[k][M]) / w1_ref for k in range(len(ts))]
        for col, k in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row, col])
            x, y = S[0], S[k]
            ax.plot(x[~M], y[~M], ".", ms=1.6, color="#b8b8b8", alpha=0.6)
            ax.plot(x[M], y[M], ".", ms=2.0, color=C_DATA, alpha=0.75)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"p={p},  t={ts[k]},  $\\widetilde W$={emd_snap[k]:.2f}", fontsize=9.5)
            if col == 0:
                ax.set_ylabel(r"$\Delta_i(t)$")
            ax.set_xlabel(r"$\Delta_i(0)$" if row == 1 else "")
        # right panel: EMD curve
        ax = fig.add_subplot(gs[row, 3])
        zm = load(tag)
        Tm = zm["T"].min()
        t = np.arange(Tm + 1)
        emd = ref0_avg(zm, "emd_pos")
        ax.plot(t, emd, color=C_DATA, label=r"$\widetilde W(t)$, average over runs")
        for k in snap_idx:
            if ts[k] <= Tm:
                ax.plot(ts[k], np.interp(ts[k], t, emd), "o", color="k", ms=5)
        tau = int(round((int(zm["N"])) / (2 * p - 2 + np.pi)))
        ax.axvline(tau, color=C_TH, ls=":", lw=1.4)
        ax.text(tau * 1.07, 0.47, fr"$\tau_{{\rm EMD}} = N/(2p{{-}}2{{+}}\pi)$"
                                  f"\n$\\approx {tau}$", color=C_TH, fontsize=9)
        ax.set_xlabel("t (accepted flips)")
        ax.set_ylabel("normalized EMD")
        ax.set_xlim(0, min(3.2 * tau, Tm))
        ax.set_ylim(0, 1.03)
        ax.set_title(f"p={p}, N={int(zm['N'])}")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_observable.png"))
    plt.close(fig)


# ===========================================================================
# F2: walk geometry (unchanged content, tighter labels)
# ===========================================================================

def f2_walk():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for row, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        Tm = z["T"].min()
        t = np.arange(Tm + 1)
        d_f = np.nanmean(z["scal_d_f"][:, :Tm + 1], axis=0)
        u0 = np.nanmean(z["scal_u0"][:, :Tm + 1], axis=0)
        ax = axes[row, 0]
        ax.plot(t, d_f / N, color="#1f77b4", label=r"$d_H(\sigma(t),\sigma_f)/N$")
        ax.plot(t, (d_f[0] - v * t) / N, "--", color="#1f77b4", alpha=0.8,
                label=fr"slope $-v$,  $v={v}$")
        ax.plot(t, u0 / N, color="#ff7f0e", label=r"$u(0,t)/N$")
        ax.plot(t, np.minimum(t, u0[-1] * 1.05) / N, "--", color="#ff7f0e", alpha=0.8,
                label="slope $+1$")
        ax.set_xlabel("t (accepted flips)")
        ax.set_title(f"p={p}, N={N}: distance to $\\sigma_f$ falls at $v$;\n"
                     "distance from start grows at speed 1")
        ax.legend(loc="center right", fontsize=8)
        ax.set_ylim(0, 0.72)
        ax = axes[row, 1]
        R2 = 4 * d_f * (N - d_f) / N
        ax.plot(t, R2 / N, color="#1f77b4", label=r"$R^2(t)/N$ (data)")
        dlin = np.maximum(d_f[0] - v * t, 0)
        ax.plot(t, 4 * dlin * (N - dlin) / N / N, "--", color=C_TH,
                label=r"parabola implied by linear $d_H(t)$")
        ax.set_xlabel("t")
        ax.set_title("shell radius: $R^2$ is a parabola in $t$")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax = axes[row, 2]
        w = np.nanmean(z["scal_spec_std"][:, :Tm + 1], axis=0)
        npos = np.nanmean(z["scal_n_pos"][:, :Tm + 1], axis=0)
        ax.plot(t / Tm, w / w[0], color="#1f77b4", label="spectrum width / initial")
        ax.plot(t / Tm, npos / npos[0], color="#9467bd", label="raisers $n_+(t)/n_+(0)$")
        ax.set_xlabel("t / T")
        ax.set_title("width stays O(1) (no global rescaling)")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f2_walk.png"))
    plt.close(fig)


# ===========================================================================
# F3: EMD money plot
# ===========================================================================

def f3_money():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        Tm = z["T"].min()
        t = np.arange(Tm + 1)
        rows = z["pair_emd_pos"][:, 0, :Tm + 1]
        emd = np.nanmean(rows, axis=0)
        sem = np.nanstd(rows, axis=0) / np.sqrt(rows.shape[0])
        u = ref0_avg(z, "u")
        kern = al.rho_unflip_theory(u, N, [p])
        law = kern * np.maximum(1 - KAPPA * u / N, 0)
        s = np.load(os.path.join(RES, f"surr_sswm_N{N}_P{p}.npz"))
        L = min(Tm, len(s["t"]) - 1)
        m = np.isfinite(s["emd_pos"][:L + 1])

        ax = axes[col]
        ax.fill_between(t, emd - sem, emd + sem, color=C_DATA, alpha=0.25, lw=0)
        ax.plot(t, emd, color=C_DATA, lw=2.2, label="(1) EMD data (mean of 10 runs)")
        ax.plot(t, law, "--", color=C_TH, lw=2,
                label=r"(2) early-time law  $q_{0t}^{\,p-1}\,(1-\pi u/N)$")
        ax.plot(t, kern, "--", color=C_KERNEL, lw=1.2, alpha=0.8,
                label=r"(3) kernel only  $q_{0t}^{\,p-1}$ (no move-reversal)")
        ax.plot(s["t"][:L + 1][m], s["emd_pos"][:L + 1][m], ".", ms=3, color=C_SURR,
                label="(4) landscape-free surrogate")
        qn = np.maximum(1 - 2 * v * t / N, 0)
        ax.plot(t, qn ** (p - 1) * np.maximum(1 - KAPPA * v * t / N, 0), ":",
                color=C_N1, lw=2, label=fr"(5) naive radial clock ($u\to v t$), $v={v}$")
        d0 = float(np.nanmean(z["scal_d_f"][:, 0]))
        ax.plot(t, np.exp(-v * t / d0), ":", color=C_N2, lw=2,
                label=r"(6) naive time-to-max  $e^{-v t/d_0}$")
        tau = N / (2 * p - 2 + np.pi)
        ax.axvline(tau, color=C_TH, ls=":", lw=1)
        ax.annotate(fr"$\tau_{{\rm EMD}} = N/(2p-2+\pi) = {tau:.0f}$",
                    xy=(tau, np.exp(-1)), xytext=(tau * 1.25, 0.55),
                    arrowprops=dict(arrowstyle="->", color=C_TH), color=C_TH, fontsize=9)
        ax.set_xlabel("t (accepted flips since the reference)")
        ax.set_ylabel(r"normalized EMD $\widetilde W(t)$")
        ax.set_xlim(0, min(int(3.4 * tau), Tm))
        ax.set_ylim(0, 1.02)
        ax.set_title(f"p={p}, N={N}: early-time EMD, reference $t_{{\\rm ref}}=0$")
        ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f3_money.png"))
    plt.close(fig)


# ===========================================================================
# F4: EMD collapse in mutual overlap / non-collapse in radial coordinate
# ===========================================================================

def f4_collapse():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        Ts = z["T"]
        cmap = cm.viridis
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for r in range(len(Ts)):
            T = Ts[r]
            d_f_r = z["scal_d_f"][r]
            for j in range(z["ref_t"].shape[1]):
                tr = z["ref_t"][r, j]
                if tr < 0 or tr / T > 0.6:
                    continue
                sl = slice(tr, T + 1, 3)
                q_mut = z["pair_q"][r, j, sl]
                emd = z["pair_emd_pos"][r, j, sl]
                q_f = 1 - 2 * d_f_r[:T + 1][sl] / N
                colr = cmap(tr / T / 0.65)
                ax_top.plot(q_mut, emd, ".", ms=1.5, color=colr, alpha=0.45)
                ax_bot.plot(q_f, emd, ".", ms=1.5, color=colr, alpha=0.45)
        qg = np.linspace(0.35, 1, 200)
        ug = (1 - qg) * N / 2
        ax_top.plot(qg, al.rho_unflip_theory(ug, N, [p]) * np.maximum(1 - KAPPA * ug / N, 0),
                    "-", color=C_TH, lw=2.2, label=r"$q^{p-1}(1-\pi u/N)$")
        ax_top.set_xlabel(r"mutual overlap $q(t_{\rm ref},t)$")
        ax_top.set_ylabel(r"$\widetilde W$")
        ax_top.set_title(f"p={p}: one curve in the mutual overlap")
        ax_top.legend(loc="upper left")
        ax_top.set_xlim(0.35, 1)
        ax_top.set_ylim(-0.05, 1.02)
        ax_bot.set_xlabel(r"overlap with the terminal maximum, $q_f(t)$")
        ax_bot.set_ylabel(r"$\widetilde W$")
        ax_bot.set_title(f"p={p}: no curve in the radial coordinate")
        ax_bot.set_xlim(0, 1)
        ax_bot.set_ylim(-0.05, 1.02)
        sm = cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 0.65))
        for a in (ax_top, ax_bot):
            cb = fig.colorbar(sm, ax=a, pad=0.01)
            cb.set_label(r"$t_{\rm ref}/T$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f4_collapse.png"))
    plt.close(fig)


# ===========================================================================
# F5: what sets the constant -- acceptance rule, not drift
# ===========================================================================

def f5_controls():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    out = {}
    for col, (tag, p, v_sswm) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        short = f"N{N}_P{p}"
        Tm = z["T"].min()

        def kap(zz, n_boot=300):
            Tmm = zz["T"].min()
            rowsU = zz["pair_u"][:, 0, :Tmm + 1]
            rowsE = zz["pair_emd_pos"][:, 0, :Tmm + 1]
            base = al.fit_kappa(np.nanmean(rowsU, 0), np.nanmean(rowsE, 0), N, p)
            rng = np.random.default_rng(1)
            bs = []
            R = rowsU.shape[0]
            for _ in range(n_boot):
                idx = rng.integers(0, R, R)
                kk = al.fit_kappa(np.nanmean(rowsU[idx], 0), np.nanmean(rowsE[idx], 0), N, p)
                if np.isfinite(kk):
                    bs.append(kk)
            return base, np.std(bs)

        entries = [("SSWM\n" + fr"$v={v_sswm:.2f}$", *kap(z))]
        for mode, label in [("uniform", "uniform greedy"), ("neutral", "neutral walk")]:
            zc = np.load(os.path.join(RES, f"ctrl_{mode}_{short}.npz"))
            if mode == "uniform":
                vs = []
                for r in range(len(zc["T"])):
                    T = zc["T"][r]
                    b = int(0.65 * T)
                    vs.append(-np.polyfit(np.arange(b + 1), zc["scal_d_f"][r, :b + 1], 1)[0])
                vv = np.mean(vs)
            else:
                vv = 0.0
            entries.append((label + f"\n$v={vv:.2f}$", *kap(zc)))
        # surrogate kappas (from precision runs)
        with open(os.path.join(RES, "kappa.json")) as fh:
            KJ = json.load(fh)
        surr_sswm = KJ[f"{short}_sswm"]["k"]
        surr_unif = KJ[f"{short}_uniform"]["k"]

        ax = axes[col]
        x = np.arange(3)
        ax.errorbar(x, [e[1] for e in entries], yerr=[e[2] for e in entries],
                    fmt="o", ms=8, color=C_DATA, capsize=3, label=r"measured $\kappa$")
        ax.plot([0], [surr_sswm], "s", ms=8, mfc="none", mec=C_SURR, mew=2,
                label="surrogate (SSWM rule)")
        ax.plot([1], [surr_unif], "s", ms=8, mfc="none", mec="#8c564b", mew=2,
                label="surrogate (uniform rule)")
        ax.axhline(np.pi, color=C_TH, ls="--")
        ax.text(2.35, np.pi + 0.04, r"$\kappa=\pi$ (SSWM)", color=C_TH, fontsize=9)
        ax.axhline(2, color="#1f77b4", ls="--")
        ax.text(2.35, 2.04, r"$\kappa=2$ ($\beta{=}1$ rules)", color="#1f77b4", fontsize=9)
        ax.set_xticks(x, [e[0] for e in entries])
        ax.set_ylabel(r"flip constant $\kappa$ in $\widetilde W=q^{p-1}(1-\kappa u/N)$")
        ax.set_title(f"p={p}, N={N}: $\\kappa$ follows the acceptance rule, not $v$")
        ax.set_ylim(0.8, 4.4)
        ax.legend(loc="lower left", fontsize=8)
        out[short] = dict(entries=[(e[0].split("\n")[0], e[1], e[2]) for e in entries],
                          surr_sswm=surr_sswm, surr_unif=surr_unif)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f5_controls.png"))
    plt.close(fig)
    return out


# ===========================================================================
# F6: kappa across system sizes
# ===========================================================================

def f6_scaling():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    table = []
    for p, colr, mk in [(2, "#1f77b4", "o"), (3, "#d62728", "s")]:
        Ns, ks, es = [], [], []
        for f in sorted(glob.glob(os.path.join(RES, f"meas_N*_P{p}_pure.npz"))):
            z = np.load(f)
            N = int(z["N"])
            Tm = z["T"].min()
            rowsU = z["pair_u"][:, 0, :Tm + 1]
            rowsE = z["pair_emd_pos"][:, 0, :Tm + 1]
            k = al.fit_kappa(np.nanmean(rowsU, 0), np.nanmean(rowsE, 0), N, p)
            if not np.isfinite(k):
                continue
            rng = np.random.default_rng(2)
            bs = []
            for _ in range(300):
                idx = rng.integers(0, rowsU.shape[0], rowsU.shape[0])
                kk = al.fit_kappa(np.nanmean(rowsU[idx], 0), np.nanmean(rowsE[idx], 0), N, p)
                if np.isfinite(kk):
                    bs.append(kk)
            Ns.append(N); ks.append(k); es.append(np.std(bs))
            table.append(dict(p=p, N=N, kappa=k, err=float(np.std(bs))))
        ax.errorbar(np.array(Ns) * (1.0 if p == 2 else 1.03), ks, yerr=es, fmt=mk,
                    color=colr, ms=7, capsize=3, label=f"p={p} data")
    ax.axhline(np.pi, color="k", ls="--")
    ax.text(120, np.pi + 0.06, r"$\kappa=\pi$", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"fitted $\kappa$")
    ax.set_title(r"fitted flip constant vs system size: consistent with $\kappa=\pi$")
    ax.set_ylim(1.5, 4.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f6_scaling.png"))
    plt.close(fig)
    return table


# ===========================================================================
# F7: late-time EMD
# ===========================================================================

def f7_late():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    taus = {2: [], 3: []}
    floors = {2: [], 3: []}
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        targets = [(N // 32 - 3, N // 32 + 4), (N // 16 - 4, N // 16 + 6),
                   (N // 8 - 8, N // 8 + 8)]
        ax_c, ax_t = axes[0, col], axes[1, col]
        colors = ["#1f77b4", "#2ca02c", "#9467bd"]
        for (d_lo, d_hi), colr in zip(targets, colors):
            cur = _late_curves(z, N, d_lo, d_hi)
            if cur is None:
                continue
            d = cur["d_ref"]
            tau_pr = 2 * d / (1 + cur["v_rem"])
            x = cur["dt"] / tau_pr
            ok = np.isfinite(cur["emd"])
            ax_c.plot(x[ok], cur["emd"][ok], color=colr,
                      label=fr"$d_{{\rm ref}}\approx{d:.0f}$, $v_{{\rm rem}}={cur['v_rem']:.2f}$")
            ax_c.plot(x, cur["predA"], "--", color=colr, alpha=0.85)
            # floor: mean of last valid quarter
            e = cur["emd"][ok]
            if len(e) > 8:
                fl = float(np.nanmean(e[-max(3, len(e) // 6):]))
                floors[p].append((d, fl))
                ax_c.plot(x[ok][-1], fl, "<", color=colr, ms=7)
            s_emd = al.fit_initial_slope(cur["dt"], cur["emd"], 0.97, 0.55)
            s_prA = al.fit_initial_slope(cur["dt"], cur["predA"], 0.97, 0.55)
            taus[p].append(dict(d=d, v_rem=cur["v_rem"], c=cur["c"],
                                tau_emd=1 / s_emd if s_emd and np.isfinite(s_emd) else np.nan,
                                tau_ampl_angle=1 / s_prA if s_prA and np.isfinite(s_prA) else np.nan))
        xg = np.linspace(0, 2.2, 100)
        ax_c.plot(xg, np.exp(-xg), ":", color="k", alpha=0.75, label=r"$e^{-x}$")
        ax_c.set_xlabel(r"$\Delta t \,/\, [2d_{\rm ref}/(1+v_{\rm rem})]$")
        ax_c.set_ylabel(r"normalized EMD")
        ax_c.set_title(f"p={p}: late-time EMD, lag rescaled by $R^2/2(1+v_{{\\rm rem}})$\n"
                       "(dashed: amplitude$\\times$angle prediction; $\\blacktriangleleft$: floor)")
        ax_c.legend(fontsize=8)
        ax_c.set_xlim(0, 2.2)
        ax_c.set_ylim(0, 1.02)

        ds = np.array([r["d"] for r in taus[p]])
        ax_t.plot(ds, [r["tau_emd"] for r in taus[p]], "o", ms=9, color=C_DATA,
                  label=r"$\tau_{\rm EMD}$ (data)")
        ax_t.plot(ds, [r["tau_ampl_angle"] for r in taus[p]], "x", ms=9, mew=2.2,
                  color="#8c564b", label="amplitude$\\times$angle")
        dg = np.linspace(ds.min() * 0.6, ds.max() * 1.5, 60)
        vr = np.mean([r["v_rem"] for r in taus[p]])
        ax_t.plot(dg, 2 * dg / (1 + vr), "--", color="k",
                  label=fr"$2d/(1+v_{{\rm rem}})$, $\bar v_{{\rm rem}}={vr:.2f}$")
        ax_t.plot(dg, 2 * dg, ":", color="k", alpha=0.6, label=r"$2d = R^2/2$")
        ax_t.set_xscale("log"); ax_t.set_yscale("log")
        ax_t.xaxis.set_major_formatter(ScalarFormatter())
        ax_t.yaxis.set_major_formatter(ScalarFormatter())
        ax_t.set_xlabel(r"$d_{\rm ref} = d_H(t_{\rm ref},\sigma_f)$")
        ax_t.set_ylabel(r"$\tau_{\rm EMD}$ (accepted moves)")
        ax_t.set_title(f"p={p}: late-time EMD timescale")
        ax_t.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f7_late.png"))
    plt.close(fig)
    return taus, floors


# ===========================================================================
# F8: the whole walk -- EMD timescale vs position (crossover)
# ===========================================================================

def f8_crossover():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    curves = {}
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        Ts = z["T"]
        bins = {}
        for r in range(len(Ts)):
            T = Ts[r]
            for j in range(z["ref_t"].shape[1]):
                tr, dfr = z["ref_t"][r, j], z["ref_d_f"][r, j]
                if tr < 0 or tr >= T - 4:
                    continue
                key = int(np.floor(np.log2(max(dfr, 2))))
                bins.setdefault(key, {"emd": [], "d": [], "vr": []})
                bins[key]["emd"].append(z["pair_emd_pos"][r, j, tr:T + 1])
                bins[key]["d"].append(dfr)
                bins[key]["vr"].append(dfr / (T - tr))
        ds, taus, vrs = [], [], []
        for key, b in sorted(bins.items()):
            Lm = min(len(a) for a in b["emd"])
            if Lm < 6:
                continue
            emd = np.nanmean([a[:Lm] for a in b["emd"]], axis=0)
            se = al.fit_initial_slope(np.arange(Lm), emd, 0.97, 0.55)
            if not (se and np.isfinite(se)):
                continue
            ds.append(np.mean(b["d"])); taus.append(1 / se); vrs.append(np.mean(b["vr"]))
        ds = np.array(ds); taus = np.array(taus); vrs = np.array(vrs)
        ax = axes[col]
        ax.plot(ds / N, taus, "o-", ms=8, color=C_DATA, label=r"$\tau_{\rm EMD}$ (data)")
        kappa_tot = 2 * (p - 1) + np.pi
        dg = np.geomspace(max(ds.min() * 0.7, 2), N * 0.45, 150)
        vr = 0.87
        ax.axhline(N / kappa_tot, color=C_TH, ls="--",
                   label=fr"kernel regime: $\tau=N/(2p-2+\pi)={N/kappa_tot:.0f}$")
        ax.plot(dg / N, 2 * dg / (1 + vr), ":", color="k",
                label=r"basin regime: $\tau=2d/(1+v_{\rm rem})$")
        comb = 1.0 / (kappa_tot / N + (1 + vr) / (2 * dg))
        ax.plot(dg / N, comb, "-", color=C_TH, alpha=0.35, lw=3.2,
                label="rates added (both mechanisms)")
        dstar = N * (1 + vr) / (2 * kappa_tot)
        ax.axvline(dstar / N, color="#666", lw=1, ls="-.")
        ax.text(dstar / N * 0.97, N / kappa_tot * 1.5, fr"$d^*/N={dstar/N:.2f}$",
                fontsize=9, color="#444", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"$d_H(t_{\rm ref},\sigma_f)/N$  (how far the reference is from $\sigma_f$)")
        ax.set_ylabel(r"$\tau_{\rm EMD}(t_{\rm ref})$")
        ax.set_title(f"p={p}, N={N}: the EMD timescale along the walk")
        ax.legend(fontsize=8, loc="lower right")
        curves[tag] = dict(d=list(map(float, ds)), tau=list(map(float, taus)),
                           v_rem=list(map(float, vrs)), dstar=float(dstar))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f8_crossover.png"))
    plt.close(fig)
    return curves


# ===========================================================================
# F9: direction symmetry (EMD only)
# ===========================================================================

def f9_symmetry():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        Ts = z["T"]
        R = len(Ts)
        j = int(np.argmin(np.abs(z["ref_t"][0] / max(Ts[0], 1) - 0.4)))
        smax = 10 ** 9
        for r in range(R):
            T = Ts[r]; tr = z["ref_t"][r, j]
            smax = min(smax, min(tr, T - tr))
        fwd, bwd, fwd_n = [], [], []
        for r in range(R):
            T = Ts[r]; tr = z["ref_t"][r, j]
            row = z["pair_emd_pos"][r, j, :T + 1]
            row_n = z["pair_emd_neg"][r, j, :T + 1]
            fwd.append(row[tr:tr + smax]); bwd.append(row[tr::-1][:smax])
            fwd_n.append(row_n[tr:tr + smax])
        s = np.arange(smax)
        ax = axes[col]
        ax.plot(s, np.nanmean(fwd, axis=0), color=C_DATA, label="forward (toward $\\sigma_f$)")
        ax.plot(s, np.nanmean(bwd, axis=0), "--", color=C_DATA, label="backward (toward start)")
        dev = np.nanmax(np.abs(np.nanmean(fwd, axis=0) - np.nanmean(fwd_n, axis=0)))
        ax.text(0.03, 0.06, f"raiser vs lowerer subset: identical (max dev {dev:.1e})",
                transform=ax.transAxes, fontsize=8)
        ax.set_xlabel(r"$|t-t_{\rm ref}|$ (accepted moves)")
        ax.set_ylabel("normalized EMD")
        ax.set_title(f"p={p}: reference at $0.4\\,T$, read both ways")
        ax.legend(fontsize=8.5)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f9_symmetry.png"))
    plt.close(fig)


# ===========================================================================
# Appendix figures: correlation ingredients / rho collapse / geometry / mixed
# ===========================================================================

def a1_ingredients():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Tm = z["T"].min()
        t = np.arange(Tm + 1)
        u = ref0_avg(z, "u")
        th = al.rho_unflip_theory(u, N, [p])
        ax = axes[col]
        ax.plot(t, ref0_avg(z, "rho_unflip"), color="#1f77b4",
                label=r"$\rho_{\rm unflipped}$ data")
        ax.plot(t, th, "--", color=C_TH, label=r"$K_{p-1}(u)/\binom{N-1}{p-1}\to q^{p-1}$")
        ax.plot(t, ref0_avg(z, "rho_flip"), color="#ff7f0e", label=r"$\rho_{\rm flipped}$ data")
        ax.plot(t, -th, "--", color="#ff7f0e", alpha=0.6, label=r"$-q^{p-1}$")
        ax.plot(t, ref0_avg(z, "rho_pool"), color="#17becf", label=r"$\rho_{\rm pool}$ data")
        ax.plot(t, th * np.maximum(1 - 4 * u / N, 0), "--", color="#e377c2",
                label=r"$q^{p-1}(1-4u/N)$  [$2\beta=4$]")
        ax.set_xlabel("t")
        ax.set_ylabel("Pearson correlation")
        ax.set_xlim(0, min(int(0.35 * N), Tm))
        ax.set_ylim(-1.02, 1.02)
        ax.set_title(f"p={p}, N={N}: the two correlation branches and the pool")
        ax.legend(fontsize=8, loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a1_ingredients.png"))
    plt.close(fig)


def a2_rho_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"]); Ts = z["T"]
        cmap = cm.viridis
        ax = axes[col]
        for r in range(len(Ts)):
            T = Ts[r]
            for j in range(z["ref_t"].shape[1]):
                tr = z["ref_t"][r, j]
                if tr < 0 or tr / T > 0.75:
                    continue
                sl = slice(tr, T + 1, 3)
                ax.plot(z["pair_q"][r, j, sl], z["pair_rho_unflip"][r, j, sl],
                        ".", ms=1.4, color=cmap(tr / T / 0.8), alpha=0.45)
        qg = np.linspace(0, 1, 200)
        ax.plot(qg, al.rho_unflip_theory((1 - qg) * N / 2, N, [p]), "-", color=C_TH,
                lw=2, label=r"$\xi'(q)/\xi'(1)$")
        ax.set_xlabel(r"mutual overlap $q(t_{\rm ref},t)$")
        ax.set_ylabel(r"$\rho_{\rm unflipped}$")
        ax.set_title(f"p={p}: kernel collapse of the raw correlation")
        ax.legend()
        ax.set_ylim(-0.1, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a2_rho_collapse.png"))
    plt.close(fig)


def a3_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for col, (tag, p, v) in enumerate(MAIN):
        z = load(tag)
        N = int(z["N"])
        targets = [(N // 32 - 3, N // 32 + 4), (N // 16 - 4, N // 16 + 6),
                   (N // 8 - 8, N // 8 + 8)]
        ax = axes[col]
        colors = ["#1f77b4", "#2ca02c", "#9467bd"]
        for (d_lo, d_hi), colr in zip(targets, colors):
            cur = _late_curves(z, N, d_lo, d_hi)
            if cur is None:
                continue
            x = cur["dt"] / (2 * cur["d_ref"])
            ax.plot(x, cur["tr"], color=colr, label=fr"$d_{{\rm ref}}\approx{cur['d_ref']:.0f}$")
            ax.plot(x, cur["pred"], "--", color=colr, alpha=0.9)
        ax.set_xlabel(r"$\Delta t/2d_{\rm ref}$")
        ax.set_ylabel(r"$\rho$ of the transient part")
        ax.set_title(f"p={p}: transient correlation (solid) vs\n"
                     r"disagreement-set formula $\frac{d_1+d_2-u_{12}}{2\sqrt{d_1d_2}}$ (dashed)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a3_geometry.png"))
    plt.close(fig)


def a4_mixed():
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    z = np.load(os.path.join(RES, "meas_N400_P3_mixed.npz"))
    N = int(z["N"])
    u = ref0_avg(z, "u"); q = 1 - 2 * u / N
    ax.plot(q, ref0_avg(z, "rho_unflip"), color="#1f77b4", label="mixed 1+2+3-spin (data)")
    qg = np.linspace(0, 1, 200)
    ax.plot(qg, al.rho_unflip_theory((1 - qg) * N / 2, N, [1, 2, 3]), "--", color=C_TH,
            label=r"$\xi'(q)/\xi'(1)=(1+2q+3q^2)/6$")
    zp = np.load(os.path.join(RES, "meas_N400_P3_pure.npz"))
    up = ref0_avg(zp, "u")
    ax.plot(1 - 2 * up / 400, ref0_avg(zp, "rho_unflip"), color="#17becf", alpha=0.85,
            label="pure 3-spin (data)")
    ax.plot(qg, qg ** 2, ":", color="#17becf", label=r"$q^2$")
    ax.axhline(1 / 6, color="k", ls="--", alpha=0.6)
    ax.text(0.42, 1 / 6 - 0.05, "additive floor 1/6: the $p{=}1$ part never scrambles",
            fontsize=8.5)
    ax.set_xlabel(r"mutual overlap $q(0,t)$")
    ax.set_ylabel(r"$\rho_{\rm unflipped}(0,t)$")
    ax.set_title("mixed kernel: same law, nonzero floor (N=400)")
    ax.legend(fontsize=8.5)
    ax.set_xlim(0.25, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a4_mixed.png"))
    plt.close(fig)


if __name__ == "__main__":
    f1_observable(); print("F1", flush=True)
    f2_walk(); print("F2", flush=True)
    f3_money(); print("F3", flush=True)
    f4_collapse(); print("F4", flush=True)
    STATS["controls"] = f5_controls(); print("F5", flush=True)
    STATS["kappa_vs_N"] = f6_scaling(); print("F6", flush=True)
    taus, floors = f7_late(); print("F7", flush=True)
    STATS["late"] = {f"p{p}": taus[p] for p in taus}
    STATS["floors"] = {f"p{p}": floors[p] for p in floors}
    STATS["crossover"] = f8_crossover(); print("F8", flush=True)
    f9_symmetry(); print("F9", flush=True)
    a1_ingredients(); a2_rho_collapse(); a3_geometry(); a4_mixed()
    print("appendix figs", flush=True)
    with open(os.path.join(RES, "stats_emd.json"), "w") as fh:
        json.dump(STATS, fh, indent=1, default=float)
    print(json.dumps(STATS["late"], indent=1, default=float))
    print("floors:", json.dumps(STATS["floors"], default=float))
