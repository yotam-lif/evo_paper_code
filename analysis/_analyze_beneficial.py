import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res = pickle.load(open(os.path.join(OUT, "beneficial_results.pkl"), "rb"))

def key(name):
    r = res[name]; return (r["P"], r["N"])
names = sorted(res.keys(), key=key)

# ---- table: L/N, initial slope, and b at a few tau ----
print(f"{'N':>6} {'P':>2} {'L/N':>7} {'slope0':>8} {'b@.1':>7} {'b@.2':>7} {'b@.3':>7}")
for nm in names:
    r = res[nm]; N=r["N"]; tN=r["tN"]; b=r["b_vs_tN"]
    # initial slope over first ~0.05 in tau where cnt full
    full = r["cnt_tN"] >= r["n_reps"]      # all reps still alive
    tau = tN[full]; bb = b[full]
    m = tau <= 0.08
    slope0 = np.polyfit(tau[m], bb[m], 1)[0] if m.sum()>3 else np.nan
    def bat(x):
        return np.interp(x, tau, bb) if tau[-1]>=x else np.nan
    print(f"{N:>6} {r['P']:>2} {r['Ls'].mean()/N:>7.4f} {slope0:>8.3f} "
          f"{bat(.1):>7.4f} {bat(.2):>7.4f} {bat(.3):>7.4f}")

# ---- plots ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) b vs t/N, p=2, all N  (collapse test in tau=t/N)
ax = axes[0,0]
for nm in names:
    r = res[nm]
    if r["P"] != 2: continue
    full = r["cnt_tN"] >= r["n_reps"]
    ax.plot(r["tN"][full], r["b_vs_tN"][full], label=f"N={r['N']}")
tt = np.linspace(0,0.5,50)
ax.plot(tt, 0.5-tt, 'k--', lw=1, label='1/2 - tau')
ax.set_xlabel("tau = t/N"); ax.set_ylabel("fraction beneficial b"); ax.set_title("p=2: b vs t/N (all reps alive)")
ax.legend(fontsize=7); ax.grid(alpha=.3)

# (b) b vs fraction-of-walk, p=2 (normalized time) -> should collapse trivially
ax = axes[0,1]
TAU = np.linspace(0,1,201)
for nm in names:
    r = res[nm]
    if r["P"] != 2: continue
    ax.plot(TAU, r["b_vs_frac_mean"], label=f"N={r['N']}")
ax.set_xlabel("t / L (fraction of walk)"); ax.set_ylabel("b"); ax.set_title("p=2: b vs fraction of walk")
ax.legend(fontsize=7); ax.grid(alpha=.3)

# (c) p=3
ax = axes[1,0]
for nm in names:
    r = res[nm]
    if r["P"] != 3: continue
    full = r["cnt_tN"] >= r["n_reps"]
    ax.plot(r["tN"][full], r["b_vs_tN"][full], label=f"N={r['N']}")
ax.set_xlabel("tau = t/N"); ax.set_ylabel("b"); ax.set_title("p=3: b vs t/N")
ax.legend(fontsize=7); ax.grid(alpha=.3)

# (d) largest N each p, with linear guide
ax = axes[1,1]
for P,col in [(2,'C0'),(3,'C1')]:
    cand = [nm for nm in names if res[nm]["P"]==P]
    nm = max(cand, key=lambda n: res[n]["N"])
    r = res[nm]; full = r["cnt_tN"]>=r["n_reps"]
    ax.plot(r["tN"][full], r["b_vs_tN"][full], col, label=f"p={P}, N={r['N']}")
ax.plot(tt, 0.5-tt, 'k--', lw=1, label='1/2 - tau')
ax.set_xlabel("tau = t/N"); ax.set_ylabel("b"); ax.set_title("largest N per p")
ax.legend(fontsize=8); ax.grid(alpha=.3)

# overlay MF theory on panels (c-like) : new figure
mf = pickle.load(open(os.path.join(OUT,"mf_curves.pkl"),"rb"))
for P in (2,3):
    ax = axes[1,1] if False else None
for P,ax in [(2,axes[0,0]),(3,axes[1,0])]:
    if P in mf:
        tau,b = mf[P]
        ax.plot(tau,b,'r-',lw=2,alpha=.7,label='MF theory')
        # tangent -p at origin
        tt=np.linspace(0,0.15,10); ax.plot(tt,0.5-P*tt,'g:',lw=1.5,label=f'slope -{P}')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 0.8 if P==2 else 0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "beneficial_collapse.png"), dpi=110)
print("saved plot")

# quantitative MF vs sim at several tau (largest N)
print("\nMF vs sim b(tau), largest N:")
for P in (2,3):
    cand=[nm for nm in names if res[nm]["P"]==P]
    nm=max(cand,key=lambda n:res[n]["N"]); r=res[nm]
    full=r["cnt_tN"]>=r["n_reps"]; tau_s=r["tN"][full]; b_s=r["b_vs_tN"][full]
    tmf,bmf=mf[P]
    print(f" p={P} N={r['N']}: "+" ".join(
        f"t={x}:sim{np.interp(x,tau_s,b_s):.3f}/mf{np.interp(x,tmf,bmf):.3f}"
        for x in (0.05,0.1,0.15,0.2,0.3,0.4)))

# ---- L/N vs N trend ----
print("\nL/N vs N:")
for P in (2,3):
    row=[(res[nm]["N"], res[nm]["Ls"].mean()/res[nm]["N"]) for nm in names if res[nm]["P"]==P]
    print(f" p={P}:", " ".join(f"{N}:{v:.3f}" for N,v in row))
