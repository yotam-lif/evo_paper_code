import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res=pickle.load(open(os.path.join(OUT,"beneficial_results.pkl"),"rb"))
names=sorted(res,key=lambda n:(res[n]["P"],res[n]["N"]))

def curve(nm):
    r=res[nm]; full=r["cnt_tN"]>=r["n_reps"]
    return r["tN"][full], r["b_vs_tN"][full], r["N"], r["P"]

def quad(tau,P):  # db/dtau=-p sqrt(2b)
    return 0.5*np.clip(1-P*tau,0,None)**2
def expo(tau,P):  # db/dtau=-1-2(p-1)b
    a=2*(P-1); return np.clip((0.5+1/a)*np.exp(-a*tau)-1/a,0,None)

fig,ax=plt.subplots(2,2,figsize=(14,10))

for col,(P,bigN) in enumerate([(2,2000),(3,300)]):
    # (top) b vs tau: data (all N faint, biggest bold) + both laws
    a=ax[0,col]
    Ns=[res[n]["N"] for n in names if res[n]["P"]==P]
    for nm in names:
        if res[nm]["P"]!=P: continue
        tau,b,N,_=curve(nm)
        if N==bigN:
            a.plot(tau,b,'k-',lw=2.6,label=f"data N={N}",zorder=5)
        else:
            a.plot(tau,b,'-',color='0.7',lw=1,alpha=.8,
                   label="data (smaller N)" if N==min(Ns) else None)
    tt=np.linspace(0,0.75,300)
    a.plot(tt,quad(tt,P),'r--',lw=2.2,label=r"quadratic  $\frac{1}{2}(1-p\tau)^2$")
    a.plot(tt,expo(tt,P),'b-.',lw=2.2,label=r"exponential  $-1-2(p{-}1)b$")
    a.axvline(1/P,color='r',ls=':',alpha=.5); a.axvline(np.log(P)/(2*(P-1)),color='b',ls=':',alpha=.5)
    a.set_xlabel(r"$\tau=t/N$"); a.set_ylabel("fraction beneficial  $b$")
    a.set_title(f"p={P}:  b vs $\\tau$"); a.set_xlim(0,0.72); a.set_ylim(0,0.52)
    a.legend(fontsize=8); a.grid(alpha=.3)

    # (bottom) linearization: which functional form is straight?
    tau,b,N,_=curve(nm if False else max([n for n in names if res[n]["P"]==P],key=lambda k:res[k]["N"]))
    m=b>0.02
    a2=ax[1,col]
    a2.plot(tau[m], np.sqrt(2*b[m]),'r.',ms=4,label=r"$\sqrt{2b}$  (straight $\Rightarrow$ quadratic)")
    a2.plot(tt[tt<1/P], 1-P*tt[tt<1/P],'r--',lw=1.5,alpha=.7)
    ap=2*(P-1)
    a2.plot(tau[m], np.log(b[m]+1/ap),'b.',ms=4,label=r"$\ln(b+\frac{1}{2(p{-}1)})$  (straight $\Rightarrow$ exp)")
    # exp reference line: ln(b+1/a) = ln(1/2+1/a) - a tau
    a2.plot(tt, np.log(0.5+1/ap)-ap*tt,'b-.',lw=1.5,alpha=.7)
    a2.set_xlabel(r"$\tau=t/N$"); a2.set_title(f"p={P}: linearization test")
    a2.legend(fontsize=8); a2.grid(alpha=.3); a2.set_xlim(0,0.55)

fig.suptitle("Data vs the two candidate laws (dotted verticals = each law's predicted convergence time)",fontsize=13)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(OUT,"compare_laws.png"),dpi=120)
print("saved compare_laws.png")

# numeric residual summary over the WHOLE measured range
print("\nRMSE over full measured range (b>0.01):")
for P in (2,3):
    nm=max([n for n in names if res[n]["P"]==P],key=lambda k:res[k]["N"])
    tau,b,N,_=curve(nm); m=b>0.01
    print(f"  p={P} N={N}: quadratic={np.sqrt(np.mean((b[m]-quad(tau[m],P))**2)):.4f}  "
          f"exponential={np.sqrt(np.mean((b[m]-expo(tau[m],P))**2)):.4f}")
