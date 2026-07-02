import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res=pickle.load(open(os.path.join(OUT,"beneficial_results.pkl"),"rb"))
names=sorted(res,key=lambda n:(res[n]["P"],res[n]["N"]))

fig,axes=plt.subplots(1,2,figsize=(13,5.2))
for P,ax in [(2,axes[0]),(3,axes[1])]:
    cs=plt.cm.viridis(np.linspace(0,.85,len([n for n in names if res[n]["P"]==P])))
    i=0
    for nm in names:
        r=res[nm]
        if r["P"]!=P: continue
        full=r["cnt_tN"]>=r["n_reps"]
        ax.plot(r["tN"][full],r["b_vs_tN"][full],color=cs[i],lw=1.3,label=f"N={r['N']}")
        i+=1
    tau=np.linspace(0,1/P,100)
    ax.plot(tau,0.5*(1-P*tau)**2,'r--',lw=2.5,label=r"$\frac{1}{2}(1-p\tau)^2$")
    ax.plot([0,0.12],[0.5,0.5-P*0.12],'k:',lw=1.5,label=f"initial slope $-p={-P}$")
    ax.set_xlabel(r"$\tau = t/N$  (flips per spin)",fontsize=11)
    ax.set_ylabel("fraction of beneficial spins  $b$",fontsize=11)
    ax.set_title(f"pure p-spin, p={P}",fontsize=12)
    ax.set_xlim(0,0.7 if P==2 else 0.5); ax.set_ylim(0,0.52)
    ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.3)
fig.suptitle(r"Fraction of beneficial spins vs rescaled time:  $b(\tau)\approx\frac{1}{2}(1-p\tau)_+^2$,   $b(0)=\frac{1}{2}$,   $db/d\tau|_0=-p$",fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT,"FINAL_beneficial.png"),dpi=120)
print("saved FINAL_beneficial.png")
