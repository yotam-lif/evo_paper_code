import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res=pickle.load(open(os.path.join(OUT,"beneficial_results.pkl"),"rb"))
names=sorted(res,key=lambda n:(res[n]["P"],res[n]["N"]))
def curve(nm):
    r=res[nm]; full=r["cnt_tN"]>=r["n_reps"]; return r["tN"][full], r["b_vs_tN"][full]

fig,ax=plt.subplots(1,2,figsize=(14,5.2))
for j,(P,bigN) in enumerate([(2,2000),(3,300)]):
    a=ax[j]
    nm=max([n for n in names if res[n]["P"]==P],key=lambda k:res[k]["N"])
    tau,b=curve(nm)
    a.plot(tau,b,'k-',lw=2.4,label=f"data N={bigN}",zorder=5)
    tt=np.linspace(0,0.75,400)
    # quadratic
    a.plot(tt,0.5*np.clip(1-P*tt,0,None)**2,'r--',lw=2,label=r"quadratic $\frac{1}{2}(1-p\tau)^2$")
    # exponential UNCLIPPED (let it go negative to its floor)
    ap=2*(P-1); C=0.5+1/ap; D=1/ap
    be=C*np.exp(-ap*tt)-D
    a.plot(tt,be,'b-.',lw=2,label=r"exponential $Ce^{-2(p-1)\tau}-D$ (unclipped)")
    # markers
    tstar=np.log(P)/ap
    a.axhline(0,color='0.5',lw=.8)
    a.axhline(-D,color='b',ls=':',lw=1,alpha=.7,label=f"exp floor $-1/2(p-1)={-D:.2f}$")
    a.plot([tstar],[0],'bo',ms=7,label=f"exp zero-crossing $\\tau^*={tstar:.2f}$")
    a.set_xlabel(r"$\tau=t/N$"); a.set_ylabel("b"); a.set_title(f"p={P}")
    a.set_xlim(0,0.72); a.set_ylim(-D-0.05,0.52); a.legend(fontsize=8); a.grid(alpha=.3)
fig.suptitle("The 'exponential' is a real exponential heading to a NEGATIVE floor — it cuts through 0 at an angle (then gets clipped)",fontsize=11)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT,"exp_unclipped.png"),dpi=120)
print("saved")
