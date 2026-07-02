"""Direct test of the closed form:  db/dtau = -p * sqrt(2 b).
If true, plotting -db/dtau against sqrt(2b) gives a straight line of slope p.
Use the rep-averaged b(t/N) for the largest N, lightly smoothed."""
import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res=pickle.load(open(os.path.join(OUT,"beneficial_results.pkl"),"rb"))
names=sorted(res,key=lambda n:(res[n]["P"],res[n]["N"]))

def smooth(y,k):
    ker=np.ones(k)/k
    return np.convolve(y,ker,mode='same')

fig,axes=plt.subplots(1,2,figsize=(13,5))
for P,ax in [(2,axes[0]),(3,axes[1])]:
    cand=[nm for nm in names if res[nm]["P"]==P]
    nm=max(cand,key=lambda n:res[n]["N"]); r=res[nm]
    full=r["cnt_tN"]>=r["n_reps"]
    tau=r["tN"][full]; b=r["b_vs_tN"][full]
    # smooth b then differentiate
    bs=smooth(b, 21)
    dbt=np.gradient(bs,tau)
    m=(b>0.03)&(np.arange(len(b))>10)&(np.arange(len(b))<len(b)-10)
    x=np.sqrt(2*bs[m]); y=-dbt[m]
    # fit slope through origin
    slope=np.sum(x*y)/np.sum(x*x)
    ax.plot(x,y,'.',ms=3,label='data -db/dtau')
    xx=np.linspace(0,x.max(),20); ax.plot(xx,P*xx,'r-',label=f'slope p={P}')
    ax.plot(xx,slope*xx,'g--',label=f'best slope={slope:.2f}')
    ax.set_xlabel('sqrt(2 b)'); ax.set_ylabel('-db/dtau'); ax.set_title(f'p={P}, N={r["N"]}')
    ax.legend(); ax.grid(alpha=.3)
    print(f"p={P} N={r['N']}: best-fit slope (should be p) = {slope:.3f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"ode_test.png"),dpi=110)
print("saved ode_test.png")
