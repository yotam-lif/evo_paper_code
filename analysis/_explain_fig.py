import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res=pickle.load(open(os.path.join(OUT,"beneficial_results.pkl"),"rb"))
shp=pickle.load(open(os.path.join(OUT,"spectrum_shape.pkl"),"rb"))
names=sorted(res,key=lambda n:(res[n]["P"],res[n]["N"]))

fig,ax=plt.subplots(1,3,figsize=(16,4.8))

# A: sqrt(2b) vs tau is a straight line of slope -p  -> this is the meaning of the ODE
for P,col in [(2,'C0'),(3,'C1')]:
    nm=max([n for n in names if res[n]["P"]==P],key=lambda n:res[n]["N"]); r=res[nm]
    full=r["cnt_tN"]>=r["n_reps"]; tau=r["tN"][full]; b=r["b_vs_tN"][full]
    ax[0].plot(tau,np.sqrt(2*b),col,lw=1.6,label=f"data p={P} (N={r['N']})")
    tt=np.linspace(0,1/P,10); ax[0].plot(tt,1-P*tt,col+'--',lw=1.2)
ax[0].set_xlabel(r"$\tau=t/N$"); ax[0].set_ylabel(r"$\sqrt{2b}$")
ax[0].set_title(r"$\sqrt{2b}=1-p\tau$  (falls linearly at rate $p$)")
ax[0].plot([],[],'k--',label=r"$1-p\tau$"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_xlim(0,.65); ax[0].set_ylim(0,1.05)

# B: width of beneficial band <Delta>_+ is proportional to sqrt(2b) (self-similar width)
b=shp["b"]; s2b=np.sqrt(2*b); m=b>0.01
ax[1].plot(s2b[m], shp["wpos"][m],'o-',ms=3,label=r"$\langle\Delta\rangle_+$ (band width)")
sig0=np.sqrt(4*shp["P"]); ax[1].plot(s2b[m], sig0*np.sqrt(2/np.pi)*s2b[m],'r--',
        label=r"$\sigma_0\sqrt{2/\pi}\,\sqrt{2b}$")
ax[1].set_xlabel(r"$\sqrt{2b}$"); ax[1].set_ylabel(r"$\langle\Delta\rangle_+$")
ax[1].set_title(f"p={shp['P']}: band width $\\propto\\sqrt{{2b}}$")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

# C: -db/dtau vs sqrt(2b) linear thru origin, slope p
def smooth(y,k): return np.convolve(y,np.ones(k)/k,mode='same')
for P,col in [(2,'C0'),(3,'C1')]:
    nm=max([n for n in names if res[n]["P"]==P],key=lambda n:res[n]["N"]); r=res[nm]
    full=r["cnt_tN"]>=r["n_reps"]; tau=r["tN"][full]; b=r["b_vs_tN"][full]
    bs=smooth(b,21); d=-np.gradient(bs,tau)
    mm=(b>0.03)&(np.arange(len(b))>10)&(np.arange(len(b))<len(b)-10)
    ax[2].plot(np.sqrt(2*bs[mm]),d[mm],col+'.',ms=2,alpha=.5)
    xx=np.linspace(0,1,10); ax[2].plot(xx,P*xx,col+'-',lw=2,label=f"slope $p={P}$")
ax[2].set_xlabel(r"$\sqrt{2b}$"); ax[2].set_ylabel(r"$-db/d\tau$")
ax[2].set_title(r"$-db/d\tau = p\sqrt{2b}$"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig(os.path.join(OUT,"explain_sqrt2b.png"),dpi=120)
print("saved explain_sqrt2b.png")
