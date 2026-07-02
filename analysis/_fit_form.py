import os, pickle
import numpy as np
from scipy.optimize import curve_fit

OUT = "/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
res = pickle.load(open(os.path.join(OUT,"beneficial_results.pkl"),"rb"))
names = sorted(res, key=lambda n:(res[n]["P"],res[n]["N"]))

def get_curve(nm):
    r=res[nm]; full=r["cnt_tN"]>=r["n_reps"]
    return r["tN"][full], r["b_vs_tN"][full], r["N"], r["P"]

print("=== Test zero-parameter form  b=(1/2)(1-p*tau)^2  (bulk: b>0.02) ===")
print(f"{'N':>6}{'P':>3}   {'RMSE_bulk':>10}  {'maxdev':>8}")
for nm in names:
    tau,b,N,P = get_curve(nm)
    pred = 0.5*np.clip(1-P*tau,0,None)**2
    m = b>0.02
    rmse = np.sqrt(np.mean((b[m]-pred[m])**2)); mx=np.max(np.abs(b[m]-pred[m]))
    print(f"{N:>6}{P:>3}   {rmse:>10.4f}  {mx:>8.4f}")

print("\n=== Free fit  b=(1/2)(1 - tau/tc)^beta  on bulk (b>0.03) ===")
def form(t,tc,beta): return 0.5*np.clip(1-t/tc,0,None)**beta
print(f"{'N':>6}{'P':>3}   {'tc':>7} {'1/p':>6}  {'beta':>6}   {'rmse':>8}")
for nm in names:
    tau,b,N,P=get_curve(nm)
    m=b>0.03
    try:
        popt,_=curve_fit(form,tau[m],b[m],p0=[1.0/P,2.0],maxfev=20000)
        pred=form(tau[m],*popt); rmse=np.sqrt(np.mean((b[m]-pred)**2))
        print(f"{N:>6}{P:>3}   {popt[0]:>7.4f} {1/P:>6.4f}  {popt[1]:>6.3f}   {rmse:>8.4f}")
    except Exception as e:
        print(f"{N:>6}{P:>3}  fit failed {e}")

print("\n=== Free fit exponent with tc=1/p fixed:  b=(1/2)(1-p*tau)^beta ===")
def form2(t,P): return None
for nm in names:
    tau,b,N,P=get_curve(nm)
    m=b>0.03
    g=lambda t,beta: 0.5*np.clip(1-P*t,0,None)**beta
    try:
        popt,_=curve_fit(g,tau[m],b[m],p0=[2.0],maxfev=20000)
        pred=g(tau[m],*popt); rmse=np.sqrt(np.mean((b[m]-pred)**2))
        print(f"N={N:>5} P={P}: beta={popt[0]:.3f}  rmse={rmse:.4f}")
    except Exception as e:
        print(f"N={N} P={P} failed")
