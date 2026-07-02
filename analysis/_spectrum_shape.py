"""
Test the 'self-similar collapse' picture behind b(tau) = (1/2)(1 - p tau)^2.
Along the walk we record the full flip spectrum at sampled times and measure:
  b(tau)          fraction beneficial            = (1/2) * (positive-tail area)
  rho0(tau)       density at threshold Delta=0   (height of the beneficial sliver)
  wpos(tau)       <Delta>_+  (mean positive gap) (width of the beneficial sliver)
Prediction if the sliver collapses self-similarly with linear dimension ~ sqrt(2b):
  rho0  proportional to sqrt(2b),   wpos proportional to sqrt(2b).
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, "/Users/yotamlifschytz/Desktop/untitled folder")
import helper as h

OUT = "/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"

def run(path, nsample=60, eps=0.25):
    data = pickle.load(open(path, "rb"))
    N = data[0]["J"]["N"]; P = data[0]["J"]["P"]
    frac_grid = np.linspace(0.0, 0.98, nsample)
    acc = {k: np.zeros(nsample) for k in ("b","rho0","wpos","Esel","cnt")}
    for rep in data:
        J = rep["J"]; sigma0 = rep["init_sigma"]; seq = rep["flip_seq"]
        L = len(seq)
        sample_t = np.unique((frac_grid * L).astype(int))
        state = h._initialize_relaxation_state(sigma0, J)
        spec = state["spectrum"]
        ptr = 0
        want = set(int(x) for x in sample_t)
        rec = {}
        for t in range(L + 1):
            if t in want:
                s = np.asarray(spec, float)
                pos = s[s > 0]
                b = pos.size / N
                rho0 = np.count_nonzero(np.abs(s) < eps) / (2 * eps * N)
                wpos = pos.mean() if pos.size else 0.0
                Esel = (pos ** 2).sum() / pos.sum() if pos.size else 0.0
                rec[t] = (b, rho0, wpos, Esel)
            if t < L:
                h._apply_flip(state, J, int(seq[t]))
        # map recorded sample times back onto frac_grid slots
        for i, fr in enumerate(frac_grid):
            t = int(fr * L)
            # nearest recorded
            tt = min(rec.keys(), key=lambda k: abs(k - t))
            b, rho0, wpos, Esel = rec[tt]
            acc["b"][i] += b; acc["rho0"][i] += rho0
            acc["wpos"][i] += wpos; acc["Esel"][i] += Esel; acc["cnt"][i] += 1
    for k in ("b","rho0","wpos","Esel"):
        acc[k] /= np.maximum(acc["cnt"], 1)
    acc["N"] = N; acc["P"] = P; acc["frac"] = frac_grid
    return acc

if __name__ == "__main__":
    r = run("N1000_P2_pure_repeats10.pkl")
    P = r["P"]
    b = r["b"]; s2b = np.sqrt(2 * b)
    # theoretical prefactors at tau=0
    sig0 = np.sqrt(4 * P)
    rho0_pref = 1.0 / (sig0 * np.sqrt(2 * np.pi))     # rho(0) at start (b=1/2, s2b=1)
    w_pref = sig0 * np.sqrt(2 / np.pi)                # <Delta>_+ at start
    print(f"N={r['N']} p={P}")
    print(f"{'b':>7}{'sqrt2b':>8}{'rho0':>8}{'rho0/s2b':>10}{'wpos':>8}{'wpos/s2b':>10}{'Esel':>8}")
    for i in range(0, len(b), 4):
        if b[i] < 0.01: continue
        print(f"{b[i]:>7.3f}{s2b[i]:>8.3f}{r['rho0'][i]:>8.3f}{r['rho0'][i]/s2b[i]:>10.3f}"
              f"{r['wpos'][i]:>8.3f}{r['wpos'][i]/s2b[i]:>10.3f}{r['Esel'][i]:>8.3f}")
    print(f"\npredicted rho0/sqrt(2b) = {rho0_pref:.3f} (constant if self-similar)")
    print(f"predicted wpos/sqrt(2b) = {w_pref:.3f} (constant if self-similar)")
    print(f"E_sel at start = w*sqrt(pi/2)/... check: sig0*sqrt(pi/2)={sig0*np.sqrt(np.pi/2):.3f}")
    pickle.dump(r, open(os.path.join(OUT,"spectrum_shape.pkl"),"wb"))
