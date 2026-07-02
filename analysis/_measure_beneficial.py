"""
Measure the fraction of beneficial spins (positive flip-spectrum entries) along
the recorded greedy/SSWM walks, as a function of rescaled time tau = t/N.
Replays recorded flip_seq via the incremental spectrum machinery in helper.py.
Accumulates results across invocations into beneficial_results.pkl.
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, "/Users/yotamlifschytz/Desktop/untitled folder")
import helper as h

OUT = "/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
TAU = np.linspace(0.0, 1.0, 201)   # grid as fraction-of-walk

def measure_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    N = data[0]["J"]["N"]; P = data[0]["J"]["P"]
    Ls = []
    curves_frac = []            # b on fraction-of-walk grid
    b_of_t_list = []            # raw b(t), t=0..L per rep
    for rep in data:
        J = rep["J"]; sigma0 = rep["init_sigma"]; seq = rep["flip_seq"]
        L = len(seq); Ls.append(L)
        state = h._initialize_relaxation_state(sigma0, J)
        spec = state["spectrum"]
        b = np.empty(L + 1)
        b[0] = np.count_nonzero(spec > 0) / N
        for t, site in enumerate(seq):
            h._apply_flip(state, J, int(site))
            b[t + 1] = np.count_nonzero(spec > 0) / N
        frac = np.arange(L + 1) / L
        curves_frac.append(np.interp(TAU, frac, b))
        b_of_t_list.append(b)
    curves_frac = np.array(curves_frac)
    # average b as a function of t/N (align at integer flip count t)
    Lmax = max(Ls)
    tot = np.zeros(Lmax + 1); cnt = np.zeros(Lmax + 1)
    for b in b_of_t_list:
        tot[:len(b)] += b; cnt[:len(b)] += 1
    b_vs_tN = tot / np.maximum(cnt, 1)
    tN = np.arange(Lmax + 1) / N
    return {
        "N": N, "P": P, "n_reps": len(data),
        "Ls": np.array(Ls),
        "b_vs_frac_mean": curves_frac.mean(0),
        "b_vs_frac_std": curves_frac.std(0),
        "tN": tN, "b_vs_tN": b_vs_tN, "cnt_tN": cnt,
    }

if __name__ == "__main__":
    outpath = os.path.join(OUT, "beneficial_results.pkl")
    results = pickle.load(open(outpath, "rb")) if os.path.exists(outpath) else {}
    for path in sys.argv[1:]:
        name = os.path.basename(path)
        print(f"processing {name} ...", flush=True)
        res = measure_file(path)
        print(f"  N={res['N']} P={res['P']} reps={res['n_reps']}  "
              f"L/N={res['Ls'].mean()/res['N']:.4f} +/-{res['Ls'].std()/res['N']:.4f}  "
              f"b(0)={res['b_vs_tN'][0]:.4f}", flush=True)
        results[name] = res
    pickle.dump(results, open(outpath, "wb"))
    print("saved", len(results), "files", flush=True)
