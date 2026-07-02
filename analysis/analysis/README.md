# Scrambling analysis (p-spin SSWM walks)

Main deliverable: **`../scrambling_report.html`** — self-contained (MathJax + embedded
figures); open in a browser. Companion: **`../transient_law_derivation.html`** — complete
step-by-step derivation of the all-times kernel-difference law (every integral, the
Krawtchouk counting, the beta=2 size-bias computation, the four-term covariance algebra,
a worked numeric example, and the formula anatomy on real data). Built by
`build_derivation.py`.

## Pipeline (reproduce from scratch)

```bash
.venv/bin/python analysis/run_measurements.py main    # N1000 p=2, N300 p=3, full ref grid
.venv/bin/python analysis/run_measurements.py sweep   # all other datasets (slope scaling)
.venv/bin/python analysis/run_controls.py             # neutral / uniform walks + surrogate
.venv/bin/python analysis/make_figures.py             # figures/ + results/stats.json
.venv/bin/python analysis/build_report.py             # ../scrambling_report.html
```

Each step skips work whose output already exists in `results/`.

## Files

- `analysis_lib.py` — walk replay (via `../helper.py`), scrambling observables
  (pool/unflipped/flipped Pearson, subset EMD, transient Pearson), Krawtchouk/kernel
  theory curves, matched-protocol slope fits.
- `run_measurements.py` — measurement driver (`meas_*.npz`, `snap_*.npz`).
- `run_controls.py` — neutral walk (v=0), uniform-acceptance greedy, and the
  landscape-free Gaussian kernel surrogate (`ctrl_*.npz`, `surr_*.npz`).
- `make_figures.py` — all figures + `results/stats.json` (headline numbers).
- `build_report.py` — assembles the HTML report.

## Headline results (EMD-centric; report v2)

- Early-time EMD law: W~ = q_mutual^(p-1) * (1 - kappa*u/N), with
  kappa = 2 E[Delta_selected]/E[Delta|Delta>0] = pi for SSWM (2 for uniform/neutral).
  tau_EMD = N/(2(p-1)+pi): 194 (p=2,N=1000), 42 (p=3,N=300). No v, R, d_H, d_0.
  (Earlier draft's constant 2p+pi/2 was an incomplete bookkeeping; corrected to
  2(p-1)+pi and pinned by high-precision surrogates: kappa = 3.12/3.15 vs pi.)
- Exact facts: normalized EMD = raiser-vs-lowerer separation (so raisers/lowerers give
  identical curves); with no move-reversal the EMD equals the kernel correlation exactly.
- Controls: kappa follows the acceptance rule (pi -> 2), not the drift (v = 0.62 -> 0.47
  -> 0 leaves it unchanged). Underlying spin-level laws: unflipped +q^(p-1) (Krawtchouk),
  flipped -q^(p-1), pool q^(p-1)(1-2*beta*u/N) with beta=2 (second vs first moment).
- Basin regime (d <~ d* ~ 0.15N, last ~40% of walk): tau_EMD = 2d/(1+v_rem) =
  R^2/(2(1+v_rem)) ~ d (amplitude sqrt(d2/d1) x angle (disagreement-set renewal); drift
  cancels in the angle, enters via the amplitude). EMD floor 0.28-0.56 for basin refs.
- Crossover d* = N(1+v_rem)/(2(2p-2+pi)) ~ 0.18N (p=2), 0.13N (p=3), reached at ~60%
  of the walk for both p.
- Intermediate regime, tested densely (run_intermediate.py + intermediate_analysis.py,
  report Sec 8.1): at d ~ d* the measured EMD half-time is ~half of either single
  mechanism (both act at once); rates-added reproduces t_1/2 to 10-25% (no free params)
  over d in [d*/2, ~2d*]; far field: drop the basin term (kernel-only accurate to 5-6%;
  rates-added over-adds there by ~30-40% -- angle factor double-counts at the equator);
  deep basin: basin-only. Composite prescription accurate to <~15% everywhere measured.
- All-times theory status (Sec 8.2; unified_test.py, process_model.py, alltimes_figs.py,
  transient_law_test.py): DERIVED at all depths: transient correlation = kernel-difference
  law rho_trans = [rho_p(u12)+1-rho_p(d1)-rho_p(d2)]/(2 sqrt((1-rho_p(d1))(1-rho_p(d2)))),
  rho_p(u) = Krawtchouk(u)*(1-4u/N)+ (the verified pool kernel). Three lines:
  a_i = Delta_i(t)-Delta_i(T); expand Cov(a1,a2) into four pool covariances; u(t,T) = d_H(t).
  Accuracy <= 0.013-0.04 pointwise for d/N <= 0.34 (both p); 0.09 at the p=2 equator bin.
  The naive p=2 eigen-identity (fixed move vectors) is WORSE (up to 0.29): A-set parity
  matters and rho_p carries it.
- Position-only predictive form (derivation doc Sec V; position_only.py): substituting the
  typical trajectory, the drift cancels exactly at the tangent:
  rate(d) = c*Lambda/(2N(1-rho_p(d))), Lambda = 2(p-1)+4; tau depends only on current depth.
  t_half of the d-only law matches measured transient t_half to <=9% at all depths, both p
  (p=2 equator bin: 0.84). Depth from current spectrum: n_+(d) calibration (fig d5).
  RULED OUT for the full EMD: (i) endpoint-conditioned Gaussian / pinned-kernel partial
  regression (0.43-0.70 data/pred through crossover - real pinning ~2x stronger than
  Gaussian endpoint conditioning); (ii) EBBM-type mean-field process (far field OK,
  t_half 120 vs 109; basin impossible: convection = common shift invisible to EMD,
  fresh kicks, all rates O(1/N); 0/140 runs terminate, stalls at n+/N ~ 0.07).
  Open: non-Gaussian selection split between one-sided pseudogapped frozen part and
  Gaussian transient (frozen-transient anti-correlation at equator: 1.1+3.2-2.3 ~= 2).
