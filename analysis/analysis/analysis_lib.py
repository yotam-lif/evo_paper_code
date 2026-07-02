"""
Library for measuring scrambling along stored SSWM greedy-ascent walks
on pure/mixed p-spin landscapes (data produced by gen_dat_pspin.py).

Conventions
-----------
- Time t = number of accepted flips (0 .. T).
- S[t, i] = Delta_i(sigma(t)) : energy change of flipping spin i at time t.
- parity[t, i] = True if spin i has been flipped an odd number of times in (0, t].
- d_f(t)  = Hamming distance to the terminal configuration sigma(T).
- u(t1,t2) = Hamming distance between sigma(t1) and sigma(t2) (mutual displacement).
- q(t1,t2) = 1 - 2 u(t1,t2)/N   (mutual overlap).

Couplings in the data follow helper.py: Var(J) = p!/N^(p-1)  (twice the
convention of the context document; irrelevant for all normalized measures).
"""

import os
import sys

import numpy as np
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import helper  # noqa: E402


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_walk(rec, flip_seq=None):
    """Replay a stored walk; return spectra, parities, energies.

    rec : dict with 'init_sigma', 'J', 'flip_seq' (flip_seq overridable).
    """
    J, sig0 = rec["J"], rec["init_sigma"]
    seq = list(rec["flip_seq"]) if flip_seq is None else list(flip_seq)
    N, T = J["N"], len(seq)

    state = helper._initialize_relaxation_state(sig0, J)
    S = np.empty((T + 1, N), np.float32)
    E = np.empty(T + 1, np.float64)
    parity = np.zeros((T + 1, N), bool)
    S[0] = state["spectrum"]
    E[0] = state["energy"]
    par = np.zeros(N, bool)
    for t, k in enumerate(seq):
        helper._apply_flip(state, J, int(k))
        par[k] = ~par[k]
        S[t + 1] = state["spectrum"]
        E[t + 1] = state["energy"]
        parity[t + 1] = par
    return dict(S=S, E=E, parity=parity, seq=np.asarray(seq, int), N=N, T=T,
                sig0=np.asarray(sig0, np.int8))


def validate_replay(rw):
    """Check the replay is a genuine greedy ascent ending at a local max."""
    S, seq = rw["S"], rw["seq"]
    flipped_pos = all(S[t, k] > 0 for t, k in enumerate(seq))
    terminal_max = not np.any(S[-1] > 0)
    return flipped_pos and terminal_max


# ---------------------------------------------------------------------------
# Walk-level scalars
# ---------------------------------------------------------------------------

def walk_scalars(rw):
    S, parity, N, T = rw["S"], rw["parity"], rw["N"], rw["T"]
    par_f = parity[-1]
    d_f = np.count_nonzero(parity ^ par_f[None, :], axis=1)          # dist to terminal
    u0 = np.count_nonzero(parity, axis=1)                            # dist to start
    n_pos = np.count_nonzero(S > 0, axis=1)
    spec_mean = S.mean(axis=1)
    spec_std = S.std(axis=1)
    return dict(d_f=d_f, u0=u0, n_pos=n_pos, spec_mean=spec_mean,
                spec_std=spec_std, E=rw["E"], N=N, T=T)


# ---------------------------------------------------------------------------
# Pairwise measures
# ---------------------------------------------------------------------------

def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / d) if d > 0 else np.nan


def measure_from_ref(rw, t_ref, min_subpool=12, with_transient=True):
    """All scrambling observables between t_ref and every t in [0, T].

    Returns dict of arrays of length T+1 (both directions of time included).
    """
    S, parity, N, T = rw["S"], rw["parity"], rw["N"], rw["T"]
    s_ref = S[t_ref].astype(np.float64)
    par_ref = parity[t_ref]

    M_pos = s_ref > 0                      # the distinguished subset (raisers)
    M_neg = s_ref < 0                      # energy-lowering subset
    a_final = S[-1].astype(np.float64)     # frozen terminal spectrum
    a_ref = s_ref - a_final                # transient part at the reference

    nt = T + 1
    out = {k: np.full(nt, np.nan) for k in
           ("u", "rho_pool", "rho_unflip", "rho_flip", "emd_pos", "emd_neg",
            "rho_trans", "frac_pos_still")}

    full_ref = s_ref
    emd_pos_ref = wasserstein_distance(full_ref, full_ref[M_pos]) if M_pos.sum() >= min_subpool else np.nan
    emd_neg_ref = wasserstein_distance(full_ref, full_ref[M_neg]) if M_neg.sum() >= min_subpool else np.nan

    for t in range(nt):
        s_t = S[t].astype(np.float64)
        flip_mask = parity[t] ^ par_ref            # flipped odd # times between
        u = int(np.count_nonzero(flip_mask))
        out["u"][t] = u
        out["rho_pool"][t] = _pearson(s_ref, s_t)
        unflip = ~flip_mask
        if unflip.sum() >= min_subpool:
            out["rho_unflip"][t] = _pearson(s_ref[unflip], s_t[unflip])
        if flip_mask.sum() >= min_subpool:
            out["rho_flip"][t] = _pearson(s_ref[flip_mask], s_t[flip_mask])
        if np.isfinite(emd_pos_ref) and emd_pos_ref > 0:
            out["emd_pos"][t] = wasserstein_distance(s_t, s_t[M_pos]) / emd_pos_ref
        if np.isfinite(emd_neg_ref) and emd_neg_ref > 0:
            out["emd_neg"][t] = wasserstein_distance(s_t, s_t[M_neg]) / emd_neg_ref
        if with_transient:
            out["rho_trans"][t] = _pearson(a_ref, s_t - a_final)
        if M_pos.sum() > 0:
            out["frac_pos_still"][t] = np.count_nonzero(s_t[M_pos] > 0) / M_pos.sum()

    out["q"] = 1.0 - 2.0 * out["u"] / N
    return out


def default_ref_grid(scal, fracs=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
                                  0.5, 0.6, 0.7, 0.8, 0.9),
                     d_targets=()):
    """Reference times: fixed fractions of T plus last times with d_f >= target."""
    T, d_f = scal["T"], scal["d_f"]
    refs = sorted({int(round(f * T)) for f in fracs})
    for d in d_targets:
        idx = np.nonzero(d_f >= d)[0]
        if idx.size:
            refs.append(int(idx[-1]))
    return sorted(set(refs))


# ---------------------------------------------------------------------------
# Krawtchouk / kernel theory curves
# ---------------------------------------------------------------------------

def krawtchouk_ratio(k, u, M):
    """K_k(u; M) / C(M, k): exact correlation of a degree-k spin product
    between two configurations at Hamming distance u on M sites."""
    u = np.asarray(u, float)
    if k == 0:
        return np.ones_like(u)
    if k == 1:
        return 1.0 - 2.0 * u / M
    if k == 2:
        return 1.0 - 4.0 * u * (M - u) / (M * (M - 1.0))
    # generic (small k): sum_j (-1)^j C(u,j) C(M-u,k-j) / C(M,k)
    from scipy.special import comb
    tot = np.zeros_like(u)
    for j in range(k + 1):
        tot += (-1) ** j * comb(u, j) * comb(M - u, k - j)
    return tot / comb(M, k)


def rho_unflip_theory(u, N, orders):
    """xi'(q)-weighted exact kernel prediction for unflipped spins.

    orders: list of interaction orders p present (equal helper.py weights
    Var_p = p!/N^(p-1), giving per-spin field variance sum_p p and
    covariance sum_p p * K_{p-1}(u; N-1)/C(N-1, p-1)."""
    num = np.zeros_like(np.asarray(u, float))
    den = 0.0
    for p in orders:
        num += p * krawtchouk_ratio(p - 1, u, N - 1)
        den += p
    return num / den


def rho_flip_theory(u, N, orders):
    """Same for spins flipped an odd number of times (sign reversal),
    evaluated at u-1 disagreements among the other N-1 spins."""
    u = np.asarray(u, float)
    return -rho_unflip_theory(np.maximum(u - 1.0, 0.0), N, orders)


# ---------------------------------------------------------------------------
# Slope fitting
# ---------------------------------------------------------------------------

def fit_initial_slope(dt, y, y_hi=0.97, y_lo=0.55, min_pts=3):
    """Fit ln y = a - s*dt on the window y in [y_lo, y_hi], dt >= 0.
    Returns slope s (decay rate per accepted move) or nan."""
    dt = np.asarray(dt, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & (dt >= 0)
    dt, y = dt[ok], y[ok]
    # keep the contiguous initial stretch below y_hi and above y_lo
    sel = (y <= y_hi) & (y >= y_lo)
    if sel.sum() < min_pts:
        return np.nan
    # restrict to before y first drops below y_lo (avoid re-entries)
    first_low = np.argmax(y < y_lo) if np.any(y < y_lo) else len(y)
    sel &= np.arange(len(y)) < first_low
    if sel.sum() < min_pts:
        return np.nan
    A = np.vstack([np.ones(sel.sum()), dt[sel]]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y[sel]), rcond=None)
    return -coef[1]


def fit_kappa(u, y, N, p, y_min=0.82):
    """Least-squares fit of the flip-bookkeeping constant kappa in
    y = [K_{p-1}(u;N-1)/C] * (1 - kappa*u/N), using points with y >= y_min.

    kappa is protocol-free (a curve-level parameter, not a window slope):
    theory kappa = 2 E[Delta_selected]/E[Delta|Delta>0]  (= pi for SSWM,
    2 for uniform/neutral acceptance)."""
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & (y >= y_min) & (u > 0)
    if ok.sum() < 3:
        return np.nan
    base = rho_unflip_theory(u[ok], N, [p])
    x = u[ok] / N
    r = 1 - y[ok] / base
    return float(np.dot(x, r) / np.dot(x, x))


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def process_dataset(pkl_path, fracs=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    d_target_fracs=(1 / 8, 1 / 16, 1 / 32),
                    d_target_abs=(12, 8),
                    max_repeats=None, verbose=True):
    """Replay every repeat in a pickle and measure everything.

    Returns a dict ready for np.savez (ragged arrays padded with NaN).
    """
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if max_repeats is not None:
        data = data[:max_repeats]

    N = data[0]["J"]["N"]
    orders = [s["order"] for s in data[0]["J"]["sectors"]]
    reps = []
    for r, rec in enumerate(data):
        rw = replay_walk(rec)
        assert validate_replay(rw), f"replay failed for repeat {r}"
        scal = walk_scalars(rw)
        d_targets = sorted({int(round(f * N)) for f in d_target_fracs} |
                           {d for d in d_target_abs if d < N // 4})
        refs = default_ref_grid(scal, fracs, d_targets)
        per_ref = []
        for t_ref in refs:
            m = measure_from_ref(rw, t_ref)
            m["t_ref"] = t_ref
            m["d_f_ref"] = int(scal["d_f"][t_ref])
            per_ref.append(m)
        reps.append(dict(scal=scal, refs=refs, per_ref=per_ref))
        if verbose:
            print(f"  repeat {r}: T={rw['T']}, d0={scal['d_f'][0]}, "
                  f"E_f/N={scal['E'][-1]/N:.4f}, refs={len(refs)}", flush=True)
        del rw
    return dict(N=N, orders=orders, reps=reps)


def pack_results(res):
    """Flatten process_dataset output into arrays for np.savez."""
    N = res["N"]
    reps = res["reps"]
    R = len(reps)
    Tmax = max(rep["scal"]["T"] for rep in reps)
    nR = max(len(rep["refs"]) for rep in reps)
    nt = Tmax + 1

    scal_keys = ("d_f", "u0", "n_pos", "spec_mean", "spec_std", "E")
    packed = {f"scal_{k}": np.full((R, nt), np.nan) for k in scal_keys}
    packed["T"] = np.array([rep["scal"]["T"] for rep in reps])
    packed["N"] = N
    packed["orders"] = np.array(res["orders"])

    pair_keys = ("u", "q", "rho_pool", "rho_unflip", "rho_flip",
                 "emd_pos", "emd_neg", "rho_trans", "frac_pos_still")
    for k in pair_keys:
        packed[f"pair_{k}"] = np.full((R, nR, nt), np.nan, np.float32)
    packed["ref_t"] = np.full((R, nR), -1, int)
    packed["ref_d_f"] = np.full((R, nR), -1, int)

    for r, rep in enumerate(reps):
        T = rep["scal"]["T"]
        for k in scal_keys:
            packed[f"scal_{k}"][r, :T + 1] = rep["scal"][k]
        for j, m in enumerate(rep["per_ref"]):
            packed["ref_t"][r, j] = m["t_ref"]
            packed["ref_d_f"][r, j] = m["d_f_ref"]
            for k in pair_keys:
                packed[f"pair_{k}"][r, j, :T + 1] = m[k]
    return packed
