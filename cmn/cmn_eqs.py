import time
from pathlib import Path

import numpy as np

def theta(s):
    """Heaviside step function."""
    return 0.5 * (np.sign(s) + 1)


def positive_integral(s, p, ds, eps=1e-10):
    """Compute the integral of p(s) over positive part of s."""
    integrand = p
    integral = np.sum(integrand[s > 0]) * ds
    return integral if integral > eps else eps  # Prevent division by zero


def flip_term(s: np.ndarray, p: np.ndarray) -> np.ndarray:
    ds = s[1] - s[0]
    dp_pos = theta(s) * p
    dp_neg = np.flip(dp_pos)
    flip_term = (dp_neg - dp_pos) * np.abs(s)
    flip_term /= positive_integral(s, p, ds)
    return flip_term


def drift_term(p: np.ndarray, ds, c):
    dpdx = np.zeros_like(p)
    if c > 0:
        # Backward difference for c > 0
        dpdx[1:] = (p[1:] - p[:-1]) / ds
    elif c < 0:
        # Forward difference for c < 0
        dpdx[:-1] = (p[1:] - p[:-1]) / ds
    else:
        dpdx[:] = 0.0
    return c * dpdx

def diff_term(p: np.ndarray, ds, D):
    dpdx2 = np.zeros_like(p)
    dpdx2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / ds ** 2
    return (D/2) * dpdx2

def rhs(t, p, s, ds, c, D):
    return flip_term(s, p) + drift_term(p, ds, c) + diff_term(p, ds, D)

def msd_fit_func(t, m, a):
    return m * t**a

def normalize(p: np.ndarray, ds: float) -> np.ndarray:
    """
    Normalize a probability density p defined on a grid of spacing ds,
    so that sum(p)*ds == 1.
    """
    total = np.sum(p) * ds
    if total > 0:
        return p / total
    else:
        return p


# ---------------------------------------------------------------------------
# p-spin two-point-function DFE solver
# ---------------------------------------------------------------------------

DEFAULT_PSPIN_R_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "pspin_R_solver"
)

DEFAULT_PSPIN_R_TAU_CFG = {
    2: ([0.05, 0.125, 0.25, 0.5, 0.546], 0.56),
    3: ([1 / 30, 1 / 12, 1 / 6, 1 / 3, 0.434], 0.45),
    4: ([0.025, 0.0625, 0.125, 0.25, 0.285], 0.30),
}


def pspin_R_schedule(p):
    """Return tau checkpoints and stopping tau for the p-spin R solver."""
    if p in DEFAULT_PSPIN_R_TAU_CFG:
        return DEFAULT_PSPIN_R_TAU_CFG[p]
    return [0.25 / p, 0.5 / p, 0.75 / p, 1.0 / p, 1.3 / p], 1.7 / p


def pspin_R_state_path(p, output_dir=DEFAULT_PSPIN_R_DIR, suffix="_relu"):
    """State/cache path for a p-spin R(k,k') solver run."""
    return Path(output_dir) / f"R2d_p{int(p)}{suffix}.npz"


def tridiag_const(alpha, X, axis):
    """Solve (I - alpha*D2) Y = X along one axis, with Neumann boundaries."""
    if alpha <= 0:
        return X
    Y = np.moveaxis(X, axis, 0).astype(np.float64, copy=True)
    n = Y.shape[0]
    b = np.full(n, 1 + 2 * alpha)
    b[0] = b[-1] = 1 + alpha
    a = -alpha
    cp = np.empty(n)
    denom = np.empty(n)
    denom[0] = b[0]
    cp[0] = a / b[0]
    for i in range(1, n):
        denom[i] = b[i] - a * cp[i - 1]
        cp[i] = a / denom[i]
    Y[0] = Y[0] / denom[0]
    for i in range(1, n):
        Y[i] = (Y[i] - a * Y[i - 1]) / denom[i]
    for i in range(n - 2, -1, -1):
        Y[i] = Y[i] - cp[i] * Y[i + 1]
    return np.moveaxis(Y, 0, axis)


def div_flux_1d(v, U, dk):
    """Conservative local Lax-Friedrichs divergence of the flux v*U."""
    vf = 0.5 * (v[1:] + v[:-1])
    Jf = (
        0.5 * (v[1:] * U[1:] + v[:-1] * U[:-1])
        - 0.5 * np.abs(vf) * (U[1:] - U[:-1])
    )
    out = np.empty_like(U)
    out[1:-1] = (Jf[1:] - Jf[:-1]) / dk
    out[0] = Jf[0] / dk
    out[-1] = -Jf[-1] / dk
    return out


def div_flux_axis(v, R, dk, axis):
    """Conservative flux divergence along one axis of a 2D array."""
    Rm = np.moveaxis(R, axis, 0)
    vf = 0.5 * (v[1:] + v[:-1])[:, None]
    Jf = (
        0.5 * (v[1:, None] * Rm[1:] + v[:-1, None] * Rm[:-1])
        - 0.5 * np.abs(vf) * (Rm[1:] - Rm[:-1])
    )
    out = np.empty_like(Rm)
    out[1:-1] = (Jf[1:] - Jf[:-1]) / dk
    out[0] = Jf[0] / dk
    out[-1] = -Jf[-1] / dk
    return np.moveaxis(out, 0, axis)


def init_pspin_R_state(p, npts=251, nsig=5.0):
    """
    Initial uncorrelated p-spin DFE state for P(k,tau) and R(k,k',tau).

    The Gaussian P0 has variance p/2. R0 is the Wick/uncorrelated two-point
    function projected onto the sum rule used by the PDE.
    """
    p = int(p)
    lam = 0.5 * p * (p - 1)
    kk = np.linspace(-nsig * np.sqrt(p / 2), nsig * np.sqrt(p / 2), int(npts))
    dk = kk[1] - kk[0]
    P = np.exp(-kk ** 2 / p) / np.sqrt(np.pi * p)
    P /= P.sum() * dk
    dP = np.gradient(P, dk)
    R = -lam * (np.outer(dP, P) + np.outer(P, dP))
    return {
        "kk": kk,
        "dk": dk,
        "P": P,
        "R": R,
        "t": 0.0,
        "tau": 0.0,
        "step": 0,
        "lam": lam,
        "tau_c": [0.0],
        "f_c": [0.5],
        "var_c": [p / 2.0],
        "kv_c": [0.0],
        "e_c": [0.0],
        "snaps": {},
        "done": False,
    }


def save_pspin_R_state(path, state):
    """Save a resumable p-spin R solver state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        kk=state["kk"],
        dk=state["dk"],
        P=state["P"],
        R=state["R"],
        t=state["t"],
        tau=state["tau"],
        step=state["step"],
        lam=state["lam"],
        done=state["done"],
        tau_c=np.asarray(state["tau_c"]),
        f_c=np.asarray(state["f_c"]),
        var_c=np.asarray(state["var_c"]),
        kv_c=np.asarray(state["kv_c"]),
        e_c=np.asarray(state["e_c"]),
        **{f"snap_{k}": v for k, v in state["snaps"].items()},
    )


def load_pspin_R_state(path):
    """Load a p-spin R solver state, or return None if it is absent."""
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path)
    return {
        "kk": z["kk"],
        "dk": float(z["dk"]),
        "P": z["P"].copy(),
        "R": z["R"].copy(),
        "t": float(z["t"]),
        "tau": float(z["tau"]),
        "step": int(z["step"]),
        "lam": float(z["lam"]),
        "done": bool(z["done"]),
        "tau_c": list(z["tau_c"]),
        "f_c": list(z["f_c"]),
        "var_c": list(z["var_c"]),
        "kv_c": list(z["kv_c"]),
        "e_c": list(z["e_c"]),
        "snaps": {k[5:]: z[k] for k in z.files if k.startswith("snap_")},
    }


def run_pspin_R_solver(
    p,
    output_dir=DEFAULT_PSPIN_R_DIR,
    suffix="_relu",
    npts=251,
    nsig=5.0,
    budget_seconds=30.0,
    t_max=600.0,
    progress=True,
):
    """
    Evolve P(k,t) and the full R(k,k',t) for the pure p-spin DFE PDE.

    The acceptance rate is the exact ReLU/SSWM rate r(k)=max(-2k,0). The state
    is resumable and is saved under ``output_dir`` after each call. Returns True
    when the requested p has reached its stopping criterion.
    """
    p = int(p)
    fp = pspin_R_state_path(p, output_dir=output_dir, suffix=suffix)
    tau_targets, tau_stop = pspin_R_schedule(p)
    state = load_pspin_R_state(fp)
    if state is None:
        state = init_pspin_R_state(p, npts=npts, nsig=nsig)
    if state["done"]:
        if progress:
            print(f"[R2d] p={p} already done: tau={state['tau']:.4f}")
        return True

    kk, dk, lam = state["kk"], state["dk"], state["lam"]
    P, R = state["P"], state["R"]
    t, tau, step = state["t"], state["tau"], state["step"]
    rev = slice(None, None, -1)

    r = np.where(kk < 0, -2.0 * kk, 0.0)
    r_rev = r[rev].copy()
    rmax = r.max()
    targets = [tt for tt in tau_targets if f"{tt:.6f}" not in state["snaps"]]
    floor = 1e-13
    t_stop = time.time() + float(budget_seconds)

    while True:
        if step % 20 == 0 and time.time() > t_stop:
            break
        rr = (r * P).sum() * dk
        if (rr < 1e-5 and tau > 0.4 / p) or t > t_max or tau > tau_stop:
            state["done"] = True
            break
        D = 2 * lam * rr
        dPk = np.gradient(P, dk)

        Jd = -2.0 * (R @ r) * dk
        v = Jd / np.maximum(P, floor)
        vc = 4.0 * D / dk + 2 * np.abs(kk).max()
        np.clip(v, -vc, vc, out=v)

        gainP = r_rev * P[rev]
        lossP = r * P
        dPdt = gainP - lossP - div_flux_1d(v, P, dk)

        masterR = (
            -(r_rev[:, None]) * R[rev, :]
            - (r[:, None]) * R
            - (r_rev[None, :]) * R[:, rev]
            - (r[None, :]) * R
        )
        srcR = -2 * lam * (
            np.outer(r_rev * P[rev], dPk) + np.outer(dPk, r_rev * P[rev])
        )
        advR = div_flux_axis(v, R, dk, 0) + div_flux_axis(v, R, dk, 1)
        dRdt = masterR + srcR - advR - 2 * (p - 2) * rr * R

        vmax = max(np.abs(v).max(), 1e-9)
        dt = 0.3 * min(dk / vmax, 0.5 / rmax, 0.02)

        Pn = P + dt * dPdt
        Rn = R + dt * dRdt
        alpha = D * dt / dk ** 2
        Pn = tridiag_const(alpha, Pn, 0)
        Rn = tridiag_const(alpha, Rn, 0)
        Rn = tridiag_const(alpha, Rn, 1)
        np.clip(Pn, 0, None, out=Pn)
        Pn /= Pn.sum() * dk

        rho = (p - 1) * kk * Pn - Rn.sum(1) * dk
        tot = rho.sum() * dk
        Rn += np.outer(rho, Pn) + np.outer(Pn, rho) - tot * np.outer(Pn, Pn)
        Rn = 0.5 * (Rn + Rn.T)

        P, R = Pn, Rn
        t += dt
        tau += rr * dt
        step += 1
        if step % 25 == 0:
            m1 = (kk * P).sum() * dk
            state["tau_c"].append(tau)
            state["f_c"].append(P[kk < 0].sum() * dk)
            state["var_c"].append((kk ** 2 * P).sum() * dk - m1 ** 2)
            state["kv_c"].append((kk * Jd).sum() * dk)
            state["e_c"].append(m1 / p)
        while targets and tau >= targets[0]:
            state["snaps"][f"{targets.pop(0):.6f}"] = P.copy()

    if state["done"]:
        for tt in targets:
            state["snaps"][f"{tt:.6f}"] = P.copy()
    state.update(P=P, R=R, t=t, tau=tau, step=step)
    save_pspin_R_state(fp, state)
    if progress:
        print(
            f"[R2d] p={p}: step={step} t={t:.2f} tau={tau:.4f} "
            f"Var={state['var_c'][-1]:.3f} done={state['done']}",
            flush=True,
        )
    return state["done"]


def ensure_pspin_R_states(
    ps,
    output_dir=DEFAULT_PSPIN_R_DIR,
    suffix="_relu",
    npts=251,
    nsig=5.0,
    budget_seconds=30.0,
    max_passes=20,
    progress=True,
):
    """Run/resume solver states for every p until all are complete."""
    paths = {}
    for p in ps:
        done = False
        passes = 0
        while not done and passes < max_passes:
            done = run_pspin_R_solver(
                p,
                output_dir=output_dir,
                suffix=suffix,
                npts=npts,
                nsig=nsig,
                budget_seconds=budget_seconds,
                progress=progress,
            )
            passes += 1
        if not done:
            raise RuntimeError(f"p-spin R solver did not finish for p={p}")
        paths[int(p)] = pspin_R_state_path(p, output_dir=output_dir, suffix=suffix)
    return paths
