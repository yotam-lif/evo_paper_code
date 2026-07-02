"""
Mean-field (N-independent) theory for the fraction of beneficial spins b(tau)
along an SSWM (fitness-gain-weighted) adaptive walk in the pure p-spin model.
tau = t/N (flips per spin).

State: rho(Delta, tau) = density of flip-spectrum values Delta_i = -2 sigma_i h_i,
normalized to 1.  b(tau) = int_{Delta>0} rho.

Per flip (dtau = 1/N):
  * the flipped spin moves +Delta_sel -> -Delta_sel      (mirror; loses 1 from b)
  * every other spin gets a kick.  Because the flipped spin k was *selected* for
    S_k = -Delta_k/2 < 0 (beneficial) and weighted by |Delta_k|, the shared-coupling
    piece of each neighbour's field is biased negative:
        E[kick_j | k] = -2(p-1) Delta_sel / N   (uniform drift, Delta-independent)
    -> drift velocity in tau:  v(tau) = -2(p-1) E_sel,  E_sel = <Delta^2>_+/<Delta>_+
  * kick variance gives diffusion:  D/2 = 8 p (p-1).

Selection density (which Delta is flipped):  w(Delta) = Delta rho(Delta)/Z, Delta>0.

  d rho/dtau = 8p(p-1) rho''  +  2(p-1)E_sel rho'  -  w(Delta)1_{>0} + w(-Delta)1_{<0}

Initial slope (rho = Gaussian(0, sigma0^2=4p)):  db/dtau|_0 = -p   (derived analytically).
"""
import numpy as np

def solve_mf(p, Lgrid=30.0, nx=1501, dtau=None, tau_max=1.2):
    # explicit-diffusion stability: dtau < dx^2/(2 D)
    if dtau is None:
        _dx = 2*Lgrid/(nx-1)
        dtau = 0.2 * _dx**2 / (2*8.0*p*(p-1.0))
    x = np.linspace(-Lgrid, Lgrid, nx)
    dx = x[1] - x[0]
    sig0 = np.sqrt(4.0 * p)
    rho = np.exp(-x**2 / (2*sig0**2)) / (sig0*np.sqrt(2*np.pi))
    rho /= rho.sum()*dx

    Dcoef = 8.0*p*(p-1.0)
    pos = x > 0
    xrev = x[::-1]                      # for mirror reinsertion
    taus, bs = [], []
    tau = 0.0
    nsteps = int(tau_max/dtau)
    for it in range(nsteps):
        b = rho[pos].sum()*dx
        taus.append(tau); bs.append(b)
        if b <= 1e-4:
            break
        # selection weight w(Delta) on Delta>0
        wnum = np.where(pos, x*rho, 0.0)
        Z = wnum.sum()*dx
        if Z <= 0: break
        w = wnum / Z                                   # density, int w dx =1
        E_sel = (x*w)[pos].sum()*dx                    # <Delta^2>_+/<Delta>_+
        # ---- operators ----
        d2 = np.zeros_like(rho)
        d2[1:-1] = (rho[2:] - 2*rho[1:-1] + rho[:-2]) / dx**2
        v = -2.0*(p-1.0)*E_sel                          # drift velocity (<0)
        # upwind for drift term  d rho/dtau += -v * d rho/dx  (conservative: -(v rho)')
        # v<0 -> information moves left; use upwind
        drho = np.zeros_like(rho)
        # flux form: -(v*rho)' with v constant -> -v*rho'
        # central diff is fine with the diffusion present for stability
        drho[1:-1] = -v*(rho[2:]-rho[:-2])/(2*dx)
        sel = np.zeros_like(rho)
        sel[pos] -= w[pos]                              # remove
        sel += (w[::-1])                                # reinsert at -Delta (mirror)
        # careful: reinsertion should only add the mirrored positive part
        selr = np.zeros_like(rho)
        selr[pos] -= w[pos]
        wrev = w[::-1]
        selr[~pos] += wrev[~pos]
        rho = rho + dtau*(Dcoef*d2 + drho + selr)
        rho[rho < 0] = 0.0
        rho[0] = rho[-1] = 0.0
        tau += dtau
    return np.array(taus), np.array(bs)

if __name__ == "__main__":
    import pickle, os
    OUT = "/private/tmp/claude-501/-Users-yotamlifschytz-Desktop-untitled-folder/a51c0110-5f40-4c81-bd87-3e7fbfb22fc3/scratchpad"
    out = {}
    for p in (2, 3, 4, 5):
        tau, b = solve_mf(p)
        tau_end = tau[-1]
        # initial slope
        m = tau < 0.03
        slope0 = np.polyfit(tau[m], b[m], 1)[0]
        out[p] = (tau, b)
        print(f"p={p}: b(0)={b[0]:.4f} slope0={slope0:.3f} (predict -{p})  "
              f"tau_max(b->0)={tau_end:.4f}")
    pickle.dump(out, open(os.path.join(OUT,"mf_curves.pkl"),"wb"))
    print("saved mf curves")
