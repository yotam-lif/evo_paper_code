r"""Shared FGM experimental-DFE machinery: data loaders + moment-locked sigma-profile fit.

This module holds everything the FGM experimental-DFE analysis needs that is NOT figure
drawing, so both a figure script and a table script can share one implementation:

    code_figs/figS6_fgm_exper.py       the figure (imports the fit + loaders from here)
    code_figs/TableS3_fgm_fit_params.py the parameter table (same fit + loaders)

Contents:
    * Data loaders -- one cleaned array of fitness effects per DFE (Couce, Ascensao, Limdi).
    * Measurement-error convolution of the model DFE (context manager over cmn_fgm).
    * The "sigma profile" (moment-locked) estimator + its bootstrap-over-genes CIs.

Estimator -- "sigma profile" (moment-locked).  Rather than a fragile 3-D (n,sigma,r) grid
(whose r pins to the support edge = the single most-beneficial gene), we use the FGM moment
identities (alpha=1/2):

    E = -n sigma^2 / 2,   V = sigma^2 (|E| + 2 s0),   s0 = r^2/2,

so the SAMPLE mean+variance fix two of the three parameters for free: given sigma,

    n = 2|E|/sigma^2,   s0 = (V/sigma^2 - |E|)/2,   r = sqrt(2 s0).

Only sigma is inferred -- a 1-D binned-multinomial likelihood along this moment-locked
curve. r is COMPUTED from the moments, never slammed onto the support edge.
"""
import contextlib
import os
import sys

import numpy as np
from scipy.signal import fftconvolve

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_fgm, cmn_exper

# ── tail trimming (per data source) ───────────────────────────────────────────
# The log-fitness FGM DFE has one-sided support s <= s_max = r^2/2. The strongly-
# DELETERIOUS extremes are largely lethals / essential-gene knockouts / noisy large-
# magnitude estimates that FGM is not meant to model; dropping a small lower-tail
# fraction lets the fit match the bulk. The BENEFICIAL tail carries the distance-to-
# optimum signal, so it is kept (bar a tiny Ascensao upper trim for isolated outliers).
# Each entry is (frac_deleterious, frac_beneficial).
TRIM_COUCE = (0.02, 0.001)
TRIM_ASENCAO = (0.007, 0.001)
NBINS = 250                  # multinomial bins over [min(data), max(data)]

# ── Limdi TnSeq-LTEE config ───────────────────────────────────────────────────
# The Limdi knockout DFE per clone = per-gene transposon-insertion fitness effects
# (selection coefficients). Green/Red are two libraries of the SAME clone (Pearson
# ~0.955), averaged per gene. Drop the worst few % of the deleterious tail per clone.
POOL_REPLICATES = "mean"          # "mean" (average Green+Red per gene) | "concat"
TRIM_DEFAULT = (0.1, 0.001)        # per-clone (deleterious frac, beneficial frac)
TRIM_LIMDI = {"REL606": (0.12, 0.005)}   # per-clone overrides; fallback = TRIM_DEFAULT
#                                  REL606: drop the top 1% beneficials (extreme
#                                  most-beneficial genes; see the trim sweep).
MEAS_ERR = 0.005                  # Gaussian measurement-error s.d. on each effect; >0
#                                   convolves the model DFE with N(0, MEAS_ERR^2) and
#                                   deconvolves the sample variance so the fit describes
#                                   the TRUE DFE. See measurement_error(). 0.0 disables.
# Ara-2 (known-anomalous beneficial tail) and Ara+4 (heavy deleterious load) are excluded.
EXCLUDE = {"Ara-2", "Ara+4"}
# Ancestry: Ara- descend from REL606, Ara+ from the Ara+ revertant REL607.
_ANCESTORS_RAW = {"REL606": [f"Ara-{i}" for i in range(1, 7)],
                  "REL607": [f"Ara+{i}" for i in range(1, 7)]}
ORDER = [p for p in (["REL606"] + _ANCESTORS_RAW["REL606"]
                     + ["REL607"] + _ANCESTORS_RAW["REL607"]) if p not in EXCLUDE]

# ── sigma-profile inference (n/s0/r locked to the sample mean+variance) ────────
NSIG_PROFILE = 400           # 1-D sigma grid resolution along the moment-locked curve
N_FLOOR = 1.6                # floor on the effective dimension n (caps sigma from above,
#                              since n = 2|E|/sigma^2); keeps n >= N_FLOOR. Note tau blows up
#                              (-> nan) as n -> 1, since tau1 = 2 r^2 / ((n-1) sigma^2)
N_CAP = 200.0                # cap n (=> small-sigma floor) so the curve stays finite
BOOT_B = 300                 # bootstrap-over-genes resamples for the CIs
BOOT_SEED = 0
FLOOR_FRAC_FLAG = 0.20       # bootstrap floor-fraction above which r is "unidentified"


# ══════════════════════════════════════════════════════════════════════════════
# Data loading -- one cleaned array of fitness effects per DFE
# ══════════════════════════════════════════════════════════════════════════════
def _trim(v, trim):
    """Drop ``trim[0]`` off the deleterious (lower) tail and ``trim[1]`` off the
    beneficial (upper) tail (asymmetric robustness to non-FGM outliers)."""
    frac_del, frac_ben = trim
    lo = np.quantile(v, frac_del) if frac_del > 0.0 else -np.inf
    hi = np.quantile(v, 1.0 - frac_ben) if frac_ben > 0.0 else np.inf
    return v[(v >= lo) & (v <= hi)]


def load_couce():
    """Couce et al. DFEs for the three backgrounds (0K / 2K / 15K), tail-trimmed for FGM."""
    return [(label, _trim(cmn_exper.load_couce_effects(label), TRIM_COUCE))
            for label in ("0K", "2K", "15K")]


def load_asencao():
    """Ascensao et al. DFEs: one trimmed array per background (L / R / S) per experiment."""
    out = []
    for d in cmn_exper.asencao_experiments():
        for arr in ("L", "R", "S"):
            v = cmn_exper.load_asencao_array(d, arr)
            if v is None:
                continue
            v = v[np.isfinite(v)]
            out.append((f"Asc {d} {arr}", _trim(v, TRIM_ASENCAO)))
    return out


def load_limdi(populations=None, trim=None):
    """Limdi TnSeq-LTEE DFEs: ``{population: effects}`` for the kept populations.

    Green/Red libraries are pooled per gene per ``POOL_REPLICATES``; each clone gets
    its own per-clone tail trim from ``TRIM_LIMDI`` (fallback ``TRIM_DEFAULT``).

    ``populations`` selects which population labels to keep (defaults to ``ORDER``, which
    applies the figure's Ara-2/Ara+4 exclusion). Pass an explicit list to override the
    exclusion, e.g. the full LTEE panel used by the p/N table.

    ``trim`` overrides the per-clone tail trim for ALL requested clones: pass a
    ``(frac_deleterious, frac_beneficial)`` tuple, or ``(0.0, 0.0)`` to keep the raw
    DFE with both tails intact. ``None`` (default) uses the per-clone TRIM_LIMDI/TRIM_DEFAULT.
    """
    keep = ORDER if populations is None else list(populations)
    frame = cmn_exper.load_limdi_frame()
    present = set(frame["Population"].astype(str))
    out = {}
    for pop in keep:
        if pop not in present:
            continue
        if POOL_REPLICATES == "mean":
            v = cmn_exper.limdi_gene_series(frame, pop).to_numpy(float)
        else:
            v = frame.loc[frame["Population"] == pop, "Fitness estimate"].to_numpy(float)
        v = v[np.isfinite(v)]
        t = trim if trim is not None else TRIM_LIMDI.get(pop, TRIM_DEFAULT)
        out[pop] = _trim(v, t)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Measurement error: convolve the FGM model DFE with N(0, eps^2) noise.
# Builds the model's fine-bin pmf (cmn_fgm.fgm_bin_probs), convolves each grid row with
# a Gaussian kernel on the uniform bin grid, then takes the multinomial log-likelihood.
# Activated by swapping cmn_fgm.fgm_bin_loglik. eps<=0 is a no-op.
# ══════════════════════════════════════════════════════════════════════════════
def _make_noisy_bin_loglik(eps):
    def noisy(counts, edges, n, sigma, r):
        counts = np.asarray(counts, float)
        edges = np.asarray(edges, float)
        P = cmn_fgm.fgm_bin_probs(edges, n, sigma, r)        # (G, B) model pmf
        dx = float(edges[1] - edges[0])                      # uniform bins (linspace)
        J = int(np.ceil(5.0 * eps / dx))
        off = np.arange(-J, J + 1) * dx
        ker = np.exp(-0.5 * (off / eps) ** 2)
        ker /= ker.sum()
        Pc = np.clip(fftconvolve(P, ker[None, :], mode="same", axes=1), 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            logP = np.where(Pc > 0.0, np.log(Pc), -np.inf)
            contrib = np.where(counts[None, :] > 0.0, counts[None, :] * logP, 0.0)
        return contrib.sum(axis=1)
    return noisy


@contextlib.contextmanager
def measurement_error(eps):
    """Within the block, fit the FGM DFE *plus* N(0, eps^2) measurement noise."""
    if not eps:
        yield
        return
    orig_ll = cmn_fgm.fgm_bin_loglik
    cmn_fgm.fgm_bin_loglik = _make_noisy_bin_loglik(eps)
    try:
        yield
    finally:
        cmn_fgm.fgm_bin_loglik = orig_ll


def _prep(effects):
    """Fine histogram (counts, edges) -- the binned-multinomial likelihood inputs."""
    edges = np.linspace(float(effects.min()), float(effects.max()), NBINS + 1)
    counts, _ = np.histogram(effects, bins=edges)
    return {"edges": edges, "counts": counts}


# ══════════════════════════════════════════════════════════════════════════════
# sigma profile: infer only sigma; lock (n, s0, r) to the sample mean+variance.
# ══════════════════════════════════════════════════════════════════════════════
def _tau(n, sigma, r):
    """Combined timescale, the harmonic sum of two scales:

        tau^-1 = (2 r^2 / ((n-1) sigma^2))^-1 + (sqrt(2/pi) * r/sigma)^-1.

    Returns nan if any input is non-finite, sigma/r <= 0, or n -> 1 (first scale blows up).
    """
    if not (np.isfinite(n) and np.isfinite(sigma) and np.isfinite(r)) \
            or sigma <= 0.0 or r <= 0.0:
        return float("nan")
    denom1 = (n - 1.0) * sigma * sigma
    if abs(denom1) < 1e-12:
        return float("nan")
    tau1 = 2.0 * r * r / denom1                          # diffusive / curvature scale
    tau2 = np.sqrt(2.0 / np.pi) * r / sigma               # ballistic / drift scale
    inv = 1.0 / tau1 + 1.0 / tau2
    return 1.0 / inv if abs(inv) > 1e-12 else float("nan")


def _model_pdf(xs, n, s, r):
    """FGM density on ``xs``, convolved with N(0, MEAS_ERR^2) when MEAS_ERR>0 so the
    plotted curve matches the (noisy) data the fit was made against."""
    pdf = cmn_fgm.fgm_dfe_pdf(xs, n, s, r)
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    if MEAS_ERR > 0.0 and np.asarray(xs).size > 1:
        dx = float(xs[1] - xs[0])
        J = int(np.ceil(5.0 * MEAS_ERR / dx))
        off = np.arange(-J, J + 1) * dx
        ker = np.exp(-0.5 * (off / MEAS_ERR) ** 2)
        ker /= ker.sum()
        pdf = np.convolve(pdf, ker, mode="same")
    return pdf


def _moment_locked(sigma, absE, V):
    """(n, s0, r) locked to mean+var at given sigma (vectorised over sigma)."""
    n = 2.0 * absE / (sigma * sigma)
    s0 = np.clip(0.5 * (V / (sigma * sigma) - absE), 0.0, None)
    return n, s0, np.sqrt(2.0 * s0)


def sigma_profile(effects, eps=None, full=False):
    """1-D sigma posterior along the moment-locked curve for one DFE.

    Only sigma is inferred; n, s0, r follow from the sample mean+variance. Returns the
    MAP (sigma, n, s0, r), the moment summaries (E, V, sigma_max, n_e) and a ``floor``
    flag (MAP pinned at the small-sigma end -> DFE too symmetric for FGM to place it).
    ``eps`` defaults to MEAS_ERR; the model variance is deconvolved (V_true=V_obs-eps^2).
    """
    if eps is None:
        eps = MEAS_ERR
    e = np.asarray(effects, float)
    E = float(e.mean())
    absE = abs(E)
    V = max(float(e.var()) - eps * eps, 1e-12)          # deconvolve measurement error
    sig_max = float(np.sqrt(V / absE)) if absE > 0.0 else 0.0
    out = {"E": E, "V": V, "sigma_max": sig_max,
           "n_e": float(2.0 * E * E / V) if V > 0.0 else float("nan"),
           "sigma": float("nan"), "n": float("nan"), "s0": float("nan"),
           "r": float("nan"), "floor": True}
    if not (absE > 0.0 and sig_max > 0.0):
        return out
    # cap sigma from above so n = 2|E|/sigma^2 >= N_FLOOR (also never exceed sigma_max,
    # the at-optimum s0=0 edge). The small-sigma end (n -> N_CAP) is unchanged, so only
    # the lower-n boundary moves. The MAP then lives in n in [N_FLOOR, N_CAP].
    sig_hi = sig_max
    if N_FLOOR > 0.0:
        sig_hi = min(sig_hi, float(np.sqrt(2.0 * absE / N_FLOOR)))
    sig_lo = max(sig_max / 12.0, float(np.sqrt(2.0 * absE / N_CAP)))
    if sig_lo >= sig_hi:
        sig_lo = sig_hi / 12.0
    sig = np.linspace(sig_lo, sig_hi, NSIG_PROFILE)
    n, s0, r = _moment_locked(sig, absE, V)
    p = _prep(e)
    with measurement_error(eps):
        ll = cmn_fgm.fgm_bin_loglik(p["counts"], p["edges"], n, sig, r)
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    if not np.isfinite(ll).any():
        return out
    imap = int(np.argmax(ll))
    post = np.exp(ll - ll[imap])
    post = np.where(np.isfinite(post), post, 0.0)
    tot = post.sum()
    post = post / tot if tot > 0.0 else np.full_like(post, 1.0 / post.size)
    out.update({"sigma": float(sig[imap]), "n": float(n[imap]),
                "s0": float(s0[imap]), "r": float(r[imap]), "floor": imap <= 1})
    if full:
        out["_sig"], out["_post"], out["_r"], out["_s0"] = sig, post, r, s0
    return out


def bootstrap_sigma_profile(effects, B=BOOT_B, seed=BOOT_SEED, eps=None):
    """Bootstrap-over-genes CIs for the sigma-profile estimator.

    Resamples genes with replacement, re-runs ``sigma_profile`` (recomputing the sample
    moments each time, so the CIs include moment sampling error). Returns
    ``{param: [2.5, 50, 97.5] percentiles}`` (params: sigma, n, s0, r, tau) and the
    fraction of resamples pinned at the small-sigma floor (the identifiability signal).
    """
    rng = np.random.default_rng(seed)
    e = np.asarray(effects, float)
    keys = ("sigma", "n", "s0", "r")
    acc = {k: [] for k in keys}
    acc["tau"] = []
    floors = []
    for _ in range(B):
        s = e[rng.integers(0, e.size, e.size)]
        f = sigma_profile(s, eps=eps)
        if not np.isfinite(f["r"]):
            floors.append(True)
            continue
        for k in keys:
            acc[k].append(f[k])
        t = _tau(f["n"], f["sigma"], f["r"])
        if np.isfinite(t):
            acc["tau"].append(t)
        floors.append(bool(f["floor"]))

    def pct(a):
        return [float(np.percentile(a, q)) for q in (2.5, 50.0, 97.5)] if a \
            else [float("nan")] * 3
    return ({k: pct(acc[k]) for k in (*keys, "tau")},
            float(np.mean(floors)) if floors else 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# moment-prior Bayesian estimator: reparametrize the DFE by its moments (E, V, sigma).
# The moment-locked sigma_profile is the rho->0 (delta-prior) limit of this. Here E and V
# float near their sample values (Gaussian priors, std = rho*|measured|) and sigma gets a
# half-normal (scale = sigma_max), so r is NO LONGER rigidly slaved to the sample variance.
# Each (E, V, sigma) maps to an FGM triple by the same inversion the sigma_profile uses:
#     n = 2|E|/sigma^2,  s0 = (V/sigma^2 - |E|)/2,  r = sqrt(2 s0),
# so the same fgm_bin_loglik (ε-convolved) scores it. Uncertainty is read straight off the
# normalized 3-D posterior grid (MAP + marginal credible intervals) -- no bootstrap.
# ══════════════════════════════════════════════════════════════════════════════
NE_GRID = 41                 # E-axis grid points (around the sample mean)
NV_GRID = 41                 # V-axis grid points (around the sample variance)
NSIG_GRID = 160              # sigma-axis grid points
PRIOR_SPAN = 4.0             # grid half-width in units of the prior std
RHO_DEFAULT = 0.2            # shared relative prior width (std = rho*|measured|)


def _tau_grid(n, sigma, r):
    """Vectorized :func:`_tau`; nan where sigma<=0, r<=0, non-finite, or n->1."""
    n, sigma, r = np.broadcast_arrays(np.asarray(n, float), np.asarray(sigma, float),
                                      np.asarray(r, float))
    out = np.full(n.shape, np.nan)
    denom1 = (n - 1.0) * sigma * sigma
    ok = (np.isfinite(n) & np.isfinite(sigma) & np.isfinite(r)
          & (sigma > 0.0) & (r > 0.0) & (np.abs(denom1) > 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        tau1 = 2.0 * r * r / denom1                       # diffusive / curvature scale
        tau2 = np.sqrt(2.0 / np.pi) * r / sigma            # ballistic / drift scale
        inv = 1.0 / tau1 + 1.0 / tau2
        val = np.where(np.abs(inv) > 1e-12, 1.0 / inv, np.nan)
    out[ok] = val[ok]
    return out


def _weighted_pctl(vals, w, qs=(2.5, 50.0, 97.5)):
    """Weighted percentiles of ``vals`` with weights ``w`` (flattened; non-finite/0-weight
    entries dropped). Used for posterior marginal credible intervals off the grid."""
    vals = np.asarray(vals, float).ravel()
    w = np.asarray(w, float).ravel()
    m = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not m.any():
        return [float("nan")] * len(qs)
    vals, w = vals[m], w[m]
    order = np.argsort(vals)
    vals, cw = vals[order], np.cumsum(w[order])
    cw = cw / cw[-1]
    return [float(np.interp(q / 100.0, cw, vals)) for q in qs]


def moment_prior_map(effects, rho=RHO_DEFAULT, eps=None, full=False):
    """MAP + posterior credible intervals of the FGM DFE under moment priors.

    Reparametrizes by (E, V, sigma): given a triple, n=2|E|/sigma^2,
    s0=(V/sigma^2-|E|)/2, r=sqrt(2 s0). Priors: E~N(Ehat,(rho|Ehat|)^2),
    V~N(Vhat,(rho Vhat)^2) with Ehat, Vhat the sample mean and (ε-deconvolved) variance,
    and sigma~HalfNormal(scale=sigma_max=sqrt(Vhat/|Ehat|)). The MAP and marginal
    2.5/50/97.5 credible intervals of (n, sigma, r, s0, tau) are read off the normalized
    3-D posterior grid. ``eps`` defaults to MEAS_ERR (sample variance deconvolved). The
    ``edge`` flag is True when the MAP sits on a grid boundary (grid too narrow).
    """
    if eps is None:
        eps = MEAS_ERR
    e = np.asarray(effects, float)
    Ehat = float(e.mean())
    absEhat = abs(Ehat)
    Vhat = max(float(e.var()) - eps * eps, 1e-12)          # deconvolve measurement error
    sig_max = float(np.sqrt(Vhat / absEhat)) if absEhat > 0.0 else 0.0
    out = {"E": Ehat, "V": Vhat, "sigma_max": sig_max, "rho": float(rho),
           "n_e": float(2.0 * Ehat * Ehat / Vhat) if Vhat > 0.0 else float("nan"),
           "map": {k: float("nan") for k in ("E", "V", "sigma", "n", "s0", "r")},
           "ci": {k: [float("nan")] * 3 for k in ("n", "sigma", "r", "s0", "tau")},
           "edge": True}
    if not (absEhat > 0.0 and sig_max > 0.0 and rho > 0.0):
        return out

    # ── grids (E kept strictly negative; V kept positive) ──────────────────────
    sdE, sdV = rho * absEhat, rho * Vhat
    Egrid = np.linspace(Ehat - PRIOR_SPAN * sdE, min(Ehat + PRIOR_SPAN * sdE, -1e-9), NE_GRID)
    Vgrid = np.linspace(max(Vhat - PRIOR_SPAN * sdV, 1e-12), Vhat + PRIOR_SPAN * sdV, NV_GRID)
    sig_lo = max(sig_max / 12.0, float(np.sqrt(2.0 * absEhat / N_CAP)))
    sig_hi = sig_max * 1.05                                 # sigma_max is the s0=0 edge
    if sig_lo >= sig_hi:
        sig_lo = sig_hi / 12.0
    Sgrid = np.linspace(sig_lo, sig_hi, NSIG_GRID)

    # (E, V) plane, shared across sigma slices
    absE_col = -Egrid                                      # |E| on the negative E grid
    EE, VV = np.meshgrid(absE_col, Vgrid, indexing="ij")   # (NE, NV)
    Eg2, Vg2 = np.meshgrid(Egrid, Vgrid, indexing="ij")
    lp_EV = -0.5 * ((Eg2 - Ehat) / sdE) ** 2 - 0.5 * ((Vg2 - Vhat) / sdV) ** 2

    p = _prep(e)
    logpost = np.full((NE_GRID, NV_GRID, NSIG_GRID), -np.inf)
    with measurement_error(eps):
        for k, sig in enumerate(Sgrid):
            s2 = sig * sig
            n = 2.0 * EE / s2
            s0 = 0.5 * (VV / s2 - EE)
            valid = (s0 >= 0.0) & (n >= N_FLOOR) & (n <= N_CAP)
            if not valid.any():
                continue
            r = np.sqrt(2.0 * np.clip(s0, 0.0, None))
            ll = np.full(EE.shape, -np.inf)
            nv = n[valid]
            llv = cmn_fgm.fgm_bin_loglik(p["counts"], p["edges"], nv,
                                         np.full(nv.shape, sig), r[valid])
            ll[valid] = np.where(np.isfinite(llv), llv, -np.inf)
            lp_sig = -0.5 * (sig / sig_max) ** 2           # half-normal, scale = sigma_max
            logpost[:, :, k] = ll + lp_EV + lp_sig
    if not np.isfinite(logpost).any():
        return out

    iE, iV, iS = np.unravel_index(int(np.argmax(logpost)), logpost.shape)
    Emap, Vmap, smap = float(Egrid[iE]), float(Vgrid[iV]), float(Sgrid[iS])
    absEmap = -Emap
    nmap = 2.0 * absEmap / (smap * smap)
    s0map = max(0.5 * (Vmap / (smap * smap) - absEmap), 0.0)
    rmap = float(np.sqrt(2.0 * s0map))
    out["map"] = {"E": Emap, "V": Vmap, "sigma": smap, "n": float(nmap),
                  "s0": float(s0map), "r": rmap}
    out["edge"] = bool(iS in (0, NSIG_GRID - 1) or iE in (0, NE_GRID - 1)
                       or iV in (0, NV_GRID - 1))

    # ── marginal credible intervals off the normalized posterior ───────────────
    post = np.exp(logpost - logpost.max())
    post = np.where(np.isfinite(post), post, 0.0)
    absE3 = (-Egrid)[:, None, None]
    V3 = Vgrid[None, :, None]
    S3 = Sgrid[None, None, :]
    n3 = 2.0 * absE3 / (S3 * S3)
    s03 = np.clip(0.5 * (V3 / (S3 * S3) - absE3), 0.0, None)
    r3 = np.sqrt(2.0 * s03)
    quant = {"n": np.broadcast_to(n3, post.shape),
             "sigma": np.broadcast_to(S3, post.shape),
             "s0": np.broadcast_to(s03, post.shape),
             "r": np.broadcast_to(r3, post.shape),
             "tau": _tau_grid(n3, S3, r3)}
    out["ci"] = {k: _weighted_pctl(v, post) for k, v in quant.items()}
    if full:
        out["_grids"] = {"E": Egrid, "V": Vgrid, "sigma": Sgrid}
        out["_logpost"] = logpost
    return out
