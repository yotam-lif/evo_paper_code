r"""Fisher's Geometric Model DFE fit to ancestor genotypes -> figS5_exper_bayes.pdf.

For each ancestor DFE (Couce 0K/2K, the Ascensao R's, Limdi REL606/REL607) we fit the
analytic isotropic FGM distribution of fitness effects in LOG-fitness (the selection
coefficient competition assays measure):

    s = log(w(r+delta)/w(r)) = (r^2 - |r+delta|^2)/2,   delta ~ N(0, sigma^2 I_n),

with w(x) = exp(-|x|^2/2), so s = r^2/2 - (sigma^2/2) X, X ~ ncx2(n, r^2/sigma^2). The
three parameters are n (phenotypic dimension), sigma (mutation-step s.d.) and r (distance
to the optimum).

Estimator -- "sigma profile" (moment-locked).  Rather than a fragile 3-D (n,sigma,r)
grid (whose r pins to the support edge = the single most-beneficial gene), we use the FGM
moment identities (alpha=1/2):

    E = -n sigma^2 / 2,   V = sigma^2 (|E| + 2 s0),   s0 = r^2/2,

so the SAMPLE mean+variance fix two of the three parameters for free: given sigma,

    n = 2|E|/sigma^2,   s0 = (V/sigma^2 - |E|)/2,   r = sqrt(2 s0).

Only sigma is inferred -- a 1-D binned-multinomial likelihood along this moment-locked
curve. r is COMPUTED from the moments, never slammed onto the support edge. s0 >= 0 caps
sigma at sigma_max = sqrt(V/|E|) (there s0 = 0 = at the optimum, n = n_e = 2 E^2/V); an
n-floor (N_FLOOR) caps it the other way so n stays >= N_FLOOR. CIs come from a
bootstrap-over-genes. A non-negative sample skew or a large bootstrap floor-fraction flags
a DFE that is not FGM-shaped (the FGM log-fitness DFE is always negatively skewed:
m3 = -8 b^3 n - 24 b^2 s0 < 0). A measurement-error convolution (MEAS_ERR; the model DFE is
convolved with N(0, MEAS_ERR^2) before the likelihood, and the sample variance is
deconvolved) describes the true, de-noised DFE.

Robustness: the strongly-deleterious extremes are mostly lethals / essential-gene knockouts
FGM does not model, so a small lower-tail fraction is dropped; the beneficial tail is kept.
See the TRIM_* config.

Outputs:
    figs_paper/figS5_exper_bayes.pdf       per-clone data + moment-locked FGM fit (the figure)
    data/fgm_dfe_sigma_profile.json        per-DFE summaries + bootstrap CIs
    data/fgm_dfe_sigma_profile_params.txt  human-readable parameter table

Run from anywhere:  python code_figs/figS5_fgm_exper.py
"""
import contextlib
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import fftconvolve
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_fgm

DATA = os.path.join(REPO_DIR, "data")
FIGS = os.path.join(REPO_DIR, "figs_paper")
ASENCAO_DIR = os.path.join(DATA, "asencao_dfe_arrays")
COUCE_DIR = os.path.join(DATA, "alex_code")
LIMDI_CSV = os.path.join(
    DATA, "anurag_data", "Analysis", "Part_3_TnSeq_analysis",
    "Processed_data_for_plotting", "dfe_data_pandas.csv")
SIGMA_JSON = os.path.join(DATA, "fgm_dfe_sigma_profile.json")
SIGMA_TXT = os.path.join(DATA, "fgm_dfe_sigma_profile_params.txt")
FIG_PATH = os.path.join(FIGS, "figS5_exper_bayes.pdf")

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
TRIM_LIMDI = {}                   # per-clone overrides; fallback = TRIM_DEFAULT
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
N_FLOOR = 1.8                # floor on the effective dimension n (caps sigma from above,
#                              since n = 2|E|/sigma^2); keeps n >= N_FLOOR so tau stays finite
N_CAP = 200.0                # cap n (=> small-sigma floor) so the curve stays finite
BOOT_B = 300                 # bootstrap-over-genes resamples for the CIs
BOOT_SEED = 0
FLOOR_FRAC_FLAG = 0.20       # bootstrap floor-fraction above which r is "unidentified"

# House style (matches fig1/fig4/figS3/figS4/figSX_peak_dfe_bayes): sans-serif, large
# axis labels, mid-size ticks/legends.
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13})
TITLE_FS = 15                # per-panel clone/title text
LABEL_FS = 15               # x-axis label
TICK_FS = 12               # tick labels
ANNOT_FS = 11              # in-panel parameter box / annotations
XLABEL_S = r"Fitness effect $(s)$"   # paper convention is "Fitness effect $(\Delta)$"
_CMR = sns.color_palette("CMRmap", 5)
DATA_FILL = (0.5, 0.5, 0.5, 0.35)
MODEL_COLOR = _CMR[2]


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
    """Couce et al. marginal DFEs for the three backgrounds (0K / 2K / 15K).

    fitted1 is the inferred selection coefficient per transposon-insertion site;
    keep abn > 1, drop NaN / failed-fit sentinels, de-duplicate identical fits.
    """
    files = [("0K", "Rfitted_fil.txt"),
             ("2K", "2Kfitted_fil.txt"),
             ("15K", "15Kfitted_fil.txt")]
    out = []
    for label, fname in files:
        t = pd.read_csv(os.path.join(COUCE_DIR, fname), sep="\t").dropna(subset=["fitted1"])
        t = t.drop_duplicates(subset=["fitted1"])
        t = t[t["abn"] > 1]
        v = t["fitted1"].to_numpy(float)
        v = v[v > -100.0]                     # drop -107 failed-fit sentinel
        out.append((label, _trim(v, TRIM_COUCE)))
    return out


def load_asencao():
    """Ascensao et al. DFEs: one array per background (L / R / S) per experiment."""
    out = []
    for d in sorted(os.listdir(ASENCAO_DIR)):
        sub = os.path.join(ASENCAO_DIR, d)
        if not os.path.isdir(sub):
            continue
        for arr in ("L", "R", "S"):
            p = os.path.join(sub, f"{arr}.npy")
            if not os.path.exists(p):
                continue
            v = np.load(p).astype(float)
            v = v[np.isfinite(v)]
            out.append((f"Asc {d} {arr}", _trim(v, TRIM_ASENCAO)))
    return out


def load_limdi():
    """Limdi TnSeq-LTEE DFEs: ``{population: effects}`` for the kept populations.

    Green/Red libraries are pooled per gene per ``POOL_REPLICATES``; each clone gets
    its own per-clone tail trim from ``TRIM_LIMDI`` (fallback ``TRIM_DEFAULT``).
    """
    df = pd.read_csv(LIMDI_CSV).dropna(subset=["Fitness estimate"])
    out = {}
    for pop, sub in df.groupby("Population"):
        pop = str(pop)
        if pop not in ORDER:
            continue
        if POOL_REPLICATES == "mean":
            v = sub.groupby("Genes")["Fitness estimate"].mean().to_numpy(float)
        else:
            v = sub["Fitness estimate"].to_numpy(float)
        v = v[np.isfinite(v)]
        out[pop] = _trim(v, TRIM_LIMDI.get(pop, TRIM_DEFAULT))
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
# Figure: per-clone data histogram + moment-locked FGM fit (figS5_exper_bayes.pdf)
# Display names: Couce 0K is the REL607 ancestor of the Ara+2 line, so it and the Limdi
# REL607 are two measurements of REL607 -> (1)/(2); Couce 2K is the evolved Ara+2 at 2000
# generations. PQT/SLR (not FGM-shaped) are dropped from the figure.
# ══════════════════════════════════════════════════════════════════════════════
SIGMA_FIG_NAMES = {"Couce 0K": "REL607 (1)", "Couce 2K": "ARA+2 (2K)",
                   "Asc GHI R": "GHI R", "Asc MNO R": "MNO R",
                   "REL606": "REL606", "REL607": "REL607 (2)"}
SIGMA_FIG_DROP = ("Asc PQT R", "Asc SLR R")


def plot_ancestors_sigma(results, data_map, order, path):
    """Per-clone panel: data histogram + moment-locked FGM fit, with a parameter box."""
    names = [nm for nm in order if nm not in SIGMA_FIG_DROP]
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        eff = data_map[name]
        e = results[name]
        n, s, r = e["map"]["n"], e["map"]["sigma"], e["map"]["r"]
        b = e["boot"]
        ax.hist(eff, bins=70, density=True, color=DATA_FILL, edgecolor="none", label="Data")
        if np.isfinite(r) and r > 0.0:
            dlo, _ = cmn_fgm.fgm_support(r)
            xs = np.linspace(max(dlo, eff.min()), eff.max(), 600)
            ax.plot(xs, _model_pdf(xs, n, s, r), color=MODEL_COLOR, lw=2.0, label="Fit")
        ax.axvline(0, color="k", lw=0.5, ls=":")

        # parameter box (value [95% bootstrap CI]) in the upper-left corner
        tau = _tau(n, s, r)
        def line(sym, val, ci, fmt):
            return rf"${sym}={val:{fmt}}$ [{ci[0]:{fmt}}, {ci[2]:{fmt}}]"
        txt = "\n".join([
            line("n", n, b["n"], ".2f"),
            line(r"\sigma", s, b["sigma"], ".3f"),
            line("r", r, b["r"], ".3f"),
            line(r"\tau_s", tau, b["tau"], ".2f"),
        ])
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=ANNOT_FS - 2, color="0.15",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
        ax.set_title(SIGMA_FIG_NAMES.get(name, name), fontsize=TITLE_FS, color="black")
        ax.set_xlabel(XLABEL_S, fontsize=LABEL_FS)
        ax.set_yticks([])
        ax.tick_params(labelsize=TICK_FS)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Text export + driver
# ══════════════════════════════════════════════════════════════════════════════
def write_sigma_profile_txt(results, order, path):
    """Human-readable table of the sigma-profile (moment-locked) ancestor fits."""
    out = [
        "Fisher's Geometric Model DFE -- 1-D sigma profile, (n,s0,r) locked to mean+variance",
        "Generated by code_figs/figS5_fgm_exper.py",
        "",
        "Identities (alpha=1/2):  E=-n sigma^2/2,  V=sigma^2(|E|+2 s0),  s0=r^2/2",
        "  => given sigma: n=2|E|/sigma^2, s0=(V/sigma^2-|E|)/2, r=sqrt(2 s0).  Only "
        "sigma is inferred.",
        f"Measurement error: model DFE convolved with N(0, eps^2), eps={MEAS_ERR} "
        f"(variance deconvolved).",
        f"CIs: bootstrap-over-genes (B={BOOT_B}).  sigma_max=sqrt(V/|E|) is the at-optimum "
        f"(s0=0) edge.",
        "id? = NO when sample skew >= 0 or bootstrap floor-fraction > "
        f"{FLOOR_FRAC_FLAG:.0%} (DFE not FGM-shaped; FGM is always negatively skewed).",
        "",
        "=" * 112,
        f"{'ancestor':<11}{'N':>6}{'skew':>7}  {'r [95% boot CI]':<26}{'n':>6}{'n_e':>6}"
        f"{'s0':>9}{'sigma':>9}{'floor%':>8}  id?",
    ]
    for k in order:
        e = results[k]
        mp, b = e["map"], e["boot"]
        rci = b["r"]
        rcol = f"{rci[1]:.3f} [{rci[0]:.3f}, {rci[2]:.3f}]"
        out.append(
            f"{k:<11}{e['data']['N']:>6}{e['data']['skew']:>7.2f}  {rcol:<26}"
            f"{mp['n']:>6.1f}{e['n_e']:>6.2f}{mp['s0']:>9.4f}{mp['sigma']:>9.4f}"
            f"{100 * e['floor_frac']:>7.0f}%  {'yes' if e['identified'] else 'NO'}")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Saved {path}")


def run_sigma_profile(specs, data_map, order):
    """Moment-locked sigma-profile fit + bootstrap for the ancestors; writes the figure."""
    print(f"1-D sigma profile (moment-locked n,s0,r); bootstrap B={BOOT_B}, eps={MEAS_ERR}")
    print(f"{'ancestor':<11}{'N':>6}{'skew':>7}  {'r [95% boot CI]':<26}{'n':>6}"
          f"{'s0':>9}{'sigma':>9}{'floor%':>8}  id?")
    print("-" * 100)
    results = {}
    for name, eff in specs:
        f = sigma_profile(eff, full=True)
        boot, floor_frac = bootstrap_sigma_profile(eff)
        sk = float(skew(eff))
        identified = bool((sk < 0.0) and (floor_frac <= FLOOR_FRAC_FLAG))
        rci = boot["r"]
        print(f"{name:<11}{eff.size:>6}{sk:>7.2f}  "
              f"{f'{rci[1]:.3f} [{rci[0]:.3f}, {rci[2]:.3f}]':<26}"
              f"{f['n']:>6.1f}{f['s0']:>9.4f}{f['sigma']:>9.4f}"
              f"{100 * floor_frac:>7.0f}%  {'yes' if identified else 'NO'}")
        results[name] = {
            "data": {"N": int(eff.size), "skew": sk},
            "E": f["E"], "V": f["V"], "sigma_max": f["sigma_max"], "n_e": f["n_e"],
            "map": {"sigma": f["sigma"], "n": f["n"], "s0": f["s0"], "r": f["r"]},
            "boot": boot, "floor_frac": floor_frac, "identified": identified,
        }
    with open(SIGMA_JSON, "w") as fh:
        json.dump({"per_dfe": results,
                   "config": {"meas_err": MEAS_ERR, "boot_B": BOOT_B,
                              "floor_frac_flag": FLOOR_FRAC_FLAG,
                              "method": "sigma_profile_moment_locked"}}, fh, indent=2)
    print(f"\nSaved {SIGMA_JSON}")
    write_sigma_profile_txt(results, order, SIGMA_TXT)
    plot_ancestors_sigma(results, data_map, order, FIG_PATH)
    return results


def ancestor_dfes():
    """The ancestor DFEs to fit: Couce 0K & 2K, the Ascensao R's, Limdi REL606 & REL607."""
    couce = dict(load_couce())                 # 0K, 2K, 15K
    asc = dict(load_asencao())                 # all L/R/S
    limdi = load_limdi()                        # kept clones (dict)
    specs = [("Couce 0K", couce["0K"]), ("Couce 2K", couce["2K"])]
    specs += [(k, v) for k, v in sorted(asc.items()) if k.endswith(" R")]
    specs += [("REL606", limdi["REL606"]), ("REL607", limdi["REL607"])]
    return specs


def main():
    os.makedirs(FIGS, exist_ok=True)
    specs = ancestor_dfes()
    order = [name for name, _ in specs]
    data_map = {name: eff for name, eff in specs}
    run_sigma_profile(specs, data_map, order)


if __name__ == "__main__":
    main()
