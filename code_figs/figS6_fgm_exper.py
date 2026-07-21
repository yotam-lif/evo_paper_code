r"""Fisher's Geometric Model DFE fit to ancestor genotypes -> figS6_exper_bayes.pdf.

For each ancestor DFE (Couce 0K/2K, Limdi REL606/REL607) we fit the
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
See the TRIMS array (one (frac_del, frac_ben) row per dataset).

Outputs:
    figs_paper/figS6_exper_bayes.pdf           per-clone data + moment-locked FGM fit (the figure)
    figs_paper/figS6_sigma_loglik_profile.pdf  per-clone sigma likelihood profile (bimodality check)
    data/fgm_dfe_sigma_profile.json            per-DFE summaries + bootstrap CIs
    data/fgm_dfe_sigma_profile_params.txt      human-readable parameter table

Run from anywhere:  python code_figs/figS6_fgm_exper.py
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_fgm
# Data loaders + the moment-locked sigma-profile fit live in cmn_fgm_exper so the parameter
# table (TableS3_fgm_fit_params.py) shares one implementation with this figure.
from cmn.cmn_fgm_exper import (  # noqa: E402  (shared fit + loaders; figure code stays here)
    load_couce, load_limdi, sigma_profile, bootstrap_sigma_profile,
    _tau, _model_pdf, ORDER, MEAS_ERR, BOOT_B, FLOOR_FRAC_FLAG,
)

DATA = os.path.join(REPO_DIR, "data")
FIGS = os.path.join(REPO_DIR, "figs_paper")
SIGMA_JSON = os.path.join(DATA, "fgm_dfe_sigma_profile.json")
SIGMA_TXT = os.path.join(DATA, "fgm_dfe_sigma_profile_params.txt")
FIG_PATH = os.path.join(FIGS, "figS6_exper_bayes.pdf")
FIG_PROFILE_PATH = os.path.join(FIGS, "figS6_sigma_loglik_profile.pdf")

# House style (matches fig1/fig4/figS3/figS4/figSX_peak_dfe_bayes): sans-serif, large
# axis labels, mid-size ticks/legends.
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 16,
                     "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "legend.fontsize": 14})
TITLE_FS = 16                # per-panel clone/title text
LABEL_FS = 16               # x-axis label
TICK_FS = 16               # tick labels
ANNOT_FS = 12              # in-panel parameter box / annotations
XLABEL_S = r"Fitness effect $(s)$"   # paper convention is "Fitness effect $(\Delta)$"
_CMR = sns.color_palette("CMRmap", 5)
DATA_FILL = (0.5, 0.5, 0.5, 0.35)
MODEL_COLOR = _CMR[2]


# ══════════════════════════════════════════════════════════════════════════════
# Figure: per-clone data histogram + moment-locked FGM fit (figS6_exper_bayes.pdf)
# Display names: Couce 0K is the REL607 ancestor of the Ara+2 line, so it and the Limdi
# REL607 are two measurements of REL607 -> (1)/(2); Couce 2K is the evolved Ara+2 at 2000
# generations.
# ══════════════════════════════════════════════════════════════════════════════
SIGMA_FIG_NAMES = {"Couce 0K": "REL607 (1)", "Couce 2K": "ARA+2 (2K)",
                   "REL606": "REL606", "REL607": "REL607 (2)"}
# Panel layout (2x2, row-major): row 1 = REL607 (1), REL607 (2);
#                                row 2 = ARA+2 (2K), REL606.
SIGMA_FIG_ORDER = ["Couce 0K", "REL607", "Couce 2K", "REL606"]


def plot_ancestors_sigma(results, data_map, order, path):
    """Per-clone panel: data histogram + moment-locked FGM fit, with a parameter box."""
    names = list(SIGMA_FIG_ORDER)
    ncol, nrow = 2, 2
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


def plot_sigma_profiles(profiles, order, path):
    """Per-clone sigma likelihood profile along the moment-locked curve (bimodality check).

    Plots the RELATIVE profile likelihood L(sigma)/L_max vs sigma -- the exact 1-D slice
    sigma_profile maximises (with n, s0, r locked to the sample moments at each sigma). A
    single clean peak => sigma is well identified; two humps / a shoulder => bimodal or a
    flat ridge; a peak jammed against sigma_max (the s0=0 edge) or the small-sigma end
    => the MAP is at a boundary. sigma_hat (MAP) and sigma_max are marked.
    """
    names = list(SIGMA_FIG_ORDER)
    ncol, nrow = 2, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        pr = profiles[name]
        sig, post = pr["sigma"], pr["post"]
        ax.set_title(SIGMA_FIG_NAMES.get(name, name), fontsize=TITLE_FS, color="black")
        ax.set_xlabel(r"$\sigma$", fontsize=LABEL_FS)
        if sig is None or post is None or not np.isfinite(post).any() or post.max() <= 0.0:
            ax.text(0.5, 0.5, "no profile\n(floored / unfit)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=ANNOT_FS, color="0.4")
            ax.set_yticks([])
            continue
        rel = post / post.max()                       # exp(loglik - max loglik): peak = 1
        ax.plot(sig, rel, color=MODEL_COLOR, lw=2.0)
        ax.fill_between(sig, 0.0, rel, color=MODEL_COLOR, alpha=0.12)
        ax.plot(pr["sigma_map"], 1.0, "o", color=MODEL_COLOR, ms=5, zorder=5)  # MAP (razor
        #                                        spikes pinned at an edge stay visible here)
        ax.axvline(pr["sigma_map"], color="k", lw=1.0, ls="--", label=r"$\hat{\sigma}$")
        ax.set_xlim(float(sig.min()), float(sig.max()))   # zoom to the sigma grid so the
        smax = pr["sigma_max"]                             # profile shape fills the panel
        if np.isfinite(smax) and smax > 0.0:
            if smax <= float(sig.max()):                  # sigma_max in range -> draw it
                ax.axvline(smax, color="0.55", lw=0.9, ls=":", label=r"$\sigma_{\max}$")
            else:                                         # off to the right -> annotate
                ax.text(0.97, 0.55, rf"$\sigma_{{\max}}={smax:.2g}\!\rightarrow$",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=ANNOT_FS - 2, color="0.5")
        ax.set_ylabel("rel. likelihood", fontsize=ANNOT_FS)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=ANNOT_FS - 2, loc="upper right", frameon=False)
        for sp in ("top", "right"):
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
        "Generated by code_figs/figS6_fgm_exper.py",
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
    """Moment-locked sigma-profile fit + bootstrap for the ancestors; writes the data + fit
    figure and the per-clone sigma likelihood-profile diagnostic."""
    print(f"1-D sigma profile (moment-locked n,s0,r); bootstrap B={BOOT_B}, eps={MEAS_ERR}")
    print(f"{'ancestor':<11}{'N':>6}{'skew':>7}  {'r [95% boot CI]':<26}{'n':>6}"
          f"{'s0':>9}{'sigma':>9}{'floor%':>8}  id?")
    print("-" * 100)
    results = {}
    profiles = {}
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
        profiles[name] = {"sigma": f.get("_sig"), "post": f.get("_post"),
                          "sigma_map": f["sigma"], "sigma_max": f["sigma_max"]}
    with open(SIGMA_JSON, "w") as fh:
        json.dump({"per_dfe": results,
                   "config": {"meas_err": MEAS_ERR, "boot_B": BOOT_B,
                              "floor_frac_flag": FLOOR_FRAC_FLAG,
                              "method": "sigma_profile_moment_locked"}}, fh, indent=2)
    print(f"\nSaved {SIGMA_JSON}")
    write_sigma_profile_txt(results, order, SIGMA_TXT)
    plot_ancestors_sigma(results, data_map, order, FIG_PATH)
    plot_sigma_profiles(profiles, order, FIG_PROFILE_PATH)
    return results


# Per-dataset tail trims (frac_deleterious, frac_beneficial) -- one row per dataset, in
# ancestor_dfes() order (Couce 0K, Couce 2K, REL606, REL607). Handed straight to the
# cmn_fgm_exper loaders (see _resolve_trims), so the trims live here with the figure.
TRIMS = np.array([
    [0.02, 0.001],   # Couce 0K  -> REL607 (1)
    [0.02, 0.001],   # Couce 2K  -> ARA+2 (2K)
    [0.1, 0.001],   # REL606
    [0.1, 0.001],   # REL607 (2)
])

# TRIMS = np.array([
#     [0.09, 0.001],   # Couce 0K  -> REL607 (1)
#     [0.09, 0.001],   # Couce 2K  -> ARA+2 (2K)
#     [0.15, 0.005],   # REL606
#     [0.16, 0.005],   # REL607 (2)
# ])


def ancestor_dfes(trims=TRIMS):
    """The ancestor DFEs to fit: Couce 0K & 2K, Limdi REL606 & REL607.

    ``trims`` is a 2-D array of (frac_deleterious, frac_beneficial), one row per dataset in
    the order (Couce 0K, Couce 2K, REL606, REL607); each row is passed to the matching
    cmn_fgm_exper loader as its tail trim.
    """
    trims = np.asarray(trims, float)
    couce = dict(load_couce(trim=trims[0:2], labels=("0K", "2K")))
    limdi = load_limdi(populations=["REL606", "REL607"], trim=trims[2:4])
    specs = [("Couce 0K", couce["0K"]), ("Couce 2K", couce["2K"]),
             ("REL606", limdi["REL606"]), ("REL607", limdi["REL607"])]
    return specs


def main():
    os.makedirs(FIGS, exist_ok=True)
    specs = ancestor_dfes()
    order = [name for name, _ in specs]
    data_map = {name: eff for name, eff in specs}
    run_sigma_profile(specs, data_map, order)


if __name__ == "__main__":
    main()
