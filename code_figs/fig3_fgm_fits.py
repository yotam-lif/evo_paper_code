r"""Figure 3: canonical and heavy-tailed FGM fits to two ancestral DFEs.

One row, three panels -- the whole DFE of each ancestor on a single log density axis.

    A  REL606        -- Limdi ancestor, 3488 genes above the cut.
    B  REL607        -- Limdi ancestor, 3497 genes above the cut.
    C  REL607 (DM25) -- Couce 0K ancestor, 13258 segments above the cut.

The three panels share a y axis, so only the leftmost carries tick labels and only the
leftmost carries the legend; the fitted values are repeated in every panel.

Each panel spans the deleterious tail, the bulk and the beneficial tail at once.  A
single bin width cannot do that -- the bulk needs bins the deep tail would leave empty --
so the histogram is adaptive: fine bins are merged left to right until each carries at
least MIN_COUNT observations.  Horizontal bars are the bin extents, so a wide point in
the tail reads as the wide bin it is.

Two models, both fitted by full unbinned maximum likelihood directly to the measured
effects.  Neither is convolved with the published measurement errors, so each curve is a
prediction for the pooled histogram it is drawn against.

    canonical    Fisher's Geometric Model with isotropic Gaussian mutation vectors:
                 delta ~ N(0, sigma^2 I_n), s = -x.delta - ||delta||^2 / 2.
    heavy-tailed The same geometry with a beta-prime radial mixture on the mutation
                 length -- delta = sqrt(q) omega with q = sigma^2 T and
                 T ~ BetaPrime(n/2, mu).  mu = 1/2 recovers a radial Cauchy; the
                 canonical model is the mu -> infinity limit at fixed sigma^2.

Both likelihoods are conditional on the observed effect clearing LOWER_CUT = -0.5, which
is where both assays stop resolving deleterious effects.

Both fits use every retained effect -- no tail trimming -- so the two logliks are
maximised on the same sample and are directly comparable.

Fits are cached in data/fig3_fgm_fits.json; pass --refit to recompute them.

Run from anywhere:  python code_figs/fig3_fgm_fits.py
Output:             figs_paper/fig3_fgm_fits.pdf
"""

import argparse
import json
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_fgm  # noqa: E402
from cmn.cmn_cauchy_fgm import (  # noqa: E402
    cauchy_fgm_dfe_logpdf, cauchy_fgm_dfe_pdf, cauchy_fgm_survival,
)

# ───────────────────────────────────── Style ─────────────────────────────────────
# Typography, spine weight and tick geometry are all kept identical to fig1 so the
# two figures read as one pair.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14

DATA_COLOR = "#666666"
DATA_EDGE = "#4A4A4A"
CANONICAL_COLOR = "#FF4126"
HEAVY_COLOR = "#2166AC"

# ─────────────────────────────── Fit configuration ────────────────────────────────
LOWER_CUT = -0.5             # both assays stop resolving below this
N_BOUNDS = (2.0, 500.0)
R_BOUNDS = (1.0e-5, 5.0)
C_BOUNDS = (1.0e-6, 0.2)     # C = n sigma^2
A_BOUNDS = (1.0e-5, 0.2)     # A = r sigma
MU_BOUNDS = (0.10, 5.0)
PLOT_DX = 1.0e-4
Y_LIMITS = (1.0e-2, 1.0e2)   # shared by all three panels
MAX_INTEGER_EVALUATIONS = 10  # ceiling on the integer-n profile search

FIT_JSON = os.path.join(_REPO_ROOT, "data", "fig3_fgm_fits.json")
OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")


# ─────────────────────────────────── Data loading ─────────────────────────────────
def limdi_effects(population="REL607"):
    effects = cmn_exper.limdi_gene_series(population).to_numpy(float)
    return effects[np.isfinite(effects) & (effects >= LOWER_CUT)]


def couce_effects(timepoint="0K"):
    effects = cmn_exper.load_couce_segment_series(timepoint).to_numpy(float)
    return effects[np.isfinite(effects) & (effects >= LOWER_CUT)]


PLOTTED = ("limdi_REL606", "limdi_REL607", "couce_0K")

DATASETS = {
    "limdi_REL607": {
        "title": "Limdi REL607 ancestor",
        "load": limdi_effects,
        "panel_title": "REL607",
        "xlim": (-0.52, 0.106),
        "fine_bin_width": 0.0025,
        "min_count": 8,
    },
    "couce_0K": {
        "title": "Couce 0K ancestor",
        "load": couce_effects,
        "panel_title": "REL607 (DM25)",
        "xlim": (-0.245, 0.102),
        "fine_bin_width": 0.0020,
        "min_count": 12,
    },
    # REL606 is also the ancestor whose no-error heavy-tailed MLE the REL606 -> Ara-1
    # walks in fig4 are simulated from.
    "limdi_REL606": {
        "title": "Limdi REL606 ancestor",
        "load": lambda: limdi_effects("REL606"),
        "panel_title": "REL606",
        "xlim": (-0.52, 0.106),
        "fine_bin_width": 0.0025,
        "min_count": 8,
    },
    # Fitted but not plotted.  2K is an EVOLVED background, not an ancestor, so it does
    # not belong in a figure of ancestral DFEs -- but the 2K -> 15K walk has to start
    # from the landscape the population was actually on at 2K, and that means its own
    # MLE rather than the 0K one.  It is also the only place in either dataset where the
    # same lineage is fitted twice, so the pair 0K/2K is the one direct check on whether
    # the fitted radius really falls as the population climbs.
    "couce_2K": {
        "title": "Couce 2K background",
        "load": lambda: couce_effects("2K"),
        "panel_title": "ARA+2 2K (DM25)",
        "xlim": (-0.245, 0.102),
        "fine_bin_width": 0.0020,
        "min_count": 12,
    },
}


# ─────────────────────────────────── Likelihoods ──────────────────────────────────
class CanonicalLikelihood:
    """Exact conditional Gaussian-mutation FGM likelihood, no error convolution."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)

    @staticmethod
    def survival(n, r, sigma):
        return float(cmn_fgm.fgm_fitness_survival_many_eps(
            LOWER_CUT, n=n, sigma=sigma, r=r, eps=np.array([0.0]))[0])

    def loglik(self, n, r, sigma):
        log_density = cmn_fgm.fgm_fitness_dfe_logpdf(
            self.effects, n=n, sigma=sigma, r=r)
        survival = self.survival(n=n, r=r, sigma=sigma)
        if (np.any(~np.isfinite(log_density))
                or not np.isfinite(survival) or survival <= 0.0):
            return -np.inf
        return float(np.sum(log_density) - self.effects.size * np.log(survival))

    def pdf(self, grid, n, r, sigma):
        return (cmn_fgm.fgm_fitness_dfe_pdf(grid, n=n, sigma=sigma, r=r)
                / self.survival(n=n, r=r, sigma=sigma))


class HeavyLikelihood:
    """Exact conditional beta-prime radial FGM likelihood, no error convolution."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)

    @staticmethod
    def survival(n, r, sigma, mu):
        return float(cauchy_fgm_survival(
            LOWER_CUT, n=n, sigma=sigma, r=r, eps=0.0, mu=mu))

    def loglik(self, n, r, sigma, mu):
        log_density = cauchy_fgm_dfe_logpdf(
            self.effects, n=n, sigma=sigma, r=r, mu=mu)
        survival = self.survival(n=n, r=r, sigma=sigma, mu=mu)
        if (np.any(~np.isfinite(log_density))
                or not np.isfinite(survival) or survival <= 0.0):
            return -np.inf
        return float(np.sum(log_density) - self.effects.size * np.log(survival))

    def pdf(self, grid, n, r, sigma, mu):
        return (cauchy_fgm_dfe_pdf(grid, n=n, sigma=sigma, r=r, mu=mu)
                / self.survival(n=n, r=r, sigma=sigma, mu=mu))


# ──────────────────────────────────── Optimisers ──────────────────────────────────
# Both models are fitted in log(n, C, A[, mu]) coordinates.  Bounds are used only to
# discover local basins; each interior candidate is then polished without bounds, so the
# reported point is a genuine stationary point and not a boundary artefact.

def unpack(theta, heavy, fixed_n=None):
    """Read a log-coordinate vector.  With ``fixed_n`` set, n is pinned and dropped
    from the vector, which then holds log(C, A[, mu]) alone."""
    values = np.exp(np.asarray(theta, dtype=float))
    if fixed_n is None:
        n, c_scale, a_scale = values[:3]
        rest = values[3:]
    else:
        n, (c_scale, a_scale), rest = float(fixed_n), values[:2], values[2:]
    sigma = np.sqrt(c_scale / n)
    parameters = {"n": float(n), "r": float(a_scale / sigma), "sigma": float(sigma)}
    if heavy:
        parameters["mu"] = float(rest[0])
    return parameters


def bounds_for(heavy, fixed_n=None):
    box = [] if fixed_n is not None else [(np.log(N_BOUNDS[0]), np.log(N_BOUNDS[1]))]
    box += [(np.log(C_BOUNDS[0]), np.log(C_BOUNDS[1])),
            (np.log(A_BOUNDS[0]), np.log(A_BOUNDS[1]))]
    if heavy:
        box.append((np.log(MU_BOUNDS[0]), np.log(MU_BOUNDS[1])))
    return box


def multistart_interior(likelihood, starts, heavy, label, fixed_n=None):
    box = bounds_for(heavy, fixed_n)
    lower, upper = np.asarray(box)[:, 0], np.asarray(box)[:, 1]

    def objective(theta):
        parameters = unpack(theta, heavy, fixed_n)
        if not R_BOUNDS[0] <= parameters["r"] <= R_BOUNDS[1]:
            return 1.0e100
        value = likelihood.loglik(**parameters)
        return -value if np.isfinite(value) else 1.0e100

    def interior(result):
        theta = np.asarray(result.x, dtype=float)
        parameters = unpack(theta, heavy, fixed_n)
        margin = 2.0e-3
        return bool(np.isfinite(result.fun)
                    and np.all(theta > lower + margin)
                    and np.all(theta < upper - margin)
                    and (fixed_n is not None
                         or N_BOUNDS[0] * 1.002 < parameters["n"] < N_BOUNDS[1] / 1.002)
                    and R_BOUNDS[0] * 1.002 < parameters["r"] < R_BOUNDS[1] / 1.002)

    bounded = [minimize(objective, np.clip(np.log(start), lower, upper),
                        method="L-BFGS-B", bounds=box,
                        options={"ftol": 1.0e-12, "gtol": 1.0e-7,
                                 "maxiter": 2000, "maxls": 60})
               for start in starts]
    candidates = [result for result in bounded if interior(result)]
    if not candidates:
        raise RuntimeError(f"{label}: no finite interior maximum was found.")

    polished = [minimize(objective, result.x, method="BFGS",
                         options={"gtol": 2.0e-6, "maxiter": 1500})
                for result in candidates]
    eligible = [result for result in polished if interior(result)] or candidates
    best = min(eligible, key=lambda result: result.fun)

    fit = unpack(best.x, heavy, fixed_n)
    fit["loglik"] = float(-best.fun)
    fit["C_n_sigma2"] = fit["n"] * fit["sigma"] ** 2
    fit["A_r_sigma"] = fit["r"] * fit["sigma"]
    diagnostics = {
        "converged": bool(best.success),
        "message": str(best.message),
        "starts": len(starts),
        "interior_bounded_candidates": len(candidates),
        "interior_polished_candidates": len(
            [result for result in polished if interior(result)]),
        "selection_rule": ("highest interior local maximum from multistart; "
                           "final polish has no parameter bounds"),
    }
    return fit, diagnostics


CANONICAL_STARTS = ((2.2, 3.0e-2, 4.0e-2), (3.0, 2.0e-2, 3.0e-2),
                    (5.0, 1.0e-2, 2.0e-2), (10.0, 5.0e-3, 1.0e-2),
                    (30.0, 5.0e-3, 1.0e-2), (100.0, 5.0e-3, 1.0e-2),
                    (5.0, 1.0e-3, 1.0e-2), (12.0, 2.0e-3, 1.5e-2))

HEAVY_STARTS = ((4.0, 1.0e-4, 4.0e-3, 0.13), (4.0, 1.0e-4, 4.0e-3, 0.25),
                (6.0, 1.2e-4, 2.0e-3, 0.13), (7.0, 1.2e-4, 2.0e-3, 0.21),
                (10.0, 2.5e-4, 4.0e-3, 0.20), (12.0, 2.0e-4, 2.5e-3, 0.35),
                (30.0, 2.0e-4, 2.0e-3, 0.20), (10.0, 9.0e-3, 1.6e-2, 1.30),
                (6.0, 2.6e-3, 1.0e-2, 1.02), (20.0, 5.0e-3, 1.2e-2, 0.80))


def integer_n_fit(likelihood, continuous, heavy, label):
    """Refit with n pinned to each integer neighbouring the continuous MLE.

    n counts phenotypic dimensions, so only integers are realisable.  Pinning it to
    floor and ceil of the continuous optimum and re-maximising the remaining parameters
    at each is a profile likelihood in n; the better of the two is the integer fit.
    """
    seed = [continuous["C_n_sigma2"], continuous["A_r_sigma"]]
    if heavy:
        seed.append(continuous["mu"])
    # The profile at a pinned integer sits right next to the continuous optimum, so a
    # modest spread of perturbations around it is enough to rule out a second basin.
    starts = [tuple(seed)]
    for c_factor in (0.5, 1.0, 2.0):
        for a_factor in (0.7, 1.4):
            for mu_factor in ((0.6, 1.5) if heavy else (1.0,)):
                scaled = [seed[0] * c_factor, seed[1] * a_factor]
                if heavy:
                    scaled.append(seed[2] * mu_factor)
                starts.append(tuple(scaled))

    candidates = {}

    def evaluate(n):
        if n in candidates or not N_BOUNDS[0] <= n <= N_BOUNDS[1]:
            return
        fit, diagnostics = multistart_interior(
            likelihood, starts, heavy, f"{label} at n={n}", fixed_n=n)
        candidates[n] = {"fit": fit, "diagnostics": diagnostics}

    neighbours = sorted({int(np.floor(continuous["n"])), int(np.ceil(continuous["n"]))})
    for n in neighbours:
        evaluate(n)
    if not candidates:
        raise RuntimeError(f"{label}: no integer neighbour of n is inside the bounds.")

    def loglik(n):
        return candidates[n]["fit"]["loglik"]

    # The two neighbours are the answer whenever the continuous search really found the
    # maximum.  When it did not -- it can stall in a poor basin -- both neighbours can
    # sit on the same side of the integer profile's peak, so climb outward from the
    # better of them until neither step up nor step down improves.
    chosen = max(candidates, key=loglik)
    improved = True
    while improved and len(candidates) < MAX_INTEGER_EVALUATIONS:
        improved = False
        for step in (chosen - 1, chosen + 1):
            evaluate(step)
            if step in candidates and loglik(step) > loglik(chosen):
                chosen, improved = step, True

    print(f"  {label} integer n: "
          + ",  ".join(f"n={n} loglik={loglik(n):.2f}" for n in sorted(candidates))
          + f"   -> n={chosen}", flush=True)
    return {
        "continuous_n": continuous["n"],
        "neighbours_of_continuous_n": neighbours,
        "n_tested": sorted(candidates),
        "chosen_n": chosen,
        # True when the integer profile beat the continuous optimum, i.e. the continuous
        # search had stalled below the real maximum.
        "beat_continuous_mle": bool(loglik(chosen) > continuous["loglik"]),
        "candidates": {str(n): entry for n, entry in candidates.items()},
        "fit": candidates[chosen]["fit"],
    }


def fit_dataset(label, config):
    effects = config["load"]()
    print(f"{label}: N={effects.size} "
          f"range=({effects.min():.4f}, {effects.max():.4f})", flush=True)

    canonical_fit, canonical_diagnostics = multistart_interior(
        CanonicalLikelihood(effects), CANONICAL_STARTS,
        heavy=False, label=f"{label} canonical")
    # Seed the heavy search with the canonical solution as well as the fixed starts.
    heavy_starts = list(HEAVY_STARTS) + [(
        canonical_fit["n"], canonical_fit["C_n_sigma2"],
        max(canonical_fit["A_r_sigma"], A_BOUNDS[0]), 0.25)]
    heavy_fit, heavy_diagnostics = multistart_interior(
        HeavyLikelihood(effects), heavy_starts, heavy=True,
        label=f"{label} heavy-tailed")

    print(f"  canonical    n={canonical_fit['n']:.3f}  r={canonical_fit['r']:.4f}  "
          f"sigma={canonical_fit['sigma']:.5f}  loglik={canonical_fit['loglik']:.2f}")
    print(f"  heavy-tailed n={heavy_fit['n']:.3f}  r={heavy_fit['r']:.4f}  "
          f"sigma={heavy_fit['sigma']:.5f}  mu={heavy_fit['mu']:.3f}  "
          f"loglik={heavy_fit['loglik']:.2f}", flush=True)

    # n is a dimension count, so the reported fit is the better of its two integer
    # neighbours, with the remaining parameters re-maximised at each.
    canonical_integer = integer_n_fit(
        CanonicalLikelihood(effects), canonical_fit, heavy=False,
        label=f"{label} canonical")
    heavy_integer = integer_n_fit(
        HeavyLikelihood(effects), heavy_fit, heavy=True,
        label=f"{label} heavy-tailed")

    # Both models are maximised on the same sample, so their logliks compare directly.
    comparison = {
        "sample": "all effects above the cut",
        "N": int(effects.size),
        "canonical_loglik": canonical_integer["fit"]["loglik"],
        "heavy_loglik": heavy_integer["fit"]["loglik"],
        "continuous_n_canonical_loglik": canonical_fit["loglik"],
        "continuous_n_heavy_loglik": heavy_fit["loglik"],
    }
    comparison["heavy_minus_canonical"] = (
        comparison["heavy_loglik"] - comparison["canonical_loglik"])
    print(f"  heavy - canonical at integer n = "
          f"{comparison['heavy_minus_canonical']:+.1f} nats "
          f"(N={comparison['N']})", flush=True)
    return {
        "dataset": {
            "name": config["title"],
            "N": int(effects.size),
            "observed_lower_cut": LOWER_CUT,
            "minimum": float(effects.min()),
            "maximum": float(effects.max()),
        },
        "canonical_full_mle": {"fit": canonical_fit,
                               "diagnostics": canonical_diagnostics},
        "heavy_tailed_full_mle": {"fit": heavy_fit, "diagnostics": heavy_diagnostics},
        "canonical_integer_n": canonical_integer,
        "heavy_tailed_integer_n": heavy_integer,
        "loglik_comparison": comparison,
    }


def load_or_fit(refit):
    """Return the stored fits, fitting only the datasets that are missing.

    Fitting is INCREMENTAL rather than all-or-nothing.  Adding a dataset used to
    invalidate the whole cache, so every already-published number was re-optimised
    alongside the new one -- and a multistart search is not bit-reproducible across
    SciPy versions, so figures and tables that quote these fits could move for a reason
    that has nothing to do with the change being made.  Only ``--refit`` refits
    everything now; otherwise a stored entry is kept exactly as it is.
    """
    stored = None
    if os.path.exists(FIT_JSON):
        with open(FIT_JSON, encoding="utf-8") as handle:
            stored = json.load(handle)
    entries = dict(stored.get("datasets", {})) if (stored and not refit) else {}
    missing = [label for label in DATASETS
               if label not in entries or "canonical_integer_n" not in entries[label]]
    if not missing:
        print(f"Loaded cached fits from {FIT_JSON}")
        return stored
    if entries:
        print(f"Cached: {', '.join(sorted(entries))}; fitting: {', '.join(missing)}")

    started = time.perf_counter()
    for label in missing:
        entries[label] = fit_dataset(label, DATASETS[label])
    payload = {
        "analysis": ("Canonical Gaussian and heavy-tailed beta-prime FGM DFE fits, "
                     "full unbinned MLE, no measurement-error convolution"),
        "likelihood": {
            "measurement_error_convolution": False,
            "conditional_on_s_at_least": LOWER_CUT,
            "lower_tail_trim": None,
        },
        "datasets": entries,
    }
    payload["elapsed_seconds"] = time.perf_counter() - started
    os.makedirs(os.path.dirname(FIT_JSON), exist_ok=True)
    with open(FIT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {FIT_JSON}  (fitted {len(missing)} of {len(DATASETS)} datasets)")
    return payload


# ──────────────────────────────────── Plotting ────────────────────────────────────
def adaptive_histogram(effects, fine_bin_width, min_count):
    """Variable-width histogram: fine in the bulk, merged where the data run out.

    Fine bins of ``fine_bin_width`` are merged left to right until each carries at least
    ``min_count`` observations.  The bulk keeps the fine width -- it has counts to spare
    -- while the deleterious and beneficial tails coarsen until their Poisson errors mean
    something, which is what lets one panel show the whole DFE on a log axis.
    """
    left = fine_bin_width * np.floor(float(effects.min()) / fine_bin_width)
    right = fine_bin_width * np.ceil(float(effects.max()) / fine_bin_width)
    fine_edges = np.arange(left, right + 1.0001 * fine_bin_width, fine_bin_width)
    fine_counts, fine_edges = np.histogram(effects, bins=fine_edges)

    edges, counts, running = [float(fine_edges[0])], [], 0
    for index, count in enumerate(fine_counts):
        running += int(count)
        if running >= min_count:
            edges.append(float(fine_edges[index + 1]))
            counts.append(running)
            running = 0
    if running > 0 and counts:          # fold the short last bin into its neighbour
        counts[-1] += running
        edges[-1] = float(fine_edges[-1])
    elif running > 0:
        edges.append(float(fine_edges[-1]))
        counts.append(running)

    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (effects.size * widths)
    error = np.sqrt(counts) / (effects.size * widths)
    return centers, density, error, counts, widths


def draw_panel(axis, effects, grid, canonical, heavy, config):
    xlim = config["xlim"]
    centers, density, error, counts, widths = adaptive_histogram(
        effects, config["fine_bin_width"], config["min_count"])
    visible = (counts > 0) & (centers >= xlim[0]) & (centers <= xlim[1])
    axis.errorbar(centers[visible], density[visible], yerr=error[visible],
                  xerr=0.5 * widths[visible], fmt="o",
                  ms=4.2, mfc=DATA_COLOR, mec=DATA_EDGE, mew=0.45, ecolor=DATA_COLOR,
                  elinewidth=0.8, capsize=0, alpha=0.9, zorder=5)

    shown = (grid >= xlim[0]) & (grid <= xlim[1])
    axis.plot(grid[shown], canonical[shown], color=CANONICAL_COLOR, lw=2.4, zorder=3)
    axis.plot(grid[shown], heavy[shown], color=HEAVY_COLOR, lw=2.6, zorder=4)

    axis.set_yscale("log")
    axis.set_xlim(*xlim)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_position(("outward", 10))
    axis.spines["left"].set_position(("outward", 10))
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
    axis.tick_params(axis="both", which="major", length=10, width=1.5)
    axis.tick_params(axis="both", which="minor", length=5, width=1.6)


def parameter_block(fit, heavy):
    """Fitted values only -- the colour matches the curve's legend entry."""
    lines = [rf"$n={int(round(fit['n']))}$", rf"$\sigma={fit['sigma']:.2f}$",
             rf"$r={fit['r']:.2f}$"]
    if heavy:
        lines.append(rf"$\mu={fit['mu']:.2f}$")
    return "\n".join(lines)


def build(payload, path):
    figure, axes = plt.subplots(1, 3, figsize=(17.6, 5.6), sharey=True,
                               gridspec_kw={"wspace": 0.13})
    grid = np.arange(-0.56, 0.1201, PLOT_DX)

    blocks = []
    for axis, label in zip(axes, PLOTTED):
        config = DATASETS[label]
        entry = payload["datasets"][label]
        effects = config["load"]()
        canonical_fit = entry["canonical_integer_n"]["fit"]
        heavy_fit = entry["heavy_tailed_integer_n"]["fit"]

        canonical = CanonicalLikelihood(effects).pdf(
            grid, n=canonical_fit["n"], r=canonical_fit["r"],
            sigma=canonical_fit["sigma"])
        heavy = HeavyLikelihood(effects).pdf(
            grid, n=heavy_fit["n"], r=heavy_fit["r"], sigma=heavy_fit["sigma"],
            mu=heavy_fit["mu"])

        draw_panel(axis, effects, grid, canonical, heavy, config)
        blocks.append((axis, heavy_fit, canonical_fit))
        # set_ticks_position("left") undoes the shared-axis label suppression, so the
        # labels have to be switched off again on every panel but the first.
        if axis is not axes[0]:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_xlabel(r"Fitness effect $(s)$")
        axis.set_title(config["panel_title"], pad=10)

    # One shared y range.  Both models dive by decades past the edges of the measured
    # support; the curves are allowed to leave the frame rather than set the scale.
    axes[0].set_ylim(*Y_LIMITS)
    axes[0].set_ylabel("Probability density")

    # The legend goes in the first panel only; the fitted values are repeated in each,
    # in the colour of the curve they belong to, so the blocks need no headings.
    legend = axes[0].legend(
        [Line2D([], [], color=HEAVY_COLOR, lw=2.6),
         Line2D([], [], color=CANONICAL_COLOR, lw=2.4),
         Line2D([], [], marker="o", linestyle="none", ms=4.2, mfc=DATA_COLOR,
                mec=DATA_EDGE)],
        ["HT", "Canonical", "Data"],
        loc="upper left", bbox_to_anchor=(0.012, 0.99), frameon=False,
        fontsize=14, handlelength=2.0, labelspacing=0.42,
        handletextpad=0.7, borderpad=0.0)
    figure.canvas.draw()
    # Value blocks line up across the row, just under where the legend ends.
    below = legend.get_window_extent().transformed(axes[0].transAxes.inverted()).y0
    for axis, heavy_fit, canonical_fit in blocks:
        for offset, (fit, is_heavy, colour) in enumerate((
                (heavy_fit, True, HEAVY_COLOR),
                (canonical_fit, False, CANONICAL_COLOR))):
            axis.text(0.040 + 0.285 * offset, below - 0.035,
                      parameter_block(fit, heavy=is_heavy),
                      transform=axis.transAxes, ha="left", va="top",
                      fontsize=13.5, linespacing=1.35, color=colour)

    # Sit the letters above the frame, level with the titles, so all three share one
    # offset from their panel -- A no longer has to dodge the 10^2 tick label.
    for axis, tag in zip(axes, "ABC"):
        axis.text(-0.075, 1.035, tag, transform=axis.transAxes, ha="left",
                  va="bottom", fontsize=18, fontweight="heavy")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true",
                        help="Recompute the fits instead of using the cached JSON.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = load_or_fit(args.refit)
    build(payload, os.path.join(OUT_DIR, "fig3_fgm_fits.pdf"))


if __name__ == "__main__":
    main()
