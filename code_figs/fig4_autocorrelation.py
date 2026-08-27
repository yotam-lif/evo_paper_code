r"""Figure 4: simulated Pearson autocorrelation along an adaptive walk, against experiment.

Two panels stacked vertically, sharing an x-axis of fixed background mutations.

    A  Couce Ara+2, 0K -> 15K, assayed in DM25.
    B  Limdi REL606 -> Ara-1 at 50K, assayed in LB.

Each panel shows SSWM adaptive walks in a heavy-tailed (radial beta-prime) FGM whose
parameters were fitted to that ancestor's DFE *without* measurement-error convolution --
data/fig3_fgm_fits.json for Couce, the REL606 no-error MLE for Limdi.  Two probe subsets are
drawn per panel: all probes (r100) and the probes surviving a cut of the largest-|s| ancestral
effects, defined once from the noisy ancestral measurement and then held fixed.

The cut follows cmn_scatter, so each panel matches its own scatter panels in fig1 and figs
S1-S4: 2% for the Couce data, whose effects are compact enough that a 10% cut would reach
inside the bulk, and 10% for the Limdi data, whose effects run out to |s| = 0.65.  Hence r98
in panel A and r90 in panel B.  Both rank on |s| and drop the LARGEST-magnitude fraction, not
on the signed effect -- the caches used here (``*_absmag.npz``) were regenerated under that
rule so the measured dots and the curves they sit on share one definition.  The caches carry
a third, unused 10% cut for Couce and 2% for Limdi.

Curves are smoothed for display with a Gaussian filter (sigma = 1.6 steps), with BOTH
endpoints pinned to their raw values -- t = 0 because r = 1 there by construction, and the
last step because that is where the measured dots sit and where the plateau is read off.
This is cosmetic and rounds the corners of the steep stretch around t = 10-16.  It is defensible because the step-to-step wiggle it removes is
Monte-Carlo noise and nothing else: splitting the 500 walks into two disjoint halves gives
medians that disagree by 0.01-0.02 at exactly the steps where the kinks appear, and the rms
second difference of the raw curves (0.005 for r100, 0.015 for r90) is BELOW what sampling
noise in the median alone would produce (0.011 and 0.022).  The honest fix is more walks --
the noise falls as 1/sqrt(N) -- which means regenerating the caches, not editing this script.

Solid curves are the latent correlation -- the model's own effects, with no measurement
error anywhere.  Dashed curves add rank-matched measurement noise: each simulated mutation
is assigned the published error of the empirical gene at its own effect rank, drawn fresh at
every step for the endpoint and once per replicate for the ancestor.  The band is the
16-84% interval over walk x noise replicates.  Filled dots are the measured correlations.

Where the dots sit
------------------
Panel A's dots sit at 22 fixed mutations, the count the 0K -> 15K walk cache was built around.
(The main text quotes roughly 8 fixed mutations for the 0-2K interval and 22 for 2K-15K, so
this is the later interval rather than the cumulative total; keep the two consistent when the
caption is written.)  Panel B's dots sit at the right-hand edge, labelled with the substitution count they really
correspond to, so that their position is not read as a claim about how many mutations fixed.  That is not a substitution count: Ara-1 is a point-mutator
that carries of order a thousand mutations by 50K, almost all of them hitchhikers that SSWM
would never fix.  Fourteen is inside the range over which the simulated walk reaches its
plateau, so the dots are plateau references placed where the curves have levelled off.

The two panels stop at different times.  The Couce cache holds a walk at its peak once it runs
out of beneficial mutations, so all 500 walks contribute at every step.  The Limdi cache writes
NaN instead, and those walks peak after a median of 19 steps, so beyond about 15 the median is
taken over a shrinking and increasingly atypical set of survivors and starts to rattle.  Panel
B therefore stops at 15, where 446 of the 500 walks are still going.

One asymmetry is inherited from the two simulation pipelines and is not a choice made here:
Couce's probe library is ascertained on the observed 0K effect window, Limdi's on the latent
ancestral effect clearing the -0.5 assay cut.

Reads cached walks from data/FGM_HEAVY_TAILED; it does not re-run them.

Run from anywhere:  python code_figs/fig4_autocorrelation.py
Output:             figs_paper/fig4_autocorrelation.pdf
"""

import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from scipy.ndimage import gaussian_filter1d

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from code_figs import TableS1_limdi_autocorr as table_s1  # noqa: E402

# ───────────────────────────────────── Style ─────────────────────────────────────
# Typography, spine weight and tick geometry are all kept identical to fig1 and fig3
# so the three figures read as one set.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14

# The two band colours of fig1's row 2, with the same meaning: the all-pairs r in the colour
# of the points its subset adds, the retained-subset r in the colour of the bulk it covers.
CURVE_COLORS = (cmn_scatter.EXCLUDED_COLOR, cmn_scatter.RETAINED_COLOR)
# The cached cut axis is (0%, 2%, 10%) of the largest |s| dropped.  Each panel names the two
# columns it draws as (cache index, dropped fraction); the third is loaded but not shown.
CACHED_EXCLUSIONS = (0.00, 0.02, 0.10)

# Measured markers: triangles, area in points^2.  Line2D takes a diameter instead, so the
# legend handle is sized as sqrt(area) to match what the panels draw.
MEASURED_MARKER_AREA = 210.0

# Display smoothing -- see the note in the module docstring on why this is cosmetic only.
# A Gaussian kernel rather than a Savitzky-Golay fit: it has no polynomial to overshoot with
# and gives a visibly cleaner line, at the cost of rounding the corners of the steep stretch
# around t = 10-16 rather than tracking them.  Lower SMOOTH_SIGMA to about 1.0 to stay closer
# to the raw medians.
SMOOTH_SIGMA = 1.6


def smooth(trace):
    """Gaussian filter along the step axis, per cut, with BOTH endpoints held exact.

    The kernel has no data past either end of the trace, so both endpoints are where it is
    least trustworthy -- and both are where the figure makes a claim: t = 0 is r = 1 by
    construction, and the last step is where the measured dots sit and where the plateau is
    read off.

    Left alone when the trace carries a NaN, which is what a cache whose walks have run out
    of survivors looks like; the filter would spread that NaN across the whole kernel.
    """
    if not np.isfinite(trace).all():
        return trace
    smoothed = gaussian_filter1d(trace, SMOOTH_SIGMA, axis=0, mode="nearest")
    smoothed[0] = trace[0]
    smoothed[-1] = trace[-1]
    return smoothed

DATA_DIR = os.path.join(_REPO_ROOT, "data", "FGM_HEAVY_TAILED")
OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

PANELS = (
    {
        "cache": ("poster_fig5_couce_0K_to_15K_beta_prime_observed_window_"
                  "rank_noise_k1_w500_e10_m8429_absmag.npz"),
        "title": "ARA+2 (DM25)",
        # Couce effects are compact, so cmn_scatter drops 2% -- see SHALLOW_MAGNITUDE_EXCLUSIONS.
        "cuts": ((0, 0.00), (1, 0.02)),
        # Terminated walks are held at their peak, so all 500 contribute throughout.
        "display_steps": 25,
        # The fixed-mutation count the 0K -> 15K cache was built around.
        "markers": ({"time": 22, "pair": "couce_0K_15K"},),
    },
    {
        "cache": ("poster_fig5_limdi_REL606_to_Ara_minus_1_without_errors_"
                  "rank_noise_k1_w500_e10_m3441_absmag.npz"),
        "title": "ARA-1 (LB)",
        # Limdi effects run out to |s| = 0.65, so cmn_scatter drops 10%.
        "cuts": ((0, 0.00), (2, 0.10)),
        # These walks peak after a median of 19 steps; past 15 the median runs
        # out of surviving walks and becomes noise.
        "display_steps": 15,
        # A plateau reference, not a substitution count -- see the module docstring.
        # The note says so on the face of the figure, since the dot's position
        # would otherwise read as a claim that Ara-1 fixed 15 mutations.
        "markers": ({"time": 15, "pair": "limdi_REL606_Ara-1",
                     "note": "Measured at $t = 1100$ $\\rightarrow$"},),
    },
)


# ─────────────────────────────── Measured correlations ────────────────────────────
def couce_pair(ancestor, evolved):
    """Matched Couce segment effects, using the Table S1 gene matching."""
    early = cmn_exper.load_couce_segment_series(ancestor)
    late = cmn_exper.load_couce_segment_series(evolved)
    shared = early.index.intersection(late.index)
    return early.loc[shared].to_numpy(float), late.loc[shared].to_numpy(float)


def empirical_ladder(pair, exclusions):
    """Measured r for each retained fraction, ranked on |ancestral effect| as in fig1.

    The largest-|s| ``exclusions`` fraction is dropped, defined from the ancestor side only so
    the retained subset is not conditioned on the outcome whose correlation is reported.  This
    is the same rule cmn_scatter applies, and the same rule the cached walks were simulated
    under, so a dot and the curve it sits on mean the same thing.
    """
    if pair.startswith("couce_"):
        ancestor, evolved = couce_pair(*pair.split("_")[1:3])
    else:
        _, founder, clone = pair.split("_", 2)
        ancestor, _, evolved, _ = table_s1.limdi_pair(founder, clone)
    finite = np.isfinite(ancestor) & np.isfinite(evolved)
    ancestor, evolved = ancestor[finite], evolved[finite]
    order = np.argsort(np.abs(ancestor), kind="stable")
    ladder = {}
    for excluded in exclusions:
        kept = order[: ancestor.size - int(np.floor(excluded * ancestor.size))]
        ladder[f"r{int(round(100 * (1.0 - excluded)))}"] = float(
            np.corrcoef(ancestor[kept], evolved[kept])[0, 1])
    return ladder


# ────────────────────────────────── Cached walks ──────────────────────────────────
def load_curves(name, last_time):
    """Median latent and observed traces plus the observed 16-84% band.

    Couce caches are ``(walks, steps, cuts)``; Limdi caches carry a leading model axis,
    ``(models, walks, steps, cuts)``, with a single model in it.  Drop that axis so both
    give ``(steps, 3)`` for the medians and ``(walks x noise, steps, 3)`` for the pooled
    observations.
    """
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing walk cache: {path}")
    arrays = np.load(path)
    latent = arrays["latent_correlations"]
    observed = arrays["observed_correlations"]
    if latent.ndim == 4:          # leading model axis
        if latent.shape[0] != 1:
            raise SystemExit(f"{name} holds {latent.shape[0]} models; expected one")
        latent, observed = latent[0], observed[0]
    steps = min(last_time, latent.shape[1] - 1)
    latent = latent[:, : steps + 1, :]
    observed = observed[:, :, : steps + 1, :]
    pooled = observed.reshape(-1, observed.shape[-2], observed.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "times": np.arange(steps + 1),
            "surviving": np.isfinite(latent[:, :, 0]).sum(axis=0),
            "walks": latent.shape[0],
            "latent": smooth(np.nanmedian(latent, axis=0)),
            "observed": smooth(np.nanmedian(pooled, axis=0)),
            "lower": smooth(np.nanquantile(pooled, 0.16, axis=0)),
            "upper": smooth(np.nanquantile(pooled, 0.84, axis=0)),
        }


# ──────────────────────────────────── Plotting ────────────────────────────────────
def draw_panel(axis, curves, ladders, last_time, cuts):
    times = curves["times"]
    columns = [column for column, _ in cuts]
    keys = tuple(f"r{int(round(100 * (1.0 - excluded)))}" for _, excluded in cuts)
    for cut, color in zip(columns, CURVE_COLORS):
        axis.fill_between(times, curves["lower"][:, cut], curves["upper"][:, cut],
                          color=color, alpha=0.12, linewidth=0)
        axis.plot(times, curves["latent"][:, cut], color=color, linewidth=2.7)
        axis.plot(times, curves["observed"][:, cut], color=color, linewidth=2.5,
                  linestyle=(0, (4.0, 2.4)))

    # Only the drawn cuts set the y floor; the third cached cut is not on the figure.
    drawn = np.asarray(columns)
    floor = min(float(np.nanmin(curves["lower"][:, drawn])),
                float(np.nanmin(curves["latent"][:, drawn])))
    for time, ladder, note in ladders:
        if time > last_time:
            raise SystemExit(f"Measured marker at t={time} is outside the {last_time}-step frame")
        axis.axvline(time, color="#777777", linewidth=1.3,
                     linestyle=(0, (2.0, 2.5)), zorder=1)
        for key, color in zip(keys, CURVE_COLORS):
            # clip_on stays off so a marker sitting on the frame edge is drawn whole.
            axis.scatter([time], [ladder[key]], s=MEASURED_MARKER_AREA, marker="^",
                         facecolor=color, edgecolor="white", linewidth=1.0,
                         zorder=7, clip_on=False)
        if note:
            axis.annotate(
                note, xy=(time, max(ladder[key] for key in keys)),
                xytext=(-4, 16), textcoords="offset points", ha="right",
                va="bottom", fontsize=12.5, color="#555555", annotation_clip=False)
        floor = min(floor, min(ladder[key] for key in keys))
    return floor


def build(path):
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 9.6),
                                gridspec_kw={"hspace": 0.30})
    floor = 1.0
    cut_legends = []
    for axis, panel in zip(axes, PANELS):
        curves = load_curves(panel["cache"], panel["display_steps"])
        exclusions = tuple(excluded for _, excluded in panel["cuts"])
        ladders = [(marker["time"], empirical_ladder(marker["pair"], exclusions),
                    marker.get("note"))
                   for marker in panel["markers"]]
        last_time = int(curves["times"][-1])
        floor = min(floor,
                    draw_panel(axis, curves, ladders, last_time, panel["cuts"]))
        axis.set_title(panel["title"], pad=8)
        axis.set_ylabel("Pearson autocorrelation")
        axis.set_xlim(0, last_time)
        spacing = 5 if last_time <= 25 else 10
        axis.xaxis.set_major_locator(
            FixedLocator(list(range(0, last_time + 1, spacing))))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_position(("outward", 10))
        axis.spines["left"].set_position(("outward", 10))
        axis.xaxis.set_ticks_position("bottom")
        axis.yaxis.set_ticks_position("left")
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)
        axis.tick_params(axis="both", which="major", length=10, width=1.5)
        axis.tick_params(axis="both", which="minor", length=5, width=1.6)
        for (time, ladder, _), marker in zip(ladders, panel["markers"]):
            print(f"{panel['title']} {marker['pair']} at t={time}: measured "
                  + ", ".join(f"{key}={value:+.3f}" for key, value in ladder.items())
                  + "   simulated observed "
                  + ", ".join(f"{value:+.3f}" for value in
                              curves["observed"][time][[c for c, _ in panel["cuts"]]]))
        print(f"  walks alive at t={last_time}: "
              f"{curves['surviving'][-1]}/{curves['walks']}")
        # Panel A sets this legend aside for the style key beside it; panel B has the
        # corner to itself.
        cut_legends.append(axis.legend(
            [Line2D([], [], color=color, linewidth=2.7) for color in CURVE_COLORS],
            [rf"$r_{{{int(round(100 * (1.0 - excluded)))}\%}}$"
             for _, excluded in panel["cuts"]],
            loc="lower left", frameon=False, handlelength=2.0, labelspacing=0.35,
            **({"bbox_to_anchor": (0.34, 0.0), "bbox_transform": axis.transAxes}
               if panel is PANELS[0] else {})))

    axes[-1].set_xlabel("Fixed background mutations")
    for axis in axes:
        axis.set_ylim(min(-0.05, np.floor(20.0 * (floor - 0.02)) / 20.0), 1.04)

    # The style key -- what solid, dashed and the triangles mean -- is shown once, in panel A's
    # empty lower-left corner with panel A's colour key immediately to its right.  Panel B needs
    # only its own colour key, because its cut is 10% where panel A's is 2%.  Re-adding panel
    # A's colour key as an artist keeps this second ``legend`` call from replacing it.
    axes[0].add_artist(cut_legends[0])
    axes[0].legend(
        [Line2D([], [], color="#555555", linewidth=2.7),
         Line2D([], [], color="#555555", linewidth=2.5, linestyle=(0, (4.0, 2.4))),
         Line2D([], [], marker="^", linestyle="none",
                markersize=np.sqrt(MEASURED_MARKER_AREA),
                markerfacecolor="#777777", markeredgecolor="white")],
        ["Latent", "Noisy", "Measured"],
        loc="lower left", frameon=False, handlelength=2.4, labelspacing=0.35)

    for axis, tag in zip(axes, "AB"):
        axis.text(-0.13, 1.13, tag, transform=axis.transAxes, ha="left", va="top",
                  fontsize=18, fontweight="heavy")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {path}")


def main():
    build(os.path.join(OUT_DIR, "fig4_autocorrelation.pdf"))


if __name__ == "__main__":
    main()
