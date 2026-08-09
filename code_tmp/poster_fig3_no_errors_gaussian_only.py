r"""Poster Figure 3 without measurement errors or a heavy-tailed fit.

The Gaussian FGM is fit exactly as in ``poster_fig3_no_errors.py``: a full
unbinned MLE of the absolute-fitness DFE after excluding the most deleterious
5% of retained REL607 measurements.  The figure contains only the Gaussian
fit and the data.

Outputs:

    data/poster_fig3_no_errors_gaussian_only_fit.json
    figs_paper/poster_fig3_no_errors_gaussian_only.pdf
"""

from __future__ import annotations

import json
import os
import time

import poster_fig3_no_errors as no_error


base = no_error.base
OUT_PDF = os.path.join(
    base.FIG_DIR, "poster_fig3_no_errors_gaussian_only.pdf"
)
OUT_JSON = os.path.join(
    base.DATA_DIR, "poster_fig3_no_errors_gaussian_only_fit.json"
)


def main():
    started = time.perf_counter()
    os.makedirs(base.FIG_DIR, exist_ok=True)
    os.makedirs(base.DATA_DIR, exist_ok=True)

    all_effects, all_errors = base.rel607_effects_and_errors()
    above_cut = all_effects >= base.LOWER_FIT_CUT
    effects = all_effects[above_cut]
    errors = all_errors[above_cut]
    canonical_effects, _ = base.trim_canonical_tail(effects, errors)

    with open(base.OUT_JSON, encoding="utf-8") as handle:
        saved = json.load(handle)
    saved_canonical = saved["canonical_moment_constrained_mle"]["fit"]

    canonical_likelihood = no_error.CanonicalNoErrorLikelihood(
        canonical_effects
    )
    canonical_fit, canonical_diagnostics = (
        no_error.fit_canonical_full_mle(
            canonical_likelihood, saved_canonical
        )
    )
    canonical_intervals, canonical_ci_diagnostics = (
        no_error.canonical_bootstrap_intervals(
            effects, canonical_diagnostics
        )
    )

    base.plot_figure(
        effects,
        canonical_likelihood,
        None,
        canonical_fit,
        None,
        canonical_intervals,
        None,
        OUT_PDF,
        show_heavy=False,
        x_axis_label=r"Fitness effect $(s)$",
        y_limits=no_error.SHARED_Y_LIMITS,
    )

    elapsed = time.perf_counter() - started
    output = {
        "dataset": {
            "name": "REL607",
            "observed_lower_cut": base.LOWER_FIT_CUT,
            "N": int(effects.size),
            "N_canonical_after_5_percent_trim": int(
                canonical_effects.size
            ),
        },
        "likelihood": {
            "measurement_error_convolution": False,
            "canonical": (
                "full unbinned MLE of exact absolute-fitness DFE"
            ),
            "conditional_on_s_at_least": base.LOWER_FIT_CUT,
            "canonical_lower_tail_trim": base.CANONICAL_LOWER_TRIM,
            "heavy_tailed_fit_included": False,
        },
        "canonical_full_mle": {
            "fit": canonical_fit.serializable(),
            "confidence_intervals_95": canonical_intervals,
            "confidence_interval_diagnostics": (
                canonical_ci_diagnostics
            ),
            "diagnostics": no_error.clean_diagnostics(
                canonical_diagnostics
            ),
        },
        "elapsed_seconds": elapsed,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print(
        "Gaussian no-error full MLE: "
        f"n={canonical_fit.n:.8g}, r={canonical_fit.r:.8g}, "
        f"sigma={canonical_fit.sigma:.8g}, "
        f"logL={canonical_fit.loglik:.8f}"
    )
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
