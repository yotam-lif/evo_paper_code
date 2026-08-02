r"""Diagnostic Figure 3 variant with heavy-tail exponent mu fixed to 0.15.

This is not a refit.  It retains the fitted Gaussian curve and the fitted
heavy-tailed values of n, r, and sigma from ``poster_fig3_fit.json``, changing
only the heavy-tail exponent to mu=0.15.  The REL607 data, gene-specific error
convolution, 5% canonical tail trim, lower fit cutoff, binning, and panel
styling are identical to ``poster_fig3.py``.

Outputs:

    figs_paper/poster_fig3_mu015.pdf
    ../PhD/Posters/GRC_evo_26/poster_fig3_mu015.pdf
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

import poster_fig3_with_errors as base


MU = 0.15
OUT_PDF = os.path.join(base.FIG_DIR, "poster_fig3_mu015.pdf")
POSTER_PDF = (
    "/Users/yotamlifschytz/Desktop/PhD/Posters/"
    "GRC_evo_26/poster_fig3_mu015.pdf"
)


def parameters(values, mu=None):
    return base.ModelParameters(
        n=float(values["n"]),
        r=float(values["r"]),
        sigma=float(values["sigma"]),
        loglik=float(values["loglik"]),
        mu=mu,
    )


def main():
    with open(base.OUT_JSON, encoding="utf-8") as handle:
        saved = json.load(handle)

    all_effects, all_errors = base.rel607_effects_and_errors()
    above_cut = all_effects >= base.LOWER_FIT_CUT
    effects = all_effects[above_cut]
    errors = all_errors[above_cut]
    canonical_effects, canonical_errors = base.trim_canonical_tail(
        effects, errors
    )

    canonical_output = saved["canonical_moment_constrained_mle"]
    heavy_output = saved[
        "heavy_tailed_log_fitness_free_mu_mle"
    ]["with_gene_specific_errors"]

    canonical_fit = parameters(canonical_output["fit"])
    fitted_heavy = heavy_output["fit"]
    heavy_mu015 = parameters(fitted_heavy, mu=MU)

    canonical_likelihood = base.CanonicalGeneErrorLikelihood(
        canonical_effects,
        canonical_errors,
        dx=base.FIT_DX,
        number_error_nodes=base.FINE_ERROR_NODES,
        lower_cut=base.LOWER_FIT_CUT,
    )
    heavy_likelihood = base.HeavyLogGeneErrorLikelihood(
        effects,
        errors,
        dx=base.FIT_DX,
        number_error_nodes=base.FINE_ERROR_NODES,
        lower_cut=base.LOWER_FIT_CUT,
    )

    heavy_mu015.loglik = heavy_likelihood.heavy_loglik(
        n=heavy_mu015.n,
        r=heavy_mu015.r,
        sigma=heavy_mu015.sigma,
        mu=heavy_mu015.mu,
    )
    original_loglik = heavy_likelihood.heavy_loglik(
        n=float(fitted_heavy["n"]),
        r=float(fitted_heavy["r"]),
        sigma=float(fitted_heavy["sigma"]),
        mu=float(fitted_heavy["mu"]),
    )

    heavy_parameter_text = "\n".join([
        rf"$n={heavy_mu015.n:.2f}$",
        rf"$r={heavy_mu015.r:.3f}$",
        rf"$\sigma={heavy_mu015.sigma:.4f}$",
        rf"$\mu={heavy_mu015.mu:.2f}$ (fixed)",
    ])

    os.makedirs(base.FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(POSTER_PDF), exist_ok=True)
    base.plot_figure(
        effects,
        canonical_likelihood,
        heavy_likelihood,
        canonical_fit,
        heavy_mu015,
        canonical_output["confidence_intervals_95"],
        heavy_output["confidence_intervals_95"],
        OUT_PDF,
        heavy_parameter_text=heavy_parameter_text,
    )
    shutil.copyfile(OUT_PDF, POSTER_PDF)

    print(
        "Heavy-tailed diagnostic: "
        f"n={heavy_mu015.n:.9g}, r={heavy_mu015.r:.9g}, "
        f"sigma={heavy_mu015.sigma:.9g}, mu={heavy_mu015.mu:.9g}"
    )
    print(f"logL(mu=0.15, other parameters fixed)={heavy_mu015.loglik:.6f}")
    print(f"logL(original fitted parameters)={original_loglik:.6f}")
    print(f"delta_logL={heavy_mu015.loglik - original_loglik:.6f}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {POSTER_PDF}")


if __name__ == "__main__":
    main()
