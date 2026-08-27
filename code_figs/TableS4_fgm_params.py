#!/usr/bin/env python3
r"""Table S4: fitted FGM parameters for every background the paper simulates from.

Four DFEs, one row each, all fitted by the same full unbinned maximum likelihood as Figure 3
(``code_figs/fig3_fgm_fits.py``, cached in ``data/fig3_fgm_fits.json``).  This table does not
refit anything -- it reads that cache -- so the numbers here and the curves in Figure 3 cannot
drift apart.  Run ``python code_figs/fig3_fgm_fits.py`` first if the cache is missing an entry.

WHAT THE ROWS ARE, AND WHY THESE FOUR.  Every adaptive walk in the paper starts from a fitted
background, and there are exactly four of them:

  REL606 (Limdi, LB)      the founder of the six Ara-N transitions in TableS1
  REL607 (Limdi, LB)      the founder of the six Ara+N transitions in TableS1
  Couce 0K (REL607, DM25) the early side of 0K -> 2K and 0K -> 15K in TableS2
  Couce 2K (DM25)         the early side of 2K -> 15K in TableS2

The two Limdi ancestors are separate rows rather than one because they are separate strains --
REL607 carries the araA marker -- and because they are fitted independently on their own gene
sets, so nothing forces their parameters to agree.  That they land close together is a result,
not a construction.

    COUCE 2K IS NOT AN ANCESTOR.  It is an evolved background 2,000 generations along the
    ARA+2 lineage, fitted for the same reason the ancestors are: the 2K -> 15K walk has to
    start on the landscape the population was actually on at 2K.  Starting it from the 0K fit
    would ask the model to re-traverse 2,000 generations it has already covered, and would
    predict a decay from the wrong radius.  It is also the only background in either dataset
    that is fitted while ALSO being an endpoint of a fitted transition, which makes the 0K/2K
    pair the one direct test of the picture's central geometric claim -- that the population
    moves toward the peak, so the fitted radius should FALL between them.  It does, in both
    models.  Note this is one comparison on one lineage, not a fitted trend.

TWO MODELS, ON THE SAME SAMPLE.

  canonical     Fisher's Geometric Model with isotropic Gaussian mutations,
                delta ~ N(0, sigma^2 I_n) and s = -x.delta - ||delta||^2 / 2.
                Parameters n, r, sigma.
  heavy-tailed  the same geometry with a beta-prime radial mixture on the mutation length:
                delta = sqrt(q) omega with q = sigma^2 T and T ~ BetaPrime(n/2, mu).
                mu = 1/2 is a radial Cauchy; mu -> infinity at fixed sigma^2 recovers the
                canonical model.  Parameters n, r, sigma, mu.

Both are conditional on the observed effect clearing ``s >= -0.5``, which is where both assays
stop resolving deleterious effects, and both use every retained effect with no tail trimming --
so the two logliks are maximised on the same sample and ``dloglik`` compares them directly.
Neither is convolved with the published measurement errors, which is what makes them the right
input to a walk that RE-ADDS those errors afterwards; using an error-aware fit there would count
the measurement noise twice.

    MU IS BELOW 1 FOR BOTH LIMDI ANCESTORS AND ABOVE IT FOR BOTH COUCE BACKGROUNDS, and that
    is not a cosmetic difference.  ``T ~ BetaPrime(n/2, mu)`` has a finite mean only for
    mu > 1, so at mu ~ 0.34-0.39 the Limdi mutation-size distribution has no mean at all.
    Any quantity defined through a mean squared mutation size -- the radial and angular
    scrambling timescales of the main text among them -- is therefore undefined for those two
    rows and is deliberately not tabulated here.  Read n, r and sigma as the parameters of a
    fitted density, not as a mutation size and a distance in the canonical sense.

WHICH n IS REPORTED.  Both.  ``n_cont`` is the continuous optimum and ``n`` is the better of its
two integer neighbours with the remaining parameters re-maximised at each -- a profile likelihood
in n.  The integer is the one to quote, because n counts phenotypic dimensions and only integers
are realisable, and on all four rows it is also the n the walks are simulated at.  ``r``,
``sigma`` and ``mu`` are always taken from the INTEGER fit, never mixed across the two: at fixed
data the parameters trade off strongly against n, so pairing one fit's n with another's sigma
describes no fitted density.

    n IS WEAKLY IDENTIFIED IN THE HEAVY-TAILED MODEL.  These beta-prime fits have a shallow
    profile in n -- the loglik moves by well under a nat across neighbouring integers -- so the
    integer column should be read as the ridge's location, not as a sharp estimate.

WHAT THE WALKS ACTUALLY SIMULATE AT, which is NOT quite the reported integer fit.  The walk
drivers (``code_tmp/poster_fig5_limdi_noise.py``, ``code_tmp/poster_fig5_couce_noise.py``) take
the CONTINUOUS heavy-tailed optimum, round its n to the nearest integer, and carry that fit's r,
sigma and mu unchanged.  The reported columns instead come from the integer refit, where those
three are re-maximised at the pinned n.  The two are the same optimum approached two ways: n
agrees exactly on all four rows and r, sigma and mu differ by at most 1.4%.  They are not
identical, though, so the walk values are tabulated alongside as ``heavy_*_walk`` rather than left
implicit.  Quote the integer refit; it is the self-consistent fitted density.  Read a simulated
curve against ``heavy_*_walk``, which is what produced it.

    data/TableS4_fgm_params.csv
    columns: dataset, background, assay, N, s_min, s_max,
             canon_n_cont, canon_n, canon_r, canon_sigma, canon_loglik,
             heavy_n_cont, heavy_n, heavy_r, heavy_sigma, heavy_mu, heavy_loglik,
             heavy_n_walk, heavy_r_walk, heavy_sigma_walk, heavy_mu_walk,
             dloglik_heavy_minus_canon

Run:
    python code_figs/TableS4_fgm_params.py
    python code_figs/TableS4_fgm_params.py --latex     # a tabular for the SI
"""
import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn.cmn_exper import DATA_DIR  # noqa: E402

FIT_JSON = os.path.join(REPO_DIR, "data", "fig3_fgm_fits.json")
OUT_CSV = os.path.join(DATA_DIR, "TableS4_fgm_params.csv")

COLUMNS = ["dataset", "background", "assay", "N", "s_min", "s_max",
           "canon_n_cont", "canon_n", "canon_r", "canon_sigma", "canon_loglik",
           "heavy_n_cont", "heavy_n", "heavy_r", "heavy_sigma", "heavy_mu",
           "heavy_loglik",
           "heavy_n_walk", "heavy_r_walk", "heavy_sigma_walk", "heavy_mu_walk",
           "dloglik_heavy_minus_canon"]

# The four fitted backgrounds, in the order the paper meets them, with the medium each DFE was
# measured in.  The key is the entry name in fig3_fgm_fits.json; the label is what the row is
# called here, which is the strain rather than the fit-file key.
ROWS = (
    ("limdi_REL606", "Limdi", "REL606", "LB"),
    ("limdi_REL607", "Limdi", "REL607", "LB"),
    ("couce_0K", "Couce", "REL607 (0K)", "DM25"),
    ("couce_2K", "Couce", "ARA+2 (2K)", "DM25"),
)


def load_fits(path=FIT_JSON):
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    entries = stored.get("datasets", {})
    missing = [key for key, *_ in ROWS if key not in entries]
    if missing:
        raise SystemExit(
            f"{path} has no fit for: {', '.join(missing)}.\n"
            "Run  python code_figs/fig3_fgm_fits.py  to add them.")
    return entries


def build_rows(entries):
    """One record per background, integer-n parameters throughout.

    ``dloglik`` is taken at the integer fits, which is the pair actually reported, rather than
    at the continuous optima; the two differ by well under a nat here but the reported number
    should be the one the reported parameters produce.
    """
    rows = []
    for key, dataset, background, assay in ROWS:
        entry = entries[key]
        info = entry["dataset"]
        canonical = entry["canonical_integer_n"]["fit"]
        heavy = entry["heavy_tailed_integer_n"]["fit"]
        # What the walk drivers do with the same file: round the continuous n, keep its
        # r/sigma/mu.  See WHAT THE WALKS ACTUALLY SIMULATE AT in the docstring.
        walk = entry["heavy_tailed_full_mle"]["fit"]
        rows.append({
            "key": key,
            "dataset": dataset,
            "background": background,
            "assay": assay,
            "N": int(info["N"]),
            "s_min": float(info["minimum"]),
            "s_max": float(info["maximum"]),
            "canon_n_cont": float(entry["canonical_full_mle"]["fit"]["n"]),
            "canon_n": int(round(canonical["n"])),
            "canon_r": float(canonical["r"]),
            "canon_sigma": float(canonical["sigma"]),
            "canon_loglik": float(canonical["loglik"]),
            "heavy_n_cont": float(entry["heavy_tailed_full_mle"]["fit"]["n"]),
            "heavy_n": int(round(heavy["n"])),
            "heavy_r": float(heavy["r"]),
            "heavy_sigma": float(heavy["sigma"]),
            "heavy_mu": float(heavy["mu"]),
            "heavy_loglik": float(heavy["loglik"]),
            "heavy_n_walk": int(round(walk["n"])),
            "heavy_r_walk": float(walk["r"]),
            "heavy_sigma_walk": float(walk["sigma"]),
            "heavy_mu_walk": float(walk["mu"]),
            "dloglik_heavy_minus_canon": float(heavy["loglik"] - canonical["loglik"]),
        })
    return rows


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([
                row["dataset"], row["background"], row["assay"], row["N"],
                f"{row['s_min']:.4g}", f"{row['s_max']:.4g}",
                f"{row['canon_n_cont']:.4g}", row["canon_n"],
                f"{row['canon_r']:.4g}", f"{row['canon_sigma']:.4g}",
                f"{row['canon_loglik']:.7g}",
                f"{row['heavy_n_cont']:.4g}", row["heavy_n"],
                f"{row['heavy_r']:.4g}", f"{row['heavy_sigma']:.4g}",
                f"{row['heavy_mu']:.4g}", f"{row['heavy_loglik']:.7g}",
                row["heavy_n_walk"], f"{row['heavy_r_walk']:.4g}",
                f"{row['heavy_sigma_walk']:.4g}", f"{row['heavy_mu_walk']:.4g}",
                f"{row['dloglik_heavy_minus_canon']:.6g}"])


def latex_table(rows):
    """A PNAS-style tabular for the SI, matching the hand-written tables in sections_si."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{",
        r"        \textbf{Fitted FGM parameters for every background simulated in this work}.",
        r"        Full unbinned maximum-likelihood fits of the canonical (isotropic Gaussian)",
        r"        and heavy-tailed (beta-prime radial) FGM to each measured DFE, conditional on",
        r"        $s \geq -0.5$ and with no measurement-error convolution.",
        r"        \emph{$N$}: effects above the cut.",
        r"        \emph{$n$}: phenotypic dimension, the better of the two integer neighbours of",
        r"        the continuous optimum, with the remaining parameters re-maximised there.",
        r"        \emph{$r$}: distance to the optimum; \emph{$\sigma$}: mutation scale;",
        r"        \emph{$\mu$}: beta-prime shape, with $\mu \to \infty$ recovering the",
        r"        canonical model.",
        r"        \emph{$\Delta \ell$}: heavy-tailed minus canonical log-likelihood, both",
        r"        maximised on the same sample.",
        r"    }",
        r"    \label{tab:fgm-params}",
        r"    \begin{tabular}{lllrrrrrrr}",
        r"        \toprule",
        r"        & & & & \multicolumn{2}{c}{Canonical} & \multicolumn{3}{c}{Heavy-tailed} & \\",
        r"        \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
        r"        Data & Background & Medium & $N$ & $n$ & $r$ & $n$ & $r$ & $\mu$"
        r" & $\Delta \ell$ \\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append(
            f"        {row['dataset']} & {row['background']} & {row['assay']} & "
            f"${row['N']}$ & "
            f"${row['canon_n']}$ & ${row['canon_r']:.3f}$ & "
            f"${row['heavy_n']}$ & ${row['heavy_r']:.3f}$ & "
            f"${row['heavy_mu']:.3f}$ & "
            f"${row['dloglik_heavy_minus_canon']:+.0f}$ \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_CSV)
    parser.add_argument("--latex", action="store_true",
                        help="also print a LaTeX tabular for the SI")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows(load_fits())
    write_table(rows, args.out)

    print("\nFGM fits, integer n, conditional on s >= -0.5, no error convolution")
    print("parentheses give the continuous-n optimum the integer was profiled around")
    header = (f"{'data':<7}{'background':<14}{'medium':<7}{'N':>7}"
              f"{'canon n':>12}{'r':>8}{'sigma':>9}{'loglik':>11}"
              f"{'heavy n':>12}{'r':>8}{'sigma':>9}{'mu':>8}{'loglik':>11}{'dloglik':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['dataset']:<7}{row['background']:<14}{row['assay']:<7}{row['N']:>7}"
              f"{row['canon_n']:>6} ({row['canon_n_cont']:>4.1f})"
              f"{row['canon_r']:>8.3f}{row['canon_sigma']:>9.5f}{row['canon_loglik']:>11.1f}"
              f"{row['heavy_n']:>6} ({row['heavy_n_cont']:>4.1f})"
              f"{row['heavy_r']:>8.3f}{row['heavy_sigma']:>9.5f}{row['heavy_mu']:>8.3f}"
              f"{row['heavy_loglik']:>11.1f}"
              f"{row['dloglik_heavy_minus_canon']:>+10.1f}")

    # The one comparison in this table that is a claim rather than a description: the same
    # lineage 2,000 generations apart, so the fitted radius should have fallen.
    by_key = {row["key"]: row for row in rows}
    early, late = by_key["couce_0K"], by_key["couce_2K"]
    print("\nCouce ARA+2, 0K -> 2K, the one lineage fitted at two timepoints:")
    for model in ("canon", "heavy"):
        r0, r2 = early[f"{model}_r"], late[f"{model}_r"]
        print(f"  {model:<6} r {r0:.3f} -> {r2:.3f}  ({100 * (r2 - r0) / r0:+.1f}%)"
              f"   n {early[f'{model}_n']} -> {late[f'{model}_n']}")
    print("  The population moves toward the optimum, so a falling r is the expected sign.")
    print("  One lineage, one interval -- a consistency check, not a fitted trend.")

    # The reported integer refit against what the walks were actually run at, so the gap is
    # on the record rather than something a reader has to discover by opening a cache.
    print("\nreported integer fit vs the parameters the walk drivers simulate at:")
    hdr = (f"  {'background':<14}{'n':>4}{'n_walk':>8}{'r':>9}{'r_walk':>9}{'d%':>7}"
           f"{'sigma':>10}{'sigma_walk':>12}{'d%':>7}{'mu':>8}{'mu_walk':>9}{'d%':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in rows:
        def gap(key):
            reported, walked = row[f"heavy_{key}"], row[f"heavy_{key}_walk"]
            return 100.0 * (walked - reported) / reported if reported else float("nan")
        print(f"  {row['background']:<14}{row['heavy_n']:>4}{row['heavy_n_walk']:>8}"
              f"{row['heavy_r']:>9.4f}{row['heavy_r_walk']:>9.4f}{gap('r'):>+7.2f}"
              f"{row['heavy_sigma']:>10.5f}{row['heavy_sigma_walk']:>12.5f}"
              f"{gap('sigma'):>+7.2f}"
              f"{row['heavy_mu']:>8.4f}{row['heavy_mu_walk']:>9.4f}{gap('mu'):>+7.2f}")

    print("\nn       = phenotypic dimension: the better integer neighbour of the continuous")
    print("          optimum, with r, sigma and mu re-maximised at that n.  The heavy-tailed")
    print("          profile in n is shallow, so read it as the ridge, not a sharp estimate")
    print("r       = distance to the fitness optimum; sigma = mutation scale")
    print("mu      = beta-prime shape.  BELOW 1 for both Limdi ancestors, so their mutation-size")
    print("          distribution has no finite mean and no timescale built on one is defined")
    print("*_walk  = the continuous optimum with n rounded, which is what the walk drivers")
    print("          simulate; the reported columns re-maximise r, sigma and mu at the")
    print("          pinned integer n instead.  Quote the reported ones")
    print("dloglik = heavy-tailed minus canonical, both maximised on the same sample")
    print(f"\nSaved {args.out}")

    if args.latex:
        print("\n" + latex_table(rows))


if __name__ == "__main__":
    main()
