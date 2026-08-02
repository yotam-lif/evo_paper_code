r"""Build either measurement-error treatment of Poster Figure 3.

The two modes share the REL607 data selection, model conventions, binning,
styling, and panel layout:

* ``--errors with`` uses gene-specific Gaussian error convolution.  The
  Gaussian model is moment constrained; the heavy-tailed model is a full MLE.
* ``--errors without`` fits both models by full unbinned MLE directly to the
  measured effects, without convolving either density with measurement error.

The Gaussian fit excludes the most deleterious 5% of retained effects in both
modes; the heavy-tailed fit uses every retained effect.  Both are conditional
on the common displayed range s >= -0.5.

Examples:

    python code_tmp/poster_fig3.py --errors without --poster
    python code_tmp/poster_fig3.py --errors with

``--poster`` copies the selected result to
``../PhD/Posters/GRC_evo_26/poster_fig3.pdf``.  Without it, only the
mode-specific figure and fit-summary files are written.
"""

from __future__ import annotations

import argparse
import os
import shutil


POSTER_PDF = (
    "/Users/yotamlifschytz/Desktop/PhD/Posters/"
    "GRC_evo_26/poster_fig3.pdf"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit and plot REL607 Gaussian and heavy-tailed FGM DFEs."
        )
    )
    parser.add_argument(
        "--errors",
        choices=("with", "without"),
        default="without",
        help=(
            "Use gene-specific error convolution ('with') or fit measured "
            "effects directly ('without'; default)."
        ),
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Copy the selected mode-specific PDF into the poster.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.errors == "with":
        import poster_fig3_with_errors as backend
    else:
        import poster_fig3_no_errors as backend

    backend.main()
    if args.poster:
        os.makedirs(os.path.dirname(POSTER_PDF), exist_ok=True)
        shutil.copyfile(backend.OUT_PDF, POSTER_PDF)
        print(f"Copied selected figure to {POSTER_PDF}")


if __name__ == "__main__":
    main()
