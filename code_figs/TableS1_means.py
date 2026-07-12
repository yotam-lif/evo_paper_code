#!/usr/bin/env python3
r"""Relative difference of mean DFE across consecutive backgrounds/timepoints.

For every consecutive transition (early -> late) we take the mean fitness effect of the
early state and of the late state (each averaged over all finite entries of that state's
DFE) and report the relative difference

    rel_diff = |mean_late - mean_early| / |mean_early|

Data is loaded via ``cmn/cmn_exper.py`` (shared with TableS2_pspin_exper_fit.py).  A CSV
table is written to

    data/TableS1_means_consecutive.csv

Run:
    python code_figs/TableS1_means.py
"""
import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_exper  # noqa: E402  (shared experimental-data loaders + structure)
from cmn.cmn_exper import (  # noqa: E402
    DATA_DIR, COUCE_INTERVALS, ASENCAO_ANCESTOR, ASENCAO_EVOLVED,
    LIMDI_ANCESTORS, LIMDI_EVOLVED,
)

OUT_CSV = os.path.join(DATA_DIR, "TableS1_means_consecutive.csv")


def series_mean(a):
    """Mean over the finite entries of an array/series."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else np.nan


def rel_diff(mean_early, mean_late):
    """Relative difference: |mean_late - mean_early| / |mean_early|."""
    if not np.isfinite(mean_early) or mean_early == 0 or not np.isfinite(mean_late):
        return np.nan
    return abs(mean_late - mean_early) / abs(mean_early)


def asencao_rows():
    rows = []
    for exp in cmn_exper.asencao_experiments():
        anc = cmn_exper.load_asencao_array(exp, ASENCAO_ANCESTOR)
        if anc is None:
            continue
        m_early = series_mean(anc)
        for evo in ASENCAO_EVOLVED:
            evolved = cmn_exper.load_asencao_array(exp, evo)
            if evolved is None:
                continue
            m_late = series_mean(evolved)
            rows.append((f"Asc {exp}", f"{ASENCAO_ANCESTOR} -> {evo}", m_early, m_late))
    return rows


def couce_rows():
    strains = {name: cmn_exper.load_couce_site_series(name) for name in ("0K", "2K", "15K")}
    means = {name: series_mean(s.to_numpy()) for name, s in strains.items()}
    rows = []
    for early, late in COUCE_INTERVALS:
        rows.append(("Couce Ara+2", f"{early} -> {late}", means[early], means[late]))
    return rows


def limdi_pop_mean(tab, pop):
    """Mean fitness effect over all finite raw rows of one Limdi population."""
    return series_mean(tab[tab["Population"] == pop]["Fitness estimate"].to_numpy())


def limdi_rows():
    tab = cmn_exper.load_limdi_frame()
    rows = []
    for anc in LIMDI_ANCESTORS:
        m_early = limdi_pop_mean(tab, anc)
        for evo in LIMDI_EVOLVED[anc]:
            m_late = limdi_pop_mean(tab, evo)
            rows.append((f"Limdi {evo}", f"{anc} -> {evo}", m_early, m_late))
    return rows


def main():
    blocks = [asencao_rows(), couce_rows(), limdi_rows()]

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "transition", "mean_early", "mean_late", "rel_diff"])
        for block in blocks:
            for dataset, transition, m_early, m_late in block:
                writer.writerow([dataset, transition, f"{m_early:.4g}",
                                 f"{m_late:.4g}", f"{rel_diff(m_early, m_late):.4g}"])

    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
