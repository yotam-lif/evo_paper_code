#!/usr/bin/env python3
r"""Relative difference of mean DFE across consecutive backgrounds/timepoints.

For every consecutive transition (early -> late) we take the mean fitness effect of the
early state and of the late state (each averaged over all finite entries of that state's
DFE) and report the relative difference

    rel_diff = |mean_late - mean_early| / |mean_early|

Data loading mirrors ``code_figs/TableS1_pspin_exper_fit.py`` exactly (Ascensao, Couce,
Limdi).  A fixed-width table is written to

    data/mean_ratio_consecutive.txt

Run:
    python code_figs/mean_reldiff_consecutive.py
"""
import os

import numpy as np

from TableS1_pspin_exper_fit import (
    ASENCAO_DIR,
    ASENCAO_ANCESTOR,
    ASENCAO_EVOLVED,
    COUCE_FILES,
    COUCE_INTERVALS,
    LIMDI_ANCESTORS,
    LIMDI_EVOLVED,
    DATA_DIR,
    load_couce_strain,
    load_limdi_frame,
)

OUT_TXT = os.path.join(DATA_DIR, "mean_ratio_consecutive.txt")


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
    for exp in sorted(os.listdir(ASENCAO_DIR)):
        sub = os.path.join(ASENCAO_DIR, exp)
        anc_path = os.path.join(sub, f"{ASENCAO_ANCESTOR}.npy")
        if not (os.path.isdir(sub) and os.path.exists(anc_path)):
            continue
        m_early = series_mean(np.load(anc_path))
        for evo in ASENCAO_EVOLVED:
            evo_path = os.path.join(sub, f"{evo}.npy")
            if not os.path.exists(evo_path):
                continue
            m_late = series_mean(np.load(evo_path))
            rows.append((f"Asc {exp}", f"{ASENCAO_ANCESTOR} -> {evo}", m_early, m_late))
    return rows


def couce_rows():
    strains = {name: load_couce_strain(fname) for name, fname in COUCE_FILES.items()}
    means = {name: series_mean(s.to_numpy()) for name, s in strains.items()}
    rows = []
    for early, late in COUCE_INTERVALS:
        rows.append(("Couce Ara+2", f"{early} -> {late}", means[early], means[late]))
    return rows


def limdi_pop_mean(tab, pop):
    """Mean fitness effect over all finite raw rows of one Limdi population."""
    return series_mean(tab[tab["Population"] == pop]["Fitness estimate"].to_numpy())


def limdi_rows():
    tab = load_limdi_frame()
    rows = []
    for anc in LIMDI_ANCESTORS:
        m_early = limdi_pop_mean(tab, anc)
        for evo in LIMDI_EVOLVED[anc]:
            m_late = limdi_pop_mean(tab, evo)
            rows.append((f"Limdi {evo}", f"{anc} -> {evo}", m_early, m_late))
    return rows


def main():
    blocks = [asencao_rows(), couce_rows(), limdi_rows()]

    header = (f"{'dataset':<16}{'transition':<18}{'mean_early':>12}"
              f"{'mean_late':>12}{'rel_diff':>9}")
    sep = "-" * len(header)
    lines = [header, sep]
    for i, block in enumerate(blocks):
        for dataset, transition, m_early, m_late in block:
            lines.append(f"{dataset:<16}{transition:<18}{m_early:>12.4f}"
                         f"{m_late:>12.4f}{rel_diff(m_early, m_late):>9.3f}")
        if i < len(blocks) - 1:
            lines.append(sep)

    text = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Saved {OUT_TXT}")


if __name__ == "__main__":
    main()
