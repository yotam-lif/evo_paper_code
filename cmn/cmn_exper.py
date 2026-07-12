r"""Shared loaders for the experimental DFE datasets (Couce, Ascensao, Limdi).

One place to read and clean the raw experimental data, so every analysis script uses the
same conventions instead of copy-pasting the file parsing:

    code_figs/TableS1_means.py            mean-DFE relative differences
    code_figs/TableS2_pspin_exper_fit.py  p-spin p/N estimates (matched-DFE correlations)
    cmn/cmn_fgm_exper.py                  FGM sigma-profile fit (adds tail-trimming on top)

The loaders return the data in the minimal cleaned form each analysis builds on; analysis-
specific steps (per-site matching, tail trimming, gene aggregation) are applied by the
caller. Dataset *structure* constants (which backgrounds are ancestor vs evolved, the
consecutive intervals, the Limdi LTEE panel) also live here since they describe the data.
"""
import os

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
ASENCAO_DIR = os.path.join(DATA_DIR, "asencao_dfe_arrays")
COUCE_DIR = os.path.join(DATA_DIR, "alex_code")
LIMDI_CSV = os.path.join(
    DATA_DIR, "anurag_data", "Analysis", "Part_3_TnSeq_analysis",
    "Processed_data_for_plotting", "dfe_data_pandas.csv",
)

# ── dataset structure ─────────────────────────────────────────────────────────
# Couce Ara+2 lineage: three sequenced timepoints (0K == the REL607 ancestor).
COUCE_FILES = {"0K": "Rfitted_fil.txt", "2K": "2Kfitted_fil.txt", "15K": "15Kfitted_fil.txt"}
COUCE_INTERVALS = (("0K", "2K"), ("2K", "15K"))          # consecutive transitions

# Ascensao: per experiment the R (ancestor), L and S (evolved) arrays are index-aligned.
ASENCAO_BACKGROUNDS = ("L", "R", "S")
ASENCAO_ANCESTOR = "R"
ASENCAO_EVOLVED = ("L", "S")

# Limdi TnSeq-LTEE panel: two ancestors, each the founder of six evolved populations.
LIMDI_ANCESTORS = ("REL606", "REL607")
LIMDI_EVOLVED = {
    "REL606": tuple(f"Ara-{i}" for i in range(1, 7)),    # REL606 is the Ara- founder.
    "REL607": tuple(f"Ara+{i}" for i in range(1, 7)),    # REL607 is the Ara+ founder.
}
# Full panel (both ancestors + every evolved population), in display order.
LIMDI_PANEL = (["REL606"] + list(LIMDI_EVOLVED["REL606"])
               + ["REL607"] + list(LIMDI_EVOLVED["REL607"]))


# ══════════════════════════════════════════════════════════════════════════════
# Couce et al. -- per-site selection coefficients (fitted1) per timepoint
# ══════════════════════════════════════════════════════════════════════════════
def _couce_clean(name):
    """Cleaned Couce frame for one timepoint: fitted1 is the per-site selection coefficient.

    Keep abn > 1, drop NaN / duplicate fits and the -107 failed-fit sentinel.
    """
    path = os.path.join(COUCE_DIR, COUCE_FILES[name])
    tab = pd.read_csv(path, sep="\t").dropna(subset=["fitted1"])
    tab = tab.drop_duplicates(subset=["fitted1"])
    tab = tab[tab["abn"] > 1]
    tab = tab[np.isfinite(tab["fitted1"]) & (tab["fitted1"] > -100.0)]
    return tab


def load_couce_site_series(name):
    """Couce timepoint as a Series indexed by mutation ``site`` (one effect per site).

    De-duplicating on site makes the cross-timepoint merge unambiguous (used for matching).
    """
    tab = _couce_clean(name).drop_duplicates(subset=["site"])
    return tab.set_index("site")["fitted1"]


def load_couce_effects(name):
    """Couce timepoint as a bare array of selection coefficients (no per-site de-dup)."""
    return _couce_clean(name)["fitted1"].to_numpy(float)


# ══════════════════════════════════════════════════════════════════════════════
# Ascensao et al. -- one .npy array of fitness effects per background per experiment
# ══════════════════════════════════════════════════════════════════════════════
def asencao_experiments():
    """Sorted experiment sub-directory names (e.g. GHI / MNO / PQT / SLR)."""
    return [d for d in sorted(os.listdir(ASENCAO_DIR))
            if os.path.isdir(os.path.join(ASENCAO_DIR, d))]


def load_asencao_array(exp, background):
    """Raw fitness-effect array for one (experiment, background), or None if absent.

    Returned verbatim as float (NaNs preserved) so index-aligned backgrounds can be matched;
    callers filter to finite entries as needed.
    """
    path = os.path.join(ASENCAO_DIR, exp, f"{background}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# Limdi et al. -- TnSeq gene-knockout DFEs across the LTEE panel
# ══════════════════════════════════════════════════════════════════════════════
def load_limdi_frame():
    """Limdi DFE table, keeping only finite fitness estimates."""
    tab = pd.read_csv(LIMDI_CSV)
    return tab[np.isfinite(tab["Fitness estimate"])]


def limdi_gene_series(frame, pop):
    """Mean fitness effect per gene for one Limdi population (replicate markers pooled)."""
    sub = frame[frame["Population"] == pop]
    return sub.groupby("Genes")["Fitness estimate"].mean()
