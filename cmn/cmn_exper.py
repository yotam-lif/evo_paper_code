r"""Shared loaders for the experimental DFE datasets (Couce, Ascensao, Limdi).

One place to read and clean the raw experimental data, so every analysis script uses the
same conventions instead of copy-pasting the file parsing:

    code_figs/TableS1_couce_autocorr.py         DFE autocorrelation across consecutive transitions
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
LIMDI_DIR = os.path.join(
    DATA_DIR, "anurag_data", "Analysis", "Part_3_TnSeq_analysis",
    "Processed_data_for_plotting",
)
LIMDI_META = os.path.join(DATA_DIR, "anurag_data", "Metadata", "all_metadata_REL606.txt")

# ── dataset structure ─────────────────────────────────────────────────────────
# Couce Ara+2 lineage: three sequenced timepoints (0K == the REL607 ancestor).
COUCE_FILES = {"0K": "Rfitted_fil.txt", "2K": "2Kfitted_fil.txt", "15K": "15Kfitted_fil.txt"}
COUCE_INTERVALS = (("0K", "2K"), ("2K", "15K"))          # consecutive transitions
COUCE_SPAN = ("0K", "15K")                               # the whole lineage, end to end

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
# Couce et al. -- per-SEGMENT selection coefficients (fitted1) per timepoint
#
# THE UNIT OF ANALYSIS IS NOT AN INSERTION SITE.  Couce et al. "divided each locus into 5
# segments of equal length and then pooled all insertions in each segment", so one row is a
# fifth of a gene (or of an intergenic region), pooled over ``abn`` insertion mutants (median
# 7).  ``alle`` = "<ORF>-<segment 1..5>" is therefore the identity of a row and is unique
# within a file; ``site`` is just ONE representative coordinate out of the ``abn`` pooled,
# and since the three libraries were mutagenised independently the representative differs
# between timepoints for 40% of shared segments.  Match on ``alle``, never on ``site``.
#
# Note also that the Couce DFE has NO lethal tail: across all three timepoints exactly one row
# of 38882 falls below -0.3 (IR2123 at 15K, an intergenic region pooling 2 insertions with
# sterr1 = 0.056 -- noise, not lethality), and it is not present at 2K so it never enters a
# matched pair.  Otherwise the deepest segment is -0.23 / -0.24 / -0.21 at 0K / 2K / 15K.  The
# knockouts that are lethal in DM25 -- amino-acid auxotrophs above all -- are absent from the
# library rather than filtered out of the analysis.  Of the 163 genes Limdi et al. score at
# s < -0.3, only 60 appear in the Couce ancestor library at all, at a median of 1 covered
# segment and 80 reads against 4 segments and 644 reads for the rest; the survivors are the
# C-terminal insertions that leave the protein functional (serB: -0.50 in Limdi, +0.06 here),
# a phenomenon the paper itself documents.  Couce values are consequently commensurate only
# with the Limdi ``s > -0.3`` range, and even there the scales differ (on shared genes of the
# same strain, s_Limdi ~ 1.4 * s_Couce).
# ══════════════════════════════════════════════════════════════════════════════
def _couce_clean(name):
    """Cleaned Couce frame for one timepoint: fitted1 is the per-segment selection coefficient.

    Keep abn > 1, drop NaN / duplicate fits and the -107 failed-fit sentinel.  ``alle`` is
    unique in the result, so no de-duplication is needed to key on it.
    """
    path = os.path.join(COUCE_DIR, COUCE_FILES[name])
    tab = pd.read_csv(path, sep="\t").dropna(subset=["fitted1"])
    tab = tab.drop_duplicates(subset=["fitted1"])
    tab = tab[tab["abn"] > 1]
    tab = tab[np.isfinite(tab["fitted1"]) & (tab["fitted1"] > -100.0)]
    return tab


def load_couce_segment_series(name):
    """Couce timepoint as a Series indexed by ``alle`` (= "<ORF>-<segment 1..5>").

    ``alle`` is the sub-genic segment the authors' own scripts match on, and the only key
    that is comparable across independently mutagenised libraries -- see the block comment.
    """
    return _couce_clean(name).set_index("alle")["fitted1"]


def load_couce_segment_errors(name):
    """Couce per-segment standard errors (``sterr1``), indexed by ``alle``.

    Built from the same cleaned frame as :func:`load_couce_segment_series`, so the two are
    row-aligned and share an index.
    """
    return _couce_clean(name).set_index("alle")["sterr1"]


def load_couce_effects(name):
    """Couce timepoint as a bare array of per-segment selection coefficients."""
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


def load_asencao_errors(exp, background):
    """Per-gene 1-sigma measurement error (``s std``) for one (experiment, background), or None.

    The published per-gene standard error from the authors' data release
    (github.com/joaoascensao/S-L-REL606-BarSeq), aligned row-for-row to
    :func:`load_asencao_array` (NaN where the effect is unmeasured).  These ``*_std.npy`` arrays
    are built by ``data/asencao_dfe_arrays/build_stds_from_repo.py``; the value-match that fixes
    the strain mapping (each experiment's files are S/L/R in order) is verified there.
    """
    path = os.path.join(ASENCAO_DIR, exp, f"{background}_std.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# Ascensao MONOCULTURES -- combined and per-replicate fits, keyed on gene_ID
#
# The .npy arrays above are the authors' COMBINED per-experiment fits, which carry no replicate
# structure.  The release also publishes each biological replicate fit on its own, and replicate 1
# vs replicate 2 of one strain in one experiment is a pure technical control: same genotype, same
# library, same condition, nothing between the two numbers but assay noise.  Those tables are
# cached as tidy CSVs by ``data/asencao_dfe_arrays/build_monoculture_from_repo.py``.
#
# Rows are keyed on ``gene_ID`` (ECB_#####), NOT on gene_symbol and never on row position.  The
# build script verifies that gene_ID is unique and non-null in every file and that the
# gene_ID -> gene_symbol map is identical across all 41 tables.  Gene sets differ substantially
# between strains (E_SLR: 3021 for R, 3029 for S, but only 2361 for L), so callers MUST intersect
# indices rather than assume alignment -- which is the whole reason these are Series, not arrays.
# ══════════════════════════════════════════════════════════════════════════════
ASENCAO_MONO_DIR = os.path.join(DATA_DIR, "asencao_monoculture")

# Release strain letter -> (experiment folder, ecotype, media, description, n_replicates).
# Letters are unique across the whole release.  "Ecotype" is the genotype: REL606 is the ancestor,
# S and L are the two diversified Ara-2 ecotypes.  Media and description are split because they
# are independent: DM25 is used by THREE different experiments (E_SLR daily batch, E_PQT
# exponential, E_U the 8-day repeat), so the medium alone does not identify the condition and the
# growth regime alone does not identify the medium.  Both come from the release's BarSeq_meta.csv.
# ALL FIVE monoculture experiments are SERIAL DILUTION -- there is no chemostat and no continuous
# culture anywhere in the release.  What differs between them is the dilution factor and the
# transfer interval, and the physiological consequence of those two numbers is whether the culture
# ever exhausts its carbon inside a cycle.  That is the whole content of "full cycle" vs "exp only"
# below (Ascensao et al. 2023, Nat Commun 14:248, Methods "BarSeq experiments"):
#
#   E_SLR "Mono"     DM25 (25 mg/L glucose), 1:100 every 24 h, 5x50 mL flasks, 4 days.  The
#                    standard LTEE regime: exactly log2(100) = 6.6439 gen/day, glucose gone after
#                    a few hours, the rest of the day spent in stationary phase.
#   E_U   "Mono 2"   the same regime, 3x50 mL flasks, 8 days, four biological replicates.
#   E_MNO "1:10 dil" DM27.8, 1:10 every 24 h, 1 L flask (180 mL media + 20 mL culture), 8 days.
#                    Still a full batch cycle, exactly log2(10) = 3.3219 gen/day.  27.8 mg/L is
#                    chosen so 0.9 x 27.8 = 25 mg/L glucose after the transfer, i.e. the same
#                    glucose as DM25; the authors' stated purpose is to LENGTHEN stationary phase
#                    relative to exponential, since only 3.32 doublings are needed to refill.
#   E_PQT "Glu exp"  DM25 again, and 1:100 again, but transferred every 4.5-5 h (S, L) or 7.5-8 h
#                    (REL606) -- the measured length of DM25 exponential phase.  Same medium and
#                    same dilution factor as E_SLR; the culture is simply harvested at the end of
#                    exponential growth, so stationary phase is deleted.  Generations per transfer
#                    from CFU counts, not assumed.  Only REL606's first two transfers survived
#                    (the third bottlenecked), so T rests on 3 timepoints where P and Q have 5.
#   E_GHI "Ac exp"   DM2000-acetate (2000 mg/L sodium acetate), 10 mL in a 50 mL flask, transferred
#                    every 24 h by a VARIABLE volume: OD is measured and the culture diluted to a
#                    target OD_0 so that 24 h later it is at OD ~ 0.6, still mid-exponential.  80x
#                    the carbon of DM25 means it is never approached.  Generations per cycle =
#                    log2(OD_f/OD_0), which is why they differ by strain and replicate: on acetate
#                    REL606 grows at ~0.08/h against L 0.12/h and S 0.18/h.
#
# So "exp only" does NOT mean continuous culture -- it means serially diluted often enough (E_PQT)
# or into enough carbon (E_GHI) that the cells never leave exponential phase.
#
# letter -> (release folder, ecotype, media, description, n_replicates)
ASENCAO_MONO = {
    "R": ("SLR", "REL606", "DM25", "1:100/24h full cycle", 2),
    "S": ("SLR", "S",      "DM25", "1:100/24h full cycle", 2),
    "L": ("SLR", "L",      "DM25", "1:100/24h full cycle", 2),
    "I": ("GHI", "REL606", "DM2000-acetate", "variable/24h exp only", 2),
    "G": ("GHI", "S",      "DM2000-acetate", "variable/24h exp only", 2),
    "H": ("GHI", "L",      "DM2000-acetate", "variable/24h exp only", 2),
    "O": ("MNO", "REL606", "DM27.8", "1:10/24h long stat", 2),
    "M": ("MNO", "S",      "DM27.8", "1:10/24h long stat", 2),
    "N": ("MNO", "L",      "DM27.8", "1:10/24h long stat", 2),
    "T": ("PQT", "REL606", "DM25", "1:100/5-8h exp only", 2),
    "P": ("PQT", "S",      "DM25", "1:100/5-8h exp only", 2),
    "Q": ("PQT", "L",      "DM25", "1:100/5-8h exp only", 2),
    # A second, standalone REL606 monoculture in the same medium and transfer regime as E_SLR's R,
    # run for 8 days with four biological replicates instead of two.  No evolved partner, so it
    # contributes controls only.
    "U": ("U",   "REL606", "DM25", "1:100/24h full cycle 8d", 4),
}

# The four environments in which all three genotypes were assayed, as
# (folder, media, description, ancestor letter, ((ecotype, letter), ...)).  These give the
# ancestor -> evolved transitions; E_U has no evolved partner and appears only among the controls.
ASENCAO_MONO_ENVIRONMENTS = (
    ("SLR", "DM25",           "1:100/24h full cycle",  "R", (("S", "S"), ("L", "L"))),
    ("GHI", "DM2000-acetate", "variable/24h exp only", "I", (("S", "G"), ("L", "H"))),
    ("MNO", "DM27.8",         "1:10/24h long stat",    "O", (("S", "M"), ("L", "N"))),
    ("PQT", "DM25",           "1:100/5-8h exp only",   "T", (("S", "P"), ("L", "Q"))),
)

def asencao_mono_series(letter, rep=None, errors=False):
    """Selection coefficients for one Ascensao monoculture strain, indexed by ``gene_ID``.

    ``rep=None`` gives the authors' COMBINED fit -- the same values as
    :func:`load_asencao_array`, verified identical at build time.  ``rep=1``, ``rep=2`` (and 3, 4
    for U) give the individual biological-replicate fits, which is what makes a within-experiment
    technical control possible.  With ``errors=True`` returns ``(s, s_std)``.

    The index is the gene identifier, so intersecting two of these matches genes correctly no
    matter how the two strains' gene sets differ.
    """
    if letter not in ASENCAO_MONO:
        raise KeyError(f"{letter!r} is not an Ascensao monoculture; expected one of "
                       f"{sorted(ASENCAO_MONO)}")
    n_rep = ASENCAO_MONO[letter][4]
    if rep is not None and not 1 <= rep <= n_rep:
        raise ValueError(f"strain {letter} has replicates 1..{n_rep}, got rep={rep}")
    name = letter if rep is None else f"{letter}{rep}"
    path = os.path.join(ASENCAO_MONO_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run data/asencao_dfe_arrays/build_monoculture_from_repo.py")
    df = pd.read_csv(path)
    if df["gene_ID"].duplicated().any():
        raise ValueError(f"{path}: duplicated gene_ID; the cache is corrupt")
    eff = pd.Series(df["s"].to_numpy(float), index=df["gene_ID"].to_numpy(), name=name)
    if not errors:
        return eff
    return eff, pd.Series(df["s_std"].to_numpy(float), index=df["gene_ID"].to_numpy(),
                          name=f"{name}_std")


# ══════════════════════════════════════════════════════════════════════════════
# Limdi et al. -- TnSeq gene-knockout DFEs across the LTEE panel
#
# We read the .npy matrices, NOT dfe_data_pandas.csv.  That CSV is built in the upstream
# notebook (Fitness_estimation/fitness_calculations.ipynb, cell 36) with
#
#     next1["Fitness estimate"] = s_inverse_var[noness_pop, k, 0]   # numpy -> RangeIndex
#     next1["Genes"]            = names[noness_pop]                 # Series -> ALIGNS ON INDEX
#
# so pandas aligns the gene names on the index and CSV row i gets ``names[i]`` while its
# fitness value belongs to gene ``noness_pop[i]``.  14% of rows end up with a NaN gene name
# and the rest are mislabelled -- differently per population, since each has its own
# ``noness_pop``.  Matching genes across populations by that column therefore pairs genes at
# the same row *position*, which are the same real gene for only ~0.3-6.5% of rows.
#
# The .npy matrices carry no labels: they are indexed by metadata row, identically across
# populations, so matching on the row index is exact.
# ══════════════════════════════════════════════════════════════════════════════
# Column order of the population axis in every Limdi .npy matrix (notebook ``libraries``).
LIMDI_LIBRARIES = ("REL606", "REL607", "Ara-1", "Ara-2", "Ara-3", "Ara-4", "Ara-5", "Ara-6",
                   "Ara+1", "Ara+2", "Ara+3", "Ara+4", "Ara+5", "Ara+6")
# Sentinel written by the upstream notebook wherever a gene has no fitness estimate (too few
# usable TA sites, or the gene is deleted in that background). Real effects bottom out near
# -0.75, so a plain ``> LIMDI_MISSING`` test is unambiguous.
LIMDI_MISSING = -1.0

_LIMDI_CACHE = {}


def load_limdi_arrays():
    """``(fitness, error, gene_names)`` for the Limdi panel, all aligned on metadata rows.

    ``fitness`` is (n_genes, n_libraries, 2) -- the pseudogene-corrected, inverse-variance
    weighted effect per Green/Red replicate.  ``error`` is (n_genes, n_libraries), the
    inverse-variance weighted SEM over the per-TA-site estimates of both replicates (the
    ``_inv`` variant, which is the one the source paper reports).  Missing entries are
    ``LIMDI_MISSING`` in both.  ``gene_names`` is the (n_genes,) metadata name column.
    """
    if not _LIMDI_CACHE:
        _LIMDI_CACHE["fitness"] = np.load(os.path.join(LIMDI_DIR, "fitness_corrected_genes.npy"))
        _LIMDI_CACHE["error"] = np.load(os.path.join(LIMDI_DIR, "errors_genes_inv.npy"))
        meta = pd.read_csv(LIMDI_META, sep="\t")
        _LIMDI_CACHE["names"] = meta.iloc[:, 0].to_numpy(object)
    return _LIMDI_CACHE["fitness"], _LIMDI_CACHE["error"], _LIMDI_CACHE["names"]


def limdi_gene_series(pop, errors=False):
    """Fitness effect per gene for one Limdi population, indexed by metadata gene row.

    Green/Red replicates are averaged, and only genes actually measured in ``pop`` are kept.
    The integer index is the shared gene identity: intersecting two populations' indices
    matches the same genes across backgrounds.  With ``errors=True`` the paired 1-sigma
    measurement errors are returned alongside as ``(effects, sigma)``.
    """
    fitness, error, _ = load_limdi_arrays()
    k = LIMDI_LIBRARIES.index(pop)
    keep = np.where(fitness[:, k, 0] > LIMDI_MISSING)[0]
    eff = pd.Series(np.mean(fitness[keep, k, :], axis=1), index=keep, name=pop)
    if not errors:
        return eff
    return eff, pd.Series(error[keep, k], index=keep, name=f"{pop}_sigma")


def limdi_channel_series(pop):
    """The two technical replicates of one Limdi population, unaveraged, as ``(green, red)``.

    Same genes and same integer index as ``limdi_gene_series(pop)``, which is the mean of these
    two -- the sentinel marks a gene missing in both channels at once, never in just one, so the
    ``keep`` mask is identical.  Correlating green against red for a single library is a purely
    technical control: one strain, one assay, no evolution and no separate mutagenesis, so it
    measures the assay's own reproducibility and nothing else.
    """
    fitness, _, _ = load_limdi_arrays()
    k = LIMDI_LIBRARIES.index(pop)
    keep = np.where(fitness[:, k, 0] > LIMDI_MISSING)[0]
    return (pd.Series(fitness[keep, k, 0], index=keep, name=f"{pop}_green"),
            pd.Series(fitness[keep, k, 1], index=keep, name=f"{pop}_red"))
