# Baym et al. (2024) LTEE transposon-fitness data guide

This is an agent-facing guide to the data associated with:

- Couce, Limdi et al., *Changing fitness effects of mutations through long-term bacterial evolution*, Science 383, eadd1417 (2024), DOI: [10.1126/science.add1417](https://doi.org/10.1126/science.add1417)
- the earlier Couce et al. preprint, *Predictability shifts from local to global rules during bacterial adaptation*, DOI: [10.1101/2022.05.17.492360](https://doi.org/10.1101/2022.05.17.492360)
- the earlier Limdi et al. preprint, *Parallel evolution of mutational fitness effects over 50,000 generations*, DOI: [10.1101/2022.05.17.492023](https://doi.org/10.1101/2022.05.17.492023)

It was assembled by reading `Theory/Baym24.pdf`, `Theory/Baym_supp.pdf`, `Theory/couce.pdf`, and `Theory/limdi.pdf`, and by auditing the checked-in data and analysis code. File dimensions and parsing behavior below were verified against the local repository on 2026-07-24.

## The most important fact: these are two experiments, not one raw dataset

The final Science article combines two projects that were initially separate:

| Project | Local directory | Assay | Genetic backgrounds | Natural analysis unit | Main strength |
|---|---|---|---|---|---|
| Limdi | `data/anurag_data` | UMI-TnSeq | REL606, REL607, and one 50,000-generation clone from each of the 12 LTEE populations | protein-coding gene, after averaging interior TA insertions | deleterious effects, essentiality, parallel changes across all LTEE lines |
| Couce | `data/alex_code` | INSeq | locally: REL606 and Ara+2 clones at 2,000 and 15,000 generations | equal-length subgenic segment, including ORFs and intergenic loci | beneficial tail, sign epistasis, early-versus-late predictability |

Both assays used mariner-family transposons, measured abundance changes in pooled competitions in the LTEE environment, and reported slopes on the same approximate selection-coefficient scale. They differ in library construction, sequencing, normalization, time course, background sampling, quality filters, and aggregation level. Do not concatenate their rows and call the result one DFE.

Use `Theory/Baym_supp.pdf` as the definitive description of the combined publication. The two preprints are valuable for understanding the original projects, but some wording, thresholds, counts, and figure analyses changed during merger and review. Use the local source code to understand the exact checked-in files, even where it differs from the final methods.

## Quick decision guide

Use the Limdi/UMI data when the question concerns:

- fitness effects of disrupting protein-coding genes at 50,000 generations;
- the deleterious tail or gene essentiality;
- parallel changes across the 12 LTEE populations;
- deletions, duplications, homologous-gene redundancy, or RNA expression;
- comparison of both technical fitness-assay replicates.

Use the Couce/INSeq data when the question concerns:

- the beneficial tail in the ancestor, 2K, and 15K Ara+2 backgrounds;
- exact subgenic segments, intergenic regions, polar effects, or C-terminal effects;
- the same segment changing sign or magnitude through the Ara+2 lineage;
- the local plotting scripts in `data/alex_code`.

The local Couce directory is **not** the complete Couce dataset. It contains only the three filtered Ara+2 segment tables. The paper also analyzed Ara-1, but the Ara-1 tables, raw reads, full processing pipeline, operon/predictability inputs, and metagenomic inputs are absent locally.

## Repository map

```text
Theory/
  Baym24.pdf          final Science article
  Baym_supp.pdf       final supplementary methods, figures, and tables
  couce.pdf           original Couce project preprint
  limdi.pdf           original Limdi project preprint

data/anurag_data/
  Data/
    README.md         download instructions; large trajectories/WGS inputs absent
  Metadata/
    all_metadata_REL606.txt
    pseudogenes_locations_REL606.txt
    rel606_reference.fasta
    sequence_NC_12967.gb
    Favate_et_al_data_GSE164308_table_s1_read_counts.csv
    homologs_cluster_anc.tsv
    ...
  Analysis/
    Part_1_Data_to_trajectories/   FASTQ -> mapped, UMI-corrected TA counts
    Part_2_WGS_analysis/           breseq/depth workflows and saved SV calls
    Part_3_TnSeq_analysis/         fitness, essentiality, expression, homologs
    Plots_for_paper/               generated figures; outputs, not source data
    Supplementary_tables/         human-readable result tables

data/alex_code/
  Rfitted_fil.txt
  2Kfitted_fil.txt
  15Kfitted_fil.txt
  beneficials_accross_backgrounds.R
  overlaps.R
  segben.py
  overlapping_dfes.py
  *.png                existing plot outputs
```

The most important Limdi analysis sources are:

```text
Analysis/Part_3_TnSeq_analysis/Fitness_estimation/fitness_calculations.ipynb
Analysis/Part_3_TnSeq_analysis/Fitness_estimation/Essentiality Threshold.ipynb
Analysis/Part_3_TnSeq_analysis/Exploratory_analysis/expression_levels_data_reformatting.ipynb
Analysis/Part_3_TnSeq_analysis/Exploratory_analysis/homolog_pairs.ipynb
Analysis/Part_3_TnSeq_analysis/generate_figures_main.ipynb
Analysis/Part_3_TnSeq_analysis/generate_figures.ipynb
```

Those paths are relative to `data/anurag_data`. `generate_figures_main.ipynb` is the merged-paper version. `generate_figures.ipynb` is older but internally useful and still contains some supplementary analyses. Saved plot PDFs and PNGs are evidence of prior runs, not authoritative data objects.

## Shared biological and measurement framework

### LTEE environment

The Long-Term Evolution Experiment used *E. coli* B strain REL606 and 12 replicate populations. Cells were propagated at 37 degrees C in glucose-limited DM25: Davis-Mingioli minimal medium with 25 mg/L glucose. A daily 1:100 transfer corresponds to:

```text
log2(100) = 6.643856... generations per day
```

The pooled mutant competitions recreated these conditions. Limdi used five parallel 10 mL tube cultures per assay, pooled at each transfer. Couce used 10 mL in 50 mL flasks.

### What the reported fitness effect means

For an insertion allele or pooled set of insertions, the reported quantity is essentially:

```text
s = slope of ln(relative abundance) versus generations
```

Positive `s` means the insertion lineage increased relative to the internal reference; negative `s` means it declined. Limdi normalized each insertion by total library abundance and then centered gene effects using presumed-neutral pseudogene insertions. Couce normalized pooled segment trajectories to a consensus trajectory made from presumed-neutral loci.

This `s` is not numerically identical to the realized growth-rate ratio `W` commonly used in LTEE competition papers. The supplement describes an approximately `ln(2)` scaling and notes that the approximation degrades for large effects. Do not directly compare these values with LTEE `W` values without converting the definition used by the other source.

### What an insertion mutation represents

Both transposons insert at TA dinucleotides. An insertion often behaves like loss of function, but not always:

- an insertion near the C terminus may preserve partial function;
- an insertion can alter rather than eliminate function;
- the resistance cassette and terminators can create polar effects on a transcriptional unit;
- intergenic insertions can alter expression;
- mobile or repeated sequence is difficult or inappropriate to map;
- genes with few TA sites are poorly sampled;
- severe deleterious effects and lethality are censored by disappearance from the pool.

Therefore, these data describe the DFE of **recoverable mariner insertions under the assay protocol**, not the DFE of all point mutations, all deletions, or all possible loss-of-function alleles.

## Reference genome, coordinates, and stable identifiers

Both projects map to the REL606 reference, NCBI accession `NC_012967.1`.

The checked-in FASTA is:

```text
data/anurag_data/Metadata/rel606_reference.fasta
```

Local audit:

- reference length: 4,629,812 bp;
- number of `TA` strings using Python-style zero-based starts: 211,995.

For Limdi arrays, the stable entity key is the zero-based row index of:

```text
data/anurag_data/Metadata/all_metadata_REL606.txt
```

That file has 4,017 rows. Prefer its `Locus Tag (prokka_output)` as the persistent textual gene identifier. Gene names are not unique. Do not use `Gene Name` alone as a primary key.

The metadata columns are:

| Column | Meaning |
|---|---|
| `Gene Name` | familiar gene symbol where available; not unique |
| ` Locus Tag (prokka_output)` | Prokka locus tag; note the leading space in the raw header |
| `Locus Tag (K12 reference)` | mapped K-12 locus tag where available |
| `Start of Gene`, `End of Gene` | REL606 coordinates used by the notebooks |
| `Strand` | `1` forward, `-1` reverse |
| `UniProt ID` | mapped protein identifier |
| `Protein Product` | annotation |

For reverse-strand genes, `Start of Gene` is greater than `End of Gene`. Preserve the strand-aware coordinate logic from the notebooks. A safe first step after reading the file with pandas is:

```python
metadata.columns = metadata.columns.str.strip()
metadata["gene_row"] = range(len(metadata))
```

For Couce tables, the intended cross-background key is `alle`, the segment label such as `pykF-3`. The `site` field is useful for detecting duplicate annotations within one table, but it is not the segment boundary or a globally unique segment key.

## Limdi / UMI-TnSeq data

### Background index

Every Limdi array with a 14-element background axis uses this exact order:

| Index | Strain | Display name | Generation | Ara marker | Mutator status in local long-form file |
|---:|---|---|---:|---|---|
| 0 | REL606 | Anc | 0 | Ara- | non-mutator |
| 1 | REL607 | Anc* | 0 | Ara+ | non-mutator |
| 2 | REL11330 | Ara-1 | 50,000 | Ara- | mutator |
| 3 | REL11333 | Ara-2 | 50,000 | Ara- | mutator |
| 4 | REL11364 | Ara-3 | 50,000 | Ara- | mutator |
| 5 | REL11336 | Ara-4 | 50,000 | Ara- | mutator |
| 6 | REL11339 | Ara-5 | 50,000 | Ara- | non-mutator |
| 7 | REL11389 | Ara-6 | 50,000 | Ara- | non-mutator |
| 8 | REL11392 | Ara+1 | 50,000 | Ara+ | non-mutator |
| 9 | REL11342 | Ara+2 | 50,000 | Ara+ | non-mutator |
| 10 | REL11345 | Ara+3 | 50,000 | Ara+ | mutator |
| 11 | REL11348 | Ara+4 | 50,000 | Ara+ | non-mutator |
| 12 | REL11367 | Ara+5 | 50,000 | Ara+ | non-mutator |
| 13 | REL11370 | Ara+6 | 50,000 | Ara+ | mutator |

REL607 is the alternatively marked ancestral control, not an evolved population. In the final differential analyses, REL606 at index 0 is used as the ancestral comparator for both Ara- and Ara+ evolved lines; REL607 is used as a control for the neutral marker. Do not silently pair all Ara+ lines with REL607 unless deliberately performing a new analysis.

### Published exclusions

The main analyses exclude:

- Ara-2 / REL11333 / index 3: a few insertions swept during the pooled assay and depressed the relative abundance of almost everything else, including pseudogene controls. The correction cannot recover a trustworthy DFE.
- Ara+4 / REL11348 / index 11: poor agreement between the two replicate assays and unusually high within-gene variability.

The arrays still contain these backgrounds. Exclusion is the analyst's responsibility. For a published-style set of backgrounds:

```python
included_backgrounds = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13]
```

Some supplementary sensitivity analyses include them. State the choice explicitly.

### Wet-lab design

The pSC189 plasmid carries an approximately 2.2 kb kanamycin-resistant mariner transposon and cannot replicate outside the MFDpir donor. It was conjugated into each of the 14 backgrounds. Libraries were selected on LB plus kanamycin, scraped, pooled, and frozen. Each retained library had more than 100,000 unique insertions and insertions in more than 83% of genes.

For each background:

1. A frozen library aliquot supplied a common pre-competition sample.
2. Two replicate fitness assays were started from that stock. They are called `green` and `red` because of the marker colors used during the experiment.
3. Each replicate consisted of five parallel 10 mL DM25 cultures. The cultures were pooled before each 1:100 daily transfer, increasing the bottleneck population size fivefold.
4. Samples span five abundance timepoints over four days: approximately 0, 6.64, 13.28, 19.92, and 26.56 generations.

There is one common initial sample and two later trajectories. Some code duplicates the common initial counts into both color-specific matrices.

### Sequencing and molecule counting

Libraries were sequenced on two NovaSeq S4 lanes, paired-end 2 x 150 bp. The supplement reports a median of about 23 million reads per timepoint, about 85% mapping, and about 20% of mapped reads removed as PCR duplicates.

The local processing chain is:

```text
FASTQ
  -> filter R1 for the mariner end and save the first 10 bp as the UMI
  -> Bowtie2 mapping of filtered R1 and corresponding R2 to REL606
  -> retain unique mappings and merge NovaSeq lanes 3 and 4
  -> deduplicate molecules using insertion coordinate + UMI + R2 coordinate
  -> expand each sample to the complete 211,995-TA-site reference universe
```

Relevant files:

```text
data/anurag_data/Analysis/Part_1_Data_to_trajectories/Scripts/filter_trim_return_positions.py
data/anurag_data/Analysis/Part_1_Data_to_trajectories/Scripts/run_bowtie.sh
data/anurag_data/Analysis/Part_1_Data_to_trajectories/Scripts/merge_get_locations_UMI.py
data/anurag_data/Analysis/Part_1_Data_to_trajectories/Scripts/merging_counts_data.ipynb
```

Important implementation details:

- R1 is required to match the mariner-end pattern `GGGGACTTATCAGCCAACCTGTTA` with at most one edit.
- Trimming retains the terminal TA.
- The first 10 bases of the original R1 are stored as the UMI.
- A read is retained only if it maps and lacks a Bowtie2 `XS` alternative-alignment tag.
- The R1 insertion coordinate is the first mapped position on the forward strand or `pos[-2]` on the reverse strand.
- Within an insertion coordinate, the molecule key is the 10 bp UMI concatenated with the mapped R2 coordinate. This extra coordinate reduces accidental UMI collisions.
- Each intermediate `.pos` file is a `3 x number_of_observed_sites` matrix: coordinate, raw read count, and UMI-corrected molecule count.

The final per-color trajectory text file has shape `11 x 211995`:

| Row | Contents |
|---:|---|
| 0 | all REL606 TA coordinates |
| 1, 2 | raw and UMI-corrected counts for the common pre-competition sample |
| 3, 4 | raw and UMI-corrected counts after the first competition interval |
| 5, 6 | raw and UMI-corrected counts after the second interval |
| 7, 8 | raw and UMI-corrected counts after the third interval |
| 9, 10 | raw and UMI-corrected counts after the fourth interval |

The notebook calls the common initial input `tm1`, and calls the four later samples `t0` through `t3`. Downstream code treats these as experimental trajectory indices 0 through 4. Do not infer biological time from the variable names alone. Also note that the `np.savetxt` comment says "col" even though these are rows.

### Raw-data availability in this checkout

The large trajectory and WGS inputs are not checked in under `data/anurag_data/Data`. Its `README.md` instructs the user to download `Mutant_Trajectories` and `WGS_Data` from Zenodo and place them there.

Consequences:

- the raw FASTQ-to-count pipeline cannot be rerun from the local checkout alone;
- site-level trajectories for all genes are absent;
- only processed gene-level outputs, two example genes, metadata, and structural-variation summaries are locally available.

Raw reads: NCBI BioProject [PRJNA814281](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA814281).  
Processed archive: [Zenodo 10.5281/zenodo.6547536](https://doi.org/10.5281/zenodo.6547536).  
Upstream code: [baymlab/2022_Limdi-TnSeq-LTEE](https://github.com/baymlab/2022_Limdi-TnSeq-LTEE).

### From insertion trajectories to gene fitness

Only the interior of each gene is used. In transcriptional orientation, the code excludes the first 10% and last 25%, leaving the middle 65%. One notebook comment incorrectly calls this the "middle 80%"; the actual parameters are `frac5p = 0.1` and `frac3p = 0.25`.

For each insertion site and replicate:

1. Normalize the UMI-corrected count to counts per `10^7` molecules:

   ```text
   x[t] = count[t] / total_library_count[t] * 10^7
   ```

2. Require initial normalized depth strictly greater than 5 in both color trajectories.
3. Fit the slope of `ln(x[t])` against `[0, 6.64, 13.28, 19.92, 26.56]`.
4. If the trajectory first reaches zero, replace that first zero with a pseudocount of 1, fit only through that point, and discard later timepoints from the fit.
5. Weight the site estimate using:

   ```text
   w = [((1 + n0)^(-1) + (1 + n1)^(-1)) / ln(2)^2]^(-1)
   ```

   The local code uses normalized counts at trajectory indices 0 and 1 for `n0` and `n1`, and caps the weight at the value obtained from `n0 = n1 = 100`.

The strict gene estimate is an inverse-variance-weighted mean of site slopes, separately for each color. The local function requires:

- at least two qualifying sites in each replicate; and
- qualifying sites equal to at least 20% of the possible interior TA sites.

The final supplement additionally says protein-coding genes were restricted to at least five interior TA sites. The local `fitness_estimate` function does **not** explicitly apply this five-site gate; `ta_min = 5` is applied when constructing the LB coverage-fraction matrix. A local audit found 73 REL606 rows with a strict fitness estimate despite having only two to four interior TA sites under the notebook's coordinate rule. If reproducing the final methods, add the five-site eligibility rule yourself.

The relaxed estimate requires at least one qualifying site in each replicate and at least 10% of possible interior sites. It was created to make a cautious pairwise call when one background passed the strict arbitrary coverage threshold and the other narrowly failed. It is not the default DFE.

### Neutral correction

The gene estimates are centered separately for each background and replicate using insertion effects in 134 presumed pseudogenes:

- normally, only pseudogenes with uncorrected effects greater than `-0.05` in both replicates are used;
- for Ara-2, all analyzable pseudogenes are used because the whole library is pathologically shifted.

The correction is subtracted separately from the two replicate gene estimates. Pseudogene locations have no names in the local file; their row order is their key.

### Main processed arrays

All paths in this table are under:

```text
data/anurag_data/Analysis/Part_3_TnSeq_analysis/Processed_data_for_plotting
```

| File | Shape | Axes / meaning |
|---|---:|---|
| `fitness_corrected_genes.npy` | `(4017, 14, 2)` | gene row, background, replicate; strict, pseudogene-corrected `s` |
| `fitness_genes_relaxed_thresholds_updated.npy` | `(4017, 14, 2)` | same axes; strict values plus pairwise recovery using relaxed thresholds |
| `fitness_pseudogenes.npy` | `(134, 14, 2)` | pseudogene row, background, replicate; uncorrected values used for centering |
| `errors_genes.npy` | `(4017, 14)` | ordinary pooled-site SEM |
| `errors_genes_inv.npy` | `(4017, 14)` | inverse-weighted dispersion SEM used in the paper |
| `errors_genes_hybrid.npy` | `(4017, 14)` | unweighted dispersion about the inverse-weighted mean |
| `errors_pseudogenes.npy` | `(134, 14)` | ordinary pseudogene SEM |
| `errors_pseudogenes_inv.npy` | `(4017, 14)` | intended inverse-weighted pseudogene SEM, but incorrectly allocated to gene length |
| `errors_pseudogenes_hybrid.npy` | `(4017, 14)` | same allocation bug for hybrid error |
| `greensum.npy`, `redsum.npy` | `(14, 5)` | total UMI-corrected molecules by background and trajectory index |
| `fraction_t0_site_thresh_5_ta_min_5.txt` | `(4017, 14)` | mean fraction across colors of interior TA sites with normalized initial depth greater than 5 |
| `pvals_fdr_5_ta_min_5.txt` | `(4017, 14)` | within-background BH-adjusted lower-tail probabilities from a Gamma fit to high-coverage (`fraction >= 0.3`) genes |
| `expression_means.txt` | `(4017, 13)` | mean RNA TPM across two replicates; no Ara+6 column |
| `homologs_n.txt` | `(4017,)` | homolog-family integer; 0 means no assigned family |
| `homologs_two.txt` | `(4017,)` | pair identifier for genes in a family of exactly two; 0 otherwise |
| `mean_dfe.txt`, `variance_dfe.txt` | `(14,)` each | background-level summaries made from the processed gene values |
| `error_mean*.txt`, `error_variance*.txt` | `(14,)` each | propagated DFE-summary errors for the three error definitions |
| `rffG_trajectories_{green,red}.npy` | `(14, 5, 29)` | raw UMI-corrected counts for the 29 interior rffG TA sites retained by `search_gene` |
| `rffH_trajectories_{green,red}.npy` | `(14, 5, 20)` | analogous counts for 20 rffH sites |

#### Missing values

In the NumPy fitness and error arrays, exactly `-1` is a missing/not-calculated sentinel. It is not a measured selection coefficient. Convert it to `NaN` or use a mask before any mean, variance, correlation, regression, threshold, or plot.

The coverage-fraction and Gamma/FDR text matrices also use `-1` for ineligible or unavailable gene-background cells. Mask those values before applying `f < 0.1` or any probability cutoff. `expression_means.txt` does not use this sentinel.

For a gene-level comparison, normally require both replicate estimates to be present:

```python
valid = (fitness != -1).all(axis=2)
fitness_mean = fitness.mean(axis=2)
fitness_mean[~valid] = np.nan
```

Do not use a condition such as `fitness < -0.3` before removing `-1`; otherwise every missing value will look strongly deleterious.

The pseudogene inverse and hybrid error arrays were initialized with 4,017 rows by mistake. Only the first 134 rows can contain pseudogene values; the remainder are `-1`. Slice `[:134]`, or use `errors_pseudogenes.npy`, before combining them with `fitness_pseudogenes.npy`.

#### Replicate-axis color caveat

The checked-in fitness notebook defines:

```python
fitness_estimate(counts_red, counts_green, ...)
```

but calls it positionally with `counts_all_green` first and `counts_all_red` second. It then labels output axis 0 as Green and axis 1 as Red in `dfe_data_pandas.csv`. Thus, the two replicate estimates are valid as a pair, but the color labels appear reversed relative to the source trajectory filenames. This does not affect their mean or color-symmetric analyses. Do not use the saved Green/Red label to trace a value back to a raw color file without rechecking this call.

### Long-form DFE CSV

`dfe_data_pandas.csv` has 96,518 rows and columns:

```text
unnamed saved index
Fitness estimate
Population
Evolutionary History
Replicate
Genes
Mutator
Ara Phenotype
```

Use it for quick plotting only. Important limitations:

- it contains all 14 backgrounds, including Ara-2 and Ara+4;
- the generating notebook intended to exclude indices 3 and 11 but tests `k not in [103, 1011]`, so the exclusion never occurs;
- `Genes` contains non-unique gene symbols and omits the stable locus tag and original row index;
- the saved Green/Red names inherit the positional-argument color swap described above;
- the first CSV column is an old pandas index and has no biological meaning.

For serious analysis, load the NumPy array and join by the metadata row instead.

### Gene-level error values

The output error arrays have no replicate axis because each error is calculated from site-level slopes pooled across the two replicate assays. The primary inverse-weighted value is:

```text
sqrt(weighted mean((site_s - weighted_gene_mean)^2) / (number_of_site_estimates - 1))
```

It is a site-dispersion-based SEM, not simply the standard error from fitting one gene-level trajectory and not the disagreement of two color means alone.

### Differential fitness effects

The Limdi analysis generally:

1. averages the two replicate gene estimates;
2. restricts pairwise tests to genes with mean `s > -0.3` in both backgrounds;
3. computes:

   ```text
   z = (s_evolved - s_ancestor) / sqrt(error_evolved^2 + error_ancestor^2)
   ```

4. obtains a two-sided Normal p-value;
5. applies a Bonferroni threshold of `0.05 / number_of_tested_genes`.

The qualitative bins used for parallel changes include:

- roughly neutral: `s > -0.05`;
- deleterious but above the essentiality censoring region: `-0.3 < s < -0.15`.

Deleted genes must be removed before interpreting a missing insertion effect as evolved essentiality.

### Essentiality is an operational classification

The experiment cannot cleanly distinguish absolute lethality from sufficiently severe depletion. The authors therefore use conservative operational states.

#### DM25

A gene is called differentially essential between two backgrounds when:

- one background is clearly nonessential: mean `s > -0.15`; and
- the other is in the essential-like region: relaxed mean `s < -0.3`.

The reworked figure notebook additionally requires the maximum of the two relaxed replicate estimates to be below `-0.2`, the relaxed mean to be greater than the `-1` missing sentinel, and the gene not to be deleted in the background being called essential.

The `-0.3` threshold came from simulations comparing no growth after a 100-fold dilution with highly deleterious but growing mutants. It was chosen where overlap with the simulated essential distribution fell below 0.05.

#### LB library-construction stage

Some genes have no recoverable insertions before the DM25 competition. For each background:

```text
f = average across red and green of
    (number of interior TA sites with normalized initial depth > 5)
    / (number of possible interior TA sites)
```

The final supplement uses:

- `f < 0.1`: candidate LB essential;
- `f > 0.45`: clearly nonessential for calibration against the K-12 TraDIS reference.

For a candidate differential absence, the figure notebook compares the observed number of represented sites with the number in the other background using a Poisson lower-tail CDF, then applies Benjamini-Hochberg FDR at 0.05.

Do not confuse this pairwise Poisson test with `pvals_fdr_5_ta_min_5.txt`. The saved `pvals_fdr` matrix was constructed by fitting a Gamma distribution to high-coverage fractions within each background and BH-adjusting its lower-tail CDF. It is useful for within-library low-representation screening, but it is not the pairwise Poisson result.

#### Transition coding used by the notebooks

For the combined differential-essentiality state matrices:

```text
0 = no called transition
1 = essential in REL606 -> nonessential in the comparison background
2 = nonessential in REL606 -> essential in the comparison background
```

The published main set removes REL607, Ara-2, and Ara+4 from the evolved-line panel.

### Structural variation

Paths:

```text
data/anurag_data/Analysis/Part_2_WGS_analysis/output
```

Files:

| File | Meaning |
|---|---|
| `Deleted_genes_REL606_k12annotated.txt` | `(14, 4017)` binary matrix; 1 marks a gene called missing |
| `Deleted_pseudogenes_REL606_k12annotated.txt` | `(14, 134)` binary matrix |
| `Duplicated_genes_REL11342_Ara+2.txt` | zero-based gene-row indices in the Ara+2 duplication |
| `Duplicated_genes_REL11364_Ara-3.txt` | zero-based gene-row indices in the Ara-3 duplication |
| `Duplicated_genes_REL11367_Ara+5.txt` | zero-based gene-row indices in the Ara+5 duplications |
| `Duplicated_genes_REL11389_Ara-6.txt` | zero-based gene-row indices in the Ara-6 duplications |
| `ara-6_coverage_relative_to_anc.txt` | 4,017 values aligned to gene metadata rows |

WGS coverage exceeded 60x on average. Deletions were derived from breseq missing-coverage intervals at least 1 kb long. The local notebook marks a gene if either annotated endpoint lies strictly inside such an interval.

Duplications were identified visually from per-gene coverage normalized both by sample median and REL606 coverage. The final supplement summarizes a background threshold near 1.5x, but the local notebook uses region-specific cutoffs from 1.3 to 1.5 and hand-selected genomic spans. Treat the saved index lists as curated calls, not the output of one universal classifier.

The duplicated-gene files contain floating-point text representations of integer **row indices**, not gene names or genomic coordinates:

```python
dup_rows = np.loadtxt(path, dtype=int)
dup_metadata = metadata.iloc[dup_rows]
```

### Homolog families

`homologs_cluster_anc.tsv` is MMseqs2 output generated with minimum sequence identity 0.4. `homologs_n.txt` assigns the same positive family ID to all members of a detected family and 0 to unassigned genes. `homologs_two.txt` assigns a positive pair ID only when a gene has exactly one homolog in the filtered set.

These IDs are arbitrary labels, not family sizes. Calculate family size by counting rows sharing an ID. The human-readable outputs are:

```text
data/anurag_data/Analysis/Supplementary_tables/Table3-homolog_group_info.tsv
data/anurag_data/Analysis/Supplementary_tables/homolog_pairs_info.tsv
```

### RNA expression

Raw source:

```text
data/anurag_data/Metadata/Favate_et_al_data_GSE164308_table_s1_read_counts.csv
```

It contains both RNA-seq and ribosome-profiling records. Relevant columns include `repl`, `seqtype`, `line`, `target_id`, `est_counts`, `eff_length`, `length`, and `tpm`.

`expression_means.txt` contains mean RNA-seq TPM across `rep1` and `rep2`, with rows aligned to the 4,017-gene metadata and columns:

```text
0 REL606
1 REL607
2 Ara-1
3 Ara-2
4 Ara-3
5 Ara-4
6 Ara-5
7 Ara-6
8 Ara+1
9 Ara+2
10 Ara+3
11 Ara+4
12 Ara+5
```

Ara+6 is absent because that RNA library was contaminated. The conversion notebook initializes the matrix with zero, so zero can mean true zero TPM **or** failure to map that REL606 metadata row to the older expression target annotation. Do not automatically interpret every zero as confirmed absence of transcription.

### Supplementary tables

`data/anurag_data/Analysis/Supplementary_tables` contains human-readable result summaries:

| File | Content / caution |
|---|---|
| `Table1-1v1_fitness_assay_counts.csv` | colony counts from independent one-versus-one validation assays |
| `Table2A-differential_essential_genes.csv` | genes that were essential in the ancestor and became nonessential |
| `Table2B-differential_nonessential_genes.csv` | genes that were nonessential in the ancestor and became essential; its copied header incorrectly says "becomes nonessential" |
| `Table3-homolog_group_info.tsv` | variable-width homolog-family members and locus tags |
| `Table4-wecA_insertion_mutation_fitnesses.csv` | wecA insertion effects by background |
| `Table5-kdsB_data.csv` | two kdsB copies, expression, presence, and insertion-essentiality summary |
| `Table6A-parallel_essential_genes.csv` | parallel essential-to-nonessential transitions |
| `Table6B-parallel_nonessential_genes.csv` | parallel nonessential-to-essential transitions |
| `Table7A-GO_essentiality.txt` | PANTHER overrepresentation output for the first transition set |
| `Table7B-GO_nonessential.txt` | PANTHER output for the second transition set |

The GO files begin with blank lines and prose metadata before the tabular header. They are not ordinary header-on-line-1 TSV files.

### Reworked final-notebook caveats

`generate_figures_main.ipynb` was reworked for the merged paper and attempts to remove genes known from the K-12 TraDIS/Keio comparison to have only a small essential domain:

```text
mqsA, waaU, yabQ, yafF, yibJ, yqgD, ftsK, ftsN,
ftsX, lptC, ribB, rne, secM, spoT, yejM, polA
```

Only 12 of these 16 names match the local REL606 metadata by exact `Gene Name`; `waaU`, `yabQ`, `yafF`, and `yibJ` do not. The notebook subsets fitness, error, and deletion arrays by the surviving metadata indices, but its `fraction_t0` subsetting cell is commented and retains an old `IndexError` output. The notebook therefore depends on execution state and should not be assumed to run cleanly from top to bottom.

For any new analysis:

1. build one explicit Boolean gene-row mask;
2. apply it once and identically to metadata, fitness, relaxed fitness, errors, coverage fractions, p-values, deletion matrices, homolog arrays, expression rows, and any derived arrays;
3. reset metadata display indices only after preserving the original `gene_row`;
4. assert matching first dimensions after every filter.

## Couce / INSeq data

### Backgrounds and local scope

The full Couce project measured two LTEE lineages:

| Lineage | Ancestor | 2K clone | 15K clone |
|---|---|---|---|
| Ara+2 | REL606 | REL1159A | REL7184A |
| Ara-1 | REL606 | REL1164A | REL7177A |

Only the Ara+2 filtered tables are checked in:

| Local file | Background |
|---|---|
| `data/alex_code/Rfitted_fil.txt` | REL606 ancestor |
| `data/alex_code/2Kfitted_fil.txt` | Ara+2 REL1159A |
| `data/alex_code/15Kfitted_fil.txt` | Ara+2 REL7184A |

Do not generalize a local result to both Couce lineages. Full raw reads are under NCBI BioProject [PRJNA979973](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA979973); processed data are archived at [Zenodo 10.5281/zenodo.7985455](https://doi.org/10.5281/zenodo.7985455); upstream source is [ACouce/LTEE2022](https://github.com/ACouce/LTEE2022).

### Wet-lab and sequencing design

The pSAM plasmid carries the Himar1C9 transposase and a roughly 1.5 kb kanamycin-resistant INSeq transposon with two transcriptional terminators. It was conjugated from MFDpir for 3 to 4 hours. Libraries were assembled from at least 100,000 colonies obtained from at least 10 independent conjugations and selected on LB with streptomycin and kanamycin.

Competitions used daily 1:100 transfers in DM25. Ara+2 competitions lasted five days; Ara-1 competitions lasted eight days because of lower coverage. Viable counts were used to estimate actual generations rather than assuming exactly 6.64 each day.

Because DM25 supported little DNA per culture, the protocol used Phi29-based whole-genome amplification before MmeI digestion and INSeq enrichment. Ara+2 libraries were sequenced on HiSeq and Ara-1 libraries on MiSeq. After filtering, the supplement reports averages of about 3.55 million reads and 0.33 million mapped insertion sites for Ara+2, versus 0.39 million reads and 0.27 million insertion sites for Ara-1. The different coverage and capture protocols are why the paper treats the two lineages as related but distinct datasets.

The Ara+2 vectors used by the upstream fit code are:

```text
REL606: [7.8, 15.2, 21.5, 28.3, 34.6]
2K:     [6.0, 13.3, 19.6, 26.1, 32.7]
15K:    [8.7, 15.9, 22.2, 28.6, 35.7]
```

The pre-competition LB library sample is removed before pooling, so local `t1` through `t5` are the five competition samples, not a baseline-at-generation-zero series.

MmeI digestion captured the 14 genomic bases adjacent to the insertion. The pipeline:

1. finds the expected flanking sequence with at most one mismatch or indel;
2. maps the 14 bp tag to `NC_012967` with BWA, again allowing at most one edit;
3. retains a single best genomic location and reconciles paired reads;
4. excludes repetitive and mobile elements;
5. annotates both ORFs and intergenic regions;
6. divides each locus into five equal genomic-coordinate segments for Ara+2, except loci shorter than 100 bp, which are not divided;
7. pools the accepted insertion sites within a segment.

The lower-coverage Ara-1 pipeline used three segments per locus. This is another reason not to merge the two full lineages row-for-row without an explicit harmonization plan.

### How a pooled segment is formed

For local Ara+2 data, an individual insertion site enters a segment pool only if:

- its total count over the five retained timepoints is at least 10; and
- its `t1` count is at least 1.

The local pooling code uses `total >= 10`; the final supplement describes this threshold as `total > 10`. Later cleansing imposes the stronger first-timepoint requirement used for final fits.

The segment's `t1` through `t5` values are sums over those insertions. `abn` is the number of independent insertion positions pooled.

The upstream code keeps the first insertion row as a representative, then replaces its time counts and total with the pooled sums. Consequently:

- `site` is the genomic coordinate of the **first retained insertion in the pool**, not the segment start, midpoint, or mean insertion position;
- `pos` is the relative position of that representative first insertion within the full annotated locus;
- neither field fully describes the segment boundaries.

Segment numbering follows increasing reference coordinates. For a five-part forward-strand gene, `gene-1` is the low-coordinate segment and `gene-5` the high-coordinate segment. Interpret N- and C-terminal position using `strand`; do not assume `-1` always means the same protein terminus.

### Neutral normalization and fitting

The neutral reference set contains annotated cryptic genes and the L-arabinose transport/catabolism operons. Only internal segments are used to avoid terminal and polar effects.

For each timepoint:

1. collect neutral segments with sufficiently high initial counts;
2. determine the initial-count 25th percentile;
3. among neutral segments above that cutoff, take the 25th percentile count at that timepoint;
4. use those five values as the consensus wild-type trajectory.

Each segment's pooled raw counts are divided by the consensus. The slope of `ln(segment / consensus)` versus the measured generation vector is fitted three ways:

| Field | Regression |
|---|---|
| `fitted` | unweighted |
| `fitted1` | weighted by normalized abundance to the first power |
| `fitted2` | weighted by normalized abundance squared |

`fitted1` is the primary estimate used in the paper and local scripts.

At least two nonzero points are technically sufficient in the upstream code (`sum(zeros) <= 3` for five points), although the final supplement says mutants with fewer than three nonzero timepoints were discarded. The later quality filters remove many unstable cases.

The `sterr*` fields are slope standard errors and `pval*` are raw regression slope p-values, not multiple-testing-adjusted values. The `rmse*` names are misleading: the upstream code stores:

```text
max(Cook's distance) / min(Cook's distance)
```

They are not root mean squared errors.

### Exact local TSV schema

The header has 25 fields, while every data row has 26. R wrote row names as an unlabeled leading field. A generic parser must prepend `row_id`:

```text
row_id, ORF, alle, site, abn,
t1, t2, t3, t4, t5, tot, pos, strand,
fitted, fitted1, fitted2,
sterr, sterr1, sterr2,
pval, pval1, pval2,
rmse, rmse1, rmse2, qlty
```

Field definitions:

| Field | Meaning |
|---|---|
| `row_id` | R output row name; bookkeeping only |
| `ORF` | parent annotated locus label; despite the name, can be an ORF, intergenic region, pseudogene-like label, etc. |
| `alle` | pooled segment label and intended cross-background key |
| `site` | coordinate of first retained insertion in the pooled segment |
| `abn` | number of independent insertion positions pooled |
| `t1` ... `t5` | pooled raw read counts |
| `tot` | sum of `t1` through `t5` |
| `pos` | representative first insertion's relative coordinate within the full locus |
| `strand` | `F` forward or `C` complementary/reverse |
| `fitted*` | the three slope estimates |
| `sterr*` | corresponding slope standard errors |
| `pval*` | corresponding slope p-values |
| `rmse*` | Cook's-distance ratios, despite their names |
| `qlty` | locally appended placeholder; every checked-in value is 0 and it is not used upstream |

Pandas normally notices the extra leading field and uses it as the index, which makes the named columns appear correctly aligned. That inference is convenient but fragile. A deterministic loader is:

```python
from pathlib import Path
import pandas as pd

COUCE_COLUMNS = [
    "row_id", "ORF", "alle", "site", "abn",
    "t1", "t2", "t3", "t4", "t5", "tot", "pos", "strand",
    "fitted", "fitted1", "fitted2",
    "sterr", "sterr1", "sterr2",
    "pval", "pval1", "pval2",
    "rmse", "rmse1", "rmse2", "qlty",
]

df = pd.read_csv(
    Path("data/alex_code/Rfitted_fil.txt"),
    sep="\t",
    header=None,
    skiprows=1,
    names=COUCE_COLUMNS,
    na_values=["NA"],
)
assert len(df.columns) == 26
```

### Filtering encoded in the local files

The upstream cleansing logic:

- drops missing fits and segments with low first-timepoint counts;
- for positive estimates, rejects slope standard error above 0.01;
- removes poorly supported positive singleton loci;
- compares segments within a locus to identify suspected hitchhikers or artifacts;
- flags internal segments more than 0.01 from the locus average when the deviation is at least 25% of that average;
- handles terminal segments specially because real polar or change-of-function effects are possible;
- sets primary effects at or below `-0.5` to missing because that tail is not reliably measurable.

Rows rejected by within-locus filtering remain in the checked-in file. Their non-primary fit/error columns contain negative codes:

| Code | Upstream meaning |
|---:|---|
| `-103` | no reliable choice among a two-segment comparison, or the unchosen segment |
| `-105` | within-locus deviating segment |
| `-107` | insufficiently supported positive singleton or terminal segment |

The final `fitted1 <= -0.5` cleanup converts `fitted1` for these rows to `NA`, but other numeric fields can retain the negative codes. Therefore:

```text
valid local row = fitted1 is not NA
```

Never treat `-103`, `-105`, or `-107` in `fitted`, `fitted2`, `sterr*`, `pval*`, or `rmse*` as biological measurements.

### Local file audit

| File | Rows | Valid `fitted1` | `fitted1` NA | Unique parent labels | Excess rows after `site` deduplication |
|---|---:|---:|---:|---:|---:|
| `Rfitted_fil.txt` | 16,368 | 13,695 | 2,673 | 5,754 | 74 |
| `2Kfitted_fil.txt` | 16,832 | 13,820 | 3,012 | 5,783 | 69 |
| `15Kfitted_fil.txt` | 15,742 | 12,942 | 2,800 | 5,617 | 74 |

Duplicate `site` values arise largely because overlapping annotations can claim the same representative insertion coordinate. The local beneficial-tail scripts remove duplicate `site` values after filtering. Decide and document whether a new analysis is segment-, locus-, or physical-insertion-based.

### Operational effect thresholds

The Couce neutral calibration is:

```text
-0.015 < s < 0.015
```

The checked-in scripts define a high-confidence beneficial segment as:

```text
fitted1 > 0.015
fitted1 <= 0.3
abn > 1
fitted1 is not NA
then drop duplicate site values
```

That yields, locally:

| Background | Before `site` deduplication | After deduplication |
|---|---:|---:|
| Ancestor | 907 | 902 |
| 2K | 572 | 570 |
| 15K | 395 | 391 |

The UMI experiment uses other noise thresholds, including `s > 0.03` for a reliably beneficial UMI effect in a main-paper comparison. Do not transplant `0.015` between platforms or use `0.03` on the Couce files without explaining the changed estimand and sensitivity.

### Cross-background joins and sign epistasis

Use `alle`, not `site`, to join the same annotated segment across ancestor, 2K, and 15K:

```python
paired = (
    ancestor[["alle", "fitted1"]]
    .rename(columns={"fitted1": "s_anc"})
    .merge(two_k[["alle", "fitted1"]].rename(columns={"fitted1": "s_2k"}),
           on="alle", how="inner")
    .merge(fifteen_k[["alle", "fitted1"]].rename(columns={"fitted1": "s_15k"}),
           on="alle", how="inner")
)
```

Filter `fitted1.notna()` in each source before joining. Use an inner join for a fully paired comparison. If using a left or outer join, missing segments mean unmeasured/filtered, not neutral; do not fill them with zero.

The local plotting scripts select beneficial segments in one background and then look up their `alle` values in other backgrounds. They do not always require presence in all three backgrounds. Be explicit about whether percentages use the selected set, the intersection, or all measurable segments.

### Local plotting scripts are illustrative, not a canonical pipeline

Files:

```text
data/alex_code/beneficials_accross_backgrounds.R
data/alex_code/overlaps.R
data/alex_code/segben.py
data/alex_code/overlapping_dfes.py
```

They reproduce selected figures from the three filtered Ara+2 tables. They are not the raw-to-fitness pipeline and omit the Ara-1 and predictability analyses.

Both the R and Python versions allocate the 15K comparison object `Fepi` using `length(Tnames)` rather than `length(Fnames)`. Rebuild paired data with explicit joins instead of copying this allocation pattern into a new analysis.

## Comparing or combining the two experiments

### Reasonable common quantities

At a high level, both provide a selection-coefficient-like slope in DM25 on an REL606 coordinate system. Reasonable comparisons include:

- sign of an effect;
- whether the beneficial tail contracts with adaptation;
- whether a gene or locus changes qualitative fitness class;
- rank-based comparisons after defining a common gene/locus aggregation;
- presence versus absence of a broad target among high-confidence effects.

### Quantities that are not directly exchangeable

| Limdi / UMI | Couce / INSeq |
|---|---|
| protein-coding genes only in processed fitness array | ORFs plus intergenic regions and other locus labels |
| interior 65% of gene | equal genomic segments, including terminal segments |
| per-site slopes averaged to gene | insertions pooled first, then one segment slope |
| total-library normalization plus pseudogene centering | neutral-consensus normalization |
| two replicate assays | no equivalent replicate axis in local filtered table; within-locus segments provide a different reproducibility check |
| idealized 6.64 generations/day | measured generation vectors |
| five points from common baseline through four days | five post-LB competition samples over five days |
| primary focus at 50K across all lines | local focus at 0, 2K, and 15K in Ara+2 |
| `-1` missing sentinel | `NA` primary fit plus negative codes in secondary fields |

### If a gene-level harmonized table is required

Make the transformation explicit:

1. Start from valid Couce `fitted1` rows.
2. Decide whether to exclude terminal segments, intergenic regions, overlaps, and segments with `abn == 1`.
3. Map Couce `ORF` or segment labels to REL606 Prokka locus tags using coordinates/annotation, not gene symbol alone.
4. Aggregate Couce segments to one gene value with a predeclared rule: mean, median, internal-segment mean, maximum beneficial value, or another biologically motivated summary.
5. Keep Limdi's strict and relaxed estimates in separate fields.
6. Add platform, background generation, aggregation rule, coverage, and number of contributing insertions as columns.
7. Use platform-specific quality flags and thresholds.
8. Prefer within-platform effect changes before comparing cross-platform absolute values.

Exact Couce segment boundaries are not stored in the three local filtered tables: `site` and `pos` describe only the first retained insertion. A rigorous coordinate remap requires the upstream divided-annotation file, named `parsed_R606genoscope_IUD.txt` for Ara+2, or an equivalent reconstruction from the same reference annotation. Do not infer a segment interval from `site` alone.

For the paper's predictability analysis, Couce effects were eventually aggregated at the operon level, often taking the maximum fitness among relevant constituents and using summed target length. The required operon and metagenomic inputs are not present in `data/alex_code`; the local three files alone cannot reproduce that result.

## Minimal safe loading recipe for Limdi arrays

```python
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/anurag_data")
META = ROOT / "Metadata" / "all_metadata_REL606.txt"
PROC = ROOT / "Analysis" / "Part_3_TnSeq_analysis" / "Processed_data_for_plotting"

backgrounds = [
    "REL606", "REL607", "Ara-1", "Ara-2", "Ara-3", "Ara-4", "Ara-5",
    "Ara-6", "Ara+1", "Ara+2", "Ara+3", "Ara+4", "Ara+5", "Ara+6",
]

metadata = pd.read_csv(META, sep="\t")
metadata.columns = metadata.columns.str.strip()
metadata.insert(0, "gene_row", np.arange(len(metadata)))

fitness = np.load(PROC / "fitness_corrected_genes.npy")
relaxed = np.load(PROC / "fitness_genes_relaxed_thresholds_updated.npy")
error = np.load(PROC / "errors_genes_inv.npy")
deleted = np.loadtxt(
    ROOT / "Analysis" / "Part_2_WGS_analysis" / "output"
    / "Deleted_genes_REL606_k12annotated.txt"
)

assert fitness.shape == (len(metadata), len(backgrounds), 2)
assert relaxed.shape == fitness.shape
assert error.shape == (len(metadata), len(backgrounds))
assert deleted.shape == (len(backgrounds), len(metadata))

valid = (fitness != -1).all(axis=2)
s_mean = fitness.mean(axis=2)
s_mean[~valid] = np.nan

# Main-paper background exclusion:
included = np.ones(len(backgrounds), dtype=bool)
included[[3, 11]] = False
```

Before analyzing essentiality, add the five-interior-TA-site gate if the goal is to follow the final supplement, apply the structural deletion mask, and keep strict versus relaxed values distinct.

## Analysis checklist

Before reporting any result, answer all of these:

1. Which experiment and local files are being used?
2. Is the unit an insertion, pooled segment, gene, homolog family, or operon?
3. Which backgrounds and generations are included?
4. Were Ara-2 and Ara+4 excluded where appropriate?
5. How were `-1`, `NA`, and negative Couce filter codes handled?
6. Was the stable key a Limdi metadata row/Prokka tag or a Couce `alle` label?
7. Were overlapping Couce annotations deduplicated, and by what rule?
8. Were deleted genes removed before essentiality or cross-background comparisons?
9. Was the final-paper five-interior-TA-site rule applied to the older local Limdi arrays?
10. Were strict and relaxed Limdi estimates kept conceptually separate?
11. Which neutral/beneficial/deleterious thresholds were used, and are they platform-specific?
12. Was `s` compared only with other values on the same definition/scale?
13. If expression zeros were used, were unmapped rows distinguished from true zero TPM?
14. Was one row mask applied consistently to every row-aligned Limdi object?
15. Is the local-data limitation, especially missing Couce Ara-1 or missing Limdi trajectories, stated?

## Source hierarchy and provenance

Use this priority order:

1. `Theory/Baym_supp.pdf` for the final published experimental and statistical methods.
2. The actual checked-in data values and local generating code for the schema and behavior of local files.
3. `Theory/Baym24.pdf` for the final narrative, included/excluded analyses, and reported results.
4. `Theory/couce.pdf` and `Theory/limdi.pdf` for the history and fuller framing of the original separate projects.
5. The archived GitHub and Zenodo records when a missing raw/intermediate file or exact upstream pipeline stage is required.

When these disagree, do not silently choose one. State whether the goal is to reproduce the final publication, reproduce the checked-in intermediate, or conduct a new analysis. The local Limdi arrays and notebooks preserve some pre-final implementation details, while the local Couce directory is a narrow, filtered Ara+2 extract rather than the complete published data package.
