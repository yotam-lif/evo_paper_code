## Part 3: TnSeq Analysis and Plotting

In this section, I do the final analysis of the TnSeq data: estimating fitness effects of insertion mutations and inferring gene essentiality. I also do additional exploratory analyses looking at gene essentiality and gene expression levels, and interplay between structural variation and gene essentiality.

### Exploratory_analysis

This contains jupyter notebooks that carry out the following data reformatting and trimming

- expression_levels_data_reformatting.ipynb: here, I convert the RNAseq data from the Favate et al study to a matrix containing gene expression levels for every protein coding gene (excluding anything related to mobile genetic elements) that we analyze in our project
- homolog_pairs.ipynb: here, I use the output from running mmseqs on the E. Coli REL606 reference genome (which detects potential homologs from amino acid identity), and write an output file which contains only gene pairs which have exactly one homolog.

### Fitness_estimation

This contains jupyter notebooks that carry out the following analysis

- fitness_calculations.ipynb: here, I calculate the fitness effect of transposon insertion mutations by fitting a line to log(frequency) vs time. 
    - also includes a panel for Fig S4
- Essentiality threshold.ipynnb: here, I simulate the expected variability in the fitness estimates from TnSeq assays for essential genes and genes with significant growth deficit. The goal is to identify a threshold at which we can reliably distinguish between essentiality and fitness deficit.
    - also includes a panel for Fig S8

### Processed_data_for_plotting

All the intermediate data needed to generate the final plots is saved here such as fitness estimates, RNAseq expression levels, homolog pairs, etc.

### generate_figures_main.ipynb

this is the main jupyter notebook where the analysis comes together. I do the following analyses, and generate the corresponding plots for the paper:
- comparing the distribution of fitness effects for the ancestors and evolved clones
- inferring differential gene essentiality between ancestors and evolved clones
- parallelism analysis of gene essentiality changes
- gene expression levels and gene essentiality changes

### generate_figures.ipynb

this notebook has the older versions of the plots that are now reimplemented in generate_figures_main.ipynb, but include a few supplementary figures which have remained unchanged through the review