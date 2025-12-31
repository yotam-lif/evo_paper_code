## LTEE-TnSeq Data Analysis, Part 1: From sequencing data to mutant trajectories

NOTE: Because there is a lot of data, I split up the data into three sets of samples to process:
- anc_methods: ancestor and methods analysis data
- ara_minus: all the ara-minus clones derived from REL606
- ara_plus: all the ara-plus clones derived from the ancestor

### Step 1: Filtering R1 reads containing transposon sequence (permitting one mismatch)

Run bash script run_filter_positions.sh

- This will run the python script filter_trim_positions.py
- Returns filtered reads, UMIs, and coordinates of reads which passed the filter

### Step 2: Mapping R1 filtered reads and all R2 reads to reference genome

Run bash script run_bowtie.sh

- Returns .sam files for each sample

### Step 3: Counting number of reads mapping to each TA insertion site, correcting for PCR amplification bias

Run bash script run_merge_get_locations.sh

- This will run the python script merge_get_locations.py
- This script merges data from Lane 3 and Lane 4 of the NovaSeq (note that we ran two lanes for our samples)
- This returns text files (.pos) which include TA sites represented in data, and number of counts over the fitness assay

### Step 4: Merged mutant trajectory

Run the Jupyter notebook merging_counts_data.ipynb

- This will convert the (.pos) files to text files containing number of counts over time for all TA sites (including ones that are not represented in the sample)
- This is to standardize data and make sure we can combine data from all libraries in a three dimensional matrix (with 5 time points, 211995 TA sites, 14 libraries X 2 replicates) 
- Replicates are labelled as red or green (because I used those colored sharpies when doing the experiments)

