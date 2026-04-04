#!/bin/bash

#BSUB -q long
#BSUB -J pspin
#BSUB -n 10
#BSUB -R "span[hosts=1] rusage[mem=3000]"
#BSUB -M 3000
#BSUB -o pspin.%J.out
#BSUB -e pspin.%J.err

# Load Miniconda module
module load miniconda/24.9.2_environmentally

## Initialize Conda for this non-interactive shell
#source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate my_env

# Run the Python script
python gen_dat_pspin.py --N 400 --P 3 --n_repeats 10