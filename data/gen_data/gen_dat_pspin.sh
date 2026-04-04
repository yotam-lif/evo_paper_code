#!/bin/bash

# Load Miniconda module
module load miniconda/24.9.2_environmentally

# Activate the Conda environment
conda activate my_env

# Run the Python script
python gen_dat_pspin.py --N 40 --P 1 --n_repeats 2

# Deactivate the environment after the script finishes
conda deactivate
