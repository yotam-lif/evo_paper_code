#!/usr/bin/env bash
#SBATCH -c 2    # Request 2 cores
#SBATCH -t 0-11:59   # Set max time to be 11 hours 59 minutes
#SBATCH -p short     # Use the short partition
#SBATCH --mem=10G    # Request 240 Gigs of memory
#SBATCH -o est_%j.out
#SBATCH -e est_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alimdi@g.harvard.edu

# script to take in reference genome and list of fastqs and run breseq
# first, I need to activate the environment containing breseq
source activate env_tnseq

# include the if condition to check for number of input parameters
if [[ "$#" -ne 5 ]]; then
    >&2 echo "usage:" $(basename $0) "reference_genome fastq_file1 fastq_file2 fastq_file3 fastq_file4"
    exit 1
fi

# run breseq using the following command: 
breseq -r "$1" "$2" "$3" "$4" "$5" -j 2
