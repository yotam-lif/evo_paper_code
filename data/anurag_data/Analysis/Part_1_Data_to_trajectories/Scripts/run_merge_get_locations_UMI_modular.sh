#!/bin/bash
#SBATCH -c 2    # Request 15 cores
#SBATCH -t 0-11:59   # Set max time to be 11 hours 59 minutes
#SBATCH -p short     # Use the short partition
#SBATCH --mem=64G    # Request 150 Gigs of memory
#SBATCH -o est_%j.out
#SBATCH -e est_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alimdi@g.harvard.edu

set -e #when something fails, stop immediately
set -o pipefail #when there's a pipe and something in the pipe fails, stop there

# include the if condition to check for number of input parameters
if [[ "$#" -ne 6 ]]; then
    >&2 echo "usage:" $(basename $0) "script path_sam_r1 path_sam_r2 path_UMI_list path_indices_PF output_path"
    exit 1
fi

# first, open the envtnseq environment using the following command
source activate env_tnseq

#folder name:
script="$1"
sam_r1="$2"
sam_r2="$3"
umi="$4"
indices="$5"
output="$6"

# run the python script: note that the script will run on all the sam files in a directory
# arguments: location of r1 sam files, r2 sam files, umi list, indices, output

python "$script" "$sam_r1" "$sam_r2" "$umi" "$indices" "$output"

