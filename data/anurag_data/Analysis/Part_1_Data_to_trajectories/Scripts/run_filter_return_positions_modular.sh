#!/bin/bash
#SBATCH -c 4    # Request 4 cores
#SBATCH -t 1-00:00   # Set max time to be 1 day
#SBATCH -p medium     # Use the medium partition
#SBATCH --mem=10G    # Request 40 Gigs of memory
#SBATCH -o est_%j.out
#SBATCH -e est_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="your email here"

set -e #when something fails, stop immediately
set -o pipefail #when there's a pipe and something in the pipe fails, stop there

# include the if condition to check for number of input parameters
if [[ "$#" -ne 6 ]]; then
    >&2 echo "usage:" $(basename $0) "script input_path output_path"
    exit 1
fi

#folder name:
script="$1"
input="$2"
output="$3"

# first, open the kraken environment using the following command
source activate env_tnseq

python "$script" "$input" "$output"
 
