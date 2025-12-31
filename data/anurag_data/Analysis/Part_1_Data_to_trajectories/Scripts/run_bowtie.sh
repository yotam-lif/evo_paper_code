#!/usr/bin/env bash
#script to run bowtie on every file in a directory, and save the sam files to
#another directory

set -e #when something fails, stop immediately
set -o pipefail #when there's a pipe and something in the pipe fails, stop there

#include the if condition to check for number of input parameters
if [[ "$#" -ne 3 ]]; then
    >&2 echo "usage:" $(basename $0) "input_path output_path reference_genome"
    exit 1
fi

reference="$3"
input="$1"
output="$2"

for file in "$input"/*.fastq;
do
    echo "$file"
    int=$(basename "$file" .fastq)
    out=$int.sam
    echo "$out"
    full="$output/$out"
    echo "$full"
    bowtie2 -q -x "$reference" -U "$file" -S "$full";
done