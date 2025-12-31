#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#filter (and trim, set this as optional) reads for all fastq files in a folder.
#addition: for library preps with UMIs, also output the UMI in a different text file
#second addition: also output the indices of reads which pass the filter

import time
from Bio.SeqIO.QualityIO import FastqGeneralIterator
import os
import numpy as np 
import sys
import regex

mariner_seq = "(GGGGACTTATCAGCCAACCTGTTA)"
#time how long the trimming takes!
t0 = time.time()

#parameters from user
params = sys.argv

#this asserts that we have four input params from command line. And breaks
#the program if condition is not satisfied.
assert (len(params)==3), "Usage: %s input_directory output_directory" % (params[0])


#iterate over all the fastq files in the directory and filter
for filename in os.listdir(params[1]):
    #checking for correct filename extension
    if filename.endswith(".fastq"):

        indices = []
        counter = 0

        out_name = params[2] + "/" + filename
        #the UMIs all go into one text file
        out_name_2 = params[2] + "/UMI_" + filename[:-6] + '.txt'
        #creating a new file with list of indices
        out_name_3 = params[2] + "/PF_" + filename[:-6] + '.index'
        with open(out_name_2, 'x') as out_handle_2:
            #creating an new file where I will write the trimmed code: x is the argument for creating/writing mode
            with open(out_name, "x") as out_handle:
                #opening the sequencing data files. Opening the read 1 file here
                with open(os.path.join(params[1], filename)) as in_handle:
                    #iterate over each read in the FastQ file. 
                    for title, seq, qual in FastqGeneralIterator(in_handle):
                        #increase iterator by 1
                        counter+=1
                        #by adding this if statement, we are getting rid of all the reads which are Ns
                        #or don't contain the mariner sequence
                        out = regex.search(mariner_seq+"{e<=1}", seq)
                        if out!=None:
                            #the following line defines where to trim out the transposon sequence. Doing this
                            #retains the TA sequence right at the end of the transposon
                            start = out.span()[1]-2
                            out_handle.write("@%s\n%s\n+\n%s\n" % (title, seq[start:], qual[start:]))
                            #UMIs are in the first 10 bases of the read. There's no additional processing that
                            #needs to be done here.
                            out_handle_2.write("%s\n" % seq[:10])
                            indices.append(counter-1)
        indices = np.array(indices)
        np.savetxt(out_name_3, indices)
t1 = time.time()
print(t1-t0)