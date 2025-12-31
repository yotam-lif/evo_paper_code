#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#in this script, I'm going to merge the data from lane 3 and lane 4 of the novaseq
#for each sample.
#then I'm going to make a list of the UMI+read2 mapping location. This will serve as
#a final unique molecular identifier.
#I will then discard PCR duplicates and then output a file with the number of reads for 
#each TA site represented in the data.

import pysam
import numpy as np 
import sys
import os

#list of all parameters from command line
params = sys.argv

#this asserts that we have at least four input params from command line. And breaks
#the program if condition is not satisfied.
assert (len(params)==6), "Usage: %s sam_input_directory_read1 sam_input_directory_read2 UMI_list_input_directory indices_PF_input_directory output_directory" % (params[0])
print(params[1])

#defining a function to return the array with positions of all the mapped reads
def get_positions(sam_file):
    positions = []
    for read in sam_file:
        pos = pysam.AlignedSegment.get_reference_positions(read, full_length=False)
        #check that the read is mapped (first statement), check that the read is uniquely aligned (second statement)
        #if the read is mapped to multiple locations in the genome, it'll have an XS tag
        #uniquely mapped reads don't have the XS tag
        if pos != [] and read.has_tag('XS') == False:
            #if read is on the forward strand
            if read.flag==0:
                positions.append(pos[0])
            #if read is on the reverse strand
            elif read.flag==16:
                positions.append(pos[-2]) 
        #if the read is not mapped or multiply mapped, I am excluding it from analysis by appending a -1 to the positions matrix
        else:
            positions.append(-1)

    positions = np.array(positions)
    return positions

#Iterate over every sam file in the input folder
for filename in os.listdir(params[1]):
    #checking for correct filename extension
    if filename.endswith(".sam"):
        #Only merging each time I have an l3 dataset. This way, the script only runs the following code once for 
        #each sample in sequencing data
        if 'L003' in filename: 
            #deducing the filename for the lane 4 data
            l3name = filename
            l4name = filename[:-12]+'4'+filename[-11:]

            l3name_r2 = l3name[:-9]+'2'+l3name[-8:]
            l4name_r2 = l4name[:-9]+'2'+l4name[-8:]

            #opening the sam file corresponding to the read 1 data
            sam3_r1 = pysam.AlignmentFile(os.path.join(params[1], l3name), 'r')
            sam4_r1 = pysam.AlignmentFile(os.path.join(params[1], l4name), 'r')
            #opening the sam file corresponding to the read 2 data
            #note that the read2 sam files may be in a different directory
            sam3_r2 = pysam.AlignmentFile(os.path.join(params[2], l3name_r2), 'r')
            sam4_r2 = pysam.AlignmentFile(os.path.join(params[2], l4name_r2), 'r')

            #positions array where I store the location for the mapped reads
            positions_l3_r1 = get_positions(sam3_r1)
            positions_l4_r1 = get_positions(sam4_r1)
            positions_l3_r2 = get_positions(sam3_r2)
            positions_l4_r2 = get_positions(sam4_r2)

            #loading the UMI list and indices of reads passing filter
            #now, let's open the corresponding UMI list file:
            UMI_fname_3 = os.path.join(params[3], "UMI_"+l3name[:-4]+ ".txt")
            print(UMI_fname_3)
            with open(UMI_fname_3, 'r') as f:
                UMI_list_3 = f.read().splitlines()  

            UMI_fname_4 = os.path.join(params[3], "UMI_"+l4name[:-4]+ ".txt")
            print(UMI_fname_4)
            with open(UMI_fname_4, 'r') as f:
                UMI_list_4 = f.read().splitlines() 

            #now loading the indices of reads PF
            print(os.path.join(params[4], "PF_"+l3name[:-4]+ ".index"))
            indices_pf3 = np.loadtxt(os.path.join(params[4], "PF_"+l3name[:-4]+ ".index"))
            indices_pf4 = np.loadtxt(os.path.join(params[4], "PF_"+l4name[:-4]+ ".index"))
            #converting data type to ints
            indices_pf3 = indices_pf3.astype('int')
            indices_pf4 = indices_pf4.astype('int')

            #slicing out the data for read2 corresponding to reads which passed the filter in 
            #read 1 (i.e. contain the transposon sequence)
            positions_l3_r2_PF = positions_l3_r2[indices_pf3]
            positions_l4_r2_PF = positions_l4_r2[indices_pf4]

            #assert that the length of the r1 and r2_PF positions array is equal
            assert (len(positions_l3_r1)==len(positions_l3_r2_PF)), "Number of reads in R1 and R2 data is not equal"
            assert (len(positions_l4_r1)==len(positions_l4_r2_PF)), "Number of reads in R1 and R2 data is not equal"

            #now let's create a list of UMI's where we incorporate the r2 mapping coordinate
            #in addition to the sequence of the UMI, for extra information
            UMI_list_l3_r2 = [UMI_list_3[x]+str(positions_l3_r2_PF[x]) for x in range(0,len(UMI_list_3))]
            UMI_list_l4_r2 = [UMI_list_4[x]+str(positions_l4_r2_PF[x]) for x in range(0,len(UMI_list_4))]

            #Now, combining the data from the two lanes into a single array
            positions_r1_combined = np.hstack([positions_l3_r1,positions_l4_r1])
            #note that I'm not doing this for the r2 data as it's sole purpose in this
            #analysis is to add an extra layer of information to the UMI data
            umi_list_r2_combined = UMI_list_l3_r2+UMI_list_l4_r2

            #assert that the combined positions and the UMI list have the same length
            assert (len(positions_r1_combined)==len(umi_list_r2_combined)), "Number of UMIs and mapped reads is not equal"

            #counting how many unique data points we have, and corresponding counts
            unique, counts = np.unique(positions_r1_combined, return_counts=True)
            #cumulative sum
            cumulative = np.cumsum(counts)
            #array where we'll be storing the UMI corrected data
            counts_UMI = np.zeros([len(unique)])
            #now I'm sorting all the UMIs according to the position in the genome where the corresponding read maps
            sorted_UMIs = [x for _,x in sorted(zip(positions_r1_combined,umi_list_r2_combined))]
            start = 0
            for i in range(0, len(unique)):
                end = cumulative[i]
                #now I'll slice out the part of the UMI list corresponding to this start and end point,
                #convert it to a set (this will automatically remove duplicates)
                #and then store the length of the set in the counts_UMI array
                UMI_check = set(sorted_UMIs[start:end])
                counts_UMI[i] = len(UMI_check)
                start = end
            
            #now I'm going to construct a matrix with the following structure:
            #row 0 - list of all the unique positions
            #row 1 - counts corresponding to those positions
            #row 2 - UMI corrected counts corresponding to those positions

            #the first element of unique is -1, corresponding to all the unmapped reads.
            #those will not be a part of my analysis in any case.
            #hence I splice as [1:]
            position_counts = np.zeros([3, len(unique)-1])
            position_counts[0, :] = unique[1:]
            position_counts[1, :] = counts[1:]
            position_counts[2, :] = counts_UMI[1:]

            #saving the position data
            fname_out = os.path.join(params[5], l3name[:-15]+"merged.pos")
            np.savetxt(fname_out, position_counts)


