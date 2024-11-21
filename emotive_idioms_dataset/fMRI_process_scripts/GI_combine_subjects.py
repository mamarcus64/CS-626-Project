import csv
import os
import numpy as np

import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 

from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nibabel as nib
from brainiak import image, io
import h5py
import pandas as pd


# handle exclusion subjects
subj_exclude = [2,7,10,20]
# subj_exclude = []
# subj_arr = np.arange(1, 2)
subj_arr = np.arange(1, 24)
subj_arr = subj_arr[~np.isin(subj_arr, subj_exclude)]
print(subj_arr)

# # Load the mask, functional whole-brain
mask_file = "/Users/helenwu/Desktop/German_idioms/ds002727/mask.nii.gz"
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()
mask_data = mask_data.flatten()
print("mask_data.shape", mask_data.shape)


file_name_base = "/Users/helenwu/Desktop/German_idioms/ds002727/"


# outer loop for each subject
subject_list = []
for subj in subj_arr:

    # separate subjects into groups
    if subj in [1,4,8,13,17,21]:
        group_name = "GroupA"
    elif subj in [3,6,9,12,15,19,23]:
        group_name = "GroupB"
    elif subj in [5,11,14,18,22]:
        group_name = "GroupC"
    elif subj in [7,10,16,20]:
        group_name = "GroupD"

    # create subj_name
    if subj < 10:
        subj_name = "sub-0" + str(subj) + "/"
    else:
        subj_name = "sub-" + str(subj) + "/"

    print(subj_name, group_name)

    # use corresponding sentence-ev mappings 
    file_path1 = "/Users/helenwu/Desktop/German_idioms/" + group_name + "/Stimuli_" + group_name + "_run1_Sorted.csv"
    file_path2 = "/Users/helenwu/Desktop/German_idioms/" + group_name + "/Stimuli_" + group_name + "_run2_Sorted.csv"
    file_path3 = "/Users/helenwu/Desktop/German_idioms/" + group_name + "/Stimuli_" + group_name + "_run3_Sorted.csv"


    # Column index to read (0-based index) ID_Code column
    column_index = 3

    # Read the file without a header
    data1 = pd.read_csv(file_path1, header=None)
    data2 = pd.read_csv(file_path2, header=None)
    data3 = pd.read_csv(file_path3, header=None)

    # Extract the column, ID_Codes
    run1 = data1.iloc[:, column_index].tolist()
    run2 = data2.iloc[:, column_index].tolist()
    run3 = data3.iloc[:, column_index].tolist()


    # creating localizer mask for this subeject

    # get the ## pe file (pe 71)
    hashtag_pe_filename = file_name_base + subj_name + "/run3_full.feat" + "/stats/" + "pe71.nii.gz"
    hashtag_brain_img = nib.load(hashtag_pe_filename)
    hashtag_brain_data = hashtag_brain_img.get_fdata()
    hashtag_brain_data = hashtag_brain_data.flatten()

    # only get the voxels within the whole-brain mask
    hashtag_BOLD_1d = hashtag_brain_data[mask_data == 1]
    print("hashtag_BOLD_1d", hashtag_BOLD_1d.shape)

    # get the filler sentence files (pe 61 to 70)
    filler_brain_files = []
    for ev_num in range(61,71):
        filler_pe_num = (ev_num * 2) - 1
        filler_pe_filename = file_name_base + subj_name + "/run3_full.feat" + "/stats/" + "pe"+ str(filler_pe_num) + ".nii.gz"
        filler_brain_img = nib.load(filler_pe_filename)
        filler_brain_data = filler_brain_img.get_fdata()
        filler_brain_data = filler_brain_data.flatten()

        # only get the voxels within the whole-brain mask
        filler_BOLD_1d = filler_brain_data[mask_data == 1]
        filler_brain_files.append(filler_BOLD_1d)


    filler_brain_files = np.array(filler_brain_files)
    print("filler_brain_files", filler_brain_files.shape)
    mean_filler = np.mean(filler_brain_files, axis=0)
    print("mean_filler", mean_filler.shape)
    print()

    # voxels where postive activity to filler sentences > postive activity to hashtag sentences 
    semantic_mask = np.where((mean_filler > hashtag_BOLD_1d) & (hashtag_BOLD_1d > 0) & (mean_filler > 0), 1, 0)
    print("semantic_mask", semantic_mask.shape)
    count_ones = np.sum(semantic_mask)
    print("number of ones", count_ones)
    print()

    # iterate through each sentence for this subject
    sentence_list = []
    for i in range(1,181):

        pe_num = -1

        # find the sentence id code
        if i in run1:
            feat_run = "run1_full.feat"
            row_idx = run1.index(i)
            pe_num = data1.iloc[row_idx, 4]
            pe_num = (pe_num * 2) - 1
            pe_num = int(pe_num)
            # print("1", pe_num)

        elif i in run2:
            feat_run = "run2_full.feat"
            row_idx = run2.index(i)
            pe_num = data2.iloc[row_idx, 4]
            pe_num = (pe_num * 2) - 1
            pe_num = int(pe_num)
            # print("2", pe_num)

        elif i in run3:
            feat_run = "run3_full.feat"
            row_idx = run3.index(i)
            pe_num = data3.iloc[row_idx, 4]
            pe_num = (pe_num * 2) - 1
            pe_num = int(pe_num)
            # print("3", pe_num)

        # Load the whole-brain image (ev file)
        pe_file = file_name_base + subj_name + feat_run + "/stats/" + "pe"+ str(pe_num) + ".nii.gz"
        # print(pe_file)
        whole_brain_img = nib.load(pe_file)
        whole_brain_data = whole_brain_img.get_fdata()
        whole_brain_data = whole_brain_data.flatten()

        # print("whole_brain_data", whole_brain_data.shape)

        # mask the data, and append it to the sentence*voxels matrix
        bold_1d = whole_brain_data[mask_data == 1]  # first get wholebrain
        bold_1d = bold_1d[semantic_mask == 1]       # second get semantic-specific voxels
        bold_1d = bold_1d[0:400]                    # third ensure the same shape
        sentence_list.append(bold_1d)

    # sentence_list now contains neural patterns for all sentences for 1 subject
    sentence_list = np.array(sentence_list)
    print("sentence_list", sentence_list.shape, "subj", subj)

    # append this subject's matrix to previous subjects
    subject_list.append(sentence_list)

subject_list = np.array(subject_list)
print("subject_list", subject_list.shape, "all")




# sanity check --------------------------------------------------
print(subject_list[0,0,100:110])
transposed_subject_list = np.transpose(subject_list, (2, 0, 1))

print("transposed_subject_list shape", transposed_subject_list.shape)
print(transposed_subject_list[100:110,0,0])

h5_filename = "/Users/helenwu/Desktop/German_idioms/fMRI_masked_semantic_data.h5"
with h5py.File(h5_filename, 'w') as f:
    f.create_dataset('fMRI_sentences', data=transposed_subject_list)

print()

# h5_filename = "path/to/fMRI_masked_semantic_data"
with h5py.File(h5_filename, 'r') as f:
    loaded_array = f['fMRI_sentences'][:]
    
print("loaded_array",loaded_array.shape)
# print(loaded_array[100:110,0,0])








