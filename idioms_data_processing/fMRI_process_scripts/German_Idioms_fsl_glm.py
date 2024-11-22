#!/usr/bin/env python
from subprocess import call
import os, socket
import re

outfolder = "/Users/helenwu/Desktop/German_idioms/ds002727"

template = "/Users/helenwu/Desktop/German_idioms/German_Idioms_glm_template.fsf"

# group_A = [1,4,8,13,17,21]
# group = "GroupA"
# # group_B = [3,6,9,12,15,19,23]
# group = "GroupB"
# group_C = [5,11,14,18,22]
# group = "GroupC"
group_D = [7,10,16,20]
group = "GroupD"


for subject_num in group_D:

    if subject_num < 10:
        subject = "sub-0" + str(subject_num)
    else:
        subject = "sub-" + str(subject_num)

    for run_num in range(1,4):
        run = "run" + str(run_num)
        run_BOLD = "run-" + str(run_num)
        print("Working on run", run, run_BOLD, subject, group)

        #Create design file for this subject/run
        outfile = "%s/%s/run%d.fsf" % (outfolder,subject,run_num)
        print ("Will create %s" % outfile)
        command = "sed -e \'s/DEFINESUBJECT/%s/g\' -e \'s/DEFINEGROUP/%s/g\' -e \'s/DEFINERUN/%s/g\' -e \'s/DEFINEBOLDRUN/%s/g\' %s > %s" % (subject,group,run,run_BOLD,template,outfile)
        print(command)
        call(command, shell=True)
    
        #run feat
        command = "feat %s" % outfile
        print(command)
        call(command, shell=True)
    


