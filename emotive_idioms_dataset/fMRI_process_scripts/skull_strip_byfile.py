#!/usr/bin/env python


from __future__ import division
import sys
import os
import nibabel as nib
import numpy as np
import scipy.stats
from subprocess import call 
import argparse
import shutil


def threshold(a, threshmin=None, threshmax=None, newval=0):
    a = np.ma.array(a, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin).filled(False)

    if threshmax is not None:
        mask |= (a > threshmax).filled(False)

    a[mask] = newval
    return a



# # main stript starts from here ---------------------------------
# #command line options
parser = argparse.ArgumentParser()
parser.add_argument("--infile",help="input data file",action="store")
parser.add_argument("--outfile",help="output data file",action="store")
args = parser.parse_args()


infile = args.infile
outfile = args.outfile
print("infile", infile)


#load in mprage 
img = nib.load(infile)
xdim,ydim,zdim = img.shape
data = img.get_fdata()

#threshold to make background zero
clipped = threshold(data,250)

#find x center
begin = 0
end = 0
for x in range(xdim):

    count = clipped[x,:,:].sum()
    #print "%d: %d" % (x,count)
    if count > 0:
        if begin==0:
            begin = x
    else:
        if begin > 0 and end==0:
            end = x
if end == 0:
    end = xdim-1

xsize = end - begin
xradius = round(xsize/2)
xcenter = xradius + begin

print("Image starts at %d and ends at %d in x dimension.  Center is at %d" % (begin, end, xcenter))


#find y center
begin = 0
end = 0
for y in range(ydim):
    count = clipped[:,y,:].sum()
    if count > 0:
        if begin==0:
            begin = y
    else:
        if begin > 0 and end==0:
            end = y
if end == 0:
    end = ydim-1

ysize = end - begin
yradius = round(ysize/2)
ycenter = yradius + begin

print("Image starts at %d and ends at %d in y dimension.  Center is at %d" % (begin, end, ycenter))

#find z center
begin = 0
end = 0
for z in range(zdim):
    count = clipped[:,:,z].sum()
    if count > 0:
        if begin==0:
            begin = z
    else:
        if begin > 0 and end==0:
            end = z

if end == 0:
    end = zdim-1

zsize = end - begin
zradius = round(zsize/2)
zcenter = zradius + begin

print("Image starts at %d and ends at %d in z dimension.  Center is at %d" % (begin, end, ycenter))


#make small adjustments for head and neck
zcenter = zcenter + 10
ycenter = ycenter - 5

betcommand = "bet %s %s -c %d %d %d -B -f .38 -o" % (infile, outfile, xcenter, ycenter, zcenter)

print(betcommand)
call(betcommand,shell=True)

