#!/usr/bin/python

#merge_chains: Python script to merge MCMC chains into a single chain
#Takes a series of directories as arguments, e.g.
# python concatenate.py 185866 185867 185868 185869 186948 186949 186950 186964
#will create the directory 185866_186964 with a merged samples.npy inside
#
#The first row of each samples.npy, which holds summary information about the chain, 
#is discarded. However, a row of zeros is included at the top of the resulting samples.npy 
#for consistency with individual chains. So like individual samples.npy, rows 1: will 
#contain the MCMC samples.

import sys
import numpy as np
import os

nburn=0

if len(sys.argv)<=2:
  print "Must provide more than one directory"
else:
  outdir = sys.argv[1]+"_"+sys.argv[-1]

  #keeping the first row here to keep consistency with structure of 
  #single runs but zeroing it out because it doesn't mean anything
  samples = np.load(sys.argv[1]+"/samples.npy")[nburn:][:]
  samples[0,:]=0.

  for dir in sys.argv[2:]:
    samp = np.load(dir+"/samples.npy")[1+nburn:][:]
    samples = np.concatenate((samples,samp),axis=0)

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  np.save(outdir+'/samples.npy', samples)
  print "saved "+outdir+"/samples.npy"
