#!/usr/bin/python

import sys
import numpy as np
import os

nburn=0

if len(sys.argv)<=2:
  print("Must provide more than one directory")
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
  print("saved "+outdir+"/samples.npy")
