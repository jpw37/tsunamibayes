#!/usr/bin/python

#plot_chains: Python script to plot a 2D scatter plot of MCMC chains, 
#with the points colored by MCMC chain (or by groups of samples). 
#
#The script takes two parameters as arguments, e.g.
# python plot_chains.py lat long
#to make a scatter plot of latitude/longitude values.
#
#The number of chains (nchains) and samples per chain (sampPerChain) 
#are hardcoded below. 

import sys 

import matplotlib
matplotlib.use('agg',warn=False, force=True)

from matplotlib import pyplot as plt 
import numpy as np

A = np.load('samples.npy') # Must be in the same directory
#execfile("make_plots.py")
d = dict()
d['strike'] = 0
d['length'] = 1
d['width'] = 2
d['depth'] = 3
d['slip'] = 4
d['rake'] = 5
d['dip'] = 6
d['longitude'] = 7
d['latitude'] = 8

nchains=8
sampPerChain=10001
#param1="latitude"
#param2="longitude"
if len(sys.argv) < 2:
  print "must provide two parameters"
else:
  param1 = sys.argv[1]
  if param1 == "long" or param1 == "lat": # allow for shorthand
      param1 += "itude"
  param2 = sys.argv[2]
  if param2 == "long" or param2 == "lat": # allow for shorthand
      param2 += "itude"
  
  col1 = d[param1]
  col2 = d[param2]
  for i in range(0,nchains):
    rws=range(i*sampPerChain+1,(i+1)*sampPerChain)
    print rws[0],rws[-1]
    plt.plot(A[rws,col1],A[rws,col2],".")
  
  #plt.xlabel('width')
  #plt.ylabel('length')
  #plt.show()
  plt.xlabel(param1)
  plt.ylabel(param2)
  plt.savefig("chains_"+param1+"_"+param2+".png")
