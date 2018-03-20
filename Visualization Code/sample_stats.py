#!/usr/bin/python

#Python script to compute some statistics on an MCMC chain
#Requires a directory as an argument, e.g.
# python sample_stats.py /home/jkrometi/tsunami/1820_run/185866_186964
#or simply
# python sample_stats.py 185866_186964

import sys
import numpy

if len(sys.argv)<=1:
  print "Must provide a directory"
else:
  samples = numpy.load(sys.argv[1]+"/samples.npy")
  print "Number of samples:",len(samples)-1
  
  wins = samples[samples[:,-1]!=1,-1]-1
  wins = wins[1:]
  print "Number of accepted samples:",len(wins)
  print "Accept Ratio:",len(wins)/sum(wins)
  print "Max of wins:",int(max(wins))
  print "Sum of wins:",int(sum(wins))
  print "samples[0,-1]:",samples[0,-1]

  #half = len(samples)/2
  #ar=samples[1:half,-1]-1
  #print "Accept Ratio (first half of chain ):",float(sum(ar!=0))/len(ar)
  #ar=samples[half+1:,-1]-1
  #print "Accept Ratio (second half of chain):",float(sum(ar!=0))/len(ar)

  nbreaks=5
  step=(len(samples)-1)/nbreaks
  for i in range(0,nbreaks):
    st = (i  )*step+1
    en = (i+1)*step
    ar=samples[st:en,-1]-1
    print "Accept Ratio (",st,":",en,"):",float(sum(ar!=0))/len(ar)

  print "First row of samples is:"
  print(samples[0,:])
  print "samples[int(samples[0,-1]), :] is:"
  print(samples[int(samples[0,-1]), :])
