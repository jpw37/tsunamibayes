#!/usr/bin/python

import sys
import numpy

if len(sys.argv)<=1:
  print("Must provide a directory")
else:
  samples = numpy.load(sys.argv[1]+"/samples.npy")
  print("Number of samples:",len(samples)-1)
  
  wins = samples[samples[:,-1]!=1,-1]-1
  wins = wins[1:]
  print("Number of accepted samples:",len(wins))
  print("Accept Ratio:",len(wins)/sum(wins))
  print("Max of wins:",int(max(wins)))
  print("Sum of wins:",int(sum(wins)))
  print("samples[0,-1]:",samples[0,-1])

  if len(sys.argv)>2:
    nbreaks = int(sys.argv[2])
  else:
    nbreaks=5
  step=(len(samples)-1)//nbreaks
  for i in range(0,nbreaks):
    st = (i  )*step+1
    en = (i+1)*step
    ar=samples[st:en,-1]-1
    print("Accept Ratio (",st,":",en,"):",float(sum(ar!=0))/len(ar))

  print("First sample is:")
  print(samples[1,:])
  print("samples[int(samples[0,-1]), :] is:")
  print(samples[int(samples[0,-1]), :])

  #remove header line
  s = samples[1:,:]
  #only consider finite values
  s = s[numpy.isfinite(s[:,-2]),:]
  ##only consider non-zero values (shouldn't matter except maybe chains that haven't finished yet or some kind of concatenation of chains...not sure)
  #s = s[s[:,-2]!=0.,:]

  #column of log likelihood
  llh_idx=-2

  #find index of mle
  mle_idx=numpy.argmax(s[:,llh_idx])
  print("mle index is",mle_idx)

  #print mle sample
  print("mle sample is:")
  print(s[mle_idx,:])
