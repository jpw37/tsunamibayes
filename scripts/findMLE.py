import sys

import numpy
if len(sys.argv)<=1:
  print("Must provide a job directory")
else:
  #samples = numpy.load("tsunamibayes/Model_Scripts/6_param_bootstrapped_data.npy")
  samples = numpy.load(sys.argv[1]+"/samples.npy")
  #samples = numpy.load(sys.argv[1])

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

