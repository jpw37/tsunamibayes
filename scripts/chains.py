import sys 
from matplotlib import pyplot as plt 
import numpy as np

A = np.load('samples.npy') # Must be in the same directory
#execfile("make_plots.py")

nchains=8
sampPerChain=25001
for i in range(0,nchains-1):
  rws=range(i*sampPerChain+1,(i+1)*sampPerChain)
  print rws[0],rws[-1]
  plt.plot(A[rws,2],A[rws,1],".")

plt.xlabel('width')
plt.ylabel('length')
plt.show()
