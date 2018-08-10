import sys
import numpy
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

nbins=50
density=False

def prior2DHist(prfile,prcols,col1,col2):
  print("reading prior data from",prfile)
  prdata = numpy.load(prfile).T
  
  plt.hist2d(prdata[:,col1],prdata[:,col2],nbins,cmap="coolwarm")
  plt.xlabel(prcols[col1])
  plt.ylabel(prcols[col2])

  figNm="hist2d_"+prcols[col1]+"_"+prcols[col2]+".png"
  plt.savefig(figNm)
  print("saved",figNm)
  plt.clf()

prfile = "prior_3param.npy"
prcols=["longitude","latitude","strike"]
prior2DHist(prfile,prcols,0,1)
