import sys
import numpy
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

nbins=100
density=False

def priorHist(prfile,prcols,transpose=True):
  print("reading prior data from",prfile)
  prdata = numpy.load(prfile)
  if transpose:
    prdata = prdata.T

  #loop through columns and make plots
  for col in range(prdata.shape[1]):
    plt.hist(prdata[:,col],nbins,density=density)
    plt.xlabel(prcols[col])

    figNm="hist_"+prcols[col]+".png"
    plt.savefig(figNm)
    print("saved",figNm)
    plt.clf()

prfile = "prior_6param.npy"
prcols=["dip","rake","depth","length","width","slip"]
priorHist(prfile,prcols)

prfile = "prior_3param.npy"
prcols=["longitude","latitude","strike"]
priorHist(prfile,prcols)

nbins=25
prfile = "../tsunamibayes/Model_Scripts/6_param_bootstrapped_data.npy"
prcols=["dip_raw","rake_raw","depth_raw","length_raw","width_raw","slip_raw"]
priorHist(prfile,prcols,transpose=False)

