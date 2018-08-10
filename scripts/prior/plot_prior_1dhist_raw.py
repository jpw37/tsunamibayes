import sys
import numpy
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

density=True

def priorHist(prdata1,prdata2,prcols,transpose=True):
  #loop through columns and make plots
  for col in range(prdata1.shape[1]):
    plt.hist(prdata1[:,col],100,density=density)
    plt.hist(prdata2[:,col],25,density=density)
    plt.xlabel(prcols[col])

    figNm="hist_raw_"+prcols[col]+".png"
    plt.savefig(figNm)
    print("saved",figNm)
    plt.clf()

prcols=["dip","rake","depth","length","width","slip"]

prfile1 = "prior_6param.npy"
print("reading prior data from",prfile1)
prdata1 = numpy.load(prfile1).T

prfile2 = "../tsunamibayes/Model_Scripts/6_param_bootstrapped_data.npy"
print("reading prior data from",prfile2)
prdata2 = numpy.load(prfile2)

priorHist(prdata1,prdata2,prcols)
