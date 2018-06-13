import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def build_priors():
    samplingMult = 50
    bandwidthScalar = 2
    #build longitude, latitude and strike prior 
    data = pd.read_excel('Fixed92kmFaultOffset50kmgapPts.xls')
    data = np.array(data[['POINT_X', 'POINT_Y', 'Strike']])
    distrb0 = gaussian_kde(data.T)
    
    #build dip, magnitude, rake and depth prior
    dataM = pd.read_csv("./generalPrior(momentduration).csv")
    USGSvariables = ["Dip","Rake","Depth"]
    vals = np.array(dataM[USGSvariables])
    #bootstrapping
    n = len(vals)
    r = n*samplingMult
    index = np.random.choice(np.arange(n),size=r,replace=True)
    vals = vals[index]
    #creates the KDE
    distrb1 = gaussian_kde(vals.T)
    distrb1.set_bandwidth(bw_method=distrb1.factor*bandwidthScalar)
    
    #build length, width and slip prior
    dataWells = pd.read_csv("./wellsCoppersmithSlip.csv")
    variableWells = ["Length","Width","Average Slip"]
    vals = np.array(dataWells[variableWells])
    #bootstrapping
    n = len(vals)
    r = n*samplingMult
    index = np.random.choice(np.arange(n),size=r,replace=True)
    vals = vals[index]
    #creates the KDE
    distrb2 = gaussian_kde(vals.T)
    distrb2.set_bandwidth(bw_method=distrb2.factor*bandwidthScalar)
    
    return distrb0,distrb1,distrb2