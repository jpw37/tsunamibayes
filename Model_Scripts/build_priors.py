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
    
    #build dip, rake, depth, length, width, and slip prior
    dataM = pd.read_csv("6_param_prior_data.csv")
    vars = ["Dip","Rake","depth2","Length","Width2","Slip2"]
    vals = np.array(dataM[vars])
    #bootstrapping
    n = len(vals)
    r = n*samplingMult
    index = np.random.choice(np.arange(n),size=r,replace=True)
    vals = vals[index]
    #creates the KDE
    distrb1 = gaussian_kde(vals.T)
    distrb1.set_bandwidth(bw_method=distrb1.factor*bandwidthScalar)
    
    return distrb0,distrb1