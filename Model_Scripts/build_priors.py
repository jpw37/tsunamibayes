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
    vals = np.load('6_param_bootstrapped_data.npy')
    distrb1 = gaussian_kde(vals.T)
    distrb1.set_bandwidth(bw_method=distrb1.factor*bandwidthScalar)
    
    return distrb0,distrb1