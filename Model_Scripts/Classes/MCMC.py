"""
Created 10/19/2018
"""
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class MCMC:
    """
    This Interface takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface
    """

    def __init__(self, type):
        pass

    def accept_reject(self, accept_prob):
        # Increment wins. If new, change current 'best'.
        if np.random.random() < accept_prob:  # Accept new
            samples[0] = samples[-1]
            samples[-1][-1] += 1
            samples[0][-1] = len(samples) - 1
        else:  # Reject new
            samples[int(samples[0][-1])][-1] += 1  # increment old draw wins
        np.save('samples.npy', samples)

    def build_priors(self):
        samplingMult = 50
        bandwidthScalar = 2
        # build longitude, latitude and strike prior
        data = pd.read_excel('./Data/Fixed92kmFaultOffset50kmgapPts.xls')
        data = np.array(data[['POINT_X', 'POINT_Y', 'Strike']])
        distrb0 = gaussian_kde(data.T)

        # build dip, rake, depth, length, width, and slip prior
        vals = np.load('6_param_bootstrapped_data.npy')
        distrb1 = gaussian_kde(vals.T)
        distrb1.set_bandwidth(bw_method=distrb1.factor * bandwidthScalar)

        return distrb0, distrb1

    def draw(self):
        pass

    def acceptance_prob(self):
        pass