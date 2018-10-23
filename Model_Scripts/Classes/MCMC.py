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

    def __init__(self):
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


    def default_build_priors(self):
        pass

    def draw(self):
        pass

    def acceptance_prob(self):
        pass