"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import numpy as np
import pandas as pd

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self, priors):
        """
        Initialize the class with priors
        :param priors: list (was dict):
        kde distributions to their respective parameters for sampling
        """
        #rv_continuous.__init__(self)
        self.priors = priors

    def logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        lon    = sample["Longitude"]
        lat    = sample["Latitude"]
        strike = sample["Strike"]
        mag    = sample["Magnitude"]

        if mag < 0:
            lpdf = np.NINF
        else:
            #prior for longitude, latitude, strike
            lpdf = self.priors[0].logpdf(np.array([lon,lat,strike]))[0]

            #Garret and spencer wrote this 18 June 2019
            #If mag is not constructed with gaussian KDE then this will not work
            lpdf += self.priors[1].logpdf( np.array([mag]))[0]

        return lpdf

    def rvs(self, size=1):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        # CHECK ORDER OF PRODUCED RESULTS

        samples = np.vstack((self.priors[0].resample(size),
                             np.exp(self.priors[1].resample(size)))).T
        samples = samples[0]
        samples = pd.Series(samples,["Longitude","Latitude","Strike", "Magnitude"])
        return samples

