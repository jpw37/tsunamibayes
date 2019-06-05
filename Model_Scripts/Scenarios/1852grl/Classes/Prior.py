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

    #def _pdf(self, sample):
    #    """
    #    Calculate the prior likelihood
    #    :param sample:
    #    :return:
    #    """
    #    if sample[1] < 0 or sample[2] < 0 or sample[3] < 0 or sample[4] < 0:
    #        pdf = 0
    #    else:
    #        #prior for longitude, latitude, strike
    #        pdf = self.priors[0].pdf(sample[[7,8,0]])
    #        #prior for length, width, slip
    #        #this is a lognormal so the logpdf is a little more complicated
    #        #justin sent an email to jared on 01/04/2019 documenting the formula below
    #        pdf *= self.priors[1].pdf( np.log(sample[[1,2,4]]) ) - np.log(
    #            sample[1]) - np.log(sample[2]) - np.log(sample[4]) # VERIFY THIS

    #    return pdf

    def logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        #9-parameter version
        ##make sure that sample is rejected if length, width, depth, or slip are negative
        #if sample[1] < 0 or sample[2] < 0 or sample[3] < 0 or sample[4] < 0:
        #    lpdf = np.NINF
        #else:
        #    #prior for longitude, latitude, strike
        #    lpdf = self.priors[0].logpdf(sample[[7,8,0]])
        #    #prior for length, width, slip
        #    #this is a lognormal so the logpdf is a little more complicated
        #    #justin sent an email to jared on 01/04/2019 documenting the formula below
        #    lpdf += self.priors[1].logpdf( np.log(sample[[1,2,4]]) ) - np.log(
        #        sample[1]) - np.log(sample[2]) - np.log(sample[4])
        #make sure that sample is rejected if length, width, depth, or slip are negative

        #print("sample:")
        #print(sample)
        #print("type of sample:")
        #print(type(sample))
        #GRL-style 6-parameter sampling
        lon    = sample["Longitude"]
        lat    = sample["Latitude"]
        strike = sample["Strike"]
        length = sample["Length"]
        width  = sample["Width"]
        slip   = sample["Slip"]
        #lon    = sample[4]
        #lat    = sample[5]
        #strike = sample[0]
        #length = sample[1]
        #width  = sample[2]
        #slip   = sample[3]
        if length < 0 or width < 0 or slip < 0:
            lpdf = np.NINF
        else:
            #prior for longitude, latitude, strike
            lpdf = self.priors[0].logpdf(np.array([lon,lat,strike]))[0]
            #print("lpdf is:")
            #print(lpdf)

            #prior for length, width, slip
            #this is a lognormal so the logpdf is a little more complicated
            #justin sent an email to jared on 01/04/2019 documenting the formula below
            lpdf += self.priors[1].logpdf( np.log(np.array([length,width,slip])) )[0] - np.log( length ) - np.log(width) - np.log(slip)

            #print("lpdf is:")
            #print(lpdf)

        return lpdf

    def rvs(self, size=1):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        # CHECK ORDER OF PRODUCED RESULTS
        samples = np.vstack((self.priors[0].resample(size),
                             np.exp(self.priors[1].resample(size)))).T
        samples = pd.DataFrame(samples,columns=["Longitude","Latitude","Strike","Length","Width","Slip"])
        return samples

