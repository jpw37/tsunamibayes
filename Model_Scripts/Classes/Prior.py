"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import numpy as np
from scipy.stats import rv_continuous

class Prior(rv_continuous):
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self, priors):
        """
        Initialize the class with priors
        :param priors: list (was dict):
        kde distributions to their respective parameters for sampling
        """
        rv_continuous.__init__(self)
        self.priors = priors

    def _pdf(self, sample):
        """
        Calculate the prior likelihood
        :param sample:
        :return:
        """
        if sample[1] < 0 or sample[2] < 0 or sample[3] < 0 or sample[4] < 0:
            pdf = 0
        else:
            #prior for longitude, latitude, strike
            pdf = self.priors[0].pdf(sample[[7,8,0]])
            #prior for length, width, slip
            #this is a lognormal so the logpdf is a little more complicated
            #justin sent an email to jared on 01/04/2019 documenting the formula below
            pdf *= self.priors[1].pdf( np.log(sample[[1,2,4]]) ) - np.log(
                sample[1]) - np.log(sample[2]) - np.log(sample[4]) # VERIFY THIS

        return pdf

    def _logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        #make sure that sample is rejected if length, width, depth, or slip are negative
        if sample[1] < 0 or sample[2] < 0 or sample[3] < 0 or sample[4] < 0:
            lpdf = np.NINF
        else:
            #prior for longitude, latitude, strike
            lpdf = self.priors[0].logpdf(sample[[7,8,0]])
            #prior for length, width, slip
            #this is a lognormal so the logpdf is a little more complicated
            #justin sent an email to jared on 01/04/2019 documenting the formula below
            lpdf += self.priors[1].logpdf( np.log(sample[[1,2,4]]) ) - np.log(
                sample[1]) - np.log(sample[2]) - np.log(sample[4])

        return lpdf

    def _rvs(self, size=None):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        # CHECK ORDER OF PRODUCED RESULTS
        if size is None:
            samples = np.vstack((self.priors[0].resample(1),
                                    self.priors[1].resample(1)))
            return np.exp(samples) # since based on log
        else:
            samples = np.vstack((self.priors[0].resample(size),
                                    self.priors[1].resample(size)))
            return np.exp(samples) # since based on log

    def hello(self):
        return "what da"

        # DEPRICATED
        # # prior for longitude, latitude, strike
        # lpdf = self.priors[0].logpdf(sample[[7, 8, 0]])
        #
        # # prior for length, width, slip
        # # this is a lognormal so the logpdf is a little more complicated
        # # justin sent an email to jared on 01/04/2019 documenting the formula below
        # lpdf += self.priors[ 1].logpdf(np.log(sample[[1, 2, 4]])) - np.log(sample[1]) - np.log(sample[2]) - np.log(
        #     sample[4])
        #
        # return lpdf

        # DEPRICATED?
    # def logpdf(self, params):
    #     """
    #     Takes the log pdf of the given priors for the current and proposed parameters
    #     :param proposed_params: dictionary: {distribution object: parameters}
    #     :param cur_params: current numbers for the parameters
    #     :return: llh (float) sum of the loglikelihoods for each prior dist with is corresponding parameters
    #     """
    #     llh = 0.0
    #     for prior in self.priors.keys():
    #         llh += prior.logpdf(params[self.priors[prior]].values)[0]
    #     return llh
