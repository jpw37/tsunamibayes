"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import numpy as np # testing

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self, priors):
        """
        Initialize the class with priors
        :param priors: dict: kde distributions to their respective parameters for sampling
        """
        self.priors = priors

    def prior_logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        # prior for longitude, latitude, strike
        lpdf = self.priors[0].logpdf(sample[[7, 8, 0]])

        # prior for length, width, slip
        # this is a lognormal so the logpdf is a little more complicated
        # justin sent an email to jared on 01/04/2019 documenting the formula below
        lpdf += self.priors[1].logpdf(np.log(sample[[1, 2, 4]])) - np.log(sample[1]) - np.log(sample[2]) - np.log(
            sample[4])

        return lpdf

    def logpdf(self, params):
        """
        Takes the log pdf of the given priors for the current and proposed parameters
        :param proposed_params: dictionary: {distribution object: parameters}
        :param cur_params: current numbers for the parameters
        :return: llh (float) sum of the loglikelihoods for each prior dist with is corresponding parameters
        """
        llh = 0.0
        for prior in self.priors.keys():
            llh += prior.logpdf(params[self.priors[prior]].values)[0]
        return llh

    def random_draw(self):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        pass
