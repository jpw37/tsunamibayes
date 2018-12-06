"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""


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