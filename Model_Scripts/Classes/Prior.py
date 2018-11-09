"""

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

    def logpdf(self, cur_params, proposed_params):
        """
        Takes the log pdf of the given priors for the current and proposed parameters
        :param proposed_params:
        :param cur_params:
        :return:
        """
        prop_prior_llh = 0.0
        cur_samp_prior_llh = 0.0
        for prior in self.priors.keys():
            prop_prior_llh += prior.logpdf(proposed_params[self.priors[prior]])
            cur_samp_prior_llh += prior.logpdf(cur_params[self.priors[prior]])

        return prop_prior_llh, cur_samp_prior_llh
