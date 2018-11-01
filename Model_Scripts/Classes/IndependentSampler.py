"""
Created 10/19/2018
"""
from MCMC import MCMC
import numpy as np


class IndependentSampler(MCMC):
    """
    This Interface takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface
    """

    def __init__(self, Samples):
        MCMC.__init__(Samples)
        pass

    def build_priors(self):
        pass

    def acceptance_prob(self):
        change_llh = self.change_llh_calc()
        return min(np.exp(change_llh), 1)


    """ DEPRECIATED """
    def draw(self):
        """
        Draw with the independent sampling method, using the prior
        to make each of the draws.

        Returns:
            draws (array): An array of the 9 parameter draws.
        """

        # Load distribution parameters.
        params = self.prior

        # Take a random draw from the distributions for each parameter.
        # For now assume all are normal distributions.
        draws = []
        for param in params:
            dist = stats.norm(param[0], param[1])
            draws.append(dist.rvs())
        draws = np.array(draws)
        print("independent sampler draw:", draws)
        return draws