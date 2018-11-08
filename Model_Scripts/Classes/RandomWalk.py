"""
Created 10/19/2018
"""
import numpy as np
from scipy import stats
from MCMC import MCMC


class RandomWalk(MCMC):
    """
    This Interface takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface
    """

    def __init__(self, covariance):
        MCMC.__init__(self)
        self.covariance = covariance
        pass


    def acceptance_prob(self, prior, proposed_params, cur_params):
        change_llh = self.change_llh_calc()

        # Calculate probability for the current sample and proposed sample
        cur_prior_llh, prop_prior_llh = prior.logpdf(proposed_params, cur_params)

        # Log-Likelihood
        change_prior_llh = prop_prior_llh - cur_prior_llh

        # Note we use np.exp(new - old) because it's the log-likelihood
        return min(1, np.exp(change_llh+change_prior_llh))

    def draw(self, prev_draw):
        """
        Draw with the random walk sampling method, using a multivariate_normal
        distribution with the following specified std deviations to
        get the distribution of the step size.

        Returns:
            draws (array): An array of the 9 parameter draws.
        """
        # Std deviations for each parameter, the mean is the current location
        strike = .375
        length = 4.e3
        width = 3.e3
        depth = .1875
        slip = .01
        rake = .25
        dip = .0875
        longitude = .025
        latitude = .01875
        mean = np.zeros(9)
        self.covariance = np.diag([strike, length, width, depth, slip, rake,
                       dip, longitude, latitude])

        # cov *= 16.0;

        # random draw from normal distribution
        e = stats.multivariate_normal(mean, self.covariance).rvs()
        print("Random walk difference:", e)
        print("New draw:", prev_draw + e)
        new_draw = prev_draw + e
        return new_draw
