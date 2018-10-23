"""
Created 10/19/2018
"""
import MCMC
import stats
import numpy as np


class RandomWalk(MCMC):
    """
    This Interface takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface
    """

    def __init__(self):
        pass

    def acceptance_prob(self, Samples, change_llh):
        prop_prior = Samples.get_prop_prior() # Prior for longitude, latitude, strike AND  # Prior for dip, rake, depth, length, width, slip
        cur_samp_prior = Samples.get_cur_prior() #See above
        change_prior = prop_prior - cur_samp_prior  # Log-Likelihood

        # Note we use np.exp(new - old) because it's the log-likelihood
        return min(1, np.exp(change_llh + change_prior))

    def draw(self, u):
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
        cov = np.diag([strike, length, width, depth, slip, rake,
                       dip, longitude, latitude])

        # cov *= 16.0;

        # random draw from normal distribution
        e = stats.multivariate_normal(mean, cov).rvs()
        print("Random walk difference:", e)
        print("New draw:", u + e)
        return u + e