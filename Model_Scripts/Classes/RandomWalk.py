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

    def __init__(self, Samples, priors, covariance):
        super(RandomWalk, self).__init__(Samples, priors)
        self.covariance = covariance
        pass

    def acceptance_prob(self, change_llh):
        prop_prior1, prop_prior2 = self.Samples.get_prop_prior()
        prop_prior = self.priors[0].logpdf(prop_prior1) # Prior for longitude, latitude, strike AND  # Prior for dip, rake, depth, length, width, slip
        prop_prior += self.priors[1].logpdf(prop_prior2)

        cur_samp_prior1, cur_samp_prior2 = self.Samples.get_cur_prior() # See above
        cur_samp_prior = self.priors[0].logpdf(cur_samp_prior1)  # Prior for longitude, latitude, strike AND  # Prior for dip, rake, depth, length, width, slip
        cur_samp_prior += self.priors[1].logpdf(cur_samp_prior2)

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
        self.covariance = np.diag([strike, length, width, depth, slip, rake,
                       dip, longitude, latitude])

        # cov *= 16.0;

        # random draw from normal distribution
        e = stats.multivariate_normal(mean, self.covariance).rvs()
        print("Random walk difference:", e)
        print("New draw:", u + e)
        new_draw = u + e
        self.samples.save_sample(new_draw)
        return new_draw
