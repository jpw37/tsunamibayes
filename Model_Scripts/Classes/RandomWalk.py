"""
Created 10/19/2018
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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

    def build_priors(self):
        samplingMult = 50
        bandwidthScalar = 2
        # build longitude, latitude and strike prior
        data = pd.read_excel('./Data/Fixed92kmFaultOffset50kmgapPts.xls')
        data = np.array(data[['POINT_X', 'POINT_Y', 'Strike']])
        distrb0 = gaussian_kde(data.T)

        # build dip, rake, depth, length, width, and slip prior
        vals = np.load('./Data/6_param_bootstrapped_data.npy')
        distrb1 = gaussian_kde(vals.T)
        distrb1.set_bandwidth(bw_method=distrb1.factor * bandwidthScalar)

        return distrb0, distrb1


    def acceptance_prob(self):
        change_llh = self.change_llh_calc()

        prop_prior1, prop_prior2 = self.Samples.get_prop_prior()
        prop_prior = self.priors[0].logpdf(prop_prior1) # Prior for longitude, latitude, strike
        prop_prior += self.priors[1].logpdf(prop_prior2) # Prior for dip, rake, depth, length, width, slip

        cur_samp_prior1, cur_samp_prior2 = self.Samples.get_cur_prior() # See above

        cur_samp_prior = self.priors[0].logpdf(cur_samp_prior1)  # Prior for longitude, latitude, strike
        cur_samp_prior += self.priors[1].logpdf(cur_samp_prior2)  # Prior for dip, rake, depth, length, width, slip

        change_prior = prop_prior - cur_samp_prior  # Log-Likelihood

        # Note we use np.exp(new - old) because it's the log-likelihood
        return min(1, np.exp(change_llh + change_prior))

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
