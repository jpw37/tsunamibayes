"""


"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from MCMC import MCMC

class Custom(MCMC):
    """
    Use this class to create custom build_prior, and drawing methods for the MCMC method
    When the Variable for use_custom is set to true, this class will be used as the main MCMC class for the Scenario
    """
    def __init__(self, Samples):
        MCMC.__init__(self, Samples)
        self.priors = None
        return

    def draw(self, prev_draw):
        draws = prev_draw

        return self.map_to_okada(draws)


    def map_to_okada(self, draws):
        #TODO: JARED AND JUSTIN

        self.samples.save_mcmc(draws)
        return

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
