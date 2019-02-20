"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

from MCMC import MCMC
from Prior import Prior

class Custom(MCMC):
    """
    Use this class to create custom build_prior, and drawing methods for the MCMC method
    When the Variable for use_custom is set to true, this class will be used as the main MCMC class for the Scenario
    """
    def __init__(self):
        MCMC.__init__(self)
        self.sample_cols = ['Strike', 'Length', 'Width', 'Depth', 'Slip', 'Rake', 'Dip', 'Longitude', 'Latitude']
        self.proposal_cols = ['P-Strike', 'P-Length', 'P-Width', 'P-Depth', 'P-Slip', 'P-Rake', 'P-Dip', 'P-Logitude', 'P-Latitude']

    def draw(self, prev_draw):
        """
        Draw from custom parameter space
        :param prev_draw:
        :return:
        """

        draws = prev_draw

        return self.map_to_okada(draws)

    def build_priors(self):
        samplingMult = 50
        bandwidthScalar = 2
        # build longitude, latitude and strike prior
        data = pd.read_excel('./InputData/Fixed92kmFaultOffset50kmgapPts.xls')
        data = np.array(data[['POINT_X', 'POINT_Y', 'Strike']])
        distrb0 = gaussian_kde(data.T)

        # build dip, rake, depth, length, width, and slip prior
        vals = np.load('./InputData/6_param_bootstrapped_data.npy')
        distrb1 = gaussian_kde(vals.T)
        distrb1.set_bandwidth(bw_method=distrb1.factor * bandwidthScalar)

        dists = {}
        dists[distrb0] = ['Longitude', 'Latitude', 'Strike']
        dists[distrb1] = ['Length', 'Width', 'Slip'] # 'Dip', 'Rake', 'Depth', 'Length', 'Width', 'Slip'

        self.prior = Prior(dists)

    def map_to_okada(self, draws):
        """
        TODO: JARED AND JUSTIN map to okada space
        :param draws:
        :return:
        """
        return draws

    def haversine_distance(self, p1, p2):
        """
        This function  is set up separately because the haversine distance
        likely will still be useful after we're done with this adhoc approach.

        Note, this does not account for the oblateness of the Earth. Not sure if
        this will cause a problem.
        """
        r = 6371000

        # Setting up haversine terms of distance expansion
        hav_1 = np.power(np.sin((p2[1] - p1[1]) / 2 * np.pi / 180), 2.0)
        hav_2 = np.cos(p2[1] * np.pi / 180) * np.cos(p1[1] * np.pi / 180) * np.power(
            np.sin((p2[0] - p1[0]) / 2 * np.pi / 180), 2.0)

        # taking the arcsine of the root of the sum of the haversine terms
        root = np.sqrt(hav_1 + hav_2)
        arc = np.arcsin(root)

        # return final distance between the two points
        return 2 * r * arc

    def doctored_depth_1852_adhoc(self, longitude, latitude, dip):
        """
        This is a function written specifically for our 1852 depth fix.
        We make use of the fault points used in generating our prior as
        jumping off point for fixing the depth of an event. We use a
        simple trig correction based on a 20degree dip angle and the haversine distance
        to get the depth of the earthquake in question.

        Note, this will do the dip correction regardless of which side
        of the trench our sample is on. Recognizing when the sample is
        on the wrong side seems nontrivial, so we have not implemented
        a check for this here.
        """
        # set up sample point and fault array
        p1 = np.array([longitude, latitude])
        fault_file = './InputData/fault_array.npy'
        fault_array = np.load(fault_file)
        # will store haversine distances for comparison
        dist_array = np.zeros(0.5 * len(fault_array))
        for i in range(len(dist_array)):
            x = fault_array[2 * i]
            y = fault_array[2 * i + 1]
            p2 = np.array([x, y])
            dist_array[i] = self.haversine_distance(p1, p2)

        dist = np.amin(dist_array)

        # need to add trig correction
        return (20000 + dist * np.tan(20 * np.pi / 180))
