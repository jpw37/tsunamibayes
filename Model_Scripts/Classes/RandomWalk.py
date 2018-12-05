"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
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


    def acceptance_prob(self, prop_prior_llh, cur_prior_llh):
        change_llh = self.change_llh_calc()

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
        # strike = .375
        # length = 4.e3
        # width = 3.e3
        # depth = .1875
        # slip = .01
        # rake = .25
        # dip = .0875
        # longitude = .025
        # latitude = .01875
        strike_std = 5.  # strike_std    = 1.
        length_std = 5.e3  # length_std    = 2.e4
        width_std = 2.e3  # width_std     = 1.e4
        depth_std = 1.e3  # depth_std     = 2.e3
        slip_std = 0.5  # slip_std      = 0.5
        rake_std = 0.5  # rake_std      = 0.5
        dip_std = 0.1  # dip_std       = 0.1
        longitude_std = 0.15  # longitude_std = .025
        latitude_std = 0.15  # latitude_std  = .025
        mean = np.zeros(9)
        # square for std => cov
        cov = np.diag(np.square([strike_std, length_std, width_std, depth_std, slip_std, rake_std,
                                 dip_std, longitude_std, latitude_std]))

        cov *= 0.25;

        # random draw from normal distribution
        e = stats.multivariate_normal(mean, cov).rvs()

        # does sample update normally
        print("Random walk difference:", e)
        print("New draw:", prev_draw + e)
        new_draw = prev_draw + e

        """
        Here we make some fixed changes to the dip and depth according 
        to a simple rule documented elsewhere. This fix will likely
        depreciate upon finishing proof of concept paper and work on 1852
        event.
        """
        # doctor dip to 20 degrees as discussed
        new_draw[6] = 20
        # doctor depth according to adhoc fix
        new_draw[3] = self.doctored_depth_1852_adhoc(new_draw[7], new_draw[8], new_draw[6])

        # return appropriately doctored draw
        return new_draw

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
        fault_file = './Data/fault_array.npy'
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
