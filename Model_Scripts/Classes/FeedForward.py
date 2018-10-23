"""
created 10/19/2018
"""
import numpy as np
import MakeTopo as mt
from scipy import stats
from pmf import PMFData, PMF
import os

class FeedForward:
    """
    This class Generates real physical interpretations of the prior from the MCMC class and runs GeoClaw.
    Then Calculates the log likelihood probability based on the output.
    """

    def __init__(self, mcmc):
        self.mcmc = mcmc
        os.system('rm .output')
        os.system('make .output')
        pass

    def init_guesses(self, init):
        """
        Change 6 params to 9 params for GeoClaw
        :param draws: 7 params to convert to 9 params
        :return:
        """
        if init == "manual":
          #initial guesses taken from final sample of 260911_ca/001
          strike     =  2.77152900e+02
          length     =  3.36409138e+05
          width      =  3.59633559e+04
          depth      =  2.50688161e+04
          slip       =  9.17808160e+00
          rake       =  5.96643293e+01
          dip        =  1.18889907e+01
          longitude  =  1.31448175e+02
          latitude   = -4.63296475e+00

          self.guesses = np.array([strike, length, width, depth, slip, rake, dip,
              longitude, latitude])

        elif init == "random":
            # draw initial sample at random from prior (kdes)
            priors = self.mcmc.build_priors()
            p0 = priors[0].resample(1)[:, 0]
            longitude = p0[0]
            latitude = p0[1]
            strike = p0[2]

            # draw from prior but redraw if values are unphysical
            length = -1.
            width = -1.
            depth = -1.
            slip = -1.
            rake = -1.
            dip = -1.
            while length <= 0. or width <= 0. or depth <= 0. or slip <= 0.:
                p1 = priors[1].resample(1)[:, 0]
                length = p1[3]
                width = p1[4]
                depth = p1[2]
                slip = p1[5]
                rake = p1[1]
                dip = p1[0]

            self.guesses = np.array([strike, length, width, depth, slip, rake, dip,
                                     longitude, latitude])

        elif init == "restart":
            self.guesses = np.load('../samples.npy')[0][:9]

            # np.save("guesses.npy", self.guesses)
            print("initial sample is:")
            print(self.guesses)

        return self.guesses

    def run_geo_claw(self, draws):
        """
        Runs Geoclaw
        :param draws: parameters
        :return:
        """
        # Run GeoClaw using draws
        mt.get_topo()
        mt.make_dtopo(self.draws)

        os.system('rm .output')
        os.system('make .output')

        return


    def read_gauges(self):
        """Read GeoClaw output and look for necessary conditions.
        This will find the max wave height

        Meaning of gauge<gauge_number>.txt file columns:
        - column 1 is time
        - column 2 is a scaled water height
        - column 5 is the graph that appears in plots

        Parameters:
            gauges (list): List of integers representing the gauge names
        Returns:
            arrivals (array): An array containing the arrival times for the
                highest wave for each gauge. arrivals[i] corresponds to the
                arrival time for the wave for gauges[i]
            max_heights (array): An array containing the maximum heights
                for each gauge. max_heights[i] corresponds to the maximum
                height for gauges[i]
        """
        data = np.loadtxt("_output/fort.FG1.valuemax")
        arrivals = data[:, 4]
        max_heights = data[:, 3]
        return arrivals, max_heights

    def calculate_probability(self, gauges):
        """
        Calculate the log-likelihood of the data at each of the gauges
        based on our chosen distributions for maximum wave heights and
        arrival times. Return the sum of these log-likelihoods.

        Parameters:
            gauges (list): A list of gauge objects
        Returns:
            p (float): The sum of the log-likelihoods of the data of each
                gauge in gauges.
        """
        names = []
        for gauge in gauges:
            names.append(gauge.name)
        arrivals, heights = self.read_gauges()

        # Calculate p for the arrivals and heights
        llh = 0.  # init p
        # Calculate p for the heights, using the PMFData and PMF classes
        amplification_data = np.load('../amplification_data.npy')
        row_header = amplification_data[:, 0]
        col_header = np.arange(len(amplification_data[0]) - 1) / 4
        pmfData = PMFData(row_header, col_header, amplification_data[:, 1:])
        for i, gauge in enumerate(gauges):
            if (gauge.kind[0]):
                # arrivals
                value = gauge.arrival_dist.pdf(arrivals[i])
                if np.abs(value) < 1e-20:
                    value = 1e-10
                #            llh += np.log(gauge.arrival_dist.pdf(arrivals[i]))
                llh += np.log(value)

            if (gauge.kind[1]):
                # heights
                pmf = pmfData.getPMF(gauge.distance, heights[i])
                if np.abs(heights[i]) > 999999999:
                    llh += -9999
                else:
                    llh_i = pmf.integrate(gauge.height_dist)
                    llh += np.log(llh_i)

            if (gauge.kind[2]):
                # inundation
                pmf = pmfData.getPMF(gauge.distance, heights[i])
                inun_values = np.power(pmf.vals, 4 / 3) * 0.06 * np.cos(gauge.beta) / (gauge.n ** 2)
                inun_probability = pmf.probs
                if len(inun_values) == 0:
                    print("WARNING: inun_values is zero length")
                    pmf_inundation = PMF([0., 1.], [0., 0.])
                else:
                    pmf_inundation = PMF(inun_values, inun_probability)
                llh_inundation = pmf_inundation.integrate(gauge.inundation_dist)
                llh += np.log(llh_inundation)

        return llh

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
        fault_file = 'fault_array.npy'
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
