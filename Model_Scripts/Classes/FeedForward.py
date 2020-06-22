"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""

import numpy as np
from maketopo import get_topo, make_dtopo
from scipy import stats
from scipy.integrate import quad

import os
from scipy.interpolate import interp1d

#Modelscripts/classes
from abrahamson import abrahamson
from distance import distance
from atkinson_kaka import convert_to_MMI


class FeedForward:
    """
    This class Generates real physical interpretations of the prior from the MCMC class and runs GeoClaw.
    Then Calculates the log likelihood probability based on the output.
    """

    def __init__(self):
        pass

    def run_abrahamson(self, gauges, mag, okada_params):
        """Runs Abrahamson for all observation sites and returns MMI
        
        Parameters
        ----------
        gagues : (list) of gague objects
            The set of "loose" obersvation locations with its associated probability distributions
            (see the gauge class in tsunamibayes for more details)
        mag : float
            Moment magnitude (earthquake intensity on the Richter scale)
        okada_params : (list) of floats
            A list of parameters to describe the fault plane location and composition.
            Incudes: latitude, longitude, depth, strike, rake, dip, etc. 

        Returns
        -------
        MMI_list : (list) of float values
            List of MMIs with their respective standard deviations from all observation points
        """
        MMI_list = []
        for gauge in gauges:
            # Strike, Length, width, depth, slip, rake, dip, long, lat
            D = distance(gauge.latitude, gauge.longitude, okada_params[1], okada_params[2], okada_params[0], okada_params[6], okada_params[3], okada_params[8], okada_params[7])
            logSA_g = abrahamson(mag,D,gauge.VS30)[0]
            logSA = np.log(980.665)+logSA_g
            mu_MMI = convert_to_MMI(logSA,mag,D)[0]
            MMI_list.append(mu_MMI)
        return MMI_list

    def run_geo_claw(self, okada_params):
        """Runs Geoclaw for a specific fault and its parameters.

        Parameters
        ----------
        okada_params : (list) of floats
            A list of parameters to describe the fault plane location and composition.
            Incudes: latitude, longitude, depth, strike, rake, dip, etc. 
        
        return
        ------
        none
        """
        get_topo()
        make_dtopo(okada_params)

        # os.system('make clean')
        # os.system('make clobber')
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

        Parameters
        ----------
            gauges : (list) of integers representing the gauge names

        Returns
        -------
            arrivals (array): An array containing the arrival times for the
                highest wave for each gauge. arrivals[i] corresponds to the
                arrival time for the wave for gauges[i]
            max_heights (array): An array containing the maximum heights
                for each gauge. max_heights[i] corresponds to the maximum
                height for gauges[i]
        """
        # data = np.loadtxt("./ModelOutput/geoclaw/fort.FG1.valuemax")
        # bath_data = np.loadtxt("./ModelOutput/geoclaw/fort.FG1.aux1")

        data = np.loadtxt("./fort.FG1.valuemax")
        bath_data = np.loadtxt("./fort.FG1.aux1")

        #    arrivals = data[:,4]
        arrivals = data[:, -1] / 60.  # this is the arrival time of the first wave, not the maximum wave
        # note that fgmax outputs in seconds, but our likelihood is in minutes
        max_heights = data[:, 3]
        bath_depth = bath_data[:, -1]

        max_heights[max_heights < 1e-15] = -9999  # these are locations where the wave never reached the gauge...
        max_heights[np.abs(max_heights) > 1e15] = -9999  # again places where the wave never reached the gauge...
        bath_depth[max_heights == 0] = 0
        wave_heights = max_heights + bath_depth

        return arrivals, wave_heights

    def calculate_llh(self, gauges):
        """
        Calculate the log-likelihood of the data at each of the gauges
        based on our chosen distributions for maximum wave heights and
        arrival times. Return the sum of these log-likelihoods.

        Parameters
        ----------
            gauges : (list) of gauge objects
        Returns
        -------
            llh : float
                The sum of the log-likelihoods of the data of each
                gauge in gauges list.
            arrivals : (list) of floats
                arrival times at each respective gauge
            heights : (list) of floats
                arrival heights at each respective gauge (km?)
        """
        # names = []
        # for gauge in gauges:
        #     names.append(gauge.name)
        arrivals, heights = self.read_gauges()

        llh = 0.  # init p
        heightLikelihoodTable = np.load('./InputData/gaugeHeightLikelihood.npy')
        heightValues = heightLikelihoodTable[:, 0]
        inundationLikelihoodTable = np.load('./InputData/gaugeInundationLikelihood.npy')
        inundationValues = inundationLikelihoodTable[:, 0]

        for i, gauge in enumerate(gauges):
            print("GAUGE LOG: gauge", i, "(", gauge.longitude, ",", gauge.latitude, "): arrival =", arrivals[i],
                  ", heights =", heights[i])
            # arrivals
            if (gauge.kind[0]):
                p_i = gauge.arrival_dist.logpdf(arrivals[i])
                llh += p_i
                print("GAUGE LOG: gauge", i, " (arrival)   : logpdf +=", p_i)

            # heights
            if (gauge.kind[1]):
                # special case: wave didn't arrive
                if np.abs(heights[i]) > 999999999:
                    p_i = np.NINF
                # special case: value is outside interpolation bounds
                # may need to make lower bound 0 and enable extrapolation for values very close to 0
                elif (heights[i] > max(heightValues) or heights[i] < min(heightValues)):
                    print("WARNING: height value {:.2f} is outside height interpolation range.".format(heights[i]))
                    p_i = np.NINF
                else:
                    heightLikelihoods = heightLikelihoodTable[:, i + 1]
                    f = interp1d(heightValues, heightLikelihoods, assume_sorted=True)  # ,kind='cubic')
                    p_i = np.log(f(heights[i]))

                llh += p_i
                print("GAUGE LOG: gauge", i, " (height)    : logpdf +=", p_i)

            # inundations
            if (gauge.kind[2]):
                # special case: wave didn't arrive
                if np.abs(heights[i]) > 999999999:
                    p_i = np.NINF
                # special case: value is outside interpolation bounds
                # may need to make lower bound 0 and enable extrapolation for values very close to 0
                elif (heights[i] > max(heightValues) or heights[i] < min(heightValues)):
                    print("WARNING: height value {:.2f} is outside inundation interpolation range.".format(heights[i]))
                    p_i = np.NINF
                else:
                    inundationLikelihoods = inundationLikelihoodTable[:, i + 1]
                    f = interp1d(inundationValues, inundationLikelihoods, assume_sorted=True)  # ,kind='cubic')
                    p_i = np.log(f(heights[i]))

                llh += p_i
                print("GAUGE LOG: gauge", i, " (inundation): logpdf +=", p_i)
        return llh, arrivals, heights

    def shake_llh(self, MMI, gauges, integrate=False, sigma_MMI = .73):
        """
        Calculate the log-likelihood of a sample earthquake
        based on our chosen distributions for MMI at each location.

        Parameters
        ----------
            MMI : (list) of the MMI for each gauge location
            gauges : (list) of gauge objects
            integrate : bool
                True to calculate likelihood using
                integration of the observation distribution and the
                MMI distribution, False to calculate without MMI
                uncertainty. (False is default)
            sigma_MMI : float
                standard deviation for MMI estimates
                (default of .73 from Atkinson-Kaka model)

        Returns
        -------
            llh : float
                The combined log-likelihood of the sample
                earthquake.
        """

        llh = 0

        if integrate:
            for i, gauge in enumerate(gauges):
                MMI_distribution = stats.norm(loc=MMI[i], scale=sigma_MMI)
                f = lambda x: gauge.distribution.pdf(x) * \
                    MMI_distribution.pdf(x)
                L = quad(f, 0, 12)[0]
                llh += np.log(L)

        else:
            for i, gauge in enumerate(gauges):
                llh += gauge.distribution.logpdf(MMI[i])

        return llh