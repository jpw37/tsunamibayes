import json

class BaseForwardModel:
    def __init__(self,gauges):
        self.gauges = gauges

        #Load in global parameters
        with open('parameters.txt') as json_file:
            self.global_params = json.load(json_file)

    def run(self,model_params):
        """
        Run  Model(model_params)
        Read gauges
        Return observations (arrival times, Wave heights)
        """
        raise NotImplementedError("{}.run() must be implemented in classes inheriting from BaseForwardModel".format(type(self).__name__))

    def llh(self, observations):
        """
        Parameters:
        ----------
        observations : ndarray
            arrivals , heights
        Compute/Return llh
        """
        raise NotImplementedError("{}.llh() must be implemented in classes inheriting from BaseForwardModel".format(type(self).__name__))

from scipy.interpolate import interp1d
import numpy as np
from maketopo import get_topo, make_dtopo

class ForwardGeoClaw(BaseForwardModel):
    def run(self,model_params):
        """
        Run  Model(model_params)
        Read gauges
        Return observations (arrival times, Wave heights)
        """
        get_topo()
        make_dtopo(model_params)
        os.system('rm .output')
        os.system('make .output')

        observations = np.loadtxt(self.global_params['obser_file_path'])
        bath_data    = np.loadtxt(self.global_params['bath_file_path'])

        # this is the arrival time of the first wave, not the maximum wave
        arrivals = observations[:, -1] / 60.
        # note that fgmax outputs in seconds, but our likelihood is in minutes
        max_heights = observations[:, 3]
        bath_depth = bath_data[:, -1]

         # these are locations where the wave never reached the gauge...
        max_heights[max_heights < 1e-15] = -9999
        max_heights[np.abs(max_heights) > 1e15] = -9999

        bath_depth[max_heights == 0] = 0
        wave_heights = max_heights + bath_depth

        return arrivals, wave_heights

    def llh(self, observations):
        """
        Parameters:
        ----------
        observations : ndarray
            arrivals , heights
        Compute/Return llh
        """
        arrivals, heights = observations
        llh = 0.  # init p
        heightLikelihoodTable = np.load(self.global_params['h_table_path'])
        heightValues = heightLikelihoodTable[:, 0]
        inundationLikelihoodTable = np.load(self.global_params['inu_table_path'])
        inundationValues = inundationLikelihoodTable[:, 0]

        for i, gauge in enumerate(self.gauges):
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

        return llh
