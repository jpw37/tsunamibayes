# File for all earthquake specific information
# This is the only file that needs to be changed to run GeoClaw on
# a different earthquake.
import os
import numpy as np
from gauge import Gauge
import gauge
import maketopo as mt
import json
from gauge_dist_1852 import load_gauges


class Setup:
    """
    A class for setting up the GeoClaw run.
    Attributes:
        guesses (array): An array containing the initial guesses
            for the earthquake parameters. Set in the make_input_files()
            method and used in the initial run.
    """

    def __init__(self):
        # TODO Add file_guide.txt
        """
        Initialize all necessary variables for a GeoClaw run using MCMC and
        put them in their appropriate .npy and .txt files to be used later.
        Specifically, initialize the files described in file_guide.txt.
        NOTE TO USER: Edit all of the marked variables to match the specific
        earthquake you would like to model.
        """
        # Change the following variables
        #####
        # initial guesses (mean for prior if using normal distribution)
        strike = 84.6 # 205.0
        length = 100.e3
        width = 45.e3
        depth = 5.54e3
        slip = 20. # 9.
        rake = 67.1 # 90.
        dip = 13.3
        longitude = 130.47 # 132.4
        latitude = -5.63
        self.guesses = np.array([strike, length, width, depth, slip, rake, dip,
            longitude, latitude])
        # np.save("guesses.npy", self.guesses)

        # Parameters for priors
        # Standard deviations
        strike_std = 7.5
        length_std = 75.e3
        width_std = 60.e3
        depth_std = 3.75
        slip_std = 2.
        rake_std = 5.
        dip_std = 1.75
        longitude_std = .5
        latitude_std = .375
        means = self.guesses
        stds = np.array([strike_std, length_std, width_std, depth_std,
            slip_std, rake_std, dip_std, longitude_std, latitude_std])

        # Gauge information
        # Add as many as you like, repeating this pattern to add new gauges.
        # gauges = []
        # # Set gauge values for gauge 1
        # name = 10000 # BULUKUMBA (Terang-Terang)
        # longitude = 120.205664
        # latitude = -5.571685
        # distance = 2.5 # in kilometers (max 5)
        # # normal distribution for arrival and height, respectively
        # # (also accepts chi2 and skewnorm)
        # kind = ['norm', 'norm']
        #
        # # For kind = 'norm'
        # arrival_mean = 30 # in minutes
        # arrival_std = 6.
        # height_mean = 20. # in meters
        # height_std = 1.5
        # arrival_params = [arrival_mean, arrival_std]
        # height_params = [height_mean, height_std]
        #
        # # For kind = 'chi2'
        # arrival_k = None # chi2 parameter
        # arrival_lower_bound = None # in meters
        # height_k = None # chi2 param
        # height_lower_bound = None # in meters
        # # arrival_params = [arrival_k, arrival_lower_bound]
        # # height_params = [height_k, height_lower_bound]
        #
        # # For kind = 'skewnorm'
        # arrival_skew_param = None
        # arrival_mean = None # in minutes
        # arrival_std = None
        # height_skew_param = None
        # height_mean = None # in meters
        # height_std = None
        # # arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
        # # height_params = [height_skew_param, height_mean, height_std]
        #
        # g = Gauge(name, longitude, latitude, distance,
        #                 kind, arrival_params, height_params) #, beta, n)
        # gauges.append(g.to_json())
        #
        #
        # # Set gauge values for gauge 2
        # name = 10010 # Bima
        # longitude = 118.709077
        # latitude = -8.335202
        # distance = 5 # in kilometers (max 5)
        # # normal distribution for arrival and height, respectively
        # # (also accepts chi2 and skewnorm)
        # kind = ['norm', 'chi2']
        #
        # # For kind = 'norm'
        # arrival_mean = 30. # in minutes
        # arrival_std = 6.
        # height_mean = None # in meters
        # height_std = None
        # arrival_params = [arrival_mean, arrival_std]
        # # height_params = [height_mean, height_std]
        #
        # # For kind = 'chi2'
        # arrival_k = None # chi2 parameter
        # arrival_lower_bound = None # in meters
        # height_k = 5 # chi2 param
        # height_lower_bound = 4 # in meters
        # # arrival_params = [arrival_k, arrival_lower_bound]
        # height_params = [height_k, height_lower_bound]
        #
        # # For kind = 'skewnorm'
        # arrival_skew_param = None
        # arrival_mean = None # in minutes
        # arrival_std = None
        # height_skew_param = None
        # height_mean = None # in meters
        # height_std = None
        # # arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
        # # height_params = [height_skew_param, height_mean, height_std]
        #
        # g = Gauge(name, longitude, latitude, distance,
        #                 kind, arrival_params, height_params)
        # gauges.append(g.to_json())

        #Returns a list of gauges for the 1852 Priors
        gauges = load_gauges()
        # print(gauges)


        # Set gauge values for gauge 3 following pattern
        # as set out above (if desired)

        # latitude and longitude bounds (same as etopo file)
        xlower = 125.
        xupper = 135.
        ylower = -8.
        yupper = -2.

        # Length of time to run the model (in minutes)
        time = 75.0

        #############################
        # DO NOT MODIFY BEYOND THIS POINT

        # Save files
        # Save means and stds for prior to prior.npy.
        # TODO this will need to be modified for differently
        # distributed priors
        means = self.guesses
        stds = np.array([strike_std, length_std, width_std, depth_std,
            slip_std, rake_std, dip_std, longitude_std, latitude_std])
        probability_params = np.vstack((means, stds))
        probability_params = probability_params.T
        np.save("prior.npy", probability_params)

        # Save initial guesses to samples.npy.
        init_p_w = np.array([0,1])
        sample = np.hstack((self.guesses, init_p_w))
        sample = np.vstack((sample, sample))
        np.save("samples.npy", sample)

        # Save gauges to gauges.txt
        self.gauges = gauges
        with open('gauges.txt', 'w') as outfile:
            json.dump(gauges, outfile)

        # DEPRICATED
        # output_params = []
        # for gauge in gauges:
        #     output_params.append([gauge.arrival_mean, gauge.arrival_std])
        # for gauge in gauges:
        #     output_params.append([gauge.height_mean, gauge.height_std])
        # output_params = np.array(output_params)
        # np.save("output_dist.npy", output_params)
        #
        # # Save gauge names
        # gauge_names = []
        # for gauge in gauges:
        #     gauge_names.append([gauge.name, gauge.longitude,
        #                         gauge.latitude, gauge.distance])
        # np.save("gauges.npy", np.array(gauge_names))

        # Save latitude and longitude bounds and time to run model
        data = dict()
        data['xlower'] = xlower
        data['xupper'] = xupper
        data['ylower'] = ylower
        data['yupper'] = yupper
        data['run_time'] = time
        with open('model_bounds.txt', 'w') as outfile:
            json.dump(data, outfile)

    def run_once(self):
        """
        Run Geoclaw one time using the initial data given above.
        (This initializes the samples.npy file for subsequent runs.)
        """
        os.system('rm dtopo.tt3')
        os.system('rm dtopo.data')

        # Do initial run of GeoClaw using the initial guesses.
        mt.get_topo()
        mt.make_dtopo(self.guesses)

        os.system('make clean')
        os.system('make clobber')
        os.system('make .output')

        gauges_ = []
        for g in self.gauges:
            G = Gauge(None, None, None, None, None, None, None,None,None,None,None)
            G.from_json(g)
            gauges_.append(G)
        p = gauge.calculate_probability(gauges_)

        # Change entries in samples.npy
        samples = np.load('samples.npy')
        samples[0][-2] = p
        samples[1][-2] = p
        np.save('samples.npy', samples)



if __name__ == "__main__":
    s = Setup()
#    s.make_input_files()
    s.run_once()
