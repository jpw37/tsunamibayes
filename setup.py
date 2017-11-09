# File for all earthquake specific information
# This is the only file that needs to be changed to run GeoClaw on
# a different earthquake.
import os
import numpy as np
from gauge import Gauge
import gauge
import maketopo as mt
import json
from scipy import stats

class Setup:

    def __init__(self):
        pass

    def make_input_files(self):
        """
        Initialize all necessary variables for a GeoClaw run using MCMC and
        put them in their appropriate .npy files to be used later.
        Specifically, initialize the following files:
        * guesses.npy
        * prior.npy
        * samples.npy
        * output_dist.npy
        * gauges.npy
        * model_bounds.txt
        #TODO: add file_guide.txt
        NOTE TO USER: Edit all of the marked variables to match the specific
        earthquake you would like to model.
        """
        # Change the following variables
        #####
        # initial guesses (mean for prior if using normal distribution)
        strike = 205.0
        length = 270.e3
        width = 80.e3
        depth = 20.0
        slip = 9.
        rake = 90.
        dip = 15.
        longitude = 132.4
        latitude = -6.485
        self.guesses = np.array([strike, length, width, depth, slip, rake, dip,
            longitude, latitude])
        np.save("guesses.npy", self.guesses)

        # Parameters for priors
        # Standard deviations
        strike_std = 10.
        length_std = 33.e3
        width_std = 15.e3
        depth_std = 3.3
        slip_std = 2.
        rake_std = 3.
        dip_std = 1.6
        longitude_std = .27
        latitude_std = .6
        means = self.guesses
        stds = np.array([strike_std, length_std, width_std, depth_std,
            slip_std, rake_std, dip_std, longitude_std, latitude_std])

        # Gauge information
        # Add as many as you like, repeating this pattern to add new gauges.
        gauges = []
        # Set gauge values for gauge 1
        name1 = 10000
        arrival_mean1 = 17.5 # in minutes
        arrival_std1 = 2.
        height_mean1 = 8. # in meters
        height_std1 = 2.
        longitude1 = 129.98
        latitude1 = -4.54

        gauge1 = Gauge(name1, arrival_mean1, arrival_std1, height_mean1,
                        height_std1, longitude1, latitude1)
        gauges.append(gauge1)

        # Set gauge values for gauge 2
        name2 = 10010
        arrival_mean2 = 60.
        arrival_std2 = 5.
        height_mean2 = 1.8
        height_std2 = .2
        longitude2 = 128.667
        latitude2 = -3.667

        gauge2 = Gauge(name2, arrival_mean2, arrival_std2, height_mean2,
                        height_std2, longitude2, latitude2)
        gauges.append(gauge2)

        # Set gauge values for gauge 3 (if desired)

        # latitude and longitude bounds
        xlower = 127.
        xupper = 134.5
        ylower = -7.5
        yupper = -2.5

        # Length of time to run the model (in minutes)
        time = 90.0

        #############################
        # DO NOT MODIFY BEYOND THIS POINT

        # Save files
        # Save means and stds for prior to prior.npy.
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

        # Save gauges to output_dist.npy.
        output_params = []
        for gauge in gauges:
            output_params.append([gauge.arrival_mean, gauge.arrival_std])
        for gauge in gauges:
            output_params.append([gauge.height_mean, gauge.height_std])
        output_params = np.array(output_params)
        # means = np.array([arrival1_mean, arrival2_mean, height1_mean, height2_mean])
        # stds = np.array([arrival1_std, arrival2_std, height1_std, height2_std])
        # output_params = np.vstack((means, stds))
        # output_params = output_params.T
        np.save("output_dist.npy", output_params)

        # Save gauge names
        gauge_names = []
        for gauge in gauges:
            gauge_names.append([gauge.name, gauge.longitude, gauge.latitude])
        np.save("gauges.npy", np.array(gauge_names))

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
        Runs Geoclaw one time using the initial data given above.
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

        gauges = np.load('gauges.npy')

        arrivals, max_heights = gauge.read_gauges(gauges[:,0])

        # Create probability distributions for each gauge and variable.
        # Then, multiply together the probabilities of each output
        arrivals_and_heights = np.hstack((arrivals, max_heights))
        p = 1.
        output_params = np.load('output_dist.npy')
        for i, params in enumerate(output_params):
            # Creates normal distribution with given params for each variable and
            # gauge, in this order: 1. arrival of gauge1, 2. arrival of gauge2,
            # 3. ..., n+1. max height of gauge1, n+2, max height of gauge2, ...
            dist = stats.norm(params[0], params[1])
            p_i = dist.pdf(arrivals_and_heights[i])
            p *= p_i

        # Change entries in samples.npy
        samples = np.load('samples.npy')
        samples[0][-2] = p
        samples[1][-2] = p
        np.save('samples.npy', samples)



if __name__ == "__main__":
    s = Setup()
    s.make_input_files()
    s.run_once()
