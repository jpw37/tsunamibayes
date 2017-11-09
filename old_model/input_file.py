# File for all earthquake specific information
# This is the only file that needs to be changed to run GeoClaw on
# a different earthquake.
import numpy as np

def make_input_files():
    """
    Initialize all necessary variables for a GeoClaw run using MCMC and
    put them in their appropriate .npy files to be used later.
    Specifically, initialize the following files:
    *
    *
    *
    NOTE TO USER: Edit all of the marked variables to match the specific
    earthquake you would like to model.
    """
    # Change the following variables
    #####
    # initial guesses (mean for prior if using normal distribution)
    strike = 319.667
    length = 550.e3
    width = 100.e3
    depth = 20.08
    slip = 13.
    rake = 101.5
    dip = 11.
    longitude = 99.5
    latitude = -2.
    guesses = np.array([strike, length, width, depth, slip, rake, dip,
        longitude, latitude])

    # Parameters for priors
    # Standard deviations
    strike_std = 5.
    length_std = 50.
    width_std = 20.
    depth_std = 8.
    slip_std = 2.
    rake_std = 7.
    dip_std = 3.
    longitude_std = 1.
    latitude_std = 1.
    means = guesses
    stds = np.array([strike_std, length_std, width_std, depth_std,
        slip_std, rake_std, dip_std, longitude_std, latitude_std])

    # Gauge information


if __name__ == "__main__":
    make_input_files()
