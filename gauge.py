# A file containing the Gauge class and gauge related functions
import numpy as np
from scipy import stats
from pmf import PMFData, PMF

class Gauge:
    """A gauge object class. Mainly for data storage.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        arrival_mean (float): the mean value for the arrival time. (mins)
        arrival_std (float): the standard deviation for arrival time.
        height_mean (float): the mean for wave height. (m)
        height_std (float): the standard deviation for wave height.
        longitude (float): the longitude location of the gauge.
        latitude (float): the latitude location of the gauge.
        distance (float): the distance from the gauge to shore. (km)
    """
    def __init__(self, name, arrival_mean, arrival_std, height_mean,
                    height_std, longitude, latitude, distance):
        self.name = name
        self.arrival_mean = arrival_mean
        self.arrival_std = arrival_std
        self.height_mean = height_mean
        self.height_std = height_std
        self.longitude = longitude
        self.latitude = latitude
        self.distance = distance

def read_gauges(gauges):
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
    n = len(gauges)
    base_loc = "_output/gauge"

    arrivals = np.zeros(n)
    max_heights = np.zeros(n)

    # if gauge files are separate
    for i, gauge in enumerate(gauges): # extract from all gauge files
        gauge_file = base_loc + str(int(gauge)) + ".txt"
        with open(gauge_file, 'r') as f:
            lines = f.readlines()

        # get rid of first 2 lines
        lines.remove(lines[0])
        lines.remove(lines[0])

        # extract data to array
        data = np.zeros((len(lines), 6))
        for j, line in enumerate(lines):
            data[j] = line.split()

        h = data[:,5]
        t = data[:,1]
        max_idx = np.argmax(h)
        arrivals[i] = t[max_idx]/60.
        max_heights[i] = h[max_idx]

    return arrivals, max_heights

    # NOTE GeoClaw might clump all of the gauge data into
    # _output/fort.gauge instead of into separate gauge<gauge_number>.txt
    # files. If this is the case, comment out the above code (the portion
    # below "If gauge files are separate") and use this code instead:

    # # if gauge file is together
    # gauge_file = "_output/fort.gauge"
    # lines = []
    # with open(gauge_file, 'r') as f2:
    #     lines = f2.readlines()
    #
    # A = np.zeros((len(lines),7))
    # for i in range(len(lines)):
    #     A[i,:] = map(float, lines[i].split())
    #
    # # extract wave height and arrival time from each gauge
    # # arrival time in minutes
    # arrivals = np.zeros(n)
    # max_heights = np.zeros(n)
    # for i in range(n):
    #     h = np.array([A[j,6] for j in range(len(A[:,6])) if A[j,0] == int(gauges[i])])
    #     t = np.array([A[j,2] for j in range(len(A[:,6])) if A[j,0] == int(gauges[i])])
    #     max_idx = np.argmax(h)
    #     arrivals[i] = t[max_idx]/60.
    #     max_heights[i] = h[max_idx]
    # return arrivals, max_heights

def calculate_probability(gauges):
    # TODO Change this so gauges is a list of Gauge objects?
    """
    Calculate the log-likelihood of the data at each of the gauges
    based on our chosen distributions for maximum wave heights and
    arrival times. Return the sum of these log-likelihoods.

    Parameters:
        gauges (2d array): A 2d array where each row represents a
            different gauge and contains the following information:
            gauge name, longitude, latitude, and distance from shore
            in that order.
    Returns:
        p (float): The sum of the log-likelihoods of the data of each
            gauge in gauges.
    """
    arrivals, heights = read_gauges(gauges[:,0])

    # Calculate p for the arrivals first
    arrival_params = np.load('output_dist.npy')[:len(arrivals)]
    p = -sum(((arrivals - arrival_params[:,0])/arrival_params[:,1])**2)/2

    # DEPRICATED (from before it was log-likelihood)
    # p = 1.
    # for i, params in enumerate(arrival_params):
    #     # Creates normal distribution with given params for each variable and
    #     # gauge, in this order: 1. arrival of gauge1, 2. arrival of gauge2,
    #     # 3. ...
    #     dist = stats.norm(params[0], params[1])
    #     p_i = dist.pdf(arrivals[i])
    #     p *= p_i

    # Calculate p for the heights, using the PMFData and PMF classes
    amplification_data = np.load('amplification_data.npy')
    row_header = amplification_data[:,0]
    col_header = np.arange(len(amplification_data[0]) - 1)/4
    pmfData = PMFData(row_header, col_header, amplification_data[:,1:])
    # Integrate probability distributions for each gauge, using
    # the PMF class and add together the log-likelihoods
    # with the previous p calculated above.
    height_params = np.load('output_dist.npy')[len(arrivals):]
    for i, params in enumerate(height_params):
        # Creates PMF distribution integrated with normal distribution
        # where the normal distribution is given from the gauge data
        # in output_params.npy
        pmf = pmfData.getPMF(gauges[i][3], heights[i])
        p_i = pmf.integrate(stats.norm(params[0], params[1]))
        p += np.log(p_i) # Take Log-Likelihood

    return p
