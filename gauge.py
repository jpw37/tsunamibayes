# A file containing the Gauge class
import numpy as np
from scipy import stats
from pmf import PMFData, PMF

class Gauge:
    """A gauge object class. Has a name and means and standard deviations
    for wave heights and arrival times.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        arrival_mean (float): the mean value for the arrival time.
        arrival_std (float): the standard deviation for arrival time.
        height_mean (float): the mean for wave height.
        height_std (float): the standard deviation for wave height.
        longitude (float): the longitude location of the gauge.
        latitude (float): the latitude location of the gauge.
        distance (float): the distance from the gauge to shore.
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

def read_gauges(gauges): # gauges is list of integers (gauge names)
    """Read GeoClaw output and look for necessary conditions.
    This will find the max wave height

    Meaning of gauge<gauge_number>.txt file columns:
    - column 1 is time
    - column 2 is a scaled water height
    - column 5 is the graph that appears in plots"""
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
    #     print(i, len(h))
    #     max_idx = np.argmax(h)
    #     arrivals[i] = t[max_idx]/60.
    #     max_heights[i] = h[max_idx]

    # The amplification factor will be accounted for in
    # the function below
    # max_heights *= 2.5 # Amplification factor

def calculate_probability(gauges): # gauges is list of gauge names, long, lat, dist in that order
    arrivals, heights = read_gauges(gauges[:,0])

    # First we calculate p for the arrivals
    # Create probability distributions for each gauge and variable.
    # Then, multiply together the probabilities of each output

    # DEPRICATED
    # p = 1.
    arrival_params = np.load('output_dist.npy')[:len(arrivals)]
    p = -sum(((arrivals - arrival_params[:,0])/arrival_params[:,1])**2)/2

    # DEPRICATED
    # for i, params in enumerate(arrival_params):
    #     # Creates normal distribution with given params for each variable and
    #     # gauge, in this order: 1. arrival of gauge1, 2. arrival of gauge2,
    #     # 3. ...
    #     dist = stats.norm(params[0], params[1])
    #     p_i = dist.pdf(arrivals[i])
    #     p *= p_i

    # Next, we calculate p for the heights, using the PMFData and
    # PMF classes
    amplification_data = np.load('amplification_data.npy')
    row_header = amplification_data[:,0]
    col_header = np.arange(len(amplification_data[0]) - 1)/4
    pmfData = PMFData(row_header, col_header, amplification_data[:,1:])
    # Integrate probability distributions for each gauge, using
    # the PMF class and multiply together the probabilities
    # with the previous p calculated above.
    height_params = np.load('output_dist.npy')[len(arrivals):]
    for i, params in enumerate(height_params):
        # Creates PMF distribution integrated with normal distribution
        # where the normal distribution is given from the gauge data
        # in output_params.npy
        pmf = pmfData.getPMF(gauges[i][3], heights[i]) # gauges[i][3] gets distance of that gauge
        p_i = pmf.integrate(stats.norm(params[0], params[1]))
        p += np.log(p_i) # Take Log-Likelihood

    return p

    # NOTE these are hints for how to run the code
    # pmf = self.pmfData.getPMF(distance from shore, GeoClaw output)
    # pmf.integrate(stats.norm(6,1)) where the stats.norm object is
    #  the one that we use from the original gauge data on run up height
    # #### I need to split up calculation to calculate arrival times
    #   #### and wave heights separately
