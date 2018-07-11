# A file containing the Gauge class and gauge related functions
import numpy as np
from scipy import stats
from pmf import PMFData, PMF

class Gauge:
    """A gauge object class. Mainly for data storage.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        longitude (float): the longitude location of the gauge.
        latitude (float): the latitude location of the gauge.
        distance (float): the distance from the gauge to shore. (km)
        kind (list): the type of distribution to be used for the wave
            arrival time and height respectively. ex: ['norm', 'chi2']
        arrival_params (list): list of params for the arrival time distribution
        height_params (list): list of params for the height distribution
        inundation_params (list): list of params for the inundation distribution
        arrival_dist (stats object): distribution of arrival time at gauge
        height_dist (stats object): distribution of wave height at gauge
    """
    def __init__(self, name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name):
        self.name = name
        self.city_name = city_name
        self.longitude = longitude
        self.latitude = latitude
        self.distance = distance
        self.kind = kind
        self.arrival_params = arrival_params
        self.height_params = height_params
        self.beta = beta
        self.n = n
        self.inundation_params = inundation_params
        if name is not None: # Allows for None initialized object
            # Kind[0] is for Wave Arrival Times
            # kind[1] is for Wave Height
            # kind[2] is for Inundation
            if kind[0] == 'norm':
                mean = arrival_params[0]
                std = arrival_params[1]
                self.arrival_dist = stats.norm(mean, std)
            elif kind[0] == 'chi2':
                k = arrival_params[0]
                loc = arrival_params[1]
                self.arrival_dist = stats.chi2(k, loc=loc)
            elif kind[0] == 'skewnorm':
                skew_param = arrival_params[0]
                mean = arrival_params[1]
                std = arrival_params[2]
                self.arrival_dist = stats.skewnorm(skew_param, mean, std)

            if kind[1] == 'norm':
                mean = height_params[0]
                std = height_params[1]
                self.height_dist = stats.norm(mean, std)
            elif kind[1] == 'chi2':
                k = height_params[0]
                loc = height_params[1]
                self.height_dist = stats.chi2(k, loc=loc)
            elif kind[1] == 'skewnorm':
                skew_param = height_params[0]
                mean = height_params[1]
                std = height_params[2]
                self.height_dist = stats.skewnorm(skew_param, mean, std)

            if kind[2] == 'norm':
                mean = inundation_params[0]
                std = inundation_params[1]
                self.inundation_dist = stats.norm(mean, std)
            elif kind[2] == 'chi2':
                k = inundation_params[0]
                loc = inundation_params[1]
                self.inundation_dist = stats.chi2(k, loc=loc)
            elif kind[2] == 'skewnorm':
                skew_param = inundation_params[0]
                mean = inundation_params[1]
                std = inundation_params[2]
                self.inundation_dist = stats.skewnorm(skew_param, mean, std)


    def to_json(self):
        """
        Convert object to dict of attributes for json
        """
        d = dict()
        d['name'] = self.name
        d['longitude'] = self.longitude
        d['latitude'] = self.latitude
        d['distance'] = self.distance
        d['kind'] = self.kind
        d['arrival_params'] = self.arrival_params
        d['height_params'] = self.height_params
        d['inundation_params'] = self.inundation_params
        d['beta'] = self.beta
        d['n'] = self.n
        d['city_name'] = self.city_name
        return d

    def from_json(self, d):
        """
        Converts from json file format into gauge object
        """
        self.__init__(d['name'], d['longitude'], d['latitude'],
                        d['distance'], d['kind'], d['arrival_params'],
                        d['height_params'], d['inundation_params'], d['beta'], d['n'], d['city_name'])

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
    data = np.loadtxt("_output/fort.FG1.valuemax")
    arrivals = data[:,4]
    max_heights = data[:,3]
    return arrivals,max_heights
    
    ###OLD - Now using fgmax class
    # n = len(gauges)
    # base_loc = "_output/gauge"

    # arrivals = np.zeros(n)
    # max_heights = np.zeros(n)

    # # if gauge files are separate
    # for i, gauge in enumerate(gauges): # extract from all gauge files
    #     gauge_file = base_loc + str(int(gauge)) + ".txt"
    #     with open(gauge_file, 'r') as f:
    #         lines = f.readlines()

    #     # get rid of first 2 lines
    #     lines.remove(lines[0])
    #     lines.remove(lines[0])

    #     # extract data to array
    #     data = np.zeros((len(lines), 6))
    #     for j, line in enumerate(lines):
    #         data[j] = line.split()

    #     h = data[:,5]
    #     t = data[:,1]
    #     max_idx = np.argmax(h)
    #     arrivals[i] = t[max_idx]/60.
    #     max_heights[i] = h[max_idx]

    # return arrivals, max_heights

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
    arrivals, heights = read_gauges(names)

    # Calculate p for the arrivals and heights
    p = 0. # init p
    # Calculate p for the heights, using the PMFData and PMF classes
    amplification_data = np.load('amplification_data.npy')
    row_header = amplification_data[:,0]
    col_header = np.arange(len(amplification_data[0]) - 1)/4
    pmfData = PMFData(row_header, col_header, amplification_data[:,1:])
    for i, gauge in enumerate(gauges):
        if(gauge.kind[0]):
            # arrivals
            p += np.log(gauge.arrival_dist.pdf(arrivals[i]))

        if(gauge.kind[1]):
            # heights
            pmf = pmfData.getPMF(gauge.distance, heights[i])
            p_i = pmf.integrate(gauge.height_dist)
            p += np.log(p_i)

        if(gauge.kind[2]):
            # inundation
            inun_values = np.power(pmf.vals,4/3) * 0.06 * np.cos(gauge.beta) / (gauge.n**2)
            inun_probability = pmf.probs
            pmf_inundation = PMF(inun_values, inun_probability)
            p_inundation = pmf_inundation.integrate(gauge.inundation_dist)
            p += np.log(p_inundation)

    return p

def make_input_files(self):
    pass




    # DEPRICATED
    # arrival_params = np.load('output_dist.npy')[:len(arrivals)]
    #
    # p = -sum(((arrivals - arrival_params[:,0])/arrival_params[:,1])**2)/2
    #
    #
    # # Calculate p for the heights, using the PMFData and PMF classes
    # amplification_data = np.load('amplification_data.npy')
    # row_header = amplification_data[:,0]
    # col_header = np.arange(len(amplification_data[0]) - 1)/4
    # pmfData = PMFData(row_header, col_header, amplification_data[:,1:])
    # # Integrate probability distributions for each gauge, using
    # # the PMF class and add together the log-likelihoods
    # # with the previous p calculated above.
    # height_params = np.load('output_dist.npy')[len(arrivals):]
    # for i, params in enumerate(height_params):
    #     # Creates PMF distribution integrated with normal distribution
    #     # where the normal distribution is given from the gauge data
    #     # in output_params.npy
    #     pmf = pmfData.getPMF(gauges[i][3], heights[i])
    #     p_i = pmf.integrate(stats.norm(params[0], params[1]))
    #     p += np.log(p_i) # Take Log-Likelihood
    #
    # return p
