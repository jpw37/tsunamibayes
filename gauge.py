# A file containing the Gauge class
import numpy as np

class Gauge:
    """A gauge object class. Has a name and means and standard deviations
    for wave heights and arrival times.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        arrival_mean (float): the mean value for the arrival time.
        arrival_std (float): the standard deviation for arrival time.
        height_mean (float): the mean for wave height.
        height_std (float): the standard deviation for wave height.
    """
    def __init__(self, name, arrival_mean, arrival_std, height_mean,
                    height_std, longitude, latitude):
        self.name = name
        self.arrival_mean = arrival_mean
        self.arrival_std = arrival_std
        self.height_mean = height_mean
        self.height_std = height_std
        self.longitude = longitude
        self.latitude = latitude

def read_gauges(gauges):
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
        print(i, len(h))
        max_idx = np.argmax(h)
        arrivals[i] = t[max_idx]/60.
        max_heights[i] = h[max_idx]




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

    max_heights *= 2.5 # Amplification factor TODO

    return arrivals, max_heights
