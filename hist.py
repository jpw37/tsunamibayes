# Makes a histogram of geoclaw parameter
import numpy as np
from matplotlib import pyplot as plt

def make_hist(param, bins=30):
    """Make a histogram of an individual parameter with the specified number
    of bins. Loads data from 'samples.npy'.
    Parameters:
        param (str): The name of the parameter for the histogram. If 'all' is
            used, it creates a histogram of all 9 parameters sequentially.
        bins (int): The number of bins to use for the histogram. Defaults to 30.
    """
    d = dict()
    d['strike'] = 0
    d['length'] = 1
    d['width'] = 2
    d['depth'] = 3
    d['slip'] = 4
    d['rake'] = 5
    d['dip'] = 6
    d['longitude'] = 7
    d['latitude'] = 8

    if param == 'all':
        for key in d.keys():
            make_hist(key, bins)
    else:
        column = d[param]
        A = np.load('samples.npy') # Must be in the same directory
        freq = A[1:, -1]
        values = A[1:, column]

        L = []
        for i in range(len(freq)):
            L += ([values[i]]*int(freq[i]))

        plt.hist(L, bins)
        plt.title(param)
        plt.show()
