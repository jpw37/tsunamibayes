import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix

A = np.load('samples.npy') # Must be in the same directory
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

def add_axis_label(param, axis):
    # label = param + ' '
    label = ''
    if param == 'strike' or param == 'dip' or param == 'rake' or param == 'longitude' or param == 'latitude':
        label += 'degrees'
    elif param == 'length' or param == 'width' or param == 'slip':
        label += 'meters'
    elif param == 'depth':
        label += 'kilometers'

    if axis == 'x':
        plt.xlabel(label)
    elif axis == 'y':
        plt.ylabel(label)

def make_hist(param, bins=30):
    if param not in d.keys():
        print("{} is not a valid parameter.".format(param))
        return
    column = d[param]
    freq = A[1:, -1]
    values = A[1:, column]

    L = []
    for i in range(len(freq)):
        L += ([values[i]]*int(freq[i]-1)) # -1 to get rid of rejected

    # Can add other things to the plot if desired, like x and y axis
    # labels, etc.
    plt.hist(L, bins)
    plt.title(param)
    add_axis_label(param, 'x')

def make_2dhist(param1,param2,bins):
    """Make a 2d histogram of two parameters with the specified number
    of bins. Loads data from 'samples.npy'.
    Parameters:
        param (str): The name of the parameters for the histogram.
        bins (int): The number of bins to use for the histogram. Defaults to 20.
    """
    if param1 not in d.keys():
        print("{} is not a valid parameter value.".format(param1))
    elif param2 not in d.keys():
        print("{} is not a valid parameter value.".format(param2))
    else:
        column1 = d[param1]
        column2 = d[param2]
        freq = A[1:, -1]
        values1 = A[1:, column1]
        values2 = A[1:, column2]

        L1 = []
        L2 = []
        for i in range(len(freq)):
            L1 += ([values1[i]]*int(freq[i]-1)) # -1 to get rid of rejected
            L2 += ([values2[i]]*int(freq[i]-1)) # -1 to get rid of rejected

        # Can add other things to the plot if desired, like x and y axis
        # labels, etc.
        plt.hist2d(L1,L2, bins,cmap="viridis")
        # plt.xlabel(param1)
        # plt.ylabel(param2)
        add_axis_label(param1, 'x')
        add_axis_label(param2, 'y')
        plt.title('%s vs %s' %(param1,param2))
        plt.colorbar()

def make_change_plot(param):
    if param not in d.keys():
        print("{} is not a valid parameter.".format(param))
        return
    column = d[param]
    x = [0] + [i-1 for i in range(2,len(A)) if A[i,-1] > 1]
    y = [A[1,column]] + [A[i+1,column] for i in range(1, len(A)-1) if A[i+1,-1] >1]
    plt.plot(x,y)
    plt.xlabel("Iteration")
    # plt.ylabel("Value")
    add_axis_label(param, 'y')
    plt.title(param)

def make_scatter_matrix():
    df = pandas.DataFrame(data=A[1:,:-2],columns=d.keys())
    scatter_matrix(df, alpha=0.2, diagonal="hist")

def generate_subplots(kind, bins=30):
    if kind not in ["values", "change"]:
        print("{} is not a valid plot type.".format(kind))
        return
    for i, key in enumerate(d.keys()):
        plt.subplot(3,3,int(i+1))
        if kind == "values":
            make_hist(key,bins)
        elif kind == "change":
            make_change_plot(key)
    plt.tight_layout()
    plt.show()

def plot_stuff(param1, param2, kind, bins=30):
    if param2 is not None:
        make_2dhist(param1, param2, bins)
        plt.show()
    elif param1 == "all":
        generate_subplots(kind, bins)
    elif kind == "change":
        make_change_plot(param1)
        plt.show()
    elif param1 == "scatter_matrix":
        make_scatter_matrix()
        plt.show()
    else:
        make_hist(param1, bins)
        plt.show()


if __name__ == "__main__":
    # Can be set to any of the 9 parameters in the dictionary, or
    # to "all"
    param1 = sys.argv[1]
    if param1 == "long" or param1 == "lat": # allow for shorthand
        param1 += "itude"

    # initialze values (kind to "values" and bins to 30)
    param2 = None
    kind = "values"
    bins = 30

    # extract other command line arguments
    if len(sys.argv) > 3: # gets kind and bins
        kind = sys.argv[2]
        bins = int(sys.argv[3])
    elif len(sys.argv) > 2:
        if sys.argv[2].isdigit(): # gets bins
            bins = int(sys.argv[2])
        else: # gets kind
            kind = sys.argv[2]

    if kind == "long" or kind == "lat":
        kind += "itude"
    if kind in d.keys():
        param2 = kind

    plot_stuff(param1, param2, kind, bins)
