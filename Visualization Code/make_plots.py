import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix
import operator

# A = np.load('samples.npy') # Must be in the same directory
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

def make_hist(param, bins=30, prior=False, indices=None, normed=True, title=None, matrix=False):
    if prior:
        A = np.load("prior.npy")
    else:
        A = np.load("samples.npy")
    if param not in d.keys():
        print("{} is not a valid parameter.".format(param))
        return
    column = d[param]
    if indices is None:
        freq = A[1:, -1]
        values = A[1:, column]
    else:
        freq = A[indices[0]:indices[1], -1]
        values = A[indices[0]:indices[1],-1]

    L = []
    for i in range(len(freq)):
        L += ([values[i]]*int(freq[i]-1)) # -1 to get rid of rejected

    # Can add other things to the plot if desired, like x and y axis
    # labels, etc.
    if prior:
        plt.title(param + " (prior)")
        plt.hist(L, bins, label="prior", alpha=.6, normed=normed)
    else:
        plt.title(param)
        plt.hist(L, bins, label='observations', alpha=.6, normed=normed)
    if title is not None:
        plt.title(title)

    if not matrix:
        add_axis_label(param, 'x')

def make_2dhist(param1,param2,bins, prior=False, indices=None, normed=True, colorbar=True, title=None, matrix=False):
    """Make a 2d histogram of two parameters with the specified number
    of bins. Loads data from 'samples.npy'.
    Parameters:
        param (str): The name of the parameters for the histogram.
        bins (int): The number of bins to use for the histogram. Defaults to 20.
    """
    if prior:
        A = np.load("prior.npy")
    else:
        A = np.load("samples.npy")
    if param1 not in d.keys():
        print("{} is not a valid parameter value.".format(param1))
    elif param2 not in d.keys():
        print("{} is not a valid parameter value.".format(param2))
    else:
        column1 = d[param1]
        column2 = d[param2]
        if indices is None:
            freq = A[1:, -1]
            values1 = A[1:, column1]
            values2 = A[1:, column2]
        else:
            freq = A[indices[0]:indices[1], -1]
            values1 = A[indices[0]:indices[1],column1]
            values2 = A[indices[0]:indices[1],column2]

        L1 = []
        L2 = []
        for i in range(len(freq)):
            L1 += ([values1[i]]*int(freq[i]-1)) # -1 to get rid of rejected
            L2 += ([values2[i]]*int(freq[i]-1)) # -1 to get rid of rejected

        # Can add other things to the plot if desired, like x and y axis
        # labels, etc.
        plt.hist2d(L1,L2, bins,cmap="viridis", normed=normed)
        # plt.xlabel(param1)
        # plt.ylabel(param2)
        if not matrix:
            add_axis_label(param1, 'x')
            add_axis_label(param2, 'y')
        if prior:
            plt.title('%s vs %s (prior)' %(param1,param2))
        else:
            plt.title('%s vs %s' %(param1, param2))
        if colorbar:
            plt.colorbar()
        if title is not None:
            plt.title(title)

def make_change_plot(param, prior=False):
    if prior:
        A = np.load("prior.npy")
    else:
        A = np.load("samples.npy")
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
    if prior:
        plt.title(param + " (prior)")
    else:
        plt.title(param)

def make_scatter_matrix(prior=False):
    if prior:
        A = np.load("prior.npy")
    else:
        A = np.load("samples.npy")
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    names = [row[0] for row in sorted_d]
    df = pandas.DataFrame(data=A[1:,:-2],columns=names)
    scatter_matrix(df, alpha=0.2, diagonal="hist")

def make_correlations(prior=False):
    if prior:
        A = np.load("prior.npy")
    else:
        A = np.load("samples.npy")
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    names = [row[0] for row in sorted_d]
    df = pandas.DataFrame(data=A[1:,:-2],columns=names)
    correlations = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

def make_9x9_matrix(prior=False):
    params = d.keys()
    for i, param2 in enumerate(params):
        for j, param1 in enumerate(params):
            plt.subplot(9, 9, i*9+j+1)
            title = ""
            if i == 0:
                title = param1
            if param1 == param2:
                param = param1
                # xlower, xupper = bounds[param]
                make_hist(param, prior=prior, title=title, matrix=True)
                # plt.xlim(xlower, xupper)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(param2)
            else:
                bins = 30
                # xlower, xupper = bounds[param1]
                # ylower, yupper = bounds[param2]
                make_2dhist(param1, param2, bins, prior=prior, colorbar=False, title=title, matrix = True)
                # plt.xlim(xlower, xupper)
                # plt.ylim(ylower, yupper)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(param2)

    plt.show()

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
    elif param1 == "correlations":
        make_correlations()
        plt.show()
    else:
        make_hist(param1, bins)
        plt.show()


# if __name__ == "__main__":
#     # Can be set to any of the 9 parameters in the dictionary, or
#     # to "all"
#     param1 = sys.argv[1]
#     if param1 == "long" or param1 == "lat": # allow for shorthand
#         param1 += "itude"
#
#     # initialze values (kind to "values" and bins to 30)
#     param2 = None
#     kind = "values"
#     bins = 30
#
#     # extract other command line arguments
#     if len(sys.argv) > 3: # gets kind and bins
#         kind = sys.argv[2]
#         bins = int(sys.argv[3])
#     elif len(sys.argv) > 2:
#         if sys.argv[2].isdigit(): # gets bins
#             bins = int(sys.argv[2])
#         else: # gets kind
#             kind = sys.argv[2]
#
#     if kind == "long" or kind == "lat":
#         kind += "itude"
#     if kind in d.keys():
#         param2 = kind
#
#     plot_stuff(param1, param2, kind, bins)
