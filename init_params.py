# init_params.py
import numpy as np
import maketopo as mt

"""Description of samples.npy:

The file samples.npy will keep track of important information from each GeoClaw
run. Specifically, for each run, it keeps track of the 9 earthquake parameters
drawn for that run, the probability those parameters have of producing the
expected tsunami based on information from the output, and the total number
of 'wins' that set of parameters has. In addition, it keeps track of the
current set of parameters that a new run will be compared to. The file contains
a NumPy array of size (N+1, 11), where N is the number of times GeoClaw has been
run with different parameters. Each row i, excluding the first row, represents
the ith GeoClaw run. Columns 0-8 represent the 9 earthquake parameters, in
the order as seen in the below example. Column 9 represents the probability
that the set of parameters for that run produced the expected tsunami max wave
height and arrival time. Column 10 represents the number of 'wins' that set
of parameters has (i.e. the number of times that set of parameters was chosen
over a new set of parameters). Columns 0-8 are all floats. Column 9 is a float
between 0 and 1. Column 10 is stored as a float but is an integer in effect,
and is initialized to be 1.

The first row is slightly different. It represents the current 'best' iteration,
or the one that the next run will be compared to. As such, it has a duplicate
column elsewhere in the table. Columns 0-9 are the same as that duplicate
column, but column 10 represents the index of the duplicate. For example, if
the current 'best' iteration was produced on the 4th GeoClaw run, columns 1-9
of row 0 would be identical to columns 1-9 of row 4. Column 10 of row 0 would
be 4, representing the index where the duplicate can be found, while column 10
of row 4 represents the number of 'wins' that iteration has. When a new
iteration is produced that is accepted over the current 'best', row 0 is
replaced with the information associated with the new iteration.

The following is an example table, assuming 3 runs of geoclaw have been done,
that the first iteration was preferred over the second, and that the third
iteration was preferred over the first:

Row: strike: length: width: depth: slip: rake: dip: longitude: latitude: probability: index/wins:
0     322.38    570.   180.    21.   11. 105.9  10.      101.5      -2.5          0.5          3
1     319.67    550.   200.    20.   13. 101.5  11.      100.5      -3.5          0.3          2
2     318.56    535.   213.    18.   10. 108.1   9.      100.0      -1.8          0.1          1
3     322.38    570.   180.    21.   11. 105.9  10.      101.5      -2.5          0.5          2
"""

def init_param_dist():
    """
    1. Set initial guesses for each of the 9 earthquake parameters.
    2. Set parameters for the probability distribution to be used on each
       earthquake parameter.
    3. Save these parameters as a NumPy array in
       param_dist.npy where each row is an earthquake parameter,
       and each column is a probability distribution parameter. The following
       is an example assuming each earthquake parameter will have a normal
       distribution for each of its values:
               mean:    std dev:
       strike:   50.           5.
       length:  150.          10.
       .          .           .
       .          .           .
       .          .           .
    4. Save the initial guesses to samples.npy as described above this
       function.
    5. Run mt.get_topo() and mt.make_dtopo(), passing in the initial guesses
       from step 1 into mt.make_dtopo().

    POTENTIAL CHANGES:
    * Pass in initial guesses and parameters from separate file where all
      run specific changes are made to keep them all in one place.
    * Use distributions other than the normal distribution, meaning the size of
      the NumPy array saved in param_dist.npy will have to be changed.
    # Abstract steps to functions
    """
    # Step 1
    # Initial guesses for 9 earthquake parameters
    # EARTHQUAKE SPECIFIC
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

    # Step 2
    # Set parameters for probability distributions
    # We assume for now each one has a normal distribution with a mean equal
    # to our initial guess. Here we set the standard deviations.
    # EARTHQUAKE SPECIFIC
    strike_std = 5.
    length_std = 50.
    width_std = 20.
    depth_std = 8.
    slip_std = 2.
    rake_std = 7.
    dip_std = 3.
    longitude_std = 1.
    latitude_std = 1.

    # Step 3
    # Save probability distribution parameters in param_dist.npy in format
    # described above in the docstring.
    means = guesses
    stds = np.array([strike_std, length_std, width_std, depth_std,
        slip_std, rake_std, dip_std, longitude_std, latitude_std])
    probability_params = np.vstack((means, stds))
    probability_params = probability_params.T
    np.save("param_dist.npy", probability_params)

    # Step 4
    # Save initial guesses to samples.npy as described above this function.
    init_p_w = np.array([0,1])
    sample = np.hstack((guesses, init_p_w))
    sample = np.vstack((sample, sample))
    np.save("samples.npy", sample)

    # Step 5
    # Do initial run of GeoClaw using the initial guesses.
    mt.get_topo()
    mt.make_dtopo(guesses)

if __name__ == "__main__":
    init_param_dist()
