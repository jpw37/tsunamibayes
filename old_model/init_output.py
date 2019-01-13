# init_output.py
import numpy as np
from scipy import stats

def init_output():
    """
    1. Put the numbers corresponding to the gauges that were used in the
       GeoClaw run in a list.
    2. Set probability distribution parameters for both arrival time and
       max wave height for each gauge.
    3. Save these in output_dist.npy in a similar format to the param_dist.npy
       file. The following is an example assuming we are working with
       normal distributions for each of the gauges and types of data:
                          mean:    std dev:
       arrival (gauge 1):   30.           5.
       arrival (gauge 2):   60.          10.
       .                      .           .
       .                      .           .
       .                      .           .
       height (gauge 1):     5.           1.
       height (gauge 2):     7.           0.5
       .                      .           .
       .                      .           .
       .                      .           .

    4. Extract data for arrival times and maximum wave heights for each gauge
       from _output/fort.gauge, a file produced from running the model. Save
       this information in two NumPy arrays, called arrivals and max_heights.
       These arrays will correspond to the list made in step 1. (i.e. the
       first entry in the list from step 1 will be from the same gauge as
       the first entry in arrivals, and the first entry in max_heights).
    5. Create probability distributions for each gauge and variable (arrivals
       and max heights) using the parameters from step 2. Then, find the
       probability of each variable at each gauge and multiply them
       all together to find the probability in total of the run. (We assume
       for now that all variables are independent)
    6. Read in data from samples.npy and change the entries in the second to
       last column to all be equal to the probability from step 6.

    POTENTIAL CHANGES:
    * Pass in parameters from separate file where all
      run specific changes are made to keep them all in one place.
    * Use distributions other than the normal distribution, meaning the size of
      the NumPy array saved in output_dist.npy will have to be changed.
    * If variables are not independent, a simple multiplication of all
      probabilities as described in step 5 will not suffice.
    * Abstract steps to functions.
    """
    # Step 1
    # Put gauge numbers into a list.
    # EARTHQUAKE SPECIFIC (get numbers from SetRun.py, around line 342)
    # The following are just placeholders for now
    gauges = [10000, 10010] # Can be any number of gauges (min 1)
    n = len(gauges)

    # Step 2
    # Set probability distribution parameters for arrivals and max height
    # for each gauge.
    # EARTHQUAKE SPECIFIC
    # GAUGE 10000
    arrival1_mean = 30. # Expected to arrive 30 minutes after earthquake
    arrival1_std = 3. # Accounts for possible error
    height1_mean = 4. # Expected height in meters
    height1_std = .5 # Accounts for possible error

    # GAUGE 10010
    arrival2_mean = 45.
    arrival2_std = 3.
    height2_mean = 5.
    height2_std = 1.

    # GAUGE 10020
    # REPEAT FOR AS MANY GAUGES AS THERE ARE

    # Step 3
    # Save parameters to output_dist.npy in format described in docstring.
    means = np.array([arrival1_mean, arrival2_mean, height1_mean, height2_mean])
    stds = np.array([arrival1_std, arrival2_std, height1_std, height2_std])
    output_params = np.vstack((means, stds))
    output_params = output_params.T
    np.save("output_dist.npy", output_params)

    # Step 4
    '''Read output and look for necessary conditions.
    This file will find the max wave height

    Meaning of Gauge columns:
    - column 0 is gauge number
    - column 2 is time
    - column 3 is a scaled water height
    - column 6 is the graph that appears in plots'''
    # read in file
    gauge_file = "_output/fort.gauge"
    lines = []
    with open(gauge_file, 'r') as f2:
        lines = f2.readlines()

    A = np.zeros((len(lines),7))
    for i in xrange(len(lines)):
        A[i,:] = map(float, lines[i].split())

    # extract wave height and arrival time from each gauge
    # arrival time in minutes
    arrivals = np.zeros(n)
    max_heights = np.zeros(n)
    for i in xrange(n):
        h = np.array([A[j,6] for j in xrange(len(A[:,6])) if A[j,0] == gauges[i]])
        t = np.array([A[j,2] for j in xrange(len(A[:,6])) if A[j,0] == gauges[i]])
        print i, len(h)
        max_idx = np.argmax(h)
        arrivals[i] = t[max_idx]/60.
        max_heights[i] = h[max_idx]
    max_heights *= 2.5 # Amplification factor

    # Step 5
    # Create probability distributions for each gauge and variable.
    # Then, multiply together the probabilities of each output
    arrivals_and_heights = np.hstack((arrivals, max_heights))
    p = 1.
    for i, params in enumerate(output_params):
        # Creates normal distribution with given params for each variable and
        # gauge, in this order: 1. arrival of gauge1, 2. arrival of gauge2,
        # 3. ..., n+1. max height of gauge1, n+2, max height of gauge2, ...
        dist = stats.norm(params[0], params[1])
        p_i = dist.pdf(arrivals_and_heights[i])
        p *= p_i

    # Step 6
    # Change entries in samples.npy
    samples = np.load('samples.npy')
    samples[0][-2] = p
    samples[1][-2] = p
    np.save('samples.npy', samples)

if __name__ == "__main__":
    init_output()
