# Python file for reading output of Geoclaw
import numpy as np
from scipy import stats

def read_output():
    """
    1. Put the numbers corresponding to the gauges that were used in the
       GeoClaw run in a list.
    2. Extract data for arrival times and maximum wave heights for each gauge
       from _output/fort.gauge, a file produced from running the model. Save
       this information in two NumPy arrays, called arrivals and max_heights.
       These arrays will correspond to the list made in step 1. (i.e. the
       first entry in the list from step 1 will be from the same gauge as
       the first entry in arrivals, and the first entry in max_heights).
    3. Get probability distribution parameters for both arrival time and
       max wave height for each gauge from output_dist.npy.
    4. Create probability distributions for each gauge and variable (arrivals
       and max heights) using the parameters from step 2. Then, find the
       probability of each variable at each gauge and multiply them
       all together to find the probability in total of the run (We assume
       for now that all variables are independent). Call this probability p.
    5. Read in data from samples.npy and change the entry in the second to
       last column of the last row to be equal to the probability from step 4.
    6. Use Bayesian inference to accept or reject the current draw over the
       'best' draw. Accept the new one over the old one with
       probability = min(p(new)/p(old), 1).
    7. If the old one was selected, increment the number of 'wins' of the old
       draw by one.
       If the new one was selected, increment the number of 'wins' of the new
       draw by one, then set the current 'best' draw to be the new draw in
       samples.npy.

    POTENTIAL CHANGES:
    * Change the distribution parameters for each earthquake parameter
      based on which draw was selected.
    """
    # Step 1
    # Put gauge numbers into a list.
    # EARTHQUAKE SPECIFIC (get numbers from SetGeoClaw.py, around line 342)
    # The following are just placeholders for now
    gauges = [10000, 10010] # Can be any number of gauges (min 1)
    n = len(gauges)

    # Step 2
    # Extract arrivals and max_heights and save to arrays.
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

    # Step 3
    # Get distribution parameters from output_dist.npy.
    output_params = np.load('output_dist.npy')

    # Step 4
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

    # Step 5
    # Change entry in samples.npy
    samples = np.load('samples.npy')
    samples[-1][-2] = p

    # Step 6
    # Find probability to accept new draw over the old draw.
    accept_prob = min(p/samples[0][-2], 1)

    # Step 7
    # Increment wins. If new, change current 'best'.
    if np.random.random() < accept_prob: # Accept new
        samples[0] = samples[-1]
        samples[-1][-1] += 1
        samples[0][-1] = len(samples) - 1
    else: # Reject new
        samples[int(samples[0][-1])][-1] += 1 # increment old draw wins
    np.save('samples.npy', samples)

if __name__ == "__main__":
    read_output()
