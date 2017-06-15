# set_params.py
import maketopo as mt
import numpy as np
from scipy import stats


def create_topo():
    """
    1. Load distribution parameters for each earthquake parameter from
       param_dist.npy.
    2. Create the distributions from the parameters. Then, take a random
       draw from each of the 9 distributions and save it in a NumPy array.
    3. Append the draw (as well as initialized p and w) to samples.npy.
    4. Run mt.get_topo() and mt.make_dtopo(), passing in the random draws
       from step 2 into mt.make_dtopo().

    POTENTIAL CHANGES:
    *
    """
    # Step 1
    # Load distribution parameters.
    params = np.load('param_dist.npy')

    # Step 2
    # Take a random draw from the distributions for each parameter.
    # For now assume all are normal distributions.
    draws = []
    for param in params:
        dist = stats.norm(param[0], param[1])
        draws.append(dist.rvs())
    draws = np.array(draws)

    # Step 3
    # Append draws and initialized p and w to samples.npy.
    init_p_w = np.array([0,1])
    sample = np.hstack((draws, init_p_w))
    samples = np.vstack((np.load('samples.npy'), sample))
    np.save('samples.npy', samples)

    # Step 4
    # Run GeoClaw using draws.
    mt.get_topo()
    mt.make_dtopo(draws)

if __name__ == "__main__":
    create_topo()
