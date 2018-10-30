"""
Class to handle running the scenario
"""
import sys

import Scenario

if __name__ == "__main__":
    title = sys.argv[1]

    if sys.argv[2] == 1:
        use_custom = True
    else:
        use_custom = False

    init = sys.argv[3]

    rw_covariance = sys.argv[4]

    method = sys.argv[5]

    iterations = int(sys.argv[6])

    scenario = Scenario(title, use_custom, init, rw_covariance, method, iterations)
    scenario.run_model()
