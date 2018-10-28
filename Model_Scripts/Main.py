"""
Class to handle running the scenario
"""
import sys

import Scenario

if __name__ == "__main__":
    iterations = int(sys.argv[1])
    method = sys.argv[2]
    title = "1852 Event"
    use_custom = True
    rw_covariance = 0
    init = "manual"

    scenario = Scenario(title, use_custom, init, rw_covariance, method, iterations)
    scenario.run_model(iterations)