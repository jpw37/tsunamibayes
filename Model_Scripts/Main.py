"""
Class to handle running the scenario
"""
import sys

import Scenario

if __name__ == "__main__":
    iterations = int(sys.argv[1])
    method = sys.argv[2]
    title = "1852 Event"
    use_utils = True

    scenario = Scenario(title, use_utils, method)
    scenario.run_model(iterations)