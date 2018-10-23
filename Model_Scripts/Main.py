"""
Class to handle running the scenario
"""
import sys

import Scenario

if __name__ == "__main__":
    iterations = int(sys.argv[1])
    method = sys.argv[2]

    scenario = Scenario("1852 Event", method)
    scenario.run_model(iterations)