"""
Class to handle running the scenario
"""
import sys

sys.path.append('./Classes')

from Scenario import Scenario
import json

if __name__ == "__main__":
    scenario = Scenario()

    if(len(sys.argv) > 1):
        if(sys.argv[1] == 'custom'):
            with open('./inputs.txt') as json_file:
                inputs = json.load(json_file)
            scenario = Scenario(inputs['title'], inputs['custom'], inputs['init'], inputs['rw_covariance'], inputs['method'], inputs['iterations'])

    scenario.run()
