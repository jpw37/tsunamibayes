"""The goal is for this file to be called by main.py from tsunamibayes
and to initialize implemented classes from the interfaces like BaseScenario
"""

def init_scenario():
    # Initialize specific scenario, prior, forward and pass the scenario
    # back to main.py (since scenario will contain the prior and forward that
    # can be called as needed from main.py, we don't need to return those to main)
    raise NotImplementedError