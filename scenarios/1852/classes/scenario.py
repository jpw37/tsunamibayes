import numpy as np
import pandas as pd

# TODO: GET ACCESS TO BASE CLASS
from tsunamibayes import BaseScenario

class Scenario_1852(BaseScenario):
    def propose(self,sample):
        """Propose a new sample, perhaps dependent on the current sample.
        """
        raise NotImplementedError

    def proposal_logpdf(self,u,v):
        """Evaluate the logpdf of the proposal kernel, expressed as the log-probability-density
        of moving from `v` to `u`.
        """
        raise NotImplementedError

    def map_to_model_params(self,sample):
        """Evaluate the map from sample parameters to forward model parameters.
        """
        raise NotImplementedError