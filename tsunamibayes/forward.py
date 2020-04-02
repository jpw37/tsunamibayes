import json
import numpy as np
import pandas as pd

class BaseForwardModel:
    def __init__(self,gauges):
        self.gauges = gauges

        # #Load in global parameters
        # with open('parameters.txt') as json_file:
        #     self.global_params = json.load(json_file)

    def run(self,model_params):
        """
        Run  Model(model_params)
        Read gauges
        Return observations (arrival times, Wave heights)
        """
        raise NotImplementedError("run() must be implemented in classes inheriting from BaseForwardModel")

    def llh(self, model_output):
        """
        Parameters:
        ----------
        observations : ndarray
            arrivals , heights
        Compute/Return llh
        """
        raise NotImplementedError("llh() must be implemented in classes inheriting from BaseForwardModel")

class TestForwardModel(BaseForwardModel):
    def run(self,model_params):
        d = {}
        for gauge in self.gauges:
            if 'power' in gauge.obstypes:
                d[gauge.name+' power'] = np.log(model_params['length']*model_params['width'])
        return d

    def llh(self,model_output):
        llh = 0
        for gauge in self.gauges:
            if 'power' in gauge.obstypes:
                llh += gauge.dists['power'].logpdf(model_output[gauge.name+' power'])
        return llh
