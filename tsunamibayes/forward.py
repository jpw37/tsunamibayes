import json

class BaseForwardModel:
    def __init__(self,gauges):
        self.gauges = gauges

        #Load in global parameters
        with open('parameters.txt') as json_file:
            self.global_params = json.load(json_file)

    def run(self,model_params):
        """
        Run  Model(model_params)
        Read gauges
        Return observations (arrival times, Wave heights)
        """
        raise NotImplementedError("{}.run() must be implemented in classes inheriting from BaseForwardModel".format(type(self).__name__))

    def llh(self, observations):
        """
        Parameters:
        ----------
        observations : ndarray
            arrivals , heights
        Compute/Return llh
        """
        raise NotImplementedError("{}.llh() must be implemented in classes inheriting from BaseForwardModel".format(type(self).__name__))
