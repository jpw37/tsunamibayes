import numpy as np
import pandas as pd
from tsunamibayes import BaseScenario
from tsunamibayes.utils import calc_length, calc_width, calc_slip

class BandaScenario(BaseScenario):
    sample_cols = ['latitude','longitude','magnitude','delta_logl','delta_logw',
                   'depth_offset']
    model_param_cols = ['latitude','longitude','length','width','slip','strike',
                        'dip','depth','rake','depth_offset']

    def __init__(self,prior,forward_model,covariance):
        super().__init__(prior,forward_model)
        self.fault = forward_model.fault
        self.cov = covariance

    def propose(self,sample):
        """Random walk proposal of a new sample using a multivariate normal."""
        proposal = sample.copy()
        proposal += np.random.multivariate_normal(np.zeros(len(self.sample_cols)),cov=self.cov)
        return proposal

    def proposal_logpdf(self,u,v):
        """For the random walk proposal, the kernel is symmetric. This function
        returns 0 for convenience, as k(u,v) - k(v,u) = 0.
        """
        return 0

    def map_to_model_params(self,sample):
        """Evaluate the map from sample parameters to forward model parameters.
        """
        length = calc_length(sample['magnitude'],sample['delta_logl'])
        width = calc_width(sample['magnitude'],sample['delta_logw'])
        slip = calc_slip(sample['magnitude'],length,width)
        strike = self.fault.strike_map(sample['latitude'],
                                       sample['longitude'])
        dip = self.fault.dip_map(sample['latitude'],
                                 sample['longitude'])
        depth = self.fault.depth_map(sample['latitude'],
                                     sample['longitude'])
        rake = 90

        model_params = dict()
        model_params['latitude'] = sample['latitude']
        model_params['longitude'] = sample['longitude']
        model_params['depth_offset'] = sample['depth_offset']
        model_params['length'] = length
        model_params['width'] = width
        model_params['slip'] = slip
        model_params['strike'] = strike
        model_params['dip'] = dip
        model_params['depth'] = depth
        model_params['rake'] = rake
        return model_params
