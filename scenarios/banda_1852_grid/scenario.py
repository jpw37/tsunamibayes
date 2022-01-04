import numpy as np
import pandas as pd
from tsunamibayes import BaseScenario
from tsunamibayes.utils import calc_length, calc_width, calc_slip

class BandaScenario(BaseScenario):
    sample_cols = ['latitude','longitude','magnitude','delta_logl','delta_logw',
                   'depth_offset']
    model_param_cols = ['latitude','longitude','length','width','slip','strike',
                        'dip','depth','rake','depth_offset']

    def __init__(self,prior,forward_model,covariance,start_idx):
        """Initializes all the necessary variables for the BandaScenario subclass.

        Parameters
        ----------
        prior : BandaPrior Object
            The prior object made from the scenario defaults for the prior distribution.
        forward_model : GeoClawForwardModel Object
            The forward model made with the scenario's gauge, fault, and togography data.
        covariance : array-like
            The ndarray that contains the covariances computed from the standard deviations for
            the scenario's latitude, longitude, magnitude, delta logl & logw, and depth offset.
        """
        super().__init__(prior,forward_model)
        self.fault = forward_model.fault
        self.cov = covariance
        self.grid_samples = np.load('data/grid_samples.npy')
        self.grid_idx = start_idx

    def propose(self,sample):
        """Random walk proposal of a new sample using a multivariate normal.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values.

        Returns
        -------
        proposal : pandas Series of floats
            Essentailly the same format as 'sample', we have simply added a multivariate normal
            to produce proposal values for lat, long, mag, etc.
        """
        proposal = pd.Series(
            self.grid_samples[self.grid_idx],
            index=self.sample_cols
        )
        self.grid_idx += 1
        return proposal

    def proposal_logpdf(self,u,v):
        """Evaluate the logpdf of the proposal kernel, expressed as the
        log-probability-density of proposing 'u' given current sample 'v'.
        For the random walk proposal, the kernel is symmetric. This function
        returns 0 for convenience, as k(u,v) - k(v,u) = 0.

        Parameters
        ----------
        u : pandas Series of floats
            The series that contains the float values for a proposal.
        v : pandas Series of floats
            The series that contains the float values for the current sample.
        """
        return 0

    def map_to_model_params(self,sample):
        """Evaluate the map from sample parameters to forward model parameters.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values.

        Returns
        -------
        model_params : dict
            A dictionary that builds off of the sample dictionary whose keys are the
            okada parameters: 'latitude', 'longitude', 'depth_offset', 'strike','length',
            'width','slip','depth','dip','rake',
            and whose associated values are the newly calculated values from the sample.
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
