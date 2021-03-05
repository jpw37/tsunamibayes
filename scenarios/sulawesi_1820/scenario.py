import numpy as np
import pandas as pd
from tsunamibayes import BaseScenario
from tsunamibayes.utils import calc_length, calc_width, calc_slip

class MultiFaultScenario():
    def __init__(self,scenarios):
        """Wrapper for multiple scenarios."""
        self.scenarios = scenarios

    def init_chain(self,fault_idx,u0=None,method=None,verbose=False,**kwargs):
        """Initializes a chain associated with scenarios[fault_idx]."""
        self.scenarios[fault_idx].init_chain(
            u0=u0,
            method=method,
            verbose=verbose,
            **kwargs
        )

    def resume_chain(self,fault_idx,output_dir):
        """Resumes a chain associated with scenarios[fault_idx]."""
        self.scenarios[fault_idx].resume_chain(output_dir)

    def sample(self,fault_idx,nsamples,output_dir=None,save_freq=1,verbose=False):
        """Samples from the chain at fault_idx."""
        self.scenarios[fault_idx].sample(nsamples,output_dir,save_freq,verbose)


class SulawesiScenario(BaseScenario):
    sample_cols = [
        'latitude',
        'longitude',
        'magnitude',
        'delta_logl',
        'delta_logw',
        'depth_offset',
        'dip_offset',
        'strike_offset',
        'rake_offset',
        'fault_idx'
    ]
    model_param_cols = [
        'latitude',
        'longitude',
        'length',
        'width',
        'slip',
        'strike',
        'dip',
        'depth',
        'rake',
        'depth_offset',
        'rake_offset', # TODO: do rake and dip offsets need to be Okada parameters?
        'dip_offset',
        'strike_offset'
    ]

    def __init__(self,prior,forward_model,covariance):
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
            Essentailly the same format as 'sample', we have simply added a
            multivariate normal to produce proposal values for lat, lon,
            mag, etc.
        """
        proposal = sample.copy()
        proposal[:-1] += np.random.multivariate_normal(
            np.zeros(len(self.sample_cols)-1),
            cov=self.cov
        )
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
        if sample['fault_idx'] == 0: # If we are on the Flores fault:
            strike, strike_std = self.fault.strike_map(
                sample['latitude'],
                sample['longitude'],
                return_std=True
            )
            dip, dip_std = self.fault.dip_map(
                sample['latitude'],
                sample['longitude'],
                return_std=True
            )
            depth, depth_std = self.fault.depth_map(
                sample['latitude'],
                sample['longitude'],
                return_std=True
            )
            rake = 90 # FIXME: GPFault needs to fit to rake data as well.

            # Multiply strike, dip, depth offsets by standard deviation of
            #  Gaussian process
            sample['depth_offset'] *= 2*depth_std
            sample['dip_offset'] *= 2*dip_std
            sample['strike_offset'] *= 2*strike_std

        else:
            strike = self.fault.strike_map(sample['latitude'],
                                           sample['longitude'])
            dip = self.fault.dip_map(sample['latitude'],
                                     sample['longitude'])
            depth = self.fault.depth_map(sample['latitude'],
                                         sample['longitude'])
            rake = 90 # FIXME: is this fine?

        model_params = dict()           #TODO : Would we need to add dip_offset and rake_offset as Okada or model parameters?
        model_params['latitude'] = sample['latitude']
        model_params['longitude'] = sample['longitude']
        model_params['length'] = length
        model_params['width'] = width
        model_params['slip'] = slip
        model_params['strike'] = strike
        model_params['strike_offset'] = sample['strike_offset']
        model_params['dip'] = dip
        model_params['dip_offset'] = sample['dip_offset']
        model_params['depth'] = depth
        model_params['depth_offset'] = sample['depth_offset']
        model_params['rake'] = rake
        model_params['rake_offset'] = sample['rake_offset']

        return model_params
