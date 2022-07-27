import numpy as np
import pandas as pd
from tsunamibayes import BaseScenario
from tsunamibayes.utils import calc_length, calc_width, calc_slip

class MultiFaultScenario():
    def __init__(self, scenarios):
        """Wrapper for multiple scenarios."""
        self.scenarios = scenarios

    def init_chain(self, u0=None, method=None, verbose=False, **kwargs):
        """Initializes a chain associated with scenarios[0]."""
        self.scenarios[0].init_chain(
            u0=u0,
            method=method,
            verbose=verbose,
            **kwargs
        )

    def resume_chain(self, output_dir):
        """Resumes a chain associated with scenarios[0]."""
        self.scenarios[0].resume_chain(output_dir)

    def sample(
        self,
        nsamples,
        output_dir=None,
        save_freq=1,
        verbose=False,
    ):
        """Samples from the chain."""
        self.scenarios[0].sample(
            nsamples,
            output_dir,
            save_freq,
            verbose
        )



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
        'rake_offset'
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
        'rake_offset',
        'dip_offset',
        'strike_offset'
    ]

    def __init__(self, flores_prior, flores_forward_model, flores_covariance,
                        walanae_prior, walanae_forward_model, walanae_covariance):
        """Initializes all the necessary variables for the BandaScenario
        subclass.

        Parameters
        ----------
        prior : BandaPrior Object
            The prior object made from the scenario defaults for the prior
            distribution.
        forward_model : GeoClawForwardModel Object
            The forward model made with the scenario's gauge, fault, and
            togography data.
        covariance : array-like
            The ndarray that contains the covariances computed from the
            standard deviations for the scenario's latitude, longitude,
            magnitude, delta logl & logw, and depth offset.
        """
        # It looks like this class initializes using the parent directory
        # I'm not sure what this looks like, so I'm going to make the inputs
        # lists and see what happens
        # super().__init__(prior, forward_model)
        super().__init__([flores_prior, walanae_prior], [flores_forward_model, walanae_forward_model])
        self.flores_fault = flores_forward_model.fault
        self.flores_cov = flores_covariance
        self.walanae_fault = walanae_forward_model.fault
        self.walanae_cov = walanae_covariance

    def propose(self,sample):
        """Random walk proposal of a new sample using a multivariate normal.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float
            values.

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

    def map_to_model_params(self, flores_sample, walanae_sample):
        """Evaluate the map from sample parameters to forward model parameters.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float
            values.

        Returns
        -------
        model_params : dict
            A dictionary that builds off of the sample dictionary whose keys
            are the okada parameters: 'latitude', 'longitude', 'depth_offset',
            'strike','length', 'width','slip','depth','dip','rake', and whose
            associated values are the newly calculated values from the sample.
        """
        flores_length = calc_length(flores_sample['magnitude'], flores_sample['delta_logl'])
        flores_width = calc_width(flores_sample['magnitude'], flores_sample['delta_logw'])
        flores_slip = calc_slip(flores_sample['magnitude'], flores_length, flores_width)

        walanae_length = calc_length(walanae_sample['magnitude'], walanae_sample['delta_logl'])
        walanae_width = calc_width(walanae_sample['magnitude'], walanae_sample['delta_logw'])
        walanae_slip = calc_slip(walanae_sample['magnitude'], walanae_length, walanae_width)

        flores_strike, flores_strike_std = self.flores_fault.strike_map(
            flores_sample['latitude'],
            flores_sample['longitude'],
            return_std=True
        )
        flores_dip, flores_dip_std = self.flores_fault.dip_map(
            flores_sample['latitude'],
            flores_sample['longitude'],
            return_std=True
        )
        flores_depth, flores_depth_std = self.flores_fault.depth_map(
            flores_sample['latitude'],
            flores_sample['longitude'],
            return_std=True
        )
        flores_rake, flores_rake_std = self.flores_fault.rake_map(
            flores_sample['latitude'],
            flores_sample['longitude'],
            return_std=True
        )

        # WHY IS THIS BLOCK NOT REPEATED FOR WALANAE?!?!
        ########################################################################
        # Multiply strike, dip, depth offsets by standard deviation of
        # Gaussian processes
        flores_sample['depth_offset'] *= flores_depth_std
        flores_sample['dip_offset'] *= flores_dip_std
        flores_sample['strike_offset'] *= flores_strike_std
        flores_sample['rake_offset'] *= flores_rake_std
        ########################################################################

        walanae_strike = self.walanae_fault.strike_map(
            walanae_sample['latitude'],
            walanae_sample['longitude']
        )
        walanae_dip = self.walanae_fault.dip_map(
            walanae_sample['latitude'],
            walanae_sample['longitude']
        )
        walanae_depth = self.walanae_fault.depth_map(
            walanae_sample['latitude'],
            walanae_sample['longitude']
        )
        walanae_rake = 80 # On Walanae, the rake is assumed to be 80.


        model_params = dict()
        model_params['latitude']        = [flores_sample['latitude'], walanae_sample['latitude']]
        model_params['longitude']       = [flores_sample['longitude'], walanae_sample['longitude']]
        model_params['length']          = [flores_length, walanae_length]
        model_params['width']           = [flores_width, walanae_width]
        model_params['slip']            = [flores_slip, walanae_slip]
        model_params['strike']          = [flores_strike, walanae_strike]
        model_params['strike_offset']   = [flores_sample['strike_offset'], walanae_sample['strike_offset']]
        model_params['dip']             = [flores_dip, walanae_dip]
        model_params['dip_offset']      = [flores_sample['dip_offset'], walanae_sample['dip_offset']]
        model_params['depth']           = [flores_depth, walanae_depth]
        model_params['depth_offset']    = [flores_sample['depth_offset'], walanae_sample['depth_offset']]
        model_params['rake']            = [flores_rake, walanae_rake]
        model_params['rake_offset']     = [flores_sample['rake_offset'], walanae_sample['rake_offset']]

        return model_params
