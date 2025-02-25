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
        'flores_latitude',
        'flores_longitude',
        'flores_magnitude',
        'flores_delta_logl',
        'flores_delta_logw',
        'flores_depth_offset',
        'flores_dip_offset',
        'flores_strike_offset',
        'flores_rake_offset',
        'walanae_latitude',
        'walanae_longitude',
        'walanae_magnitude',
        'walanae_delta_logl',
        'walanae_delta_logw'
        ########################################################################
        # 'walanae_depth_offset',
        # 'walanae_dip_offset',
        # 'walanae_strike_offset',
        # 'walanae_rake_offset'
        ########################################################################

    ]
    model_param_cols = [
        'flores_latitude',
        'flores_longitude',
        'flores_length',
        'flores_width',
        'flores_slip',
        'flores_strike',
        'flores_dip',
        'flores_depth',
        'flores_rake',
        'flores_depth_offset',
        'flores_rake_offset',
        'flores_dip_offset',
        'flores_strike_offset',
        'walanae_latitude',
        'walanae_longitude',
        'walanae_length',
        'walanae_width',
        'walanae_slip',
        'walanae_strike',
        'walanae_dip',
        'walanae_depth',
        'walanae_rake'
        ########################################################################
        # 'walanae_depth_offset',
        # 'walanae_rake_offset',
        # 'walanae_dip_offset',
        # 'walanae_strike_offset'
        ########################################################################
    ]


    def __init__(self, forward_model, flores_prior, flores_covariance,
                        walanae_prior, walanae_covariance):
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

        ########################################################################
        # It looks like this class initializes using the parent directory
        # I'm not sure what this looks like, so I'm going to make the inputs
        # lists and see what happens
        # super().__init__(prior, forward_model)
        super().__init__([flores_prior, walanae_prior], forward_model)

        # I'm assuming here that forward_model has two parts:
        # one one indexed at 0 is the flores forward model, and
        # the one indexed at 1 is the walanae forward model
        self.flores_fault = forward_model.fault[0]
        self.walanae_fault = forward_model.fault[1]
        ########################################################################

        # construct a single covariance matrix using the covariance matrices for flores and walanae
        self.cov = np.diag( np.hstack(( np.diag(flores_covariance), np.diag(walanae_covariance) )) )

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

    def map_to_model_params(self, sample):
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
        flores_length = calc_length(sample['flores_magnitude'], sample['flores_delta_logl'])
        flores_width = calc_width(sample['flores_magnitude'], sample['flores_delta_logw'])
        flores_slip = calc_slip(sample['flores_magnitude'], flores_length, flores_width)

        walanae_length = calc_length(sample['walanae_magnitude'], sample['walanae_delta_logl'])
        walanae_width = calc_width(sample['walanae_magnitude'], sample['walanae_delta_logw'])
        walanae_slip = calc_slip(sample['walanae_magnitude'], walanae_length, walanae_width)

        flores_strike, flores_strike_std = self.flores_fault.strike_map(
            sample['flores_latitude'],
            sample['flores_longitude'],
            return_std=True
        )
        flores_dip, flores_dip_std = self.flores_fault.dip_map(
            sample['flores_latitude'],
            sample['flores_longitude'],
            return_std=True
        )
        flores_depth, flores_depth_std = self.flores_fault.depth_map(
            sample['flores_latitude'],
            sample['flores_longitude'],
            return_std=True
        )
        flores_rake, flores_rake_std = self.flores_fault.rake_map(
            sample['flores_latitude'],
            sample['flores_longitude'],
            return_std=True
        )

        # Multiply strike, dip, depth offsets by standard deviation of
        # Gaussian processes
        sample['flores_depth_offset'] *= flores_depth_std
        sample['flores_dip_offset'] *= flores_dip_std
        sample['flores_strike_offset'] *= flores_strike_std
        sample['flores_rake_offset'] *= flores_rake_std


        walanae_strike = self.walanae_fault.strike_map(
            sample['walanae_latitude'],
            sample['walanae_longitude']
        )
        walanae_dip = self.walanae_fault.dip_map(
            sample['walanae_latitude'],
            sample['walanae_longitude']
        )
        walanae_depth = self.walanae_fault.depth_map(
            sample['walanae_latitude'],
            sample['walanae_longitude']
        )
        walanae_rake = 80 # On Walanae, the rake is assumed to be 80.

        model_params = dict()
        model_params['flores_latitude']         = sample['flores_latitude']
        model_params['flores_longitude']        = sample['flores_longitude']
        model_params['flores_length']           = flores_length
        model_params['flores_width']            = flores_width
        model_params['flores_slip']             = flores_slip
        model_params['flores_strike']           = flores_strike
        model_params['flores_strike_offset']    = sample['flores_strike_offset']
        model_params['flores_dip']              = flores_dip
        model_params['flores_dip_offset']       = sample['flores_dip_offset']
        model_params['flores_depth']            = flores_depth
        model_params['flores_depth_offset']     = sample['flores_depth_offset']
        model_params['flores_rake']             = flores_rake
        model_params['flores_rake_offset']      = sample['flores_rake_offset']
        model_params['walanae_latitude']        = sample['walanae_latitude']
        model_params['walanae_longitude']       = sample['walanae_longitude']
        model_params['walanae_length']          = walanae_length
        model_params['walanae_width']           = walanae_width
        model_params['walanae_slip']            = walanae_slip
        model_params['walanae_strike']          = walanae_strike
        # model_params['walanae_strike_offset']   = sample['walanae_strike_offset']
        model_params['walanae_dip']             = walanae_dip
        # model_params['walanae_dip_offset']      = sample['walanae_dip_offset']
        model_params['walanae_depth']           = walanae_depth
        # model_params['walanae_depth_offset']    = sample['walanae_depth_offset']
        model_params['walanae_rake']            = walanae_rake
        # model_params['walanae_rake_offset']     = sample['walanae_rake_offset']

        return model_params
