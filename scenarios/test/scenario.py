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

    def sample(self, nsamples, output_dir=None, save_freq=1, verbose=False, refinement_ratios=None,
               multi_fidelity=False, ref_rat_max_values=None, config_path=None):
        """Samples from the chain."""
        self.scenarios[0].sample(
            nsamples,
            output_dir,
            save_freq,
            verbose,
            refinement_ratios,
            multi_fidelity,
            ref_rat_max_values,
            config_path
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
        'walanae_delta_logw',
        'walanae_depth_offset',
        'walanae_dip_offset',
        'walanae_strike_offset',
        'walanae_rake_offset',
        'mystery_latitude',
        'mystery_longitude',
        'mystery_magnitude',
        'mystery_delta_logl',
        'mystery_delta_logw',
        'mystery_depth_offset',
        'mystery_dip_offset',
        'mystery_strike_offset',
        'mystery_rake_offset'
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
        'walanae_rake',
        'walanae_depth_offset',
        'walanae_rake_offset',
        'walanae_dip_offset',
        'walanae_strike_offset',
        'mystery_latitude',
        'mystery_longitude',
        'mystery_length',
        'mystery_width',
        'mystery_slip',
        'mystery_strike',
        'mystery_dip',
        'mystery_depth',
        'mystery_rake',
        'mystery_depth_offset',
        'mystery_rake_offset',
        'mystery_dip_offset',
        'mystery_strike_offset'
    ]

    def __init__(self, forward_model, prior, covariance, save_all_data=False):
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

        super().__init__(prior, forward_model, save_all_data)

        self.flores_fault = forward_model.fault[0]
        self.walanae_fault = forward_model.fault[1]
        self.mystery_fault = forward_model.fault[2]

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

        # this used to be proposal[:-1] because of the fault index at the end
        proposal += np.random.multivariate_normal(
            np.zeros(len(self.sample_cols)),
            cov=self.cov
        )
        return proposal


    def propose_hmc(self, sample, time_steps=7, epsilon=0.001, delta=0.01):
        """ q is the current position, defined to be a sample
            p is the current momentum, defined to have a zero-mean multivariate gaussian distribution
        """

        model_params = self.map_to_model_params(sample)
        q = sample.copy()
        p = np.random.multivariate_normal(np.zeros(len(self.sample_cols)), cov=self.cov)

        current_p = p.copy()
        curr_dU = dU(q,
                     self.fault.strike_map,
                     self.fault.dip_map,
                     self.fault.depth_map,
                     self.config,
                     self.fault,
                     self.model_params,
                     self.model_output)

        p = p - epsilon * curr_dU / 2
        for i in range(time_steps):
            q = q + epsilon * p
            if i != time_steps - 1:
                p = p - epsilon * dU(q,
                                     self.fault.strike_map,
                                     self.fault.dip_map,
                                     self.fault.depth_map,
                                     self.config,
                                     self.fault,
                                     self.model_params,
                                     self.model_output)
        p = p - epsilon * dU(q,
                             self.fault.strike_map,
                             self.fault.dip_map,
                             self.fault.depth_map,
                             self.config,
                             self.fault,
                             self.model_params,
                             self.model_output)
        p = -p
        return q, current_p, p

    def sample_hmc(self, nsamples, time_steps=7, epsilon=0.001, delta=0.01, output_dir=None, save_freq=1, verbose=False):
        """Draw samples from the posterior distribution using the HMC algorithm.
        
        Parameters
        ----------
        nsamples: int
            Number of samples to draw.
        time_steps: int
            Number of iterations to take with HMC algorithm.
        epsilon: float
            epsilon value for HMC algorithm.
        delta: float
            delta value for HMC algorithm.
        output_dir: None
            The name of the output directory to save the sample data.
        save_freq: int
            The integer that sets how frequently the sample data will be saved
            and written to a file. Default is 1. This also represents the number
            of rows to append when this function calls the save_data function.
        verbose: bool.
            If true, prints gague data for the loglikelihood of the forward
            model. Default is false.

        Returns
        -------
        samples: pandas DataFrame
            Pandas dataframe containing the set of samples from all of the
            accepted proposals generated, including any from prior runs (such as
            when using Scenario.restart()).

        """
        if not hasattr(self, 'samples'):
            name = type(self).__name__
            raise AttributeError("Chain must first be initialized with "
                                 "{}.init_chain() or {}.resume_chain()".format(name, name)
                                 )
        if output_dir is not None:
            if not os.path.exist(output_dir): os.mkdir(output_dir)
            self.save_data(output_dir)
            
        chain_start = time.time()
        j = 0
        for i in range(len(self.samples), len(self.samples) + nsamples):
            if verbose:
                print("\n------------\nIteration {}".format(i))
                start = time.time()
            
            # propose new sample from previous
            proposal, current_p, proposal_p = self.propose_hmc(self.samples.loc[i-1], 
                                                               time_steps=time_steps, 
                                                               epsilon=epsilon, 
                                                               delta=delta)
            if verbose: print("Proposal:"); print(proposal)

            # evaluate prior logpdf
            prior_logpdf = self.prior.logpdf(proposal)
            if verbose: print(f"Prior logpdf = {prior_logpdf}")

            # If prior logpdf is -infinity, reject proposal and bypass forward model
            if prior_logpdf == np.NINF:
                # set acceptance probability to 0
                alpha = 0
                
                # model_params, model_output, and log-likelihood are set to Nan values
                model_params = self.model_params.iloc[0].copy()
                model_params[...] = np.nan
                model_output = self.model_output.iloc[0].copy()
                model_output[...] = np.nan
                llh = np.nan
            
            # Otherwise, run the forward model, calculate the log-likelhihood, and
            # calculate the HMC acceptance probability
            else:
                if verbose: print("Running forward model...", flush=True)
                model_output = self.forward_model.run(model_params)
                if verbose: print("Evaluating log-likelhihood:")
                llh = self.forward_model.llh(model_output, verbose)
                if verbose: print("Total llh = {}".format(llh))

                # acceptance probability
                current_U = - self.bayes_data.loc[i-1]['prior_logpdf'] - \
                            self.bayes_data.loc[i-1]['llh']
                proposed_U = -prior_logpdf - llh
                current_K = np.sum((current_p**2)) / 2
                proposed_K = np.sum((proposal_p**2)) / 2

                alpha = np.exp(current_U - proposed_U + current_K - proposed_K)
                if verbose: print(f"alpha = {alpha}")
                unif_samp = np.random.uniform()

                accepted = unif_samp < alpha

            # prior, likelihood, and posterior logpdf values
            bayes_data = pd.Series(
                [prior_logpdf, llh, prior_logpdf + llh], index=self.bayes_data_cols)
            
            if accepted:
                if verbose: print("Proposal accepted", flush=True)
                self.samples.loc[i] = proposal
                self.model_params.loc[i] = model_params
                self.model_output.loc[i] = model_output
                self.bayes_data.loc[i] = bayes_data
            else:
                if verbose: print("Proposal rejected", flush=True)
                self.samples.loc[i] = self.samples.loc[i-1]
                self.model_params.loc[i] = self.model_params.loc[i-1]
                self.model_output.loc[i] = self.model_output.loc[i-1]
                self.bayes_data.loc[i] = self.bayes_data.loc[i-1]

            # generate data for debug dataframe
            hmc_data = pd.Series({'alpha': alpha, 'accepted':int(accepted),
                                  'acceptance_rate': np.nan})
            self.debug.loc[i-1] = self.gen_debug_row(self.samples.loc[i-1],
                                                     proposal,
                                                     self.model_params.loc[i-1],
                                                     model_params,
                                                     self.bayes_data.loc[i-1],
                                                     bayes_data,
                                                     hmc_data)
            
            self.debug.loc[i-1, 'acceptance_rate'] = self.debug["accepted"].mean()

            if not j % save_freq and (output_dir is not None):
                if verbose: print("Saving data...")
                self.save_data(output_dir, append_rows=save_freq)
            
            if verbose:
                print("Iteration elapsed time: {}".format(timedelta(seconds=time.time()-start)))
            
            j += 1
        
        if output_dir is not None:
            self.save_data(output_dir)
        if verbose and (output_dir is not None): print("Saving data...")

        total_time = time.time() - chain_start

        if verbose: print("Chain complete, total time: {}, time per sample: {}".format(
                timedelta(seconds=total_time),
                timedelta(seconds=total_time/nsamples)
                ))
        return self.samples

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

        mystery_length = calc_length(sample['mystery_magnitude'], sample['mystery_delta_logl'])
        mystery_width = calc_width(sample['mystery_magnitude'], sample['mystery_delta_logw'])
        mystery_slip = calc_slip(sample['mystery_magnitude'], mystery_length, mystery_width)

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


        mystery_strike, mystery_strike_std = self.mystery_fault.strike_map(
            sample['mystery_latitude'],
            sample['mystery_longitude'],
            return_std=True
        )
        mystery_dip, mystery_dip_std = self.mystery_fault.dip_map(
            sample['mystery_latitude'],
            sample['mystery_longitude'],
            return_std=True
        )
        mystery_depth, mystery_depth_std = self.mystery_fault.depth_map(
            sample['mystery_latitude'],
            sample['mystery_longitude'],
            return_std=True
        )
        mystery_rake, mystery_rake_std = self.mystery_fault.rake_map(
            sample['mystery_latitude'],
            sample['mystery_longitude'],
            return_std = True
        )
        
        # Multiply strike, dip, depth offsets by standard deviation of
        # Gaussian processes
        sample['mystery_depth_offset'] *= mystery_depth_std
        sample['mystery_dip_offset'] *= mystery_dip_std
        sample['mystery_strike_offset'] *= mystery_strike_std
        sample['mystery_rake_offset'] *= mystery_rake_std



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
        model_params['walanae_rake_offset']     = sample['walanae_rake_offset']
        model_params['mystery_latitude']        = sample['mystery_latitude']
        model_params['mystery_longitude']       = sample['mystery_longitude']
        model_params['mystery_length']          = mystery_length
        model_params['mystery_width']           = mystery_width
        model_params['mystery_slip']            = mystery_slip
        model_params['mystery_strike']          = mystery_strike
        model_params['mystery_strike_offset']   = sample['mystery_strike_offset']
        model_params['mystery_dip']             = mystery_dip
        model_params['mystery_dip_offset']      = sample['mystery_dip_offset']
        model_params['mystery_depth']           = mystery_depth
        model_params['mystery_depth_offset']    = sample['mystery_depth_offset']
        model_params['mystery_rake']            = mystery_rake
        model_params['mystery_rake_offset']     = sample['mystery_rake_offset']

        return model_params
