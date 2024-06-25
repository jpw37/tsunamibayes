import numpy as np
import pandas as pd
from tsunamibayes import BaseScenario
from tsunamibayes.utils import calc_length, calc_width, calc_slip
from gradient import dU, calc_adjoint # , gradient_setup
import time
from datetime import timedelta
import os



class BandaScenario(BaseScenario):
    # Class variables with names of sampling chain columns and model (okada) parameter columns
    sample_cols = ['latitude', 'longitude', 'magnitude', 'delta_logl', 'delta_logw',
                   'depth_offset']
    model_param_cols = ['latitude', 'longitude', 'length', 'width', 'slip', 'strike',
                        'dip', 'depth', 'rake', 'depth_offset']

    def __init__(self, prior, forward_model, covariance, config):
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
        super().__init__(prior, forward_model)
        self.fault = forward_model.fault
        self.cov = covariance
        self.config = config
#         if config.init['mcmc_mode'] == 'mala':
#             gradient_setup(self.fault.dip_map,
#                            self.fault.depth_map, config, forward_model)


    def propose(self, sample, mode='random_walk', time_steps=25, epsilon=.00001, delta=0.01):
        """Random walk proposal of a new sample using a multivariate normal.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values.
        mode : str
            The desired mcmc algorithm ('random_walk' and 'mala' are supported)
        delta : float
            Step size (only for the mala mcmc method)

        Returns
        -------
        proposal : pandas Series of floats
            Essentailly the same format as 'sample', we have simply added a multivariate normal
            to produce proposal values for lat, long, mag, etc.
        """
        # Propose a new sample based on current sample
        proposal = sample.copy()

        # For random walk method, adds Gaussian noise to current sample
        if mode == 'random_walk':
            proposal += np.random.multivariate_normal(
                np.zeros(len(self.sample_cols)), cov=self.cov)

        # Uses MALA algorithm
        elif mode == 'mala':
            # Takes a step towards negative gradient of log pdf, then adds Gaussian noise
            v = np.random.multivariate_normal(
                np.zeros(len(self.sample_cols)), cov=self.cov)
            proposal += -delta**2 / 2 * dU(q, 
                                           self.fault.strike_map, 
                                           self.fault.dip_map, 
                                           self.fault.depth_map,
                                           self.config,
                                           self.fault,
                                           self.model_params,
                                           self.model_output,
                                           self.arrival_times) + delta * v

        # Uses HMC algorithm
        elif mode == 'hmc':
            # Gets forward model parameters from sample
            model_params = self.map_to_model_params(sample)

            #Copy current sample and initialize momentum vector p for HMC
            q = sample.copy()
            #p = np.random.multivariate_normal(np.zeros(len(q)), np.eye(len(q)))
            p = np.random.multivariate_normal(np.zeros(len(q)), cov=self.cov)

            # Calculate adjoint values needed for gradient computation
            grads, outputs = calc_adjoint(self.model_params, self.model_output, self.arrival_times)

            # Test initial momentum/sample
            # print('Initial p', p)
            # print('Initial q', q)
            current_p = p.copy()
#             print('-----------------------------------------------')
#             print('COMPUTING dU ONCE')
#             print(q)
            # Compute gradient of potential energy at current sample
            curr_dU = dU(q,
                         self.fault.strike_map,
                         self.fault.dip_map,
                         self.fault.depth_map,
                         self.config,
                         self.fault,
                         self.model_params,
                         self.model_output,
                         self.arrival_times,
                         grads, outputs)
            # print('dU')
            # print(curr_dU)
            # Adjust momentum with half step of gradient
            p = p - epsilon *  curr_dU/ 2
            # print('updated p', p)

            # Leapfrog method (integration) to update sample q
            for i in range(time_steps):
#                 print('-----------------------------------------------')
# #                 print('COMPUTING dU IN LOOP')

                q = q + epsilon * p
                # print('Updated p', p)
                # print('Updated q', q)

                # If not last step, compute gradient at new sample
                if i != time_steps - 1:
                    gradient = dU(q,
                                  self.fault.strike_map,
                                  self.fault.dip_map,
                                  self.fault.depth_map,
                                  self.config,
                                  self.fault,
                                  self.model_params,
                                  self.model_output,
                                  self.arrival_times,
                                  grads, outputs)
                    # print('Gradient', gradient)

                    # Update momentum using gradient
                    p = p - epsilon * gradient
            # Calculates final half step for momentum after leapfrog method
            p = p - epsilon * dU(q,
                                 self.fault.strike_map,
                                 self.fault.dip_map,
                                 self.fault.depth_map,
                                 self.config,
                                 self.fault,
                                 self.model_params,
                                 self.model_output,
                                 self.arrival_times,
                                 grads, outputs)/2
            # Reverse momentum direction for detailed balance
            p = -p

            # Return updated sample, and initial and final momentums
            return q, current_p, p

        # Raise error if not one of the three methods used
        else:
            raise ValueError(
                'Invalid Parameter, use \'random_walk\', \'mala\', or \'hmc\'')

        # Returns proposal sample
        return proposal

# Random-walk only method, for which the log pdf difference is 0
    def proposal_logpdf(self, u, v):
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
            'delta_logw', and 'depth_offset' with their associated float values.

        Returns
        -------
        model_params : dict
            A dictionary that builds off of the sample dictionary whose keys are the
            okada parameters: 'latitude', 'longitude', 'depth_offset', 'strike','length',
            'width','slip','depth','dip','rake',
            and whose associated values are the newly calculated values from the sample.
        """
        # Calculate forward model parameters from sample parameters
        length = calc_length(sample['magnitude'], sample['delta_logl'])
        width = calc_width(sample['magnitude'], sample['delta_logw'])
        slip = calc_slip(sample['magnitude'], length, width)
        strike = self.fault.strike_map(sample['latitude'],
                                       sample['longitude'])
        dip = self.fault.dip_map(sample['latitude'],
                                 sample['longitude'])
        depth = self.fault.depth_map(sample['latitude'],
                                     sample['longitude'])
        # Set rake angle to 90
        rake = 90

        # Store model parameters in dictionary and return
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

    def sample(self, nsamples, mode='random_walk', delta=0.01, time_steps=25, epsilon=.00001, output_dir=None, save_freq=1, verbose=False):
        """Draw samples from the posterior distribution using the Metropolis-Hastings
        algorithm.

        Parameters
        ----------
        nsamples : int
            Number of samples to draw.
        mode     : str
            The desired mcmc method ('random_walk' or 'mala' at this point)
        delta    : float
            Step size (only required for the mala mcmc method)
        output_dir : string
            The name of the output directory to save the sample data.
        save_freq : int
            The integer that sets how frequently the sample data will be saved and written to a file.
            Default is 10. This also represents the number of rows to appened when
            this function calls the save_data function.
        verbose : bool
            If true, prints gague data for the loglikelihood of the forward model. Default is false.

        Returns
        -------
        samples : pandas DataFrame
            Pandas dataframe containing the set of samples from all of the accepted proposal generated,
            including any from prior runs (such as when using Scenario.restart()).
        """
        # Make sure chain has been initialized
        if not hasattr(self, 'samples'):
            raise AttributeError("Chain must first be initialized with "
                                 "{}.init_chain() or {}.resume_chain()".format(type(self).__name__, type(self).__name__))

        # Create output directory if needed
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            self.save_data(output_dir)

        # Save chain start time
        chain_start = time.time()
        j = 0
        prop_accept = 0

        # Iterate over number of samples to be drawn
        for i in range(len(self.samples), len(self.samples) + nsamples):
            if verbose:
                print(f"\n----------\nSAMPLING ITERATION {i}")
                start = time.time()

            # Propose new sample from previous and save in attribute
            if mode == 'hmc':
                proposal, current_p, proposal_p = self.propose(
                    self.samples.loc[i - 1], mode=mode, time_steps=time_steps, epsilon=epsilon, delta=delta)
                print(proposal)
            else:
                proposal = self.propose(
                    self.samples.loc[i - 1], mode=mode, delta=delta)

            # Get forward model parameters from proposal sample
            model_params = self.map_to_model_params(proposal)
            if verbose:
                print("Proposal:")
                print(proposal)

            # Evaluate prior logpdf
            prior_logpdf = self.prior.logpdf(proposal)

            if verbose:
                print("Prior logpdf = {:.3E}".format(prior_logpdf))

            # If prior logpdf is -infinity, reject proposal and bypass forward model
            if prior_logpdf == np.NINF:
                # Set acceptance probability to 0
                alpha = 0

                # Model_params, model_output and log-likelihood are set to nan values
                model_params = self.model_params.iloc[0].copy()
                model_params[...] = np.nan
                model_output = self.model_output.iloc[0].copy()
                model_output[...] = np.nan
                llh = np.nan
                accepted=False

            # Otherwise run the forward model, calculate the log-likelihood, and calculate
            # the Metropolis-Hastings acceptance probability
            else:
                if verbose:
#                     print('returning dummy for forward model')
                    print("Running forward model...", flush=True)
                model_output, arrival_times = self.forward_model.run(model_params)
                self.arrival_times = arrival_times
                if verbose:
#                       print('returning dummy for llh')
                    print("Evaluating log-likelihood:")
#                 print(model_output)
                llh = self.forward_model.llh(model_output, verbose)
                if verbose:
                    print("Total llh = {:.3E}".format(llh))

                # Acceptance probability calculation
                if mode == 'random_walk':
                    # Catch if both log-likelihoods are -inf
                    if self.bayes_data.loc[i - 1, 'llh'] == np.NINF and llh == np.NINF:
                        # Calculates acceptance probability when both log-likelihoods are -inf
                        alpha = prior_logpdf + self.proposal_logpdf(self.samples.loc[i - 1], proposal) - \
                            self.bayes_data.loc[i - 1, 'prior_logpdf'] - \
                            self.proposal_logpdf(proposal, self.samples.loc[i - 1])
                    else:
                        # Calculates acceptance probability when not both log-likelihoods are -inf
                        alpha = prior_logpdf + llh + \
                            self.proposal_logpdf(self.samples.loc[i - 1], proposal) - \
                            self.bayes_data.loc[i - 1, 'prior_logpdf'] - \
                            self.bayes_data.loc[i - 1, 'llh'] - \
                            self.proposal_logpdf(proposal, self.samples.loc[i - 1])
                    alpha = np.exp(alpha)
                    accepted = (np.random.rand() < alpha)

                elif mode == 'mala':
                    # TODO: should it be +logprior +llh or -logprior -llh???
                    U_0 = - \
                        self.bayes_data.loc[i - 1]['posterior_logpdf'] - \
                        self.bayes_data.loc[i - 1]['llh']
                    U_1 = -prior_logpdf - llh
                    x1, x0 = proposal, self.samples.loc[i - 1]
                    alpha = -U_1 - 1 / (2 * delta**2) * np.linalg.norm(x0 - x1 + delta**2 / 2 * dU(x1))**2 +\
                        U_0 + 1 / (2 * delta**2) * np.linalg.norm(x1 -
                                                                  x0 + delta**2 / 2 * dU(x0))**2
                    alpha = min(1, alpha)
                    accepted = np.log(np.random.uniform()) <= alpha

                elif mode == 'hmc':
                    current_U = - \
                        self.bayes_data.loc[i - 1]['prior_logpdf'] - \
                        self.bayes_data.loc[i - 1]['llh']
                    proposed_U = -prior_logpdf - llh
                    current_K = np.sum(current_p**2) / 2
                    proposed_K = np.sum(proposal_p**2) / 2
                    print(f'current_U: {current_U}')
                    print(f'proposed_U: {proposed_U}')
                    print(f'current_K: {current_K}')
                    print(f'proposed_K: {proposed_K}')
                    
                    alpha = np.exp(current_U-proposed_U+current_K-proposed_K)
                    unif_samp = np.random.uniform()
                    print(f'alpha: {alpha}')
                    print(f'unif_samp: {unif_samp}')

                    accepted = unif_samp < alpha
                    print(accepted)
                    print('------------------------------------')

                else:
                    raise ValueError(
                        'Invalid Mode, try random_walk or mala instead')

            if verbose:
                print("alpha = {:.3E}".format(alpha))

            # Prior, likelihood, and posterior logpdf values
            bayes_data = pd.Series(
                [prior_logpdf, llh, prior_logpdf + llh], index=self.bayes_data_cols)

            # Accept/reject
            if accepted:
                if verbose:
                    print("Proposal accepted", flush=True)
                self.samples.loc[i] = proposal
                self.model_params.loc[i] = model_params
                self.model_output.loc[i] = model_output
                self.bayes_data.loc[i] = bayes_data
                prop_accept += 1
            else:
                if verbose:
                    print("Proposal rejected", flush=True)
                self.samples.loc[i] = self.samples.loc[i - 1]
                self.model_params.loc[i] = self.model_params.loc[i - 1]
                self.model_output.loc[i] = self.model_output.loc[i - 1]
                self.bayes_data.loc[i] = self.bayes_data.loc[i - 1]

            # Generate data for debug dataframe
            metro_hastings_data = pd.Series({'alpha': alpha, 'accepted': int(accepted),
                                             'acceptance_rate': np.nan})
            self.debug.loc[i - 1] = self.gen_debug_row(self.samples.loc[i - 1],
                                                       proposal,
                                                       self.model_params.loc[i - 1],
                                                       model_params,
                                                       self.bayes_data.loc[i - 1],
                                                       bayes_data,
                                                       metro_hastings_data)
            self.debug.loc[i - 1,
                           'acceptance_rate'] = self.debug["accepted"].mean()

            if not j % save_freq and (output_dir is not None):
                if verbose:
                    print("Saving data...")
                self.save_data(output_dir, append_rows=save_freq)

            if verbose:
                print("Iteration elapsed time: {}".format(
                    timedelta(seconds=time.time() - start)))
            j += 1

        if output_dir is not None:
            self.save_data(output_dir)
        if verbose and (output_dir is not None):
            print("Saving data...")

        total_time = time.time() - chain_start
        if verbose:
            print("Chain complete. total time: {}, time per sample: {}\
                              ".format(timedelta(seconds=total_time),
                                       timedelta(seconds=total_time / nsamples)))
        print("Acceptance Proposal", prop_accept / nsamples)  
        return self.samples
