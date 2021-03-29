import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
from tsunamibayes.utils import dU, naive_gradient_setup

class BaseScenario:
    """Base class for running a tsunamibayes scenario. Contains the essential
    core routines for running the Metropolis-Hastings algorithm, as well as
    saving the various kinds of output data associated with the sampling proceedure.
    This class should never be instantiated, but should be inherited from."""

    bayes_data_cols = ["prior_logpdf","llh","posterior_logpdf"]

    # Must be defined in inherited classes. Placed here for reference.
    sample_cols = None
    model_param_cols = None

    def __init__(self,prior,forward_model):
        """Initializes variables for the base class.
        
        Parameters
        ----------
        prior : Prior object
            The prior distribution for the scenario. 
        forward_model : Forward Model object
            This is usually the GeoClawForward model.
        """
        if self.sample_cols is None or self.model_param_cols is None:
            raise NotImplementedError("sample_cols and model_param_cols must be "
                                      "defined in inherited classes")

        self.prior = prior
        self.forward_model = forward_model
        self.model_output_cols = forward_model.model_output_cols

        # generate column labels for debug dataframe
        proposal_cols = list(map(lambda x:'p_'+x,self.sample_cols))
        proposal_bayes_cols = list(map(lambda x:'p_'+x,self.bayes_data_cols))

        debug_mp_cols = list(map(lambda x:'m_'+x if x in self.sample_cols
                                 else x,self.model_param_cols))
        debug_pmp_cols = list(map(lambda x:'p_m_'+x if x in self.sample_cols
                                  else 'p_'+x,self.model_param_cols))
        self.debug_cols = self.sample_cols + proposal_cols + debug_mp_cols \
                          + debug_pmp_cols + self.bayes_data_cols + \
                          proposal_bayes_cols + ["alpha","accepted","acceptance_rate"]

    def init_chain(self,u0=None,method=None,verbose=False,**kwargs):
        """Initialize a sampling chain with a given initial sample or a string
        indicating a random initialization method. Creates DataFrames for the
        samples, Okada parameters, simulated observations, and debug information.
        Runs the forward model and evaluates the prior and likelihood for the initial
        sample.

        Parameters
        ----------
        u0 : dict, list or ndarray, optional
            Initial sample for the chain. If a dict, the keys must agree with
            self.sample_cols. Otherwise, the length must match the length of
            self.sample_cols.
        method : str, optional
            String indicating method for choosing a random initial sample. Only
            'prior_rvs' available by default. Ignored if `u0` is given.
        **kwargs
            Keyword arguments specifying initial sample parameter values. All
            keyword arguments must be in self.sample_cols. If `method` is given,
            unspecified keyword arguments will be filled in using the
            specified method. If `method` is None, everything in self.sample_cols
            must be specified with a keyword argument.
        """
        if isinstance(u0,dict):
            if set(u0.keys()) != set(self.sample_cols):
                raise ValueError("u0 must have keys: {}".format(self.sample_cols))

        elif isinstance(u0,(list,np.ndarray)):
            if len(u0) != len(self.sample_cols):
                raise ValueError("u0 must have length {}".format(len(self.sample_cols)))
            u0 = dict(zip(self.sample_cols,u0))

        elif isinstance(u0,pd.Series):
            if set(u0.index) != set(self.sample_cols):
                raise ValueError("u0 must have index: {}".format(self.sample_cols))

        elif method == 'prior_rvs':
            u0 = self.prior.rvs()
            u0 = dict(zip(self.sample_cols,u0))

        else:
            if not all(key in self.sample_cols for key in kwargs.keys()):
                raise ValueError("Valid keyword argments are: {}".format(self.sample_cols))

            missing_kwargs = list(set(self.sample_cols)-set(kwargs.keys()))

            if len(missing_kwargs) == 0:
                u0 = kwargs

            else:
                if method is None:
                    raise ValueError("Missing keyword arguments: {}".format(missing_kwargs))
                for key in kwargs.keys():
                    u0[key] = kwargs[key]

        # build dataframes
        self.samples = pd.DataFrame(columns=self.sample_cols)
        self.model_params = pd.DataFrame(columns=self.model_param_cols)
        self.model_output = pd.DataFrame(columns=self.model_output_cols)
        self.bayes_data = pd.DataFrame(columns=self.bayes_data_cols)
        self.debug = pd.DataFrame(columns=self.debug_cols)

        # save first sample
        self.samples.loc[0] = u0

        if verbose:
            print("\n----------\nInitializing chain with initial sample:")
            print(self.samples.loc[0])

        # evaluate prior logpdf
        prior_logpdf = self.prior.logpdf(u0)
        if verbose: print("Prior logpdf = {:.3E}".format(prior_logpdf))

        # raise error if prior density is zero (-infinty logpdf)
        if prior_logpdf == np.NINF:
            raise ValueError("Initial sample must result in a nonzero prior probabiity density")

        # evaluate forward model and compute log-likelihood
        if verbose: print("Running forward model...",flush=True)
        model_params = self.map_to_model_params(u0)
        self.model_params.loc[0] = model_params

        model_output = self.forward_model.run(model_params,verbose)
        self.model_output.loc[0] = model_output

        if verbose: print("Evaluating log-likelihood:")
        llh = self.forward_model.llh(model_output,verbose)
        if verbose: print("Total llh = {:.3E}".format(llh))

        # save prior logpdf, log-likelihood, and posterior logpdf
        bayes_data = pd.Series([prior_logpdf,llh,prior_logpdf+llh],index=self.bayes_data_cols)
        self.bayes_data.loc[0] = bayes_data

    def resume_chain(self,output_dir):
        """Reads DataFrames from the .csv files housing the samples, model info,
        bayes data, and debugging information that have already been stored before
        the program was paused.
        
        Parameters
        ----------
        output_dir : string
            The name of the output directory from which the function pulls
            data to resume.
        """
        self.samples = pd.read_csv(output_dir+"/samples.csv",index_col=0).reset_index(drop=True)
        self.model_params = pd.read_csv(output_dir+"/model_params.csv",index_col=0).reset_index(drop=True)
        self.model_output = pd.read_csv(output_dir+"/model_output.csv",index_col=0).reset_index(drop=True)
        self.bayes_data = pd.read_csv(output_dir+"/bayes_data.csv",index_col=0).reset_index(drop=True)
        self.debug = pd.read_csv(output_dir+"/debug.csv",index_col=0).reset_index(drop=True)

    def save_data(self,output_dir,append_rows=None):
        """Writes the DataFrames for the samples, model parameters & ouput,
        bayes data, and debugging info into .csv files and stores them in the 
        specified directory.

        Parameters
        ----------
        output_dir : string
            The name of the output directory to which the files will be stored.
        append_rows : int -or- None
            Default is None. If an integer is passed in, then this is 
            the number of rows to append to the end of the .csv files to be written.
        """
        if not append_rows:
            self.samples.to_csv(output_dir+"/samples.csv")
            self.model_params.to_csv(output_dir+"/model_params.csv")
            self.model_output.to_csv(output_dir+"/model_output.csv")
            self.bayes_data.to_csv(output_dir+"/bayes_data.csv")
            self.debug.to_csv(output_dir+"/debug.csv")
        else:
            n = -append_rows
            self.samples.iloc[n:].to_csv(output_dir+"/samples.csv",mode='a+',header=False)
            self.model_params.iloc[n:].to_csv(output_dir+"/model_params.csv",mode='a+',header=False)
            self.model_output.iloc[n:].to_csv(output_dir+"/model_output.csv",mode='a+',header=False)
            self.bayes_data.iloc[n:].to_csv(output_dir+"/bayes_data.csv",mode='a+',header=False)
            self.debug.iloc[n:].to_csv(output_dir+"/debug.csv",mode='a+',header=False)        

    def sample(self,nsamples,mode='random_walk',delta=0.01,output_dir=None,save_freq=1,verbose=False):
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
        if not hasattr(self,'samples'):
            raise AttributeError("Chain must first be initialized with "
            "{}.init_chain() or {}.resume_chain()".format(type(self).__name__,type(self).__name__))
        if output_dir is not None:
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            self.save_data(output_dir)

        chain_start = time.time()
        j = 0
        for i in range(len(self.samples),len(self.samples)+nsamples):
            if verbose: print("\n----------\nIteration {}".format(i)); start = time.time()

            # propose new sample from previous
            proposal = self.propose(self.samples.loc[i-1],mode=mode,delta=delta)
            model_params = self.map_to_model_params(proposal)
            if verbose: print("Proposal:"); print(proposal)

            # evaluate prior logpdf
            prior_logpdf = self.prior.logpdf(proposal)
            if verbose: print("Prior logpdf = {:.3E}".format(prior_logpdf))

            # if prior logpdf is -infinity, reject proposal and bypass forward model
            if prior_logpdf == np.NINF:
                # set acceptance probablity to 0
                alpha = 0

                # model_params, model_output and log-likelihood are set to nan values
                model_params = self.model_params.iloc[0].copy()
                model_params[...] = np.nan
                model_output = self.model_output.iloc[0].copy()
                model_output[...] = np.nan
                llh = np.nan

            # otherwise run the forward model, calculate the log-likelihood, and calculate
            # the Metropolis-Hastings acceptance probability
            else:
                if verbose: print("Running forward model...",flush=True)
                model_output = self.forward_model.run(model_params)
                if verbose: print("Evaluating log-likelihood:")
                llh = self.forward_model.llh(model_output,verbose)
                if verbose: print("Total llh = {:.3E}".format(llh))

                # acceptance probability
                if mode == 'random_walk': 
                    # catch if both loglikelihoods are -inf
                    if self.bayes_data.loc[i-1,'llh'] == np.NINF and llh == np.NINF:
                        alpha = prior_logpdf + self.proposal_logpdf(self.samples.loc[i-1],proposal) - \
                                self.bayes_data.loc[i-1,'prior_logpdf'] - \
                                self.proposal_logpdf(proposal,self.samples.loc[i-1])
                    else:
                        alpha = prior_logpdf + llh + \
                                self.proposal_logpdf(self.samples.loc[i-1],proposal) - \
                                self.bayes_data.loc[i-1,'prior_logpdf'] - \
                                self.bayes_data.loc[i-1,'llh'] - \
                                self.proposal_logpdf(proposal,self.samples.loc[i-1]) 
                    alpha = np.exp(alpha)
                    accepted = (np.random.rand() < alpha)
                elif mode == 'mala':
                    # TODO: should it be +logprior +llh or -logprior -llh??? 
                    U_0 = -self.bayes_data.loc[i-1]['posterior_logpdf'] -self.bayes_data.loc[i-1]['llh']
                    U_1 = -prior_logpdf - llh  
                    x1, x0 = proposal, self.samples.loc[i-1]
                    alpha = -U_1 -1/(2*delta**2) * np.linalg.norm(x0 - x1 + delta**2/2*dU(x1))**2 +\
                            U_0 + 1/(2*delta**2) * np.linalg.norm(x1 - x0 + delta**2/2*dU(x0))**2
                    alpha = min(1, alpha)
                    accepted = np.log(np.random.uniform()) <= alpha
                else:
                    raise ValueError('Invalid Mode, try random_walk or mala instead')

            if verbose: print("alpha = {:.3E}".format(alpha))

            # prior, likelihood, and posterior logpdf values
            bayes_data = pd.Series([prior_logpdf,llh,prior_logpdf+llh],index=self.bayes_data_cols)

            # accept/reject
            if accepted:
                if verbose: print("Proposal accepted",flush=True)
                self.samples.loc[i] = proposal
                self.model_params.loc[i] = model_params
                self.model_output.loc[i] = model_output
                self.bayes_data.loc[i] = bayes_data
            else:
                if verbose: print("Proposal rejected",flush=True)
                self.samples.loc[i] = self.samples.loc[i-1]
                self.model_params.loc[i] = self.model_params.loc[i-1]
                self.model_output.loc[i] = self.model_output.loc[i-1]
                self.bayes_data.loc[i] = self.bayes_data.loc[i-1]

            # generate data for debug dataframe
            metro_hastings_data = pd.Series({'alpha':alpha,'accepted':int(accepted),
                                             'acceptance_rate':np.nan})
            self.debug.loc[i-1] = self.gen_debug_row(self.samples.loc[i-1],
                                                     proposal,
                                                     self.model_params.loc[i-1],
                                                     model_params,
                                                     self.bayes_data.loc[i-1],
                                                     bayes_data,
                                                     metro_hastings_data)
            self.debug.loc[i-1,'acceptance_rate'] = self.debug["accepted"].mean()

            if not j%save_freq and (output_dir is not None):
                if verbose: print("Saving data...")
                self.save_data(output_dir,append_rows=save_freq)

            if verbose: print("Iteration elapsed time: {}".format(timedelta(seconds=time.time()-start)))
            j += 1

        if output_dir is not None: self.save_data(output_dir)
        if verbose and (output_dir is not None): print("Saving data...")

        total_time = time.time()-chain_start
        if verbose: print("Chain complete. total time: {}, time per sample: {}\
                          ".format(timedelta(seconds=total_time),
                                   timedelta(seconds=total_time/nsamples)))
        return self.samples

    def gen_debug_row(self,sample,proposal,sample_model_params,proposal_model_params,
                      sample_bayes,proposal_bayes,metro_hastings_data):
        """Combines data from a scenario iteration to be used for debugging puposes, and 
        stores the combined data in a Pandas Series object.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float values for the declared sample columns.
        proposal : pandas Series of floats
            The series that contains the float values for the declared proposal columns.
        sample_model_params : dict
            The dictionary containing the sample's specified parameters and their associated float values. 
            This differs from subclass to subclass, but generally contains parameters such as 
            magnitude, length, width, etc. etc.
        proposal_model_params :  dict
            The dictionary containing the proposal's specified parameters. This differs from subclass
            to subclass, but generally contains parameters such as magnitude, length, width, etc. etc.
        sample_bayes : pandas Series
            The pandas series that contains labels and float values for 
            the sample's prior logpdf, loglikelihood, and posterior logpdf.
        proposal_bayes : pandas Series
            The pandas series that contains labels and float values for 
            the prosal's prior logpdf, loglikelihood, and posterior logpdf.
        metro_hastings_data : pandas Series
            The series that contains a dictioanry with the acceptace probablity, acceptace state, and acceptance rate.

        Returns
        -------
        debug_row : pandas Series
            The combined/concatenated series of all the dictionaries and pd.series passed in to the function.
        """
        proposal = pd.Series(proposal).rename(lambda x:'p_'+x)
        sample_model_params = pd.Series(sample_model_params).rename(
                              lambda x:'m_'+x if x in self.sample_cols else x)
        proposal_model_params = pd.Series(proposal_model_params).rename(
                                lambda x:'p_m_'+x if x in self.sample_cols else 'p_'+x)
        proposal_bayes = pd.Series(proposal_bayes).rename(lambda x:'p_'+x)
        return pd.concat((sample,
                          proposal,
                          sample_model_params,
                          proposal_model_params,
                          sample_bayes,
                          proposal_bayes,
                          metro_hastings_data))

    def propose(self,sample,mode='random_walk',delta=0.01):
        """Propose a new sample, perhaps dependent on the current sample.
        Must be implemented in inherited classes.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float values for the declared sample columns.
        """
        raise NotImplementedError("propose() must be implemented in classes "
        "inheriting from BaseScenario")

    def proposal_logpdf(self,u,v):
        """Evaluate the logpdf of the proposal kernel, expressed as the
        log-probability-density of proposing 'u' given current sample 'v'. Must
        be implemented in inherited classes.

        Parameters
        ----------
        u : pandas Series of floats
            The series that contains the float values for a proposal.
        v : pandas Series of floats
            The series that contains the float values for the current sample.
        """
        raise NotImplementedError("proposal_logpdf() must be implemented in "
        "classes inheriting from BaseScenario")

    def map_to_model_params(self,sample):
        """Evaluate the map from sample parameters to forward model parameters.
        Must be implemented in inherited classes.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float values for the declared sample columns.
        """
        raise NotImplementedError("map_to_model_params() must be implemented in "
        "classes inheriting from BaseScenario")

class TestScenario(BaseScenario):
    sample_cols = ["magnitude"]
    model_param_cols = ["length","width",'magnitude']

    def propose(self,sample):
        """Proposes a new sample by adding a small factor multiplied by a
        random floats (sampled from a univariate "normal" (Gaussian) distribution of mean 0 and variance 1)
        to the original sample.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float values for the sample's earthquake magnitude.
        
        Returns
        -------
        proposal :  pandas Series of floats
            The random variant of the passed-in sample.
        """
        return sample + 0.1*np.random.randn()

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
        """Evaluate the map from sample's magnitude to forward model's length and width.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float values for the declared sample columns.
        
        Returns
        -------
        params : dict
            The dictionary with keys 'length', 'width', and 'magnitude' and their associated float values.
        """
        length = 2*sample["magnitude"]**0.5
        width = 0.5*sample["magnitude"]**0.5
        return {'length':length,'width':width,'magnitude':sample['magnitude']}
