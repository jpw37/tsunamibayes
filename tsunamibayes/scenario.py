import numpy as np
import pandas as pd

class BaseScenario:
    """Base class for running a tsunamibayes scenario. Contains the essential
    core routines for running the Metropolis-Hastings algorithm, as well as
    saving the various kinds of output data associated with the sampling proceedure.
    This class should never be instantiated, but should be inherited from."""

    prepend_p = lambda string: 'p_'+string
    bayes_data_cols = ["prior_logpdf","llh","posterior_logpdf"]
    # Must be defined in derived classes. Placed here for reference.
    sample_cols = None
    model_param_cols = None

    def __init__(self,name,prior,forward_model,gauges):
        self.name = name
        self.prior = prior
        self.forward_model = forward_model
        self.gauges = gauges
        self.model_output_cols = [gauge.name + " " + obstype for gauge in gauges for obstype in gauge.obstypes]

        # generate column headings for debug dataframe
        if self.sample_cols is None or self.model_param_cols is None:
            raise NotImplementedError("sample_cols and model_param_cols must be defined in inherited classes")
        proposal_cols = list(map(lambda x:'p_'+x,self.sample_cols))
        proposal_model_cols = list(map(lambda x:'p_'+x,self.model_param_cols))
        proposal_bayes_cols = list(map(lambda x:'p_'+x,self.bayes_data_cols))

        self.debug_cols = self.sample_cols + proposal_cols + self.model_param_cols + proposal_model_cols + \
                          self.bayes_data_cols + proposal_bayes_cols + ["alpha","accepted","acceptance_rate"]

    def init_chain(self,u0=None,method=None,**kwargs):
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
                raise TypeError("u0 must have keys: {}".format(self.sample_cols))
        elif isinstance(u0,(list,np.ndarray)):
            if len(u0) != len(self.sample_cols):
                raise ValueError("u0 must have length {}".format(len(self.sample_cols)))
            u0 = dict(zip(self.sample_cols,u0))
        elif method == 'prior_rvs':
            u0 = self.prior.rvs()
            u0 = dict(zip(self.sample_cols,u0))
        else:
            if not all(key in self.sample_cols for key in kwargs.keys()):
                raise TypeError("Valid keyword argments are: {}".format(self.sample_cols))
            missing_kwargs = list(set(self.sample_cols)-set(kwargs.keys()))
            if len(missing_kwargs) == 0:
                u0 = kwargs
            else:
                if method is None:
                    raise TypeError("Missing keyword arguments: {}".format(missing_kwargs))
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

        # evaluate prior logpdf
        prior_logpdf = self.prior.logpdf(u0)

        # raise error if prior density is zero (-infinty logpdf)
        if prior_logpdf == np.NINF:
            raise ValueError("Initial sample must result in a nonzero prior probabiity density")

        # evaluate forward model and compute log-likelihood
        model_params = self.map_to_model_params(u0)
        self.model_params.loc[0] = model_params
        model_output = self.forward_model.run(model_params)
        self.model_output.loc[0] = model_output
        llh = self.forward_model.llh(model_output)

        # save prior logpdf, log-likelihood, and posterior logpdf
        bayes_data = pd.Series([prior_logpdf,llh,prior_logpdf+llh],index=self.bayes_data_cols)
        self.bayes_data.loc[0] = bayes_data

    def restart(self,restart_path):
        self.samples = pd.read_csv(restart_path+"samples.csv",index_col=0)
        self.model_params = pd.read_csv(restart_path+"model_params.csv",index_col=0)
        self.model_output = pd.read_csv(restart_path+"model_output.csv",index_col=0)
        self.debug = pd.read_csv(restart_path+"debug.csv",index_col=0)

    def save_data(self,save_path):
        self.samples.to_csv(save_path+"samples.csv")
        self.model_params.to_csv(save_path+"model_params.csv")
        self.model_output.to_csv(save_path+"model_output.csv")
        self.debug.to_csv(save_path+"debug.csv")

    def sample(self,nsamples):
        """Draw samples from the posterior distribution using the Metropolis-Hastings
        algorithm.

        Parameters
        ----------
        nsamples : int
            number of samples to draw

        Returns
        -------
        samples : DataFrame
            pandas dataframe containing samples, including any from prior runs
            (such as when using Scenario.restart())
        """
        if not hasattr(self,'samples'):
            raise AttributeError("Chain must first be initialized with {}.init_chain() or {}.restart()".format(type(self).__name__,type(self).__name__))
        for i in range(len(self.samples),len(self.samples)+nsamples):
            # propose new sample from previous
            proposal = self.propose(self.samples.loc[i-1])
            model_params = self.map_to_model_params(proposal)

            # evaluate prior logpdf
            prior_logpdf = self.prior.logpdf(proposal)

            # if prior logpdf is -infinity, reject proposal and bypass forward model
            if prior_logpdf == np.NINF:
                # set acceptance probablity to 0
                alpha = 0

                # model_output and log-likelihood are set to nan values
                model_output = self.model_output.loc[i-1].copy()
                model_output[...] = np.nan
                llh = np.nan

            # otherwise run the forward model, calculate the log-likelihood, and calculate
            # the Metropolis-Hastings acceptance probability
            else:
                model_output = self.forward_model.run(model_params)
                llh = self.forward_model.llh(model_output)

                # acceptance probability
                alpha = prior_logpdf + llh + \
                        self.proposal_logpdf(self.samples.loc[i-1],proposal) - \
                        self.bayes_data.loc[i-1]['prior_logpdf'] - \
                        self.bayes_data.loc[i-1]['llh'] - \
                        self.proposal_logpdf(proposal,self.samples.loc[i-1])
                alpha = np.exp(alpha)

            bayes_data = pd.Series([prior_logpdf,llh,prior_logpdf+llh],index=self.bayes_data_cols)

            # accept/reject
            accepted = (np.random.rand() < alpha)
            if accepted:
                self.samples.loc[i] = proposal
                self.model_params.loc[i] = model_params
                self.model_output.loc[i] = model_output
                self.bayes_data.loc[i] = bayes_data
            else:
                self.samples.loc[i] = self.samples.loc[i-1]
                self.model_params.loc[i] = self.model_params.loc[i-1]
                self.model_output.loc[i] = self.model_output.loc[i-1]
                self.bayes_data.loc[i] = self.bayes_data.loc[i-1]

            # generate data for debug dataframe
            metro_hastings_data = pd.Series({'alpha':alpha,'accepted':int(accepted),'acceptance_rate':np.nan})
            self.debug.loc[i-1] = self.gen_debug_row(self.samples.loc[i-1],
                                                                 proposal,
                                                                 self.model_params.loc[i-1],
                                                                 model_params,
                                                                 self.bayes_data.loc[i-1],
                                                                 bayes_data,
                                                                 metro_hastings_data)
            self.debug.loc[i-1,'acceptance_rate'] = self.debug["accepted"].mean()

        return self.samples

    def gen_debug_row(self,sample,proposal,sample_model_params,proposal_model_params,sample_bayes,proposal_bayes,metro_hastings_data):
        """Create a Pandas Series object with the given data that is desired to
        be kept in the debug output.

        Parameters
        ----------
        # TODO: EITHER USE **kwargs OR A BUNCH OF SPECIFIC PARAMETERS?

        Returns
        -------
        Pandas.Series
            Row for the debug Dataframe
        """
        proposal = pd.Series(proposal).rename(lambda x:'p_'+x)
        proposal_model_params = pd.Series(proposal_model_params).rename(lambda x:'p_'+x)
        proposal_bayes = pd.Series(proposal_bayes).rename(lambda x:'p_'+x)
        return pd.concat((sample,
                          proposal,
                          sample_model_params,
                          proposal_model_params,
                          sample_bayes,
                          proposal_bayes,
                          metro_hastings_data))

    def propose(self,sample):
        """Propose a new sample, perhaps dependent on the current sample. Must
        be implemented in inherited classes.
        """
        raise NotImplementedError("propose() must be implemented in classes inheriting from BaseScenario")

    def proposal_logpdf(self,u,v):
        """Evaluate the logpdf of the proposal kernel, expressed as the log-probability-density
        of proposing 'u' given current sample 'v'. Must be implemented in inherited classes.
        """
        raise NotImplementedError("proposal_logpdf() must be implemented in classes inheriting from BaseScenario")

    def map_to_model_params(self,sample):
        """Evaluate the map from sample parameters to forward model parameters.
        Must be implemented in inherited classes.
        """
        raise NotImplementedError("map_to_model_params() must be implemented in classes inheriting from BaseScenario")

class TestScenario(BaseScenario):
    sample_cols = ["magnitude"]
    model_param_cols = ["length","width"]

    def propose(self,sample):
        return sample + 0.1*np.random.randn()

    def proposal_logpdf(self,u,v):
        return 0

    def map_to_model_params(self,sample):
        length = 2*sample["magnitude"]**0.5
        width = 0.5*sample["magnitude"]**0.5
        return {'length':length,'width':width}
