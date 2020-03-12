import numpy as np
import pandas as pd

class BaseScenario:
    """Base class for running a tsunamibayes scenario. Contains the essential
    core routines for running the Metropolis-Hastings algorithm, as well as
    saving the various kinds of output data associated with the sampling proceedure.
    This class should never be instantiated, but should be inherited from."""
    def __init__(self,name,prior,forward_model,gauges):
        self.name = name
        self.prior = prior
        self.forward_model = forward_model
        self.gauges = gauges
        self.observation_cols = [gauge.name + " " + obstype for gauge in gauges for obstype in gauge.obstypes]

        # Must be defined in derived classes. Placed here for reference.
        self.sample_cols = None
        self.model_param_cols = None

        # generate column headings for debug dataframe
        proposal_cols = ['P-'+label for label in self.sample_cols]
        proposal_model_cols = ['P-'+label for label in self.model_param_cols]
        self.debug_cols = self.sample_cols + self.model_param_cols + proposal_cols + proposal_model_cols + \
                          ["sample prior logpdf", "sample llh", "sample posterior logpdf"] + \
                          ["proposal prior logpdf", "proposal llh", "proposal posterior logpdf"] + \
                          ["accept/reject", "acceptance rate"]

    def debug_row(self,**kwargs):
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
        raise NotImplementedError

    def initialize_chain(self,u0=None,method=None,**kwargs):
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

        self.samples = pd.DataFrame(columns=self.sample_cols)
        self.samples.loc[0] = u0

    def restart(self,restart_path):
        self.samples = pd.read_csv(restart_path+"samples.csv",index_col=0)
        self.okada_params = pd.read_csv(restart_path+"okada_params.csv",index_col=0)
        self.debug = pd.read_csv(restart_path+"debug.csv",index_col=0)
        self.observations = pd.read_csv(restart_path+"observations.csv",index_col=0)

    def save_data(self,save_path):
        self.samples.to_csv(save_path+"samples.csv")
        self.okada_params.to_csv(save_path+"okada_params.csv")
        self.debug.to_csv(save_path+"debug.csv")
        self.observations.to_csv(save_path+"observations.csv")

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
            raise AttributeError("Chain must first be initialized with {}.initialize_chain() or {}.restart()".format(type(self).__name__,type(self).__name__))
        for _ in range(nsamples):
            # propose new sample from previous
            proposal = self.propose(self.samples.iloc[-1])

            # evaluate prior logpdf
            proposal_prior_lpdf = self.prior.logpdf(proposal)

            # if prior logpdf is -infinity, reject proposal and bypass forward model
            if prior_lpdf == np.NINF:
                # set accept/reject probablity to 0
                alpha = 0

                # generate nan Series for dataframes
            else:
                proposal_obs = self.forward_model.run(proposal)
                proposal_llh = self.forward_model.llh()

                # accept/reject probability
                alpha = proposal_prior_lpdf + proposal_llh + \
                        self.proposal_logpdf(self.samples.iloc[-1],proposal) - \
                        self.debug.iloc[-1]['sample prior logpdf'] - \
                        self.debug.iloc[-1]['sample llh'] - \
                        self.proposal_logpdf(proposal,self.samples.iloc[-1])
                alpha = np.exp(alpha)

            # accept/reject
            accepted = (np.random.rand() < alpha)
            if accepted:
                self.samples.loc[len(self.samples)] = proposal
                # update other dataframes
            else:
                self.samples.loc[len(self.samples)] = self.samples.iloc[-1]
                # update other dataframes

            return self.samples

    def propose(self,sample):
        """Propose a new sample, perhaps dependent on the current sample. Must
        be implemented in inherited classes.
        """
        raise NotImplementedError("{}.propose() must be implemented in classes inheriting from BaseScenario".format(type(self).__name__))

    def proposal_logpdf(self,u,v):
        """Evaluate the logpdf of the proposal kernel, expressed as the log-probability-density
        of moving from `v` to `u`. Must be implemented in inherited classes.
        """
        raise NotImplementedError("{}.proposal_logpdf() must be implemented in classes inheriting from BaseScenario".format(type(self).__name__))

    def map_to_model_params(self,sample):
        """Evaluate the map from sample parameters to forward model parameters.
        Must be implemented in inherited classes.
        """
        raise NotImplementedError("{}.map_to_model_params() must be implemented in classes inheriting from BaseScenario".format(type(self).__name__))
