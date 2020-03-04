import numpy as np
import pandas as pd

class BaseScenario:
    def __init__(self,name,gauges):
        self.name = name
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
        if not hasattr(self,'samples'):
            raise AttributeError("Chain must first be initialized with {}.initialize_chain() or {}.restart()".format(type(self).__name__,type(self).__name__))
        for _ in range(nsamples):
            # propose new sample from previous
            proposal = self.propose(samples.iloc[-1])

            # evaluate prior logpdf
            proposal_prior_lpdf = self.prior.logpdf(proposal)

            # if prior logpdf is -infinity, reject proposal and bypass forward model
            if prior_lpdf = np.NINF:
                alpha = 0

                # save nan values to dataframes
            else:
                proposal_obs = self.forward_model.run(proposal)
                proposal_llh = self.forward_model.llh()




    def propose(self,sample):
        pass

    def map_to_model_params(self,sample):
        pass
