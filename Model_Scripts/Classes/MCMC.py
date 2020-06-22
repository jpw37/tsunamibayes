"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""

import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

from Prior import Prior

class MCMC:
    """
    This Parent Class takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface.

    Can be overridden by the Custom Class if custom = 1 in the inputs for custom calculations
    """

    def __init__(self):
        self.samples = None
        self.sample_cols = None
        self.proposal_cols = None

    def set_samples(self, Samples):
        """
        Sets the samples loading class

        parameters
        ----------
        Samples : (Sample) Sample class
        """
        self.samples = Samples

    def change_llh_calc(self):
        """
        Calculates the change in loglikelihood between the current and the proposed llh.

        Returns
        -------
        change_llh : float
            The difference between the proposal and sample loglikelihood
        """
        sample_llh = self.samples.get_sample_llh()
        proposal_llh = self.samples.get_proposal_llh()
        print("sample_llh is:")
        print(sample_llh)
        print("proposal_llh is:")
        print(proposal_llh)

        if np.isneginf(proposal_llh) and np.isneginf(sample_llh):
            change_llh = 0
        elif np.isnan(proposal_llh) and np.isnan(sample_llh):
            change_llh = 0
            # fix situation where nan in proposal llh results in acceptance, e.g., 8855 [-52.34308085] -10110.84699320795 [-10163.19007406] [-51.76404079] nan [nan] 1 accept
        elif np.isnan(proposal_llh) and not np.isnan(sample_llh):
            change_llh = np.NINF
        elif not np.isnan(proposal_llh) and np.isnan(sample_llh):
            change_llh = np.inf
        else:
            change_llh = proposal_llh - sample_llh
        return change_llh

    def accept_reject(self, accept_prob):
        """
        Decides to accept or reject the proposal. Saves the accepted parameters as new current sample.

        Parameters
        ----------
        accept_prob: float 
            Proposal acceptance probability (between 0.0 and 1.0)
        
        Returns
        -------
        ar : bool
            Indicates whether the given proposal was accepted (ar=True) or rejected (ar=False)
        """
        if np.random.random() < accept_prob:
            # Accept and save proposal
            ar = True
            self.samples.accepts += 1
            print("Accepted new proposal")
        else:
            # Reject Proposal and Save current winner to sample list
            ar = False
            self.samples.rejects += 1
            print("Rejected new proposal")

        self.samples.accepted = ar
        return ar

    def map_to_okada(self, draws):
        pass

    def draw(self, prev_draw):
        pass

    def acceptance_prob(self, prop_prior_llh, cur_prior_llh):
        pass
