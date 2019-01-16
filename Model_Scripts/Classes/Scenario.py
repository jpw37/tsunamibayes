"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import os
import numpy as np
import sys

sys.path.append('./PreRun/Classes/')

from maketopo import get_topo, make_dtopo
from RandomWalk import RandomWalk
from IndependentSampler import IndependentSampler
from Samples import Samples
from FeedForward import FeedForward
from Custom import Custom
from Gauge import from_json

class Scenario:
    """
    Main Class for Running the MCMC Method for ...

    READ: Make sure you run the python notebook in the PreRun folder to generate necessary run files
    """

    def __init__(self, title="Default_Title", use_custom=False, init='manual', rw_covariance=1.0, method="random_walk", iterations=10):
        """
        Initialize all the correct variables for Running this Scenario
        :param title: Title for Scinerio (ex: 1852)
        :param use_custom: Bool: To use the custom methods for MCMC or not
        :param init: String: (manual, random or restart) How to initialize the parameters
        :param rw_covariance: float: covariance for the random walk method
        :param method: String: MCMC Method to use
        :param iterations: Int: Number of Times to run the model
        """
        # Clear previous files
        # os.system('rm ./Data/Topo/dtopo.tt3')
        # os.system('rm ./dtopo.data')
        gauges_file_path = './PreRun/Data/gauges.npy'

        self.title = title
        self.iterations = iterations
        self.use_custom = use_custom
        self.init = init
        self.feedForward = FeedForward()

        # Set the MCMC class based on input
        if(use_custom):
            self.mcmc = Custom()
        elif(method == "independent_sampler"):
            self.mcmc = IndependentSampler()
        else:
            self.mcmc = RandomWalk(rw_covariance)

        # Get initial draw for the initial run of geoclaw
        self.init_guesses = self.mcmc.init_guesses(self.init)
        # Initialize the Samples Class
        self.samples = Samples(title, self.init_guesses, self.mcmc.sample_cols, self.mcmc.proposal_cols)

        self.mcmc.set_samples(self.samples)

        # Load the samples
        self.init_guesses = self.samples.get_sample()

        # If using the custom methods map the initial guesses to okada parameters to save as initial sample
        if (self.use_custom):
            self.init_guesses = self.mcmc.map_to_okada(self.init_guesses)
        # Save
        self.samples.save_sample_okada(self.init_guesses)

        # Build the prior for the model, based on the choice of MCMC
        self.prior = self.mcmc.build_priors()

        # Make sure Pre-Run files have been generated
        if(os.path.isfile(gauges_file_path)):
            gauges = np.load(gauges_file_path)
            self.gauges = [from_json(gauge) for gauge in gauges]
            # Do initial run of GeoClaw using the initial guesses.
            self.setGeoClaw()
        else:
            raise ValueError("The Gauge and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")

    def setGeoClaw(self):
        """
        Runs an initial set up of GeoClaw
        :return:
        """
        # Set things up
        get_topo()
        make_dtopo(self.init_guesses)

        # Run Geoclaw
        os.system('make clean')
        os.system("make topo")
        os.system('make clobber')
        os.system('make .output')

        # Calculate the inital loglikelihood
        sample_llh = self.feedForward.calculate_llh(self.gauges)
        # Save
        self.samples.save_sample_llh(sample_llh)

    def clean_up(self):
        """
        Cleans up the unnecessary clutter geoclaw outputs
        :return: None
        """
        os.system('rm ./.data')
        os.system('rm ./amr.data')
        os.system('rm ./claw.data')
        os.system('rm ./dtopo.data')
        os.system('rm ./dtopo.tt3')
        os.system('rm ./fgmax.data')
        os.system('rm ./fixed_grids.data')
        os.system('rm ./friction.data')
        os.system('rm ./gauges.data')
        os.system('rm ./geoclaw.data')
        os.system('rm ./multilayer.data')
        os.system('rm ./qinit.data')
        os.system('rm ./refinement.data')
        os.system('rm ./regions.data')
        os.system('rm ./surge.data')
        os.system('rm ./topo.data')

    def run(self):
        """
        Runs the Scenario For the given amount of iterations
        """
        for i in range(self.iterations):

            # Remove dtopo file for each run to generate a new one
            os.system('rm ./Data/dtopo.tt3')

            # Get current Sample and draw a proposal sample from it
            sample_params = self.samples.get_sample()
            proposal_params = self.mcmc.draw(sample_params)

            # Save the proposal draw for debugging purposes
            self.samples.save_proposal(proposal_params)

            # If instructed to use the custom parameters, map parameters to Okada space (9 Dimensional)
            if(self.use_custom):
                proposal_params = self.mcmc.map_to_okada(proposal_params)

            # Save Proposal
            self.samples.save_proposal_okada(proposal_params)
            proposal_params = self.samples.get_proposal_okada()

            # Run Geo Claw on the new proposal
            self.feedForward.run_geo_claw(proposal_params)

            # Calculate the Log Likelihood for the new draw
            proposal_llh = self.feedForward.calculate_llh(self.gauges)
            sample_llh = self.samples.get_sample_llh()
            # Save
            print("_____proposal_llh_____", proposal_llh)
            self.samples.save_sample_llh(sample_llh)
            self.samples.save_proposal_llh(proposal_llh)

            # Calculate prior probability for the current sample and proposed sample
            sample_prior_llh = self.prior.logpdf(sample_params)
            proposal_prior_llh = self.prior.logpdf(proposal_params)
            # Save
            self.samples.save_sample_prior_llh(sample_prior_llh)
            self.samples.save_proposal_prior_llh(proposal_prior_llh)

            # Calculate the acceptance probability of the given proposal
            accept_prob = self.mcmc.acceptance_prob(sample_prior_llh, proposal_prior_llh)

            # Decide to accept or reject the proposal and save
            self.mcmc.accept_reject(accept_prob)

            # Calculate the sample and proposal posterior log likelihood
            sample_post_llh = sample_prior_llh + sample_llh
            proposal_post_llh = proposal_prior_llh + proposal_llh
            # Save
            self.samples.save_sample_posterior_llh(sample_post_llh)
            self.samples.save_proposal_posterior_llh(proposal_post_llh)

            # Saves the stored data for debugging purposes
            self.samples.save_debug()

            # Save to csv
            if i % 50 == 0:
                self.samples.save_to_csv()

        self.clean_up()
        self.samples.save_to_csv()
        return

