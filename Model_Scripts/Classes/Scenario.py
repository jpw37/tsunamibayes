"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import os
import numpy as np
import pandas as pd
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

    def __init__(self, title="Default_Title", use_custom=True, init='manual', rw_covariance=1.0, method="random_walk", iterations=1):
        """
        Initialize all the correct variables for Running this Scenario
        :param title: Title for Scinerio (ex: 1852)
        :param use_custom: Bool: To use the custom methods for MCMC or not
        :param init: String: (manual, random or restart) How to initialize the parameters
        :param rw_covariance: float: covariance for the random walk method
        :param method: String: MCMC Method to use
        :param iterations: Int: Number of Times to run the model
        """

        # Clean geoclaw files
        os.system('make clean')
        os.system('make clobber')

        # Clear previous files
        os.system('rm ./InputData/dtopo.tt3')
        gauges_file_path = './PreRun/InputData/gauges.npy'

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
        self.samples = Samples(title, self.init_guesses, self.mcmc.sample_cols, self.mcmc.proposal_cols, self.mcmc.observation_cols)

        self.mcmc.set_samples(self.samples)

        # Load the samples
        self.init_guesses = self.samples.get_sample()

        # Make sure Pre-Run files have been generated
        if(os.path.isfile(gauges_file_path)):
            gauges = np.load(gauges_file_path)
            self.gauges = [from_json(gauge) for gauge in gauges]
            # Do initial run of GeoClaw using the initial guesses.
            self.setGeoClaw()
        else:
            raise ValueError("The Gauge and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")

        # If using the custom methods map the initial guesses to okada parameters to save as initial sample
        if (self.use_custom):
            self.init_guesses = self.mcmc.map_to_okada(self.init_guesses)
        # Save
        self.samples.save_sample_okada(self.init_guesses)

        # Build the prior for the model, based on the choice of MCMC
        self.prior = self.mcmc.build_priors()

    def setGeoClaw(self):
        """
        Runs an initial set up of GeoClaw
        :return:
        """
        # # Set things up
        # get_topo()
        # make_dtopo(self.mcmc.map_to_okada(self.init_guesses))

        # # Run Geoclaw
        # os.system('rm .output')
        # # If you change the output directory for geoclaw in the mainfile, change this _output="" location to same one
        # os.system('make .output')

        # Get Okada parameters for initial guesses
        okada_params = self.mcmc.map_to_okada(self.init_guesses)
        print("init_guesses:")
        print(self.init_guesses)
        print("okada_params:")
        print(okada_params)
        # Run Geoclaw
        self.feedForward.run_geo_claw(okada_params)

        # Calculate the inital log likelihood
        sample_llh, sample_arr, sample_heights = self.feedForward.calculate_llh(self.gauges)
        # Save
        self.samples.save_sample_llh(sample_llh)

        # Now Save the observations based off the sample and the arrival times & wave heights
        obvs = self.mcmc.make_observations(self.init_guesses, sample_arr, sample_heights)
        self.samples.save_obvs(obvs)

    def clean_up(self):
        """
        Cleans up the unnecessary clutter geoclaw outputs
        :return: None
        """
        os.system('rm ModelOutput/geoclaw/*.data')

    def run(self):
        """
        Runs the Scenario For the given amount of iterations
        """
        for i in range(self.iterations):

            # Remove dtopo file for each run to generate a new one
            os.system('rm ./InputData/dtopo.tt3')

            # Get current Sample and draw a proposal sample from it
            sample_params = self.samples.get_sample()
            proposal_params = self.mcmc.draw(sample_params)

            # Save the proposal draw for debugging purposes
            self.samples.save_proposal(proposal_params)

            # If instructed to use the custom parameters, map parameters to Okada space (9 Dimensional)
            if(self.use_custom):
                proposal_params_okada = self.mcmc.map_to_okada(proposal_params)
            else:
                proposal_params_okada = proposal_params

            # Save Proposal
            self.samples.save_proposal_okada(proposal_params_okada)
            #proposal_params = self.samples.get_proposal_okada()

            print("proposal_params:")
            print(proposal_params)
            print("proposal_params_okada:")
            print(proposal_params_okada)

            # Run Geo Claw on the new proposal
            self.feedForward.run_geo_claw(proposal_params_okada)

            # Calculate the Log Likelihood for the new draw
            proposal_llh, proposal_arr, proposal_heights = self.feedForward.calculate_llh(self.gauges)
            sample_llh = self.samples.get_sample_llh()
            # Save
            print("_____proposal_llh_____", proposal_llh)
            self.samples.save_sample_llh(sample_llh)
            self.samples.save_proposal_llh(proposal_llh)
            proposal_obvs = self.mcmc.make_observations(proposal_params, proposal_arr, proposal_heights)
            self.samples.save_obvs(proposal_obvs)

            # Calculate prior probability for the current sample and proposed sample
            #sample_prior_lpdf = 0.0
            #proposal_prior_lpdf = 0.0
            sample_prior_lpdf = self.prior.logpdf(sample_params)
            proposal_prior_lpdf = self.prior.logpdf(proposal_params)
            #print("proposal_prior_lpdf is:")
            #print(proposal_prior_lpdf)
            #print("sample_prior_lpdf is:")
            #print(sample_prior_lpdf)

            # Save
            self.samples.save_sample_prior_lpdf(sample_prior_lpdf)
            self.samples.save_proposal_prior_lpdf(proposal_prior_lpdf)

            #print("proposal_prior_lpdf is:")
            #print(proposal_prior_lpdf)
            #print("sample_prior_lpdf is:")
            #print(sample_prior_lpdf)

            # Calculate the sample and proposal posterior log likelihood
            sample_post_lpdf = sample_prior_lpdf + sample_llh
            proposal_post_lpdf = proposal_prior_lpdf + proposal_llh
            # Save
            self.samples.save_sample_posterior_lpdf(sample_post_lpdf)
            self.samples.save_proposal_posterior_lpdf(proposal_post_lpdf)

            # Calculate the acceptance probability of the given proposal
            accept_prob = self.mcmc.acceptance_prob(sample_prior_lpdf, proposal_prior_lpdf)

            # Decide to accept or reject the proposal and save
            ar = self.mcmc.accept_reject(accept_prob)

            # Saves the stored data for debugging purposes
            self.samples.save_debug()

            # Save to csv
            if i % 50 == 0:
                self.samples.save_to_csv()

            if ar:
                self.samples.save_sample(self.samples.get_proposal())
                self.samples.save_sample_okada(self.samples.get_proposal_okada())
                self.samples.save_sample_llh(self.samples.get_proposal_llh())
            else:
                self.samples.increment_wins()
                self.samples.save_sample(self.samples.get_sample())
                self.samples.save_sample_okada(self.samples.get_sample_okada())

        self.samples.save_to_csv()
        return
