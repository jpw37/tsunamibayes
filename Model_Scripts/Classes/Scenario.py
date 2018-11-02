"""
Created 10/19/2018
BYU Mathematics Dept.
"""
import os
import numpy as np
import sys

sys.path.append('./PreRun/Classes/')

from MakeTopo import MakeTopo
from SetGeoClaw import SetGeoClaw
from RandomWalk import RandomWalk
from IndependentSampler import IndependentSampler
from Samples import Samples
from FeedForward import FeedForward
from Custom import Custom
from Guage import from_json

class Scenario:
    """
    Main Class for Running the "Paper Title"

    READ: Make sure you run the python notebook in the PreRun folder to generate necessary run files
    """

    def __init__(self, title="Default_Title", use_custom=False, init='manual', rw_covariance=1.0, method="random_walk", iterations=1):
        """
        Initialize all the correct variables for Running this Scenario
        """
        # Clear previous files
        os.system('rm ./Data/Topo/dtopo.tt3')
        os.system('rm ./Data/Topo/dtopo.data')
        guages_file_path = './PreRun/Data/gauges.npy'

        self.title = title
        self.iterations = iterations
        self.use_custom = use_custom
        self.init = init

        if(use_custom):
            self.mcmc = Custom()
        elif(method == "independent_sampler"):
            self.mcmc = IndependentSampler()
        else:
            self.mcmc = RandomWalk(rw_covariance)

        self.samples = Samples(title, self.mcmc.sample_cols, self.mcmc.proposal_cols)

        self.mcmc.set_samples(self.samples)
        self.init_guesses = self.mcmc.init_guesses(self.init)

        self.priors = self.mcmc.build_priors()
        self.samples.save_prior(self.priors)

        self.feedForward = FeedForward()

        if(os.path.isfile(guages_file_path)):
            # Make sure these Files Exist
            gauges = np.load(guages_file_path)
            self.guages = [from_json(gauge) for gauge in gauges]
            # Do initial run of GeoClaw using the initial guesses.
            self.setGeoClaw()
        else:
            raise ValueError("The Gauge and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")

    def setGeoClaw(self):
        """
        Runs an initial set up of GeoClaw
        :return:
        """
        mt = MakeTopo()
        sgc = SetGeoClaw()
        sgc.rundata.write()

        mt.get_topo()
        mt.make_dtopo(self.init_guesses)

        os.system('make clean')
        os.system('make clobber')
        os.system('make .output')

    def run(self):
        """
        Runs the Scenario For the given amount of iterations
        :return:
        """
        for _ in range(self.iterations):

            # Get current Sample and draw a proposal sample from it
            draws = self.mcmc.draw(self.samples.get_sample())

            # Save the proposal draw for debugging purposes
            self.samples.save_proposal(draws)

            # If instructed to use the custom parameters, map parameters to Okada space (9 Dimensional)
            if(self.use_custom):
                draws = self.mcmc.map_to_okada(draws)
                self.samples.save_okada(draws)

            # Run Geo Claw on the new proposal
            self.feedForward.run_geo_claw(draws)

            # Calculate the Log Likelyhood probability for the new draw
            prop_llh = self.feedForward.calculate_probability(self.guages)

            # Save the Log Likelyhood for the proposed draw
            self.samples.save_prop_llh(prop_llh)

            # Calculate the acceptance probability
            accept_prob = self.mcmc.acceptance_prob()

            # Decide to accept or reject the proposal and save
            self.mcmc.accept_reject(accept_prob)

            # Saves the stored data for debugging purposes
            self.samples.save_debug()

        return
