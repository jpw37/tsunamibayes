"""
Created 10/19/2018
BYU Mathematics Dept.
"""
import sys
import os
import MakeTopo as mt
from scipy import stats
import gauge
import numpy as np
import json
from gauge import Gauge
from build_priors import build_priors
import SetGeoClaw
import RandomWalk
import IndependentSampler
import Samples
import FeedForward
import Custom

import sys

class Scenario:
    """
    Main Class for Running the "Paper Title"

    READ: Make sure you run the python notebook in the PreRun folder to generate necessary files
    """

    def __init__(self, title="Default Title", use_custom=False, init='manual', rw_covariance=1.0, method="random_walk", iterations=1):
        """
        Initialize all the correct variables for Running this Scenario
        """
        # Clear previous files
        os.system('rm ./Data/dtopo.tt3')
        os.system('rm ./Data/dtopo.data')
        guages_file_path = '../PreRun/GeneratedGeoClawInput/gauges.npy'

        self.title = title
        self.iterations = iterations
        self.use_custom = use_custom
        self.init = init
        self.samples = Samples()
        if(use_custom):
            self.mcmc = Custom(self.samples)
        elif(method == "independent_sampler"):
            self.mcmc = IndependentSampler(self.samples)
        else:
            self.mcmc = RandomWalk(self.samples, rw_covariance)

        self.feedForward = FeedForward(self.mcmc)

        self.priors = self.mcmc.build_priors()

        self.samples.save_prior(self.priors)

        if(os.path.isfile(guages_file_path)):
            # Make sure these Files Exist
            self.guages = np.load(guages_file_path)
            # Do initial run of GeoClaw using the initial guesses.
            self.setGeoClaw()
        else:
            raise ValueError("The Gauges and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")

    def setGeoClaw(self):
        """
        Runs an initial set up of GeoClaw
        :return:
        """
        sgc = SetGeoClaw()
        sgc.setrun().write()

        init_guesses = self.feedForward.init_guesses(self.init)
        mt.get_topo()
        mt.make_dtopo(init_guesses)

        os.system('make clean')
        os.system('make clobber')
        os.system('make .output')

    def run(self):
        """
        Runs the Scenario For the given amount of iterations
        :return:
        """
        for _ in range(self.iterations):

            draws = self.mcmc.draw(self.samples.get_previous_sample())
            self.samples.save_samples(draws)

            if(self.use_custom):
                draws = self.mcmc.map_to_okada()
                self.samples.save_mapped(draws)

            self.feedForward.run_geo_claw(draws)

            prop_llh = self.feedForward.calculate_probability(self.guages)
            cur_samp_llh = self.samples.get_cur_llh()

            if np.isneginf(prop_llh) and np.isneginf(cur_samp_llh):
                change_llh = 0
            else:
                change_llh = prop_llh - cur_samp_llh

            self.samples.save_prop_llh(prop_llh)

            accept_prob = self.mcmc.acceptance_prob(change_llh)

            self.mcmc.accept_reject(accept_prob)

        return
