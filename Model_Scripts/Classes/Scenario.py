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
import Utils

import sys

class Scenario:
    """
    Main Class for Running the "Paper Title"

    READ: Make sure you run the python notebook in the PreRun folder to generate necessary files
    """

    def __init__(self, title, use_utils, init, rw_covariance, method="random_walk", iterations=1):
        """
        Initialize all the correct variables for Running this Scenario
        """
        # Clear previous files
        os.system('rm ./Data/dtopo.tt3')
        os.system('rm ./Data/dtopo.data')

        self.title = title
        self.iterations = iterations
        self.utils = Utils()
        self.use_utils = use_utils


        if(method == "random_walk"):
            self.mcmc = RandomWalk(rw_covariance)
        elif(method == "independent_sampler"):
            self.mcmc = IndependentSampler(self.priors)

        if(use_utils):
            self.priors = self.utils.build_priors()
        else:
            self.priors = self.mcmc.default_build_priors()

        self.samples = Samples()
        self.samples.save_priors(self.priors)

        self.feedForward = FeedForward(self.mcmc)

        files_exist = True
        if(files_exist): #Make sure these Files Exist
            self.guages = np.load('../PreRun/GeneratedGeoClawInput/gauges.npy')
            self.setGeoClaw()
        else:
            raise ValueError("The Gauges and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")

        self.guesses = self.feedForward.init_guesses(init)

        # Do initial run of GeoClaw using the initial guesses.
        # TODO: Make MAPPING funciton vaiable and the BUILD PRIORS variable, Definiton of Gauges


    def setGeoClaw(self):
        """
        :return:
        """
        sgc = SetGeoClaw()
        sgc.setrun().write()

        mt.get_topo()
        mt.make_dtopo(self.guesses)

        os.system('make clean')
        os.system('make clobber')
        os.system('make .output')

    def run(self):
        """

        :return:
        """
        for _ in range(self.iterations):

            self.draws = self.mcmc.draw(self.samples.get_previous_sample())

            if(self.use_utils):
                self.draws = self.utils.map_to_okada(self.draws)

            self.feedForward.run_geo_claw(self.draws)

            prop_llh = self.feedForward.calculate_probability(self.guages)
            cur_samp_llh = self.samples.get_cur_llh()

            if np.isneginf(prop_llh) and np.isneginf(cur_samp_llh):
                change_llh = 0
            else:
                change_llh = prop_llh - cur_samp_llh

            accept_prob = self.mcmc.acceptance_prob(change_llh)

            self.mcmc.accept_reject(accept_prob)

        return
