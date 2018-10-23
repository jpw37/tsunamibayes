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

import sys

class Scenario:
    """
    Main Class for Running the "Paper Title"

    READ: Make sure you run the python notebook in the PreRun folder to generate necessary files
    """

    def __init__(self, title, method="random_walk", iterations=1):
        """
        Initialize all the correct variables for Running this Scenario
        """
        # Clear previous files
        os.system('rm ./Data/dtopo.tt3')
        os.system('rm ./Data/dtopo.data')

        self.title = title
        self.iterations = iterations


        if(method == "random_walk"):
            self.mcmc = RandomWalk()
        elif(method == "independent_sampler"):
            self.mcmc = IndependentSampler()

        self.priors = self.mcmc.build_priors()
        self.draws = self.mcmc.draw()

        self.samples = Samples()
        self.samples.save_priors(self.priors)

        self.feedForward = FeedForward()

        files_exist = True
        if(files_exist): #Make sure these Files Exist
            self.guages = np.load('../PreRun/GeneratedGeoClawInput/gauges.npy')
            self.setGeoClaw()
        else:
            raise ValueError("The Gauges and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")


    def setGeoClaw(self):
        sgc = SetGeoClaw()
        sgc.setrun().write()

    def run(self, init="manual"):

        for _ in self.iterations:
            init = "manual"

            self.feedForward.run_geo_claw(init, self.draws, self.mcmc)

            prop_llh = self.feedForward.calculate_probability(self.guages)
            cur_samp_llh = self.samples.get_cur_llh()

            if np.isneginf(prop_llh) and np.isneginf(cur_samp_llh):
                change_llh = 0
            else:
                change_llh = prop_llh - cur_samp_llh

            accept_prob = self.mcmc.acceptance_prob(change_llh)

            self.mcmc.accept_reject(accept_prob)

        return
