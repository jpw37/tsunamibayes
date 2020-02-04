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
from Adjoint import Adjoint
from pandas import read_pickle

class Scenario:
	"""
	Main Class for Running the MCMC Method for ...

	READ: Make sure you run the python notebook in the PreRun folder to generate necessary run files
	"""

	def __init__(self, title="Default_Title", use_custom=True, init='manual', adjoint=False, rw_covariance=1.0, method="random_walk", iterations=1):
		"""
		Initialize all the correct variables for Running this Scenario
		:param title: Title for Scinerio (ex: 1852)
		:param use_custom: Bool: To use the custom methods for MCMC or not
		:param init: String: (manual, random or restart) How to initialize the parameters
		:param rw_covariance: float: covariance for the random walk method
		:param method: String: MCMC Method to use
		:param iterations: Int: Number of Times to run the model
		:param adjoint: Boolean: run the adjoint solver first or not
		"""

		# Clean geoclaw files
		os.system('make clean')
		os.system('make clobber')

		# Clear previous files
		os.system('rm ./InputData/dtopo.tt3')

		#Set up necessary prerun file paths
		gauges_file_path = './PreRun/InputData/gauges.npy'
		shake_gauges_file_path = './PreRun/InputData/shake_gauges.pkl'

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
		self.samples = Samples(title, self.init_guesses, self.mcmc.sample_cols, self.mcmc.proposal_cols, self.mcmc.observation_cols,self.mcmc.num_rectangles)
		# Load csv files if restart
		if self.init == 'restart':
			self.samples.load_csv()
		self.mcmc.set_samples(self.samples)

		# Make sure Pre-Run files have been generated
		if(os.path.isfile(gauges_file_path)):
			gauges = np.load(gauges_file_path, allow_pickle=True)
			self.gauges = [from_json(gauge) for gauge in gauges]
		else:
			raise ValueError("The Gauge and FG Max files have not be created.(Please see the file /PreRun/Gauges.ipynb")



#        #test shake gauge input
#        if(os.path.isfile(shake_gauges_file_path)):
#            self.shake_gauges = read_pickle(shake_gauges_file_path)
#        else:
#            raise ValueError("Shake gauge file does not exist")

		if self.init != 'restart':
			# If using the custom methods map the initial guesses to okada parameters to save as initial sample
			if (self.use_custom):
				self.init_okada_params = self.mcmc.map_to_okada(self.init_guesses)
			else:
				self.init_okada_params = self.init_guesses
			# Save
			self.samples.save_sample_okada(self.init_okada_params)
			# Load the samples
			self.init_guesses = self.samples.get_sample()

			# JW: Create the adjoint object here...right now is given as a separate class
			if adjoint:
				print("Starting adjoint computation")
				self.adjoint = Adjoint()
				self.adjoint.run_geo_claw()
				print("Finished adjoint computation")

			# Do initial run of GeoClaw using the initial guesses.
			self.setGeoClaw()

	def setGeoClaw(self):
		"""
		Runs an initial set up of GeoClaw
		:return:
		"""
		# Get Okada parameters for initial guesses pandas data frame
		okada_params = self.init_okada_params

		# Run Geoclaw
		self.feedForward.run_geo_claw(okada_params)

		# Calculate the inital log likelihood and save result
		sample_llh, sample_arr, sample_heights = self.feedForward.calculate_llh(self.gauges)
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

			# Calculate prior probability for the current sample and proposed sample
			sample_prior_lpdf = self.mcmc.prior_logpdf(sample_params)
			proposal_prior_lpdf = self.mcmc.prior_logpdf(proposal_params)

			# Save
			self.samples.save_sample_prior_lpdf(sample_prior_lpdf)
			self.samples.save_proposal_prior_lpdf(proposal_prior_lpdf)

			if proposal_prior_lpdf == np.NINF or np.isnan(proposal_prior_lpdf):
				proposal_params_okada = self.samples.get_sample_okada().copy()
				proposal_params_okada[...] = np.nan
				self.samples.save_proposal_okada(proposal_params_okada)
				proposal_llh = np.nan
				proposal_posterior_lpdf = np.nan
				self.samples.save_proposal_llh(proposal_llh)
				self.samples.save_proposal_posterior_lpdf(proposal_post_lpdf)
				proposal_obvs = self.samples.get_sample_obvs().copy()
				proposal_obvs[...] = np.nan
				self.samples.save_obvs(proposal_obvs)
				ar = 0

			else:
				# If instructed to use the custom parameters, map parameters to Okada space (9 Dimensional)
				if(self.use_custom):
					proposal_params_okada = self.mcmc.map_to_okada(proposal_params)
				else:
					proposal_params_okada = proposal_params

				# Save Proposal
				self.samples.save_proposal_okada(proposal_params_okada)

				# Run Geo Claw on the new proposal
				self.feedForward.run_geo_claw(proposal_params_okada)

				"""
				BEGIN SHAKE MODEL

				#To speed up shake model calculation set this False
				#            shake_option = True

				#            print("init_guesses:")
				#            print(self.init_guesses)
				#            print(type(self.init_guesses))
				#            self.proposal_MMI = self.feedForward.run_abrahamson(self.shake_gauges, self.init_guesses["Magnitude"], proposal_params_okada)
				#            self.proposal_shake_llh = self.feedForward.shake_llh(self.proposal_MMI, self.shake_gauges, shake_option )
				END SHAKE MODEL
				"""

				# Calculate the Log Likelihood for the new draw
				proposal_llh, proposal_arr, proposal_heights = self.feedForward.calculate_llh(self.gauges)
				sample_llh = self.samples.get_sample_llh()

				# Save SHAKE STUFF
				print("_____proposal_llh_____", proposal_llh)
				#            proposal_llh += self.proposal_shake_llh
				print("_____proposal_llh_____", proposal_llh)

				self.samples.save_sample_llh(sample_llh)
				self.samples.save_proposal_llh(proposal_llh)
				proposal_obvs = self.mcmc.make_observations(proposal_params, proposal_arr, proposal_heights)
				self.samples.save_obvs(proposal_obvs)

				# Calculate the sample and proposal posterior log likelihood
				sample_post_lpdf = sample_prior_lpdf + sample_llh
				proposal_post_lpdf = proposal_prior_lpdf + proposal_llh
				# Save
				self.samples.save_sample_posterior_lpdf(sample_post_lpdf)
				self.samples.save_proposal_posterior_lpdf(proposal_post_lpdf)

				# Calculate the acceptance probability of the given proposal
				accept_prob = self.mcmc.acceptance_prob(sample_params,proposal_params,sample_prior_lpdf, proposal_prior_lpdf)

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
				self.samples.save_sample(self.samples.get_sample())
				self.samples.save_sample_okada(self.samples.get_sample_okada())

		self.samples.save_to_csv()
		return
