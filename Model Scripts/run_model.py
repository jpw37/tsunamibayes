# File for running GeoClaw model after initial setup using setup.py
import sys
import os
import maketopo as mt
from scipy import stats
import gauge
import numpy as np
import json
from gauge import Gauge
from build_priors import build_priors

class RunModel:
    """
    A class for running the model.

    Attributes:
        iterations (int): The number of iterations to run.
        method (str): The method of sampling to use.
            Can be one of the following:
                'rw': Random walk method
                'is': Independent sampler method
        prior (2d array): Information on the prior distributions
            of the parameters for making the draws.
        gauges (list): List of gauge objects.
    """
    def __init__(self, iterations, method):
        self.iterations = iterations
        self.method = method
        self.priors = build_priors()

        with open('gauges.txt') as json_file:
            gauges_json = json.load(json_file)
        gauges = []
        for g in gauges_json:
            G = Gauge(None, None, None, None, None, None, None)
            G.from_json(g)
            gauges.append(G)
        self.gauges = gauges
        
        
    """ DEPRECIATED
    def independant_sampler_draw(self):
        """
        Draw with the independent sampling method, using the prior
        to make each of the draws.

        Returns:
            draws (array): An array of the 9 parameter draws.
        """
        # Load distribution parameters.
        params = self.prior

        # Take a random draw from the distributions for each parameter.
        # For now assume all are normal distributions.
        draws = []
        for param in params:
            dist = stats.norm(param[0], param[1])
            draws.append(dist.rvs())
        draws = np.array(draws)
        print("independent sampler draw:", draws)
        return draws
    """
    def random_walk_draw(self, u):
        """
        Draw with the random walk sampling method, using a multivariate_normal
        distribution with the following specified std deviations to
        get the distribution of the step size.

        Returns:
            draws (array): An array of the 9 parameter draws.
        """
        # Std deviations for each parameter, the mean is the current location
        strike = .375
        length = 4.e3
        width = 3.e3
        depth = .1875
        slip = .01
        rake = .25
        dip = .0875
        longitude = .025
        latitude = .01875
        mean = np.zeros(9)
        cov = np.diag([strike, length, width, depth, slip, rake,
                        dip, longitude, latitude])
        # random draw from normal distribution
        e = stats.multivariate_normal(mean, cov).rvs()
        print("Random walk difference:", e)
        print("New draw:", u+e)
        return u + e

    def one_run(self):
        """
        Run the model one time all the way through.
        """
        # Clear previous files
        os.system('rm dtopo.tt3')
        os.system('rm dtopo.data')

        # Draw new proposal
        if self.method == 'is':
            draws = self.independant_sampler_draw()
        elif self.method == 'rw':
            u = np.load('samples.npy')[0][:9]
            draws = self.random_walk_draw(u)

        # Append draws and initialized p and w to samples.npy
        init_p_w = np.array([0,1])
        sample = np.hstack((draws, init_p_w))
        samples = np.vstack((np.load('samples.npy'), sample))
        np.save('samples.npy', samples)

        # Run GeoClaw using draws
        mt.get_topo()
        mt.make_dtopo(draws)

        #os.system('make clean')
        #os.system('make clobber')
        os.system('rm .output')
        os.system('make .output')

        # Compute log-likelihood of results
        p = gauge.calculate_probability(self.gauges)

        # Change entry in samples.npy
        samples = np.load('samples.npy')
        samples[-1][-2] = p

        if self.method == 'is':
            # Find probability to accept new draw over the old draw.
            # Note we use np.exp(new - old) because it's the log-likelihood
            accept_prob = min(np.exp(p-samples[0][-2]), 1)

        elif self.method == 'rw':
            new_prob = self.priors[0].logpdf(samples[-1,[7,8,0]])
            new_prob += self.priors[1].logpdf(samples[-1,[6,5,3]])
            new_prob += self.priors[2].logpdf(samples[-1,[1,2,4]])
            old_prob = self.priors[0].logpdf(samples[0,[7,8,0]])
            old_prob += self.priors[1].logpdf(samples[0,[6,5,3]])
            old_prob += self.priors[2].logpdf(samples[0,[1,2,4]])
            #DEPRICATED
            """# Log-Likelihood of prior
            new_prob = -sum(((samples[-1][:9] - self.prior[:,0])/self.prior[:,1])**2)/2
            old_prob = -sum(((samples[0][:9] - self.prior[:,0])/self.prior[:,1])**2)/2
            """
            prior_prob = new_prob - old_prob # Log-Likelihood

            # DEPRICATED (before changed to log-likelihood)
            # prior_prob = 1.
            # for i, param in enumerate(self.prior):
            #     dist = stats.norm(param[0], param[1])
            #     new_prob = dist.pdf(samples[-1][i])
            #     old_prob = dist.pdf(samples[0][i])
            #     prior_prob *= (new_prob/old_prob)

            # Note we use np.exp(new - old) because it's the log-likelihood
            accept_prob = min(1, np.exp(p-samples[0][-2]+prior_prob))

        # Increment wins. If new, change current 'best'.
        if np.random.random() < accept_prob: # Accept new
            samples[0] = samples[-1]
            samples[-1][-1] += 1
            samples[0][-1] = len(samples) - 1
        else: # Reject new
            samples[int(samples[0][-1])][-1] += 1 # increment old draw wins
        np.save('samples.npy', samples)



    def run_model(self):
        """
        Run the model as many times as desired.
        """
        for _ in range(self.iterations):
            self.one_run()


if __name__ == "__main__":
    iterations = int(sys.argv[1])
    method = sys.argv[2]

    model = RunModel(iterations, method)
    model.run_model()
