# File for running GeoClaw model after initial setup using setup.py
import sys
import os
import maketopo as mt
from scipy import stats
import gauge
import numpy as np
# from pmf import PMFData, PMF

class RunModel:
    def __init__(self, iterations, method):
        self.iterations = iterations
        self.method = method
        self.prior = np.load('prior.npy')
        self.output_params = np.load('output_dist.npy')
        self.gauges = np.load('gauges.npy')

        # NOTE I don't think I need this here
        # amplification_data = np.load('amplification_data.npy')
        # row_header = amplification_data[:,0]
        # col_header = np.arange(len(data[0]) - 1)/4
        # self.pmfData = PMFData(row_header, col_header, data[:,1:])

        # pmf = self.pmfData.getPMF(distance from shore, GeoClaw output)
        # pmf.integrate(stats.norm(6,1)) where the stats.norm object is
        #  the one that we use from the original gauge data on run up height
        # #### I need to split up calculation to calculate arrival times
        #   #### and wave heights separately



    # Run with independant sampling
    def independant_sampler_draw(self):
        # Step 1
        # Load distribution parameters.
        params = self.prior

        # Step 2
        # Take a random draw from the distributions for each parameter.
        # For now assume all are normal distributions.
        draws = []
        for param in params:
            dist = stats.norm(param[0], param[1])
            draws.append(dist.rvs())
        draws = np.array(draws)
        print("independent sampler draw:", draws)
        return draws

    # Run with random walk
    def random_walk_draw(self, u):
        strike = 1.
        length = 2.e3
        width = 2.e3
        depth = .2
        slip = 2.
        rake = 1.
        dip = .5
        longitude = .25
        latitude = .25
        mean = np.zeros(9)
        cov = np.diag([strike, length, width, depth, slip, rake,
                        dip, longitude, latitude])# random draw from normal distribution
        e = stats.multivariate_normal(mean, cov).rvs()
        print("Random walk difference:", e)
        print("New draw:", u+e)
        return u + e



    def one_run(self):
        os.system('rm dtopo.tt3')
        os.system('rm dtopo.data')

        if self.method == 'is':
            draws = self.independant_sampler_draw()
        elif self.method == 'rw':
            u = np.load('samples.npy')[0][:9]
            draws = self.random_walk_draw(u)

        # Append draws and initialized p and w to samples.npy.
        init_p_w = np.array([0,1])
        sample = np.hstack((draws, init_p_w))
        samples = np.vstack((np.load('samples.npy'), sample))
        np.save('samples.npy', samples)

        # Run GeoClaw using draws.
        mt.get_topo()
        mt.make_dtopo(draws)

        os.system('make clean')
        os.system('make clobber')
        os.system('make .output')

        # arrivals, heights = gauge.read_gauges(self.gauges[:,0])
        #
        # # Create probability distributions for each gauge and variable.
        # # Then, multiply together the probabilities of each output
        # arrivals_and_heights = np.hstack((arrivals, heights))
        # p = 1.
        # output_params = np.load('output_dist.npy')
        # for i, params in enumerate(output_params):
        #     # Creates normal distribution with given params for each variable and
        #     # gauge, in this order: 1. arrival of gauge1, 2. arrival of gauge2,
        #     # 3. ..., n+1. max height of gauge1, n+2, max height of gauge2, ...
        #     dist = stats.norm(params[0], params[1])
        #     p_i = dist.pdf(arrivals_and_heights[i])
        #     p *= p_i
        p = gauge.calculate_probability(self.gauges)

        # Change entry in samples.npy
        samples = np.load('samples.npy')
        samples[-1][-2] = p

        if self.method == 'is':
            # Find probability to accept new draw over the old draw.
            accept_prob = min(np.exp(p-samples[0][-2]), 1) # Because it's log-likelihood

        elif self.method == 'rw':
            # prior_prob = 1.
            # Log-Likelihood
            new_prob = -sum(((samples[-1][:9] - self.prior[:,0])/self.prior[:,1])**2)/2
            old_prob = -sum(((samples[0][:9] - self.prior[:,0])/self.prior[:,1])**2)/2
            prior_prob = new_prob - old_prob # Log-Likelihood

            # DEPRICATED
            # for i, param in enumerate(self.prior):
            #     dist = stats.norm(param[0], param[1])
            #     new_prob = dist.pdf(samples[-1][i])
            #     old_prob = dist.pdf(samples[0][i])
            #     prior_prob *= (new_prob/old_prob)

            accept_prob = min(1, np.exp(p-samples[0][-2]+prior_prob)) # Because log-likelihood

        # Increment wins. If new, change current 'best'.
        if np.random.random() < accept_prob: # Accept new
            samples[0] = samples[-1]
            samples[-1][-1] += 1
            samples[0][-1] = len(samples) - 1
        else: # Reject new
            samples[int(samples[0][-1])][-1] += 1 # increment old draw wins
        np.save('samples.npy', samples)



    def run_model(self):
        for _ in range(self.iterations):
            self.one_run()


if __name__ == "__main__":
    iterations = int(sys.argv[1])
    method = sys.argv[2]

    model = RunModel(iterations, method)
    model.run_model()
