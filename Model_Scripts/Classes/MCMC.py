"""
Created 10/19/2018
"""
import numpy as np

class MCMC:
    """
    This Parent Class takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface
    """

    def __init__(self, Samples):
        self.samples = Samples

    def build_priors(self):
        pass

    def change_llh_calc(self):
        samp_llh = self.samples.get_cur_llh()
        prop_llh = self.samples.get_prop_llh()

        if np.isneginf(prop_llh) and np.isneginf(samp_llh):
            change_llh = 0
        elif np.isnan(prop_llh) and np.isnan(samp_llh):
            change_llh = 0
            # fix situation where nan in proposal llh results in acceptance, e.g., 8855 [-52.34308085] -10110.84699320795 [-10163.19007406] [-51.76404079] nan [nan] 1 accept
        elif np.isnan(prop_llh) and not np.isnan(samp_llh):
            change_llh = np.NINF
        elif not np.isnan(prop_llh) and np.isnan(samp_llh):
            change_llh = np.INF
        else:
            change_llh = prop_llh - samp_llh
        return change_llh

    def accept_reject(self, accept_prob):
        # Increment wins. If new, change current 'best'.
        if np.random.random() < accept_prob:  # Accept new
            samples[0] = samples[-1]
            samples[-1][-1] += 1
            samples[0][-1] = len(samples) - 1
        else:  # Reject new
            samples[int(samples[0][-1])][-1] += 1  # increment old draw wins
        np.save('samples.npy', samples)

    def map_to_okada(self):
        pass

    def build_priors(self):
        pass

    def draw(self, prev_draw):
        pass

    def acceptance_prob(self):
        pass
