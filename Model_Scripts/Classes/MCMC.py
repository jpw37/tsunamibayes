"""
Created 10/19/2018
"""
import numpy as np

class MCMC:
    """
    This Parent Class takes care of generating prior and calculating the probability given the prior and the observation
    Random Walk and Independent Sampler Inherit from this interface
    """

    def __init__(self):
        self.samples = None
        self.sample_cols = None
        self.proposal_cols = None

    def set_samples(self, Samples):
        self.samples = Samples

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
            change_llh = np.inf
        else:
            change_llh = prop_llh - samp_llh
        return change_llh



    def accept_reject(self, accept_prob):
        if np.random.random() < accept_prob:
            # Accept and save proposal
            self.samples.save_sample(self.samples.get_proposal())
            self.samples.save_sample_okada(self.samples.get_proposal_okada())
            self.samples.save_cur_llh(self.samples.get_prop_llh())
            self.samples.reset_wins()
        else:
            # Reject Proposal and Save current winner to sample list
            self.samples.increment_wins()
            self.samples.save_sample(self.samples.get_sample())
            self.samples.save_sample_okada(self.samples.get_sample_okada())


    def map_to_okada(self):
        pass

    def draw(self, prev_draw):
        pass

    def acceptance_prob(self):
        pass

    def init_guesses(self, init):
        """

        :param init:
        :return:
        """
        if init == "manual":
          #initial guesses taken from final sample of 260911_ca/001
          strike     =  2.77152900e+02
          length     =  3.36409138e+05
          width      =  3.59633559e+04
          depth      =  2.50688161e+04
          slip       =  9.17808160e+00
          rake       =  5.96643293e+01
          dip        =  1.18889907e+01
          longitude  =  1.31448175e+02
          latitude   = -4.63296475e+00

          guesses = np.array([strike, length, width, depth, slip, rake, dip,
              longitude, latitude])

        elif init == "random":
            # draw initial sample at random from prior (kdes)
            priors = self.build_priors()
            p0 = priors[0].resample(1)[:, 0]
            longitude = p0[0]
            latitude = p0[1]
            strike = p0[2]

            # draw from prior but redraw if values are unphysical
            length = -1.
            width = -1.
            depth = -1.
            slip = -1.
            rake = -1.
            dip = -1.
            while length <= 0. or width <= 0. or depth <= 0. or slip <= 0.:
                p1 = priors[1].resample(1)[:, 0]
                length = p1[3]
                width = p1[4]
                depth = p1[2]
                slip = p1[5]
                rake = p1[1]
                dip = p1[0]

            guesses = np.array([strike, length, width, depth, slip, rake, dip,
                                     longitude, latitude])

        elif init == "restart":
            guesses = np.load('../samples.npy')[0][:9]

            # np.save("guesses.npy", self.guesses)
            print("initial sample is:")
            print(guesses)

        self.samples.save_sample(guesses)

        return guesses
