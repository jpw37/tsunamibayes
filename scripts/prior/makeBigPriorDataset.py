import numpy as np
from build_priors import build_priors

nsamp=100000

#draw initial sample at random from prior (kdes)
priors = build_priors()
p0=priors[0].resample(nsamp)
p1=priors[1].resample(nsamp)

np.save("../../prior/prior_3param.npy",p0)
np.save("../../prior/prior_6param.npy",p1)
