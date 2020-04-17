import numpy as np
import scipy.stats as stats
import json
import tsunamibayes as tb
from prior import LatLonPrior, BandaPrior
from gauges import build_gauges
from scenario import BandaScenario

# paramters file
with open('parameters.txt') as json_file:
    global_params = json.load(json_file)

# Banda Arc fault object
arrays = np.load("data/slab2_banda.npz")
fault = tb.GridFault(bounds=global_params['bounds'],**arrays)

# Priors
# latitude/longitude
depth_mu = 30000
depth_std = 5000
mindepth = 2500
maxdepth = 50000
a,b = (mindepth - depth_mu) / depth_sigma, (maxdepth - depth_mu) / depth_sigma
depth_dist = stats.truncnorm(a,b,loc=depth_mu,scale=depth_sigma)
latlon = LatLonPrior(fault,depth_dist)

# magnitude
mag = stats.truncexpon(b=3,loc=6.5)

# delta_logl
delta_logl = stats.norm(scale=0.18842320591492676) # sample standard deviation from data

# delta_logw
delta_logw = stats.norm(scale=0.17186788334444705) # sample standard deviation from data

# depth offset
depth_offset = stats.norm(scale=5) # in km to avoid numerically singular covariance matrix
prior = BandaPrior(latlon,mag,delta_logl,delta_logw,depth_offset)

# load gauges
gauges = build_gauges()

# Forward model
forward_model = tb.GeoClawForwardModel(gauges,fault,fault,
                                       global_params['fgmax_params'],
                                       global_params['dtopo_path'],
                                       global_params['bathymetry_path'])

# Scenario
longitude_std = 0.075
latitude_std = 0.075
magnitude_std = 0.05
delta_logl_std = 0.005
delta_logw_std = 0.005
depth_offset_std = .5 #in km to avoid singular covariance matrix

# square for std => cov
covariance = np.diag(np.square([longitude_std,
                         latitude_std,
                         magnitude_std,
                         delta_logl_std,
                         delta_logw_std,
                         depth_offset_std]))

scenario = BandaScenario("Banda 1852",prior,forward_model,"output/",covariance)

if __name__ == "__main__":
    pass
