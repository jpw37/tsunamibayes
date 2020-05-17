import numpy as np
import scipy.stats as stats
import json
import tsunamibayes as tb
from prior import LatLonPrior, BandaPrior
from gauges import build_gauges
from scenario import BandaScenario

def setup(config):

    # Banda Arc fault object
    arrays = np.load(config.fault['grid_data_path'])
    fault = tb.GridFault(bounds=config.model_bounds,**arrays)

    # Priors
    # latitude/longitude
    depth_mu = config.prior['depth_mu']
    depth_std = config.prior['depth_std']
    mindepth = config.prior['mindepth']
    maxdepth = config.prior['maxdepth']
    a,b = (mindepth - depth_mu) / depth_std, (maxdepth - depth_mu) / depth_std
    depth_dist = stats.truncnorm(a,b,loc=depth_mu,scale=depth_std)
    latlon = LatLonPrior(fault,depth_dist)

    # magnitude
    mag = stats.truncexpon(b=config.prior['mag_b'],loc=config.prior['mag_loc'])

    # delta_logl
    delta_logl = stats.norm(scale=config.prior['delta_logl_std']) # sample standard deviation from data

    # delta_logw
    delta_logw = stats.norm(scale=config.prior['delta_logw_std']) # sample standard deviation from data

    # depth offset
    depth_offset = stats.norm(scale=config.prior['depth_offset_std']) # in km to avoid numerically singular covariance matrix
    prior = BandaPrior(latlon,mag,delta_logl,delta_logw,depth_offset)

    # load gauges
    gauges = build_gauges()

    # Forward model
    config.fgmax['min_level_check'] = len(config.geoclaw['refinement_ratios'])
    forward_model = tb.GeoClawForwardModel(gauges,fault,config.fgmax,
                                           config.geoclaw['dtopo_path'])

    # Proposal kernel
    lat_std = config.proposal_kernel['lat_std']
    lon_std = config.proposal_kernel['lon_std']
    mag_std = config.proposal_kernel['mag_std']
    delta_logl_std = config.proposal_kernel['delta_logl_std']
    delta_logw_std = config.proposal_kernel['delta_logw_std']
    depth_offset_std = config.proposal_kernel['depth_offset_std'] #in km to avoid singular covariance matrix

    # square for std => cov
    covariance = np.diag(np.square([lat_std,
                             lon_std,
                             mag_std,
                             delta_logl_std,
                             delta_logw_std,
                             depth_offset_std]))

    return BandaScenario(prior,forward_model,covariance)

if __name__ == "__main__":
    import os
    from tsunamibayes.utils import parser, Config
    from tsunamibayes.setrun import write_setrun

    # parse command line arguments
    args = parser.parse_args()

    # break if both resume and sequential reinit flags are set
    if args.resume_dir and args.seq_reinit_dir:
        raise ValueError("flags '-r' and '-s' cannot both be set")

    # load defaults and config file
    if args.verbose: print("Reading defaults.cfg")
    config = Config()
    config.read('defaults.cfg')
    if args.config_path:
        if args.verbose: print("Reading {}".format(args.config_path))
        config.read(args.config_path)

    # write setrun.py file
    if args.verbose: print("Writing setrun.py")
    write_setrun(args.config_path)

    # copy Makefile
    if args.verbose: print("Copying Makefile")
    makefile_path = tb.__file__[:-11]+'Makefile'
    os.system("cp {} Makefile".format(makefile_path))

    # build scenario
    scenario = setup(config)

    # initialize new chain
    if not (args.resume_dir or args.seq_reinit_dir):
        if config.init['method'] == 'manual':
            u0 = {key:val for key,val in config.init.items() if key in scenario.sample_cols}
            scenario.init_chain(u0)
        elif config.init['method'] == 'prior_rvs':
            scenario.init_chain(method='prior_rvs')
        if args.verbose: print("Initializing chain with initial sample:\n",scenario.samples.iloc[0],flush=True)

    # resume in-progress chain
    if args.resume_dir:
        if args.verbose: print("Resuming chain from: {}".format(args.resume_dir),flush=True)
        scenario.resume_chain(args.resume_dir)

    # reinitialize with sequential MCMC (after using tsunamibayes.sequential.resample)
    if args.seq_reinit_dir:
        if args.verbose: print("Reinitializing chain from Sequential MCMC",flush=True)
        scenario.seq_reinit(args.seq_reinit_dir)

    # sample
    scenario.sample(args.n_samples,output_dir=args.output_dir,
                    save_freq=args.save_freq,verbose=args.verbose)
