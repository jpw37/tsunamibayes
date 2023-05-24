from clawpack.geoclaw import topotools as topo
import shapely
import numpy as np
import scipy.stats as stats
import json
import os
import sys
import tsunamibayes as tb
from prior import LatLonPrior, BandaPrior, DepthPrior, VolumePrior, ThicknessPrior
from gauges import build_gauges
from scenario import BandaScenario
from clawpack.geoclaw import topotools as topo

def setup(config):
    """Extracts the data from the config object to create the BandaFault object, 
    and then declares the scenario's initial prior, forward model, and covariance 
    in order to create the BandaScenario. 
    
    Parameters
    ----------
    config : Config object
        The config object that contains the default scenario data to use for the sampling.
        Essentially, this sets all the initial conditions for the bounds, prior, fault, etc.
    
    Returns
    -------
    BandaScenario : BandaScenario object
    """
    # Banda Arc fault object
    # arrays = np.load(config.fault['grid_data_path'])
    # fault = tb.GridFault(bounds=config.model_bounds,**arrays)

    # load topography file
    topo_file = topo.Topography()
    topo_file.read('./data/topo/base.tt3', topo_type=3)
    # Priors
    # depth_mu = config.prior['depth_mu']
    # depth_std = config.prior['depth_std']
    # mindepth = config.prior['mindepth']
    # maxdepth = config.prior['maxdepth']

    # depth prior (to be used in lat/lon prior)
    depth_dist = DepthPrior()
    # lat/lon prior
    latlon = LatLonPrior(topo_file,depth_dist)
    # initial velocity prior
    vel = stats.uniform(10,100 - 10)
    # aspect ratio prior
    aspect_ratio = stats.uniform(.3,1 - .3)
    # volume prior
    volume = VolumePrior(topo_file)
    # thickness prior
    thickness = ThicknessPrior()
    prior = BandaPrior(latlon,vel,volume,thickness,aspect_ratio)

    # load gauges
    gauges = build_gauges()

    # Forward model
    config.fgmax['min_level_check'] = len(config.geoclaw['refinement_ratios'])+1
    # forward_model = tb.GeoClawForwardModel(gauges,fault,config.fgmax,
    #                                        config.geoclaw['dtopo_path'])
    forward_model = tb.GeoClawForwardModel(gauges,config.fgmax,
                                           config.geoclaw['dtopo_path'])
    # Proposal kernel
    lat_std = config.proposal_kernel['lat_std']
    lon_std = config.proposal_kernel['lon_std']
    vel_std = config.proposal_kernel['vel_std']
    volume_std = config.proposal_kernel['volume_std']
    thickness_std = config.proposal_kernel['thickness_std']
    aspect_ratio_std = config.proposal_kernel['aspect_ratio_std'] 

    # square for std => cov
    covariance = np.diag(np.square([lat_std,
                             lon_std,
                             vel_std,
                             volume_std,
                             thickness_std,
                             aspect_ratio_std]))

    return BandaScenario(prior,forward_model,covariance)

if __name__ == "__main__":
    import os
    from tsunamibayes.utils import parser, Config
    from tsunamibayes.setrun import write_setrun

    # parse command line arguments
    args = parser.parse_args()

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

    # resume in-progress chain
    if args.resume:
        if args.verbose: print("Resuming chain from: {}".format(args.output_dir),flush=True)
        scenario.resume_chain(args.output_dir)
    
    # initialize new chain
    else: 
        if config.init['method'] == 'manual':
            u0 = {key:val for key,val in config.init.items() if key in scenario.sample_cols}
            scenario.init_chain(u0,verbose=args.verbose)
        elif config.init['method'] == 'prior_rvs':
            scenario.init_chain(method='prior_rvs',verbose=args.verbose)
  
    scenario.sample(args.n_samples,output_dir=args.output_dir,
                    save_freq=args.save_freq,verbose=args.verbose)
