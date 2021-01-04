import numpy as np
import scipy.stats as stats
import json
import tsunamibayes as tb
from prior import LatLonPrior, BandaPrior
from gauges import build_gauges
from scenario import SulawesiScenario, MultiFaultScenarios
from enum import IntEnum

class FAULT(IntEnum): # Don't know if we'll need this. Maybe.
    FLORES = 0
    WALANAE = 1

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
    #Flores and Walinae fault objects
    fault_initialization_data = np.load(config.fault['grid_data_path']) # TODO: This will need to contain dictionaries/arrays to initialize both fault objects.
    fault = [
        tb.ReferenceCurveFault(bounds=mb, **init) for mb, init in zip(
            config.model_bounds, fault_initialization_data
        )
    ]


    # Priors
    # latitude/longitude
    depth_mu = [config.prior['depth_mu_flo'], config.prior['depth_mu_wal']]
    depth_std = [config.prior['depth_std_flo'], config.prior['depth_std_wal']]
    mindepth = [config.prior['mindepth_flo'], config.prior['mindepth_wal']]
    maxdepth = [config.prior['maxdepth_flo'], config.prior['maxdepth_wal']]
    lower_bound_depth = [
        (md-dmu)/dstd for md, dmu, dstd in zip(mindepth, depth_mu, depth_std)
    ]
    upper_bound_depth = [
        (md-dmu)/dstd for md, dmu, dstd in zip(maxdepth, depth_mu, depth_std)
    ]
    depth_dist = [
        stats.truncnorm(lb,ub,loc=dmu,scale=dstd) for lb,ub,dmu,dstd in zip(
            lower_bound_depth, upper_bound_depth, depth_mu, depth_std
        )
    ]
    latlon = [
        LatLonPrior(fault[FAULT.FLORES], depth_dist[FAULT.FLORES]),
        LatLonPrior(fault[FAULT.WALANAE], depth_dist[FAULT.WALANAE])
    ]

    # dip distrubution
    dip_mu = [config.prior['dip_mu_flo'], config.prior['dip_mu_wal']]
    dip_std = [config.prior['dip_std_flo'], config.prior['dip_std_wal']]
    mindip = [config.prior['mindip'], config.prior['mindip']]
    maxdip = [config.prior['maxdip'], config.prior['maxdip']]
    lower_bound_dip = [
        (mdip-dipmu)/dipstd for mdip, dipmu, dipstd in zip(mindip, dip_mu, dip_std)
    ]
    upper_bound_dip = [
        (mdip-dipmu)/dipstd for mdip, dipmu, dipstd in zip(maxdip, dip_mu, dip_std)
    ]

    dip_dist = [        #TODO Figure out where we are going to use this.
        stats.truncnorm(lb_dip,ub_dip,loc=dipmu,scale=dipstd) for lb_dip,ub_dip,dipmu,dipstd in zip(
            lower_bound_dip, upper_bound_dip, dip_mu, dip_std
        )
    ]

    # rake distribution
    rake_dist = [   #TODO, figure out where to put this as well. 
        stats.norm(loc=config.prior['rake_mu_flo'], scale=config.prior['rake_std_flp']),
        stats.norm(loc=config.prior['rake_mu_wal'], scale=config.prior['rake_std_wal'])
    ]
    

    # magnitude
    mag = [
        stats.truncexpon(
            b=config.prior['mag_b_flo'],
            loc=config.prior['mag_loc_flo']
        ),
        stats.truncexpon(
            b=config.prior['mag_b_wal'],
            loc=config.prior['mag_loc_wal']
        )
    ]

    # delta_logl
    # sample standard deviation from data
    delta_logl = [
        stats.norm(scale=config.prior['delta_logl_std_flo']),
        stats.norm(scale=config.prior['delta_logl_std_wal'])
    ]

    # delta_logw
    # sample standard deviation from data
    delta_logw = [
        stats.norm(scale=config.prior['delta_logw_std_flo']),
        stats.norm(scale=config.prior['delta_logw_std_wal'])
    ]

    # depth offset
    # in km to avoid numerically singular covariance matrix
    depth_offset = [
        stats.norm(scale=config.prior['depth_offset_std_flo']),
        stats.norm(scale=config.prior['depth_offset_std_wal'])
    ]

    # dip offset
    # in degrees or radians?
    dip_offset = [
        stats.norm(scale=config.prior['dip_offset_std_flo']),
        stats.norm(scale=config.prior['dip_offset_std_wal'])
    ]

    # rake offset
    # in degrees to avoid
    rake_offset = [
        stats.norm(scale=config.prior['rake_offset_std_flo']),
        stats.norm(scale=config.prior['rake_offset_std_wal'])
    ]

    prior = [
<<<<<<< Updated upstream
        SulawesiPrior(latlon,dip_dist,rake_dist,mag,delta_logl,delta_logw,depth_offset,dip_offset,rake_offset),
        SulawesiPrior(),            #TODO : Did we need to add something else here?
=======
        SulawesiPrior(latlon,mag,delta_logl,delta_logw,depth_offset),
        SulawesiPrior(latlon[FAULT.WALANAE],mag[FAULT.WALANAE],delta_logl[FAULT.WALANAE],delta_logw[FAULT.WALANAE],depth_offset[FAULT.WALANAE]),
>>>>>>> Stashed changes
    ]

    # load gauges
    gauges = build_gauges() # TODO: Ashley should be working on this.

    # Forward model
    config.fgmax['min_level_check'] = len(config.geoclaw['refinement_ratios'])+1
    forward_model = [
        tb.GeoClawForwardModel(gauges,fault[FAULT.FLORES],config.fgmax,config.geoclaw['dtopo_path']),
        tb.GeoClawForwardModel(gauges,fault[FAULT.WALANAE],config.fgmax,config.geoclaw['dtopo_path'])
    ]

    # TODO: how does proposal kernel need to change?
    # TODO: I added rake/dip offsets, but do there need to be
    #  different values for each fault?
    # Proposal kernel
    lat_std = [
        config.proposal_kernel['lat_std_flo'],
        config.proposal_kernel['lat_std_wal']
    ]
    lon_std = [
        config.proposal_kernel['lon_std_flo'],
        config.proposal_kernel['lon_std_wal']
    ]
    mag_std = [
        config.proposal_kernel['mag_std_flo'],
        config.proposal_kernel['mag_std_wal']
    ]
    delta_logl_std = [
        config.proposal_kernel['delta_logl_std_flo'],
        config.proposal_kernel['delta_logl_std_wal']
    ]
    delta_logw_std = [
        config.proposal_kernel['delta_logw_std_flo'],
        config.proposal_kernel['delta_logw_std_wal']
    ]
    # in km to avoid singular covariance matrix
    depth_offset_std = [
        config.proposal_kernel['depth_offset_std_flo'],
        config.proposal_kernel['depth_offset_std_wal']
    ]
    dip_offset_std = [
        config.proposal_kernel['dip_offset_std_flo'],
        config.proposal_kernel['dip_offset_std_wal']
    ]
    rake_offset_std = [
        config.proposal_kernel['rake_offset_std_flo'],
        config.proposal_kernel['rake_offset_std_wal']
    ]

    # square for std => cov
    covariance = [
        np.diag(np.square([a,b,c,d,e,f,g,h])) for a,b,c,d,e,f,g,h in zip([
            lat_std,
            lon_std,
            mag_std,
            delta_logl_std,
            delta_logw_std,
            depth_offset_std,
            dip_offset_std,
            rake_offset_std
        ])
    ]

    scenarios = [
        SulawesiScenario(
            prior[FAULT.FLORES],
            forward_model[FAULT.FLORES],
            covariance[FAULT.FLORES]
        ),
        SulawesiScenario(
            prior[FAULT.WALANAE],
            forward_model[FAULT.WALANAE],
            covariance[FAULT.WALANAE]
        )
    ]
    return MultiFaultScenario(scenarios)

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

    # build scenarios
    scenarios = setup(config)

    # resume in-progress chain
    if args.resume:
        if args.verbose: print("Resuming chain from: {}".format(args.output_dir),flush=True)
        scenarios.resume_chain(args.fault_idx,args.output_dir)

    # initialize new chain
    else:
        if config.init['method'] == 'manual':
            u0 = {key:val for key,val in config.init.items() if key in scenario.sample_cols}
            scenarios.init_chain(args.fault_idx, u0, verbose=args.verbose)
        elif config.init['method'] == 'prior_rvs':
            scenarios.init_chain(
                args.fault_idx,
                method='prior_rvs',
                verbose=args.verbose
            )

    scenario.sample(
        args.fault_idx,
        args.n_samples,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        verbose=args.verbose
    )
