import numpy as np
import scipy.stats as stats
import json
import pickle
import tsunamibayes as tb
from tsunamibayes.gaussian_process_regressor import GPR
from prior import LatLonPrior, SulawesiPrior            #Forgot to change this
from gauges import build_gauges
from scenario import SulawesiScenario, MultiFaultScenario
from enum import IntEnum

class FAULT(IntEnum): # Don't know if we'll need this. Maybe.
    FLORES = 0
    WALANAE = 1


# Dip angle assumed to be 25 degrees.
def walanae_dip(x):
    return np.ones(np.shape(x))*25


# Depths are assumed to be 20 km.
def walanae_depth(dist):
    """Gives depth based on distance from fault.
    A negative distance is higher than the base fault depth.
    """
    base_depth = 20000
    extra_depth = dist*np.tan(np.deg2rad(walanae_dip(dist)))
    return base_depth - extra_depth


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
    #Flores and Walanae fault objects
    with open(config.fault['walanae_data_path'], 'rb') as file:
        walanae_initialization_data = pickle.load(file)

    fault_initialization_data = [
        np.load(config.fault['flores_data_path']),
        walanae_initialization_data
    ]
    # Initialize the kernel for the Gaussian process fault. Strike, dip and
    #  depth will use the same kernel (the RBF kernel).
    flores_kernel = lambda x,y: GPR.rbf_kernel(x,y,sig=0.75)
    fault = [
        tb.fault.GaussianProcessFault( # The Flores fault uses a GaussianProcessFault
            bounds=config.model_bounds, # Model bounds are currently same for both
            kers={
                'depth': flores_kernel,
                'dip': flores_kernel,
                'strike': flores_kernel,
            },
            noise={'depth': 1, 'dip': 1, 'strike': 1},
            **fault_initialization_data[FAULT.FLORES]
        ),
        tb.fault.ReferenceCurveFault( # The Walanae fault uses a ReferenceCurveFault
            bounds=config.model_bounds,
            **fault_initialization_data[FAULT.WALANAE]
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

    strike_offset = [
        stats.norm(scale=config.prior['strike_offset_std_flo']),
        stats.norm(scale=config.prior['strike_offset_std_wal'])
    ]

    # rake offset
    # in degrees to avoid
    rake_offset = [
        stats.norm(scale=config.prior['rake_offset_std_flo']),
        stats.norm(scale=config.prior['rake_offset_std_wal'])
    ]

    prior = [
        SulawesiPrior(  latlon[FAULT.FLORES],
                        mag[FAULT.FLORES],
                        delta_logl[FAULT.FLORES],
                        delta_logw[FAULT.FLORES],
                        depth_offset[FAULT.FLORES],
                        dip_offset[FAULT.FLORES],
                        strike_offset[FAULT.FLORES],
                        rake_offset[FAULT.FLORES]
                    ) ,

        SulawesiPrior(  latlon[FAULT.WALANAE],
                        mag[FAULT.WALANAE],
                        delta_logl[FAULT.WALANAE],
                        delta_logw[FAULT.WALANAE],
                        depth_offset[FAULT.WALANAE],
                        dip_offset[FAULT.WALANAE],
                        strike_offset[FAULT.WALANAE],
                        rake_offset[FAULT.WALANAE])
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
    strike_offset_std = [
        config.proposal_kernel['strike_offset_std_flo'],
        config.proposal_kernel['strike_offset_std_wal']
    ]
    rake_offset_std = [
        config.proposal_kernel['rake_offset_std_flo'],
        config.proposal_kernel['rake_offset_std_wal']
    ]

    # square for std => cov
    covariance = [
        np.diag(np.square([
            lat_std[FAULT.FLORES],
            lon_std[FAULT.FLORES],
            mag_std[FAULT.FLORES],
            delta_logl_std[FAULT.FLORES],
            delta_logw_std[FAULT.FLORES],
            depth_offset_std[FAULT.FLORES],
            dip_offset_std[FAULT.FLORES],
            rake_offset_std[FAULT.FLORES]
        ])),
        np.diag(np.square([
            lat_std[FAULT.WALANAE],
            lon_std[FAULT.WALANAE],
            mag_std[FAULT.WALANAE],
            delta_logl_std[FAULT.WALANAE],
            delta_logw_std[FAULT.WALANAE],
            depth_offset_std[FAULT.WALANAE],
            dip_offset_std[FAULT.WALANAE],
            rake_offset_std[FAULT.WALANAE]
        ]))
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
        scenarios.resume_chain(0,args.output_dir)

    # initialize new chain
    else:
        if config.init['method'] == 'manual':
            u0 = {key:val for key,val in config.init.items() if key in scenarios.scenarios[0].sample_cols}
            scenarios.init_chain(0, u0, verbose=args.verbose)
        elif config.init['method'] == 'prior_rvs':
            scenarios.init_chain(
                args.fault_idx,
                method='prior_rvs',
                verbose=args.verbose
            )

    scenarios.sample(
        args.fault_idx,
        args.n_samples,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        verbose=args.verbose
    )
