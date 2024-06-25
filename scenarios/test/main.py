import numpy as np
import scipy.stats as stats
import json
import pickle
import tsunamibayes as tb
from tsunamibayes.gaussian_process_regressor import GPR
from prior import LatLonPrior, SulawesiPrior
from gauges import build_gauges
from scenario import SulawesiScenario, MultiFaultScenario
from enum import IntEnum

# Used to easily reference the two faults.
class FAULT(IntEnum):
    FLORES = 0
    WALANAE = 1
    MYSTERY = 2

# Dip angle assumed to be 25 degrees.
def walanae_dip(x):
    return np.ones(np.shape(x))*25

def walanae_depth(dist):
    """Gives depth based on distance from fault line."""
    depth = dist*np.tan(np.deg2rad(walanae_dip(dist)))
    return depth


def setup(config, save_all_data):
    """Extracts the data from the config object to create the SulawesiFault
    object, and then declares the scenario's initial prior, forward model, and
    covariance in order to create the SulawesiScenario.

    Parameters
    ----------
    config : Config object
        The config object that contains the default scenario data to use for
        the sampling. Essentially, this sets all the initial conditions for the
        bounds, prior, fault, etc.

    Returns
    -------
    BandaScenario : BandaScenario object
    """
    #Flores and Walanae fault objects
    with open(config.fault['walanae_data_path'], 'rb') as file:
        walanae_initialization_data = pickle.load(file)

    fault_initialization_data = [
        np.load(config.fault['flores_data_path']),
        walanae_initialization_data,
        np.load(config.fault['mystery_data_path'])
    ]
    
    geoclaw_bounds = config.geoclaw_bounds
    bounds = geoclaw_bounds
    # Initialize the kernel for the Gaussian process fault. Strike, dip and
    #  depth will use the same kernel (the RBF kernel).
    flores_kernel = lambda x,y: GPR.rbf_kernel(x,y,sig=0.75)
    mystery_kernel = lambda x,y: GPR.rbf_kernel(x,y,sig=0.75)
    fault = [
        tb.fault.GaussianProcessFault( # Flores uses a GaussianProcessFault
            bounds=geoclaw_bounds,
            model_bounds=bounds,
            kers={
                'depth': flores_kernel,
                'dip': flores_kernel,
                'strike': flores_kernel,
                'rake': flores_kernel
            },
            noise={'depth': 1, 'dip': 1, 'strike': 1, 'rake': 1},
            **fault_initialization_data[FAULT.FLORES]
        ),
        tb.fault.ReferenceCurveFault( # Walanae uses a ReferenceCurveFault
            bounds=geoclaw_bounds,
            model_bounds=bounds,
            **fault_initialization_data[FAULT.WALANAE]
        ),
        tb.fault.GaussianProcessFault( # Mystery fault uses a GaussianProcessFault
            bounds=geoclaw_bounds,
            model_bounds=bounds,
            kers={
                'depth': mystery_kernel,
                'dip': mystery_kernel,
                'strike': mystery_kernel,
                'rake': mystery_kernel
            },
            noise={'depth': 1, 'dip': 1, 'strike': 1, 'rake': 1},
            **fault_initialization_data[FAULT.MYSTERY]
        )
    ]


    # Priors
    # latitude/longitude
    depth_mu = [config.prior['depth_mu_flo'], config.prior['depth_mu_wal'],
                config.prior['depth_mu_mst']]
    depth_std = [config.prior['depth_std_flo'], config.prior['depth_std_wal'], 
                 config.prior['depth_std_mst']]
    mindepth = [config.prior['mindepth_flo'], config.prior['mindepth_wal'],
                config.prior['mindepth_mst']]
    maxdepth = [config.prior['maxdepth_flo'], config.prior['maxdepth_wal'],
                config.prior['maxdepth_mst']]

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
        LatLonPrior(fault[FAULT.FLORES], depth_dist[FAULT.FLORES],FAULT.FLORES),
        LatLonPrior(fault[FAULT.WALANAE], depth_dist[FAULT.WALANAE],FAULT.WALANAE),
        LatLonPrior(fault[FAULT.MYSTERY], depth_dist[FAULT.MYSTERY], FAULT.MYSTERY)
    ]

    # magnitude
    mag = [
        stats.truncexpon(
            b=config.prior['mag_b_flo'],
            loc=config.prior['mag_loc_flo'],
            scale=config.prior['mag_scale_flo']
        ),
        stats.truncexpon(
            b=config.prior['mag_b_wal'],
            loc=config.prior['mag_loc_wal'],
            scale=config.prior['mag_scale_wal']
        ),
        stats.truncexpon(
            b=config.prior['mag_b_mst'],
            loc=config.prior['mag_loc_mst'],
            scale=config.prior['mag_scale_mst']
        )
    ]

    # delta_logl
    # sample standard deviation from data
    delta_logl = [
        stats.norm(scale=config.prior['delta_logl_std_flo']),
        stats.norm(scale=config.prior['delta_logl_std_wal']),
        stats.norm(scale=config.prior['delta_logl_std_mst'])
    ]

    # delta_logw
    # sample standard deviation from data
    delta_logw = [
        stats.norm(scale=config.prior['delta_logw_std_flo']),
        stats.norm(scale=config.prior['delta_logw_std_wal']),
        stats.norm(scale=config.prior['delta_logw_std_mst'])
    ]

    # depth offset in km to avoid numerically singular covariance matrix
    depth_offset = [
        stats.norm(scale=config.prior['depth_offset_std_flo']),
        stats.norm(scale=config.prior['depth_offset_std_wal']),
        stats.norm(scale=config.prior['depth_offset_std_mst'])
    ]

    dip_offset = [
        stats.norm(scale=config.prior['dip_offset_std_flo']),
        stats.norm(scale=config.prior['dip_offset_std_wal']),
        stats.norm(scale=config.prior['dip_offset_std_mst'])
    ]

    strike_offset = [
        stats.norm(scale=config.prior['strike_offset_std_flo']),
        stats.norm(scale=config.prior['strike_offset_std_wal']),
        stats.norm(scale=config.prior['strike_offset_std_wal'])
    ]

    rake_offset = [
        stats.norm(scale=config.prior['rake_offset_std_flo']),
        stats.norm(scale=config.prior['rake_offset_std_wal']),
        stats.norm(scale=config.prior['rake_offset_std_mst'])
    ]

    prior = SulawesiPrior(
            latlon[FAULT.FLORES],
            mag[FAULT.FLORES],
            delta_logl[FAULT.FLORES],
            delta_logw[FAULT.FLORES],
            depth_offset[FAULT.FLORES],
            dip_offset[FAULT.FLORES],
            strike_offset[FAULT.FLORES],
            rake_offset[FAULT.FLORES],
            latlon[FAULT.WALANAE],
            mag[FAULT.WALANAE],
            delta_logl[FAULT.WALANAE],
            delta_logw[FAULT.WALANAE],
            depth_offset[FAULT.WALANAE],
            dip_offset[FAULT.WALANAE],
            strike_offset[FAULT.WALANAE],
            rake_offset[FAULT.WALANAE],
            latlon[FAULT.MYSTERY],
            mag[FAULT.MYSTERY],
            delta_logl[FAULT.MYSTERY],
            delta_logw[FAULT.MYSTERY],
            depth_offset[FAULT.MYSTERY],
            dip_offset[FAULT.MYSTERY],
            strike_offset[FAULT.MYSTERY],
            rake_offset[FAULT.MYSTERY]
        )

    # load gauges
    gauges = build_gauges()

    # Forward model
    config.fgmax['min_level_check'] = (
        len(config.geoclaw['refinement_ratios']) + 1
    )
    forward_model = tb.GeoClawForwardModel(
            gauges,
            fault,  # notice that fault is now a MultiFault object
            config.fgmax,
            config.geoclaw['dtopo_path']
        )

    # Proposal kernel
    lat_std = [
        config.proposal_kernel['lat_std_flo'],
        config.proposal_kernel['lat_std_wal'],
        config.proposal_kernel['lat_std_mst']
    ]
    lon_std = [
        config.proposal_kernel['lon_std_flo'],
        config.proposal_kernel['lon_std_wal'],
        config.proposal_kernel['lon_std_mst']
    ]
    mag_std = [
        config.proposal_kernel['mag_std_flo'],
        config.proposal_kernel['mag_std_wal'],
        config.proposal_kernel['mag_std_mst']
    ]
    delta_logl_std = [
        config.proposal_kernel['delta_logl_std_flo'],
        config.proposal_kernel['delta_logl_std_wal'],
        config.proposal_kernel['delta_logl_std_mst']
    ]
    delta_logw_std = [
        config.proposal_kernel['delta_logw_std_flo'],
        config.proposal_kernel['delta_logw_std_wal'],
        config.proposal_kernel['delta_logw_std_mst']
    ]
    # in km to avoid singular covariance matrix
    depth_offset_std = [
        config.proposal_kernel['depth_offset_std_flo'],
        config.proposal_kernel['depth_offset_std_wal'],
        config.proposal_kernel['depth_offset_std_mst']
    ]
    dip_offset_std = [
        config.proposal_kernel['dip_offset_std_flo'],
        config.proposal_kernel['dip_offset_std_wal'],
        config.proposal_kernel['dip_offset_std_mst']
    ]
    strike_offset_std = [
        config.proposal_kernel['strike_offset_std_flo'],
        config.proposal_kernel['strike_offset_std_wal'],
        config.proposal_kernel['strike_offset_std_mst']
    ]
    rake_offset_std = [
        config.proposal_kernel['rake_offset_std_flo'],
        config.proposal_kernel['rake_offset_std_wal'],
        config.proposal_kernel['rake_offset_std_mst']
    ]

    # square for std => cov
    covariance = np.diag(np.hstack((
        np.square([
            lat_std[FAULT.FLORES],
            lon_std[FAULT.FLORES],
            mag_std[FAULT.FLORES],
            delta_logl_std[FAULT.FLORES],
            delta_logw_std[FAULT.FLORES],
            depth_offset_std[FAULT.FLORES],
            dip_offset_std[FAULT.FLORES],
            strike_offset_std[FAULT.FLORES],
            rake_offset_std[FAULT.FLORES]  ]),
        np.square([
            lat_std[FAULT.WALANAE],
            lon_std[FAULT.WALANAE],
            mag_std[FAULT.WALANAE],
            delta_logl_std[FAULT.WALANAE],
            delta_logw_std[FAULT.WALANAE],
            depth_offset_std[FAULT.WALANAE],
            dip_offset_std[FAULT.WALANAE],
            strike_offset_std[FAULT.WALANAE],
            rake_offset_std[FAULT.WALANAE]  ]),
        np.square([
            lat_std[FAULT.MYSTERY],
            lon_std[FAULT.MYSTERY],
            mag_std[FAULT.MYSTERY],
            delta_logl_std[FAULT.MYSTERY],
            delta_logw_std[FAULT.MYSTERY],
            depth_offset_std[FAULT.MYSTERY],
            dip_offset_std[FAULT.MYSTERY],
            strike_offset_std[FAULT.MYSTERY],
            rake_offset_std[FAULT.MYSTERY]  ])
        )) )

    scenarios = SulawesiScenario(
            forward_model,
            prior,
            covariance,
            save_all_data=save_all_data
        )
    return MultiFaultScenario(scenarios)


if __name__ == "__main__":
    import os
    from tsunamibayes.utils import parser, Config
    from tsunamibayes.setrun import write_setrun

    MULTI_FIDELITY = True
    SAVE_ALL_DATA = True

    # parse command line arguments
    args = parser.parse_args()
    
    # load defaults and config file
    if args.verbose: print("Reading defaults.cfg")
    config = Config()
    config.read('defaults.cfg')
    if args.config_path:
        if args.verbose: print("Reading {}".format(args.config_path))
        config.read(args.config_path)

    #if not MULTI_FIDELITY:
    # write setrun.py file
    if args.verbose: print("Writing setrun.py")
    write_setrun(args.config_path)

    # copy Makefile
    if args.verbose: print("Copying Makefile")
    makefile_path = tb.__file__[:-11]+'Makefile'
    os.system("cp {} Makefile".format(makefile_path))

    # build scenarios
    scenarios = setup(config, save_all_data=SAVE_ALL_DATA)
    
    # resume in-progress chain
    if args.resume:
        if args.verbose:
            print("Resuming chain from: {}".format(args.output_dir),flush=True)
        scenarios.resume_chain(args.output_dir)

    # initialize new chain
    else:
        if config.init['method'] == 'manual':
            u0 = {
                key:val for key,val in config.init.items()
                if key in scenarios.scenarios.sample_cols   # used to be scenarios[0]
            }
                        
            scenarios.init_chain(u0, verbose=args.verbose)
        elif config.init['method'] == 'prior_rvs':
            scenarios.init_chain(
                method='prior_rvs',
                verbose=args.verbose
            )
    
    scenarios.sample(
        args.n_samples,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        verbose=args.verbose,
        refinement_ratios=list( config.geoclaw["refinement_ratios"] ),
        multi_fidelity=MULTI_FIDELITY,
        ref_rat_max_values=[4,5],   # This should be an array of ints (number of refinements)
        config_path=args.config_path
    )
