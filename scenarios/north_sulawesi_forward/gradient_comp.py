import numpy as np
import scipy.stats as stats
import json
import tsunamibayes as tb
from prior import LatLonPrior, BandaPrior
from gauges import build_gauges
from scenario import BandaScenario
from forward import NeuralNetEmulator
import datetime

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
    config.fgmax['min_level_check'] = len(config.geoclaw['refinement_ratios'])+1
    #forward_model = tb.GeoClawForwardModel(gauges,fault,config.fgmax,
    #                                       config.geoclaw['dtopo_path'])
    forward_model = NeuralNetEmulator(gauges, fault)

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

    return BandaScenario(prior,forward_model,covariance,config)

def centered_diff(config, step):
    # Create df, fill each column with height derivs wrt current param.
    # Currently deriving wrt model params, need deriv wrt sample params
    gauge_vals = ['Pulu Ai height',          
                  'Ambon height',                
                  'Banda Neira height',      
                  'Buru height',             
                  'Hulaliu height',                
                  'Saparua height',         
                  'Kulur height',           
                  'Ameth height',           
                  'Amahai height']
    
    scenario, forward_model, nn_model = setup(config)
    fault = forward_model.fault
    
    sample = {key:val for key,val in config.init.items() if key in scenario.sample_cols}
    model_params = pd.DataFrame(scenario.map_to_model_params(sample), index=[0])
    param_vals = sample.keys()

    
    nn_grad, nn_outputs = nn_model.compute_gradient(model_params)
    total_nn_grad = dict()
    for key in nn_grad.keys():
        grad = compute_nn_grads(nn_grad[key], 
                                q.values(), 
                                fault.strike_map, 
                                fault.dip_map, 
                                fault.depth_map, step=step)
        total_nn_grad[key] = grad
    
    nn_df = pd.DataFrame.from_dict(total_nn_grad).T.rename(columns={'dN_dmag': 'magnitude', 
                                                                      'dN_ddll': 'delta_logl',
                                                                      'dN_ddlw': 'delta_logw',
                                                                      'dN_dlat': 'latitude',
                                                                      'dN_dlon': 'longitude',
                                                                      'dN_ddo': 'depth_offset'})
    nn_df.loc[gauge_vals][param_vals]

    
    model_df = pd.DataFrame(columns=param_vals, index=gauge_vals)
    forward_grad = dict()
    for param in param_vals:
        fwd_params = sample.copy()
        bck_params = sample.copy()
        fwd_params[param] += step
        bck_params[param] -= step
        fwd_modelparams = pd.DataFrame(scenario.map_to_model_params(fwd_params), index=[0])
        bck_modelparams = pd.DataFrame(scenario.map_to_model_params(bck_params), index=[0])

#         fwd_out = forward_model.run(fwd_model)[gauge_vals]
#         bck_out = forward_model.run(bck_model)[gauge_vals]
        fwd_out = [5] * model_df.shape[0]
        bck_out = [5] * moded_df.shape[0]
     
        centered_diff = (fwd_out - bck_out) / step
        model_df[param] = centered_diff.values
        
    return model_df, nn_df


if __name__ == "__main__":
    import os
    from tsunamibayes.utils import parser, Config
    from tsunamibayes.setrun import write_setrun
    from gradient import compute_nn_grads

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
    print('----------------------------------------------')
    
    print('Calculating gradient...')
    timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    model_grad, nn_grad = compute_diffs(config, .0001)
    
    model_grad.to_csv(f'model_grad_{timestamp}')
    nn_grad.to_csv(f'nn_grad_{timestamp}')
    print('Done')
                 