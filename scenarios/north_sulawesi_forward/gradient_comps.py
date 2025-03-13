import numpy as np
import pandas as pd
import tsunamibayes as tb
from tsunamibayes import BaseScenario
from tsunamibayes.utils import calc_length, calc_width, calc_slip
from gradient import dU # , gradient_setup
import time
from datetime import timedelta
from main import setup
from tsunamibayes.utils import parser, Config
import numpy as np
import scipy.stats as stats
import json
import tsunamibayes as tb
from prior import LatLonPrior, BandaPrior
from gauges import build_gauges
from scenario import BandaScenario
from forward import NeuralNetEmulator
import sympy as sy


def map_to_model_params(sample):
    """Evaluate the map from sample parameters to forward model parameters.

    Parameters
    ----------
    sample : pandas Series of floats
        The series containing the arrays of information for a sample.
        Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
        'delta_logw', and 'depth_offset' with their associated float values.

    Returns
    -------
    model_params : dict
        A dictionary that builds off of the sample dictionary whose keys are the
        okada parameters: 'latitude', 'longitude', 'depth_offset', 'strike','length',
        'width','slip','depth','dip','rake',
        and whose associated values are the newly calculated values from the sample.
    """
    length = calc_length(sample['magnitude'], sample['delta_logl'])
    width = calc_width(sample['magnitude'], sample['delta_logw'])
    slip = calc_slip(sample['magnitude'], length, width)
    strike = fault.strike_map(sample['latitude'],
                                   sample['longitude'])
    dip = fault.dip_map(sample['latitude'],
                             sample['longitude'])
    depth = fault.depth_map(sample['latitude'],
                                 sample['longitude'])
    rake = 90

    model_params = dict()
    model_params['latitude'] = sample['latitude']
    model_params['longitude'] = sample['longitude']
    model_params['depth_offset'] = sample['depth_offset']
    model_params['length'] = length
    model_params['width'] = width
    model_params['slip'] = slip
    model_params['strike'] = strike
    model_params['dip'] = dip
    model_params['depth'] = depth
    model_params['rake'] = rake
    return model_params

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
    forward_model = tb.GeoClawForwardModel(gauges,fault,config.fgmax,
                                          config.geoclaw['dtopo_path'])
    nn_model = NeuralNetEmulator(gauges, fault, retain_graph=True)

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

    return BandaScenario(prior,forward_model,covariance,config), forward_model, nn_model

def compute_nn_grads(grad, sample, strike_map, dip_map, depth_map, step):
    def centered_difference(depth_map, lat, lon, step):
        """Compute an approximation to the gradient at (x,y) on a discretized domain

        Paramters
        ---------
        depth_map : map (lat,lon) -> depth

        Returns
        -------
        gradient_approx : [d depth_map / d lat, d depth_map / d_lat]
        """
        lat_deriv = (0.5 * depth_map(lat + step, lon) -
                     0.5 * depth_map(lat - step, lon)) / step
        lon_deriv = (0.5 * depth_map(lat, lon + step) -
                     0.5 * depth_map(lat, lon - step)) / step

        return [lat_deriv, lon_deriv]

    # Define functions for length, width, slip to use in differentiation
    mag, delta_logl, delta_logw, mu = sy.symbols('mag delta_logl delta_logw mu')

    length=sy.Pow(10, 0.5233956445903871 * mag +
                    1.0974498706605313 + delta_logl)
    width=sy.Pow(10, 0.29922483873212863 * mag +
                   2.608734705074858 + delta_logw)

    slip=sy.Pow(10, 1.5 * mag + 9.05 - sy.log(mu * length * width, 10))
    
    # Set sample parameters
    sample_lat, sample_lon, sample_m, sample_dll, sample_dlw, sample_do = sample
    
    sample_mu = 4e10
    
    # Tuples of inputs for cleaner code
    width_inputs = (sample_m, sample_dlw, sample_dll)
    length_inputs = (sample_m, sample_dlw, sample_dll)
    slip_inputs = (sample_m, sample_mu, sample_dlw, sample_dll)
    
    # Partial derivative functions lambdified to accept inputs
    
    # Partials of width function
    dwidth_dmag = sy.Lambda((mag,delta_logw, delta_logl), sy.diff(width, mag))
    dwidth_dlw = sy.Lambda((mag, delta_logw, delta_logl), sy.diff(width, delta_logw))
    dwidth_dll = sy.Lambda((mag, delta_logw, delta_logl), sy.diff(width, delta_logl))
    
    # Partials of length function
    dlength_dmag = sy.Lambda((mag,delta_logw, delta_logl), sy.diff(length, mag))
    dlength_dlw = sy.Lambda((mag,delta_logw, delta_logl), sy.diff(length, delta_logw))
    dlength_dll = sy.Lambda((mag,delta_logw, delta_logl), sy.diff(length, delta_logl))
    
    # Partials of slip function
    dslip_dmag = sy.Lambda((mag, mu, delta_logw, delta_logl), sy.diff(slip, mag))
    dslip_dlw = sy.Lambda((mag, mu, delta_logw, delta_logl), sy.diff(slip, delta_logw))
    dslip_dll = sy.Lambda((mag, mu, delta_logw, delta_logl), sy.diff(slip, delta_logl))
    
    # Partials of depth function
#     print(f'Centered diff output: {centered_difference(depth_map, sample_lat, sample_lon, step)}')
    ddepth_dlat, ddepth_dlon = centered_difference(depth_map, sample_lat, sample_lon, step)
    dstrike_dlat, dstrike_dlon = centered_difference(strike_map, sample_lat, sample_lon, step)
    ddip_dlat, ddip_dlon = centered_difference(dip_map, sample_lat, sample_lon, step)
                                
    
    # derivative of NN wrt magnitude
    dm = grad['length'] * float(dlength_dmag(*length_inputs)) + \
         grad['width'] * float(dwidth_dmag(*width_inputs))+     \
         grad['slip'] * float(dslip_dmag(*slip_inputs))
    
    # derivative of NN wrt delta_logl
    dll = grad['length'] * float(dlength_dll(*length_inputs)) + \
          grad['width'] * float(dwidth_dll(*width_inputs)) +    \
          grad['slip'] * float(dslip_dll(*slip_inputs))
    
    # derivative of NN wrt delta_logw
    dlw = grad['length'] * float(dlength_dlw(*length_inputs)) + \
          grad['width'] * float(dwidth_dlw(*width_inputs)) +    \
          grad['slip'] * float(dslip_dlw(*slip_inputs))

    # derivative of NN wrt latitude
    dlat = grad['latitude'] +              \
           grad['dip'] * ddip_dlat +       \
           grad['strike'] * dstrike_dlat + \
           grad['depth'] * ddepth_dlat
    
    # derivative of NN wrt longitude
    dlon = grad['longitude'] +             \
           grad['dip'] * ddip_dlon +       \
           grad['strike'] * dstrike_dlon + \
           grad['depth'] * ddepth_dlon
    
    # derivative of NN wrt depth offset
    ddo = grad['depth'] * 1
    
    return {'dN_dmag': dm, 'dN_ddll': dll, 'dN_ddlw': dlw, 'dN_dlat': dlat, 'dN_dlon': dlon, 'dN_ddo': ddo}


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
    nn_df = nn_df.loc[gauge_vals][param_vals]

    
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
        fwd_out = forward_model.compute_gradient(fwd_modelparams)[1][gauge_vals]
        bck_out = forward_model.compute_gradient(bck_modelparams)[1][gauge_vals]
     
        centered_diff = (fwd_out - bck_out) / (2*step)
        model_df[param] = centered_diff.values
        
    return model_df, nn_df

if __name__ == "__main__":
    import os
    from tsunamibayes.utils import parser, Config
    from tsunamibayes.setrun import write_setrun

    # parse command line arguments
    args = parser.parse_args()

    # load defaults and config file
    print("Reading defaults.cfg")
    config = Config()
    config.read('defaults.cfg')
    if args.config_path:
        print("Reading {}".format(args.config_path))
        config.read(args.config_path)

    # write setrun.py file
    if args.verbose: print("Writing setrun.py")
    write_setrun(args.config_path)

    # copy Makefile
    if args.verbose: print("Copying Makefile")
    makefile_path = tb.__file__[:-11]+'Makefile'
    os.system("cp {} Makefile".format(makefile_path))

    model_df, nn_df = centered_diff(config, .0001)
    
    print('Model derivative')
    print(model_df)
    
    print('NN derivative')
    print(nn_df)
    
