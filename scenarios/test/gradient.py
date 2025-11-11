from gauges import build_gauges
import sympy as sy
from vanilla_net import VanillaNet as VN
import torch
import numpy as np

global neg_llh_grad, neg_lprior_grad
neg_llh_grad = None
neg_lprior_grad = None

from forward import NeuralNetEmulator


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


def compute_nn_grads(grad, sample, strike_map, dip_map, depth_map, step, arrival=False, a_grad=None):
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
    
    if arrival:
        # Arrival time gradients
        a_dlat = a_grad['latitude'] +              \
               a_grad['dip'] * ddip_dlat +       \
               a_grad['strike'] * dstrike_dlat + \
               a_grad['depth'] * ddepth_dlat
        
        a_dlon = a_grad['longitude'] +             \
               a_grad['dip'] * ddip_dlon +       \
               a_grad['strike'] * dstrike_dlon + \
               a_grad['depth'] * ddepth_dlon

    if arrival:
        return {'arrival_grad':
                {'dN_dlat': a_dlat,
                 'dN_dlon': a_dlon},
                'height_grad':{'dN_dmag': dm, 'dN_ddll': dll, 'dN_ddlw': dlw, 'dN_dlat': dlat, 'dN_dlon': dlon, 'dN_ddo': ddo}}

    return {'dN_dmag': dm, 'dN_ddll': dll, 'dN_ddlw': dlw, 'dN_dlat': dlat, 'dN_dlon': dlon, 'dN_ddo': ddo}


def dU(sample, strike_map, dip_map, depth_map, config, fault, model_params, model_output, step=0.01):
    """Use the simplified tsunami formula to compute the gradient
    Parameters
    ----------
    dip_map :
        Map from lat,lon to dip angle
    depth_map :
        Map from lat,lon to depth
    config :
        contains variables found in the defaults.cfg file
    step :
        step size for gradient approximation of depth_map
    Returns
    -------
    gradient : numpy array
        The computed gradient
    """
    # Variables
    delta_logl, delta_logw, lat, lon, mag=sy.symbols(
        'delta_logl delta_logw lat lon mag')

    gauge_names = ['Pulu Ai', 'Ambon', 'Banda Neira', 'Buru', 'Hulaliu',
                   'Saparua', 'Kulur', 'Ameth', 'Amahai']

    # Values independent of gauge
    mu=4e10  # Scaling factor, found in code as 4e10 by default and never changed

    length = sy.Pow(10, 0.5233956445903871 * mag +
                    1.0974498706605313 + delta_logl)
    width = sy.Pow(10, 0.29922483873212863 * mag +
                   2.608734705074858 + delta_logw)
    slip = sy.Pow(10, 1.5 * mag + 9.05 - sy.log(mu * length * width, 10))

    def to_rad(x): return sy.pi * x / 180

    # get gauge info, need lat and lon
    gauges=build_gauges()
    # print(model_params)
    nn_model = NeuralNetEmulator(gauges, fault, retain_graph=True)
#     print(model_params)
#     print(type(model_params))
    grads, _  = nn_model.compute_gradient(model_params)

    # Values dependent on gauge location
    for i, gauge in enumerate(gauges):
        print(i, gauge.name)
        # parameters from gauge likelihood (in gauge.py)
        loc=gauge.loc
        scale=gauge.scale
        if i in [3, 4]:
            grad, H = grads[f'{gauge.name} height'], model_output[f'{gauge.name} height'].values
            NN_grads = compute_nn_grads(grad, sample.values, strike_map, dip_map, depth_map, step)
            # Buru and Hulaliu use the chi distribution
            df=gauge.df
            dh_dlat=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_dlat']
            dh_dlon=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_dlon']
            dh_dmag=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_dmag']
            dh_ddll=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_ddll']
            dh_ddlw=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_ddlw']

        elif i in [2,5]:
            h_grad,a_grad, H = grads[f'{gauge.name} height'], grads[f'{gauge.name} arrival'], model_output[f'{gauge.name} height'].values
            A = model_output[f'{gauge.name} arrival'].values
            print('AGRAD:', a_grad)
            NN_grads = compute_nn_grads(grad, sample.values, strike_map, dip_map, depth_map, step, arrival=True, a_grad=a_grad)
            h_NN_grads = NN_grads['height_grad']
            a_NN_grads = NN_grads['arrival_grad']
            aloc = gauge.aloc
            ascale = gauge.ascale
            
            # The rest of the gauges use a normal distribution
            dh_dlat=(H - loc) / scale**2 * h_NN_grads['dN_dlat'] + (A - aloc) / ascale**2 * a_NN_grads['dN_dlat']
            dh_dlon=(H - loc) / scale**2 * h_NN_grads['dN_dlon'] + (A - aloc) / ascale**2 * a_NN_grads['dN_dlon']
            dh_dmag=(H - loc) / scale**2 * h_NN_grads['dN_dmag']
            dh_ddll=(H - loc) / scale**2 * h_NN_grads['dN_ddll']
            dh_ddlw=(H - loc) / scale**2 * h_NN_grads['dN_ddlw']

        else:
            grad, H = grads[f'{gauge.name} height'], model_output[f'{gauge.name} height'].values
            NN_grads = compute_nn_grads(grad, sample.values, strike_map, dip_map, depth_map, step)
            # The rest of the gauges use a normal distribution
            dh_dlat=(H - loc) / scale**2 * NN_grads['dN_dlat']
            dh_dlon=(H - loc) / scale**2 * NN_grads['dN_dlon']
            dh_dmag=(H - loc) / scale**2 * NN_grads['dN_dmag']
            dh_ddll=(H - loc) / scale**2 * NN_grads['dN_ddll']
            dh_ddlw=(H - loc) / scale**2 * NN_grads['dN_ddlw']

        if i == 0:
            # set derivative of depth offset to 0
            dh_ddo = 1
            neg_llh_grad=[dh_dlat, dh_dlon,
                            dh_dmag, dh_ddll, dh_ddlw, dh_ddo]
        else:
            neg_llh_grad[0] += dh_dlat
            neg_llh_grad[1] += dh_dlon
            neg_llh_grad[2] += dh_dmag
            neg_llh_grad[3] += dh_ddll
            neg_llh_grad[4] += dh_ddlw
            
    # lambdify resulting gradient for speed improvement in computation
    for i in range(len(neg_llh_grad)):
        neg_llh_grad[i]=sy.Lambda(
            (lat, lon, mag, delta_logl, delta_logw), neg_llh_grad[i])

    def neg_llh_grad_fun(sample):
        grads = []
        for grad in neg_llh_grad:
            grad_val = grad(*sample[:-1])
            try:
                grad_val = grad_val[0]
            except:
                pass
            
            grads.append(float(grad_val))

        return grads

    # Build the gradient of the negative log prior
    # Prior parameters for depth, delta_log_length/width (dll/dlw) and depth_offset (do)
    depth_mu=config.prior['depth_mu']
    depth_std=config.prior['depth_std']
    dll_std=config.prior['delta_logl_std']
    dlw_std=config.prior['delta_logw_std']
    do_std=config.prior['depth_offset_std']

    d_depth_dlatlon=centered_difference(depth_map, sample[0], sample[1], step)

    # Precomputed derivative values based on prior distributions in main.py, and prior.py
    def d_lat_prior(lat, lon, do): 
        return float((depth_mu-(depth_map(lat, lon) + 1000 * do)) / depth_std**2 * d_depth_dlatlon[0])

    def d_lon_prior(lat, lon, do): 
        return float((depth_mu - (depth_map(lat, lon) + 1000 * do)) / depth_std**2 * d_depth_dlatlon[1])
    d_mag_prior=1
    def d_dll_prior(dll): return float(dll / dll_std**2)
    def d_dlw_prior(dlw): return float(dlw / dlw_std**2)
    def d_do_prior(lat, lon, do): return float((depth_mu - (depth_map(lat,lon)+1000*do))/depth_std**2*1000 + do/do_std**2)

    def neg_lprior_grad_fun(sample): return np.array([d_lat_prior(sample[0], sample[1], sample[5]), d_lon_prior(sample[0], sample[1], sample[5]), d_mag_prior, d_dll_prior(sample[3]), d_dlw_prior(sample[4]), d_do_prior(sample[0], sample[1], sample[5])])

#     if mode == 'height':
#         # This if statement is for debugging only, can be deleted
#         if neg_llh_grad is None or len(neg_llh_grad) == 0:
#             raise ValueError('neg_llh_grad is empty list')

    return neg_lprior_grad_fun(sample.values) + neg_llh_grad_fun(sample.values)

#     elif mode == 'both':
#         # TODO implement arrival time gradient
#         raise NotImplementedError()
#     else:
#         raise ValueError('Invalid mode, try \'naive\'')
