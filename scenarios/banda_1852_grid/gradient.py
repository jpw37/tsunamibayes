from gauges import build_gauges
import sympy as sy
import scipy.stats as stats
from vanilla_net import VanillaNet as VN
import torch
import numpy as np
from scipy import integrate
from read_file import get_grids, useful_grids, condensed_grids
from okada_jax import get_derivatives, get_okada_deriv
from tsunamibayes.utils import haversine, haversine_deriv_lat, haversine_deriv_lon
from clawpack.pyclaw.gauges import GaugeSolution
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
#     print(f'CD FUN: {step}({type(step)}), {lat}({type(lat)}), {lon}({type(lon)}, {depth_map(lat+step, lon)}')
    lat_deriv = (0.5 * depth_map(lat + step, lon) -
                 0.5 * depth_map(lat - step, lon)) / step
    lon_deriv = (0.5 * depth_map(lat, lon + step) -
                 0.5 * depth_map(lat, lon - step)) / step

    return [lat_deriv, lon_deriv]


def compute_nn_grads(grad, sample, strike_map, dip_map, depth_map, step):
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

def calc_adjoint(model_params, model_output, arrival_times):
    # Get gauge info, need lat and lon
    gauge_names = ['Pulu Ai', 'Ambon', 'Banda Neira', 'Buru', 'Hulaliu',
                   'Saparua', 'Kulur', 'Ameth', 'Amahai']
    gauges = build_gauges()
    

    grads = {}
    outputs = {}
    
    # Modify this when using rake on different scenario
    okada_params = ['length', 'width', 'depth', 'latitude', 'longitude', 'strike', 'slip', 'dip']
    print("arrival times:",arrival_times)
    for i, gauge in enumerate(gauge_names):
        print(i, gauge)
        temp_dict = {}
        print("New temp dict",gauge,"Gauge ^",temp_dict)
        
        arrival = arrival_times[i]
        down, up = np.floor(arrival), np.ceil(arrival)

        if down <= 9.1:
            lower_adjoint_file_name = f'/home/ilhah/fsl_groups/fslg_tsunami/nobackup/archive/1852_trail_run_chelsey_2/adjoint/_output/fort.q000{int(down)}'
        else:
            lower_adjoint_file_name = f'/home/ilhah/fsl_groups/fslg_tsunami/nobackup/archive/1852_trail_run_chelsey_2/adjoint/_output/fort.q00{int(down)}'
        
        if up <= 9.1:
            upper_adjoint_file_name = f'/home/ilhah/fsl_groups/fslg_tsunami/nobackup/archive/1852_trail_run_chelsey_2/adjoint/_output/fort.q000{int(up)}'
        else:
            upper_adjoint_file_name = f'/home/ilhah/fsl_groups/fslg_tsunami/nobackup/archive/1852_trail_run_chelsey_2/adjoint/_output/fort.q00{int(up)}'

        if i == 2 or i == 5:
            # Calculate both lower and upper grids for Banda and Saparua
            lower_grid_dict, lower_info_dict = get_grids(lower_adjoint_file_name)
            
            # print("Lower grid dict",lower_info_dict)
            # print("Lower griddict keys", lower_grid_dict.keys())
            # print("lower info dict", lower_grid_dict)
            # lower_desired_grid, lower_desired_info = useful_grids(lower_grid_dict, lower_info_dict, 125, 134, -9, -2)
            lower_desired_grid, lower_desired_info = useful_grids(lower_grid_dict, lower_info_dict, 130, 132.7, -6.5, -3.5)
            lower_adjoint_grid, lower_adjoint_keys = condensed_grids(lower_desired_grid)
            lower_temp_dict = {}

            upper_grid_dict, upper_info_dict = get_grids(upper_adjoint_file_name)
            # upper_desired_grid, upper_desired_info = useful_grids(upper_grid_dict, upper_info_dict, 125, 134, -9, -2)
            upper_desired_grid, upper_desired_info = useful_grids(upper_grid_dict, upper_info_dict, 130, 132.7, -6.5, -3.5)
            upper_adjoint_grid, upper_adjoint_keys = condensed_grids(upper_desired_grid)
            upper_temp_dict = {}

            derivatives = get_derivatives()
            for j, param in enumerate(okada_params):
                sum_lower = 0
                for k, grid in enumerate(lower_adjoint_keys):
                    mx, my, xlow, ylow, dx, dy = lower_info_dict[grid]
                    X = np.linspace(xlow, xlow + (dx * (mx-1)), mx)
                    Y = np.linspace(ylow, ylow + (dy * (my-1)), my)
                    # okada_deriv_param = get_okada_deriv(derivatives[j], float(np.array(model_params['length'])[-1]), 
                    #                                     float(np.array(model_params['width'])[-1]), float(np.array(model_params['depth'])[-1]), 
                    #                                     float(np.array(model_params['latitude'])[-1]), float(np.array(model_params['longitude'])[-1]), 
                    #                                     float(np.array(model_params['strike'])[-1]), float(np.array(model_params['slip'])[-1]), 
                    #                                     float(np.array(model_params['dip'])[-1]), float(np.array(model_params['rake'])[-1]), X, Y)
                    
                    okada_deriv_param = get_okada_deriv(derivatives[j], float(np.array(model_params['length'])), 
                                                        float(np.array(model_params['width'])), float(np.array(model_params['depth'])), 
                                                        float(np.array(model_params['latitude'])), float(np.array(model_params['longitude'])), 
                                                        float(np.array(model_params['strike'])), float(np.array(model_params['slip'])), 
                                                        float(np.array(model_params['dip'])), float(np.array(model_params['rake'])), X, Y)

                    x_deriv_lower = np.trapz(np.array(lower_adjoint_grid[lower_adjoint_keys[k]]).T * okada_deriv_param, dx = dx, axis = 0)
                    if j == 1 and k == 1:
                        print("Okada deriv param:", okada_deriv_param)
                        print("X deriv lower:", x_deriv_lower)
                    sum_lower += np.trapz(x_deriv_lower, dx = dy, axis = -1) 
                lower_temp_dict[param] = sum_lower

                sum_upper = 0
                for k, grid in enumerate(upper_adjoint_keys):
                    mx, my, xlow, ylow, dx, dy = upper_info_dict[grid]
                    X = np.linspace(xlow, xlow + (dx * (mx-1)), mx)
                    Y = np.linspace(ylow, ylow + (dy * (my-1)), my)
                    # okada_deriv_param = get_okada_deriv(derivatives[j], float(np.array(model_params['length'])[-1]), 
                    #                                     float(np.array(model_params['width'])[-1]), float(np.array(model_params['depth'])[-1]), 
                    #                                     float(np.array(model_params['latitude'])[-1]), float(np.array(model_params['longitude'])[-1]), 
                    #                                     float(np.array(model_params['strike'])[-1]), float(np.array(model_params['slip'])[-1]), 
                    #                                     float(np.array(model_params['dip'])[-1]), float(np.array(model_params['rake'])[-1]), X, Y)
                    
                    okada_deriv_param = get_okada_deriv(derivatives[j], float(np.array(model_params['length'])), 
                                                        float(np.array(model_params['width'])), float(np.array(model_params['depth'])), 
                                                        float(np.array(model_params['latitude'])), float(np.array(model_params['longitude'])), 
                                                        float(np.array(model_params['strike'])), float(np.array(model_params['slip'])), 
                                                        float(np.array(model_params['dip'])), float(np.array(model_params['rake'])), X, Y)

                    x_deriv_upper = np.trapz(np.array(upper_adjoint_grid[upper_adjoint_keys[k]]).T * okada_deriv_param, dx = dx, axis = 0)
                    sum_upper += np.trapz(x_deriv_upper, dx = dy, axis = -1) 
                upper_temp_dict[param] = sum_upper

            if up - arrival < arrival - down:
                temp_dict = upper_temp_dict
            else:
                temp_dict = lower_temp_dict
        else:
            # Calculate only the closest grid for other indices
            closest_adjoint_file_name = lower_adjoint_file_name if up - arrival >= arrival - down else upper_adjoint_file_name

            closest_grid_dict, closest_info_dict = get_grids(closest_adjoint_file_name)
            # closest_desired_grid, closest_desired_info = useful_grids(closest_grid_dict, closest_info_dict, 125, 134, -9, -2)
            closest_desired_grid, closest_desired_info = useful_grids(closest_grid_dict, closest_info_dict, 130, 132.7, -6.5, -3.5) 
            closest_adjoint_grid, closest_adjoint_keys = condensed_grids(closest_desired_grid)

            
            # print("closest grid dict",closest_info_dict)
            # print("closest griddict keys", closest_grid_dict.keys())
            # print("closest info dict", closest_grid_dict)
            derivatives = get_derivatives()
            for j, param in enumerate(okada_params):
                sum_closest = 0
                
                for k, grid in enumerate(closest_adjoint_keys):
                    mx, my, xlow, ylow, dx, dy = closest_info_dict[grid]
                    X = np.linspace(xlow, xlow + (dx * (mx-1)), mx)
                    Y = np.linspace(ylow, ylow + (dy * (my-1)), my)
                    # okada_deriv_param = get_okada_deriv(derivatives[j], float(np.array(model_params['length'])[-1]), 
                    #                                     float(np.array(model_params['width'])[-1]), float(np.array(model_params['depth'])[-1]), 
                    #                                     float(np.array(model_params['latitude'])[-1]), float(np.array(model_params['longitude'])[-1]), 
                    #                                     float(np.array(model_params['strike'])[-1]), float(np.array(model_params['slip'])[-1]), 
                    #                                     float(np.array(model_params['dip'])[-1]), float(np.array(model_params['rake'])[-1]), X, Y)
                    
                    okada_deriv_param = get_okada_deriv(derivatives[j], float(np.array(model_params['length'])), 
                                                        float(np.array(model_params['width'])), float(np.array(model_params['depth'])), 
                                                        float(np.array(model_params['latitude'])), float(np.array(model_params['longitude'])), 
                                                        float(np.array(model_params['strike'])), float(np.array(model_params['slip'])), 
                                                        float(np.array(model_params['dip'])), float(np.array(model_params['rake'])), X, Y)
                    x_deriv_closest = np.trapz(np.array(closest_adjoint_grid[grid]).T * okada_deriv_param, dx = dx, axis = 0)
                    sum_closest += np.trapz(x_deriv_closest, dx = dy, axis = -1)
                temp_dict[param] = sum_closest
        grads[gauge + ' height'] = temp_dict

        if i == 2 or i == 5:
            # ARRIVAL TIMES GRADIENT APPROXIMATIONS for index 2 (Banda) or 5 (Saparua)
            time_gauge = GaugeSolution(gauge_id=i, path="")
            times = time_gauge.t
            eta = time_gauge.q[-1]
            max_index = np.argmax(eta)

            # Find the time corresponding to the maximum height
            max_height = eta[max_index]
            max_time = times[max_index]
            if max_index == 0:  # Endpoint at the beginning
                h = times[1] - times[0]
                first_derivative = (eta[1] - eta[0]) / h
                time_second_derivative = (eta[2] - 2 * eta[1] + eta[0]) / h**2
            elif max_index == len(times) - 1:  # Endpoint at the end
                h = times[-1] - times[-2]
                first_derivative = (eta[-1] - eta[-2]) / h
                time_second_derivative = (eta[-1] - 2 * eta[-2] + eta[-3]) / h**2
            else:  # Use central difference for non-endpoints
                h = times[max_index + 1] - times[max_index]
                first_derivative = (eta[max_index + 1] - eta[max_index - 1]) / (2 * h)
                time_second_derivative = (eta[max_index + 1] - 2 * eta[max_index] + eta[max_index - 1]) / (h**2)


        print(f"Wave Height derivatives for {gauge}: {temp_dict}")    
        if i == 2 or i == 5:
            dt = 60
            time_temp_dict = {}
            for param in okada_params:
                time_temp_dict[param] = -((upper_temp_dict[param] - lower_temp_dict[param]) / dt) / time_second_derivative
            print(f"Arrival time derivatives for {gauge}: {time_temp_dict}")   
        print() 


        # Find the actual heights from geoclaw output
        outputs[gauge + ' height'] = model_output[gauge + ' height']
    print("GRADSGRADSSGRADSAGARSDFASGASEGFWEFAWE",grads)
    return grads, outputs


def dU(sample, strike_map, dip_map, depth_map, config, fault, model_params, model_output, arrival_times, grads, outputs, step=0.01):
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
    #print('MODEL PARAMETERS', model_params.keys())
    nn_model = NeuralNetEmulator(gauges, fault, retain_graph=True)
    
    def obs_deriv_lat(lat1, lon1, lat2, lon2, loc, scale):
        first = np.exp(-(haversine(lat1, lon1, lat2, lon2)-loc)**2/scale**2)
        second = 2/scale**2*(haversine(lat1,lon1,lat2,lon2)-loc)*haversine_deriv_lat(lat1,lon1,lat2,lon2)
        return first * second 

    def obs_deriv_lon(lat1, lon1, lat2, lon2, loc, scale):
        first = np.exp(-(haversine(lat1, lon1, lat2, lon2)-loc)**2/scale**2)
        second = 2/scale**2*(haversine(lat1,lon1,lat2,lon2)-loc)*haversine_deriv_lon(lat1,lon1,lat2,lon2)
        return first * second
    
    def obs_deriv_lat_banda():
        x = sy.symbols('x')
        f = stats.skewnorm(a=2*x,loc=15,scale=5)
        deriv = sy.Derivative(f, x)
        return deriv

    #print('Values dependent on gauge location')
    # Values dependent on gauge location
    g = 9.8
    for i, gauge in enumerate(gauges):
        grad, H = grads[f'{gauge.name} height'], model_output[f'{gauge.name} height'].values
        NN_grads = compute_nn_grads(grad, sample.values, strike_map, dip_map, depth_map, step)
        # Set sample parameters
        sample_lat, sample_lon, sample_m, sample_dll, sample_dlw, sample_do = sample.values

        # parameters from gauge likelihood (in gauge.py)
        loc=gauge.loc
        scale=gauge.scale
        if i in [3, 4]:
            # Buru and Hulaliu use the chi distribution
            df=gauge.df
            dh_dlat=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_dlat']
            dh_dlon=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_dlon']
            dh_dmag=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_dmag']
            dh_ddll=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_ddll']
            dh_ddlw=((1 - df) / (H - loc) + H / scale) * NN_grads['dN_ddlw']
        else:
            # The rest of the gauges use a normal distribution
            dh_dlat=(H - loc) / scale**2 * NN_grads['dN_dlat']
            dh_dlon=(H - loc) / scale**2 * NN_grads['dN_dlon']
            dh_dmag=(H - loc) / scale**2 * NN_grads['dN_dmag']
            dh_ddll=(H - loc) / scale**2 * NN_grads['dN_ddll']
            dh_ddlw=(H - loc) / scale**2 * NN_grads['dN_ddlw']
        #JPW:
        if gauge.name == 'Banda Neira':
            #print('H', H)
            H=7.0
            lat1, lon1, lat2, lon2 = gauge.lat, gauge.lon, sample_lat, sample_lon
            dh_dlat += haversine_deriv_lat(lat1, lon1, lat2, lon2)/(np.sqrt(g*H))*obs_deriv_lat(lat1, lon1, lat2, lon2, loc, scale)
            dh_dlon += haversine_deriv_lon(lat1, lon1, lat2, lon2)/np.sqrt(g*H)*obs_deriv_lon(lat1, lon1, lat2, lon2, loc, scale)
            #dh_dlat += obs_deriv_lat(lat1, lon1, lat2, lon2, loc, scale)
            #dh_dlon += obs_deriv_lon(lat1, lon1, lat2, lon2, loc, scale)

        if gauge.name == 'Saparua':
            H=7.0
            lat1, lon1, lat2, lon2 = gauge.lat, gauge.lon, sample_lat, sample_lon
            dh_dlat += haversine_deriv_lat(lat1, lon1, lat2, lon2)/(np.sqrt(g*H))*obs_deriv_lat(lat1, lon1, lat2, lon2, loc, scale)
            dh_dlon += haversine_deriv_lon(lat1, lon1, lat2, lon2)/np.sqrt(g*H)*obs_deriv_lon(lat1, lon1, lat2, lon2, loc, scale)
            #dh_dlat += obs_deriv_lat(lat1, lon1, lat2, lon2, loc, scale)
            #dh_dlon += obs_deriv_lon(lat1, lon1, lat2, lon2, loc, scale)
        #    calculate derivative of arrival time w.r.t. lat and lon
        #    calculate deriviative of observational distributions w.r.t. arrival time

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
        return float((depth_mu - (depth_map(lat, lon) + 1000 * do)) / depth_std**2 * d_depth_dlatlon[0])

    def d_lon_prior(lat, lon, do): return float((
        depth_mu - (depth_map(lat, lon) + 1000 * do)) / depth_std**2 * d_depth_dlatlon[1])
    d_mag_prior=1
    def d_dll_prior(dll): return float(dll / dll_std**2)
    def d_dlw_prior(dlw): return float(dlw / dlw_std**2)
    def d_do_prior(lat, lon, do): return float((
        depth_mu - (depth_map(lat, lon) + 1000 * do)) / depth_std**2 * 1000 + do / do_std**2)

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
