from gauges import build_gauges
import sympy as sy

global neg_llh_grad, neg_lprior_grad
neg_llh_grad = None
neg_lprior_grad = None

def centered_difference(depth_map, lat, lon, step):
    """Compute an approximation to the gradient at (x,y) on a discretized domain

    Paramters
    ---------
    depth_map : map (lat,lon) -> depth
    
    Returns
    -------
    gradient_approx : [d depth_map / d lat, d depth_map / d_lat]
    """
    lat_deriv = (0.5*depth_map(lat + step, lon) - 0.5*depth_map(lat - step, lon)) / step
    lon_deriv = (0.5*depth_map(lat, lon + step) - 0.5*depth_map(lat, lon - step)) / step

    return [lat_deriv, lon_deriv]


def gradient_setup(dip_map, depth_map, config, step=0.01):
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
    delta_logl, delta_logw, lat, lon, mag = sy.symbols('delta_logl delta_logw lat lon mag')

    # Values independent of gauge
    mu = 4e10 # Scaling factor, found in code as 4e10 by default and never changed
    earth_rad = 6.3781e6 # meters
    H = 1 # Can be assumed to be 1 
    length = sy.Pow(10, 0.5233956445903871 * mag + 1.0974498706605313 + delta_logl)
    width = sy.Pow(10, 0.29922483873212863 * mag + 2.608734705074858 + delta_logw)
    slip = sy.Pow(10, 1.5 * mag + 9.05 - log(mu * length * width, 10))
    to_rad = lambda x: sy.pi * x / 180

    radius_lambda = lambda gauge_lat, gauge_lon: 2 * earth_rad * asin(sy.sqrt(
        sy.Pow(sy.sin(0.5*(to_rad(lat) - to_rad(gauge_lat))), 2)
        + sy.cos(to_rad(lat)) * sy.cos(to_rad(gauge_lat)) 
        * sy.Pow(sy.sin(0.5*(to_rad(lon) - to_rad(gauge_lon))), 2)))

    # get gauge info, need lat and lon
    gauges = build_gauges()

    # Values dependent on gauge location
    for i, gauge in enumerate(gauges):
        # These can move above if not going to look up true values
        H_0 = 2_000 # TODO Look up actual depth at lat,lon
        H_bar = 2_000 # TODO Calculate actual average water depth?
        psi = 0.5 + 0.575 * sy.exp(-0.0175 * length / H_bar)

        # These definitely depend on gauge
        theta, phi = to_rad(dip_map(gauge.lat,gauge.lon)), to_rad(90) # TODO Look up actual dip and rake using lat and lon
        alpha = (1-theta/180) * sy.sin(theta) * Abs(sy.sin(phi))
        R = radius_lambda(gauge.lat, gauge.lon)

        # Combined formula for mean wave height
        denom = sy.cosh((4 * sy.pi * H_0) / (width + length))
        A = ((alpha * slip) / denom) * sy.Pow(1 + (2*R / length), -psi) * (H_0 / H)**(1/4)

        # parameters from gauge likelihood (in gauge.py)
        loc = gauge.loc
        scale = gauge.scale
        if i in [3,4]:
            # Buru and Hulaliu use the chi distribution 
            df = gauge.df
            dh_dlat = ((1-df)/(A-loc) + A/scale)*sy.diff(A, lat)
            dh_dlon = ((1-df)/(A-loc) + A/scale)*sy.diff(A, lon)
            dh_dmag = ((1-df)/(A-loc) + A/scale)*sy.diff(A, mag)
            dh_ddll = ((1-df)/(A-loc) + A/scale)*sy.diff(A, dll)
            dh_ddlw = ((1-df)/(A-loc) + A/scale)*sy.diff(A, dlw)
        else:
            # The rest of the gauges use a normal distribution
            dh_dlat = (A - loc) / scale**2 * sy.diff(A, lat)
            dh_dlon = (A - loc) / scale**2 * sy.diff(A, lon)
            dh_dmag = (A - loc) / scale**2 * sy.diff(A, mag)
            dh_ddll = (A - loc) / scale**2 * sy.diff(A, delta_logl)
            dh_ddlw = (A - loc) / scale**2 * sy.diff(A, delta_logw)

        if i == 0:
            # set derivative of depth offset to 0
            dh_ddo = lambda l1,l2,m,dll,dlw: 0
            neg_llh_grad = [dh_dlat, dh_dlon, dh_dmag, dh_ddll, dh_ddlw, dh_ddo]
        else:
            neg_llh_grad[0] *= dh_dlat
            neg_llh_grad[1] *= dh_dlon
            neg_llh_grad[2] *= dh_dmag
            neg_llh_grad[3] *= dh_ddll
            neg_llh_grad[4] *= dh_ddlw

    # lambdify resulting gradient for speed improvement in computation
    for i in range(len(neg_llh_grad)):
        neg_llh_grad[i] = sy.Lambda((lat, lon, mag, delta_logl, delta_logw), neg_llh_grad[i])
    neg_llh_grad = lambda sample: [neg_llh_grad[i](*sample[:-1]) for i in range(len(neg_llh_grad))]

    # Build the gradient of the negative log prior
    # Prior parameters for depth, delta_log_length/width (dll/dlw) and depth_offset (do)
    depth_mu = config.prior['depth_mu']
    depth_std = config.prior['depth_std']
    dll_std = config.prior['delta_logl_std']
    dlw_std = config.prior['delta_lowl_std']
    do_std = config.prior['depth_offset_std']
    d_depth_dlatlon = centered_difference(depth_map, lat, lon, step)

    # Precomputed derivative values based on prior distributions in main.py, and prior.py
    d_lat_prior = lambda lat, lon, do: (depth_mu - (depth_map(lat,lon,step) + 1000*do)) / depth_std * d_depth_dlatlon[0]
    d_lon_prior = lambda lat, lon, do: (depth_mu - (depth_map(lat,lon,step) + 1000*do)) / depth_std * d_depth_dlatlon[1]
    d_mag_prior = 1
    d_dll_prior = lambda dll: dll / dll_std**2
    d_dlw_prior = lambda dlw: dlw / dlw_std**2
    d_do_prior = lambda lat, lon, do: (depth_mu - (depth_map(lat,lon,step) + 1000*do)) / depth_std * 1000 + do / do_std**2

    neg_lprior_grad = lambda sample: np.array([d_lat_prior(sample[0],sample[1],sample[5]), d_lon_prior(sample[0],sample[1],sample[5]),
                        d_mag_prior, d_dll_prior(sample[3]), d_dlw_prior(sample[4]), d_do_prior(sample[0],sample[1],sample[5])])

def dU(sample,mode='height'):
    """Compute the gradient for the U function (logprior + llh)
    for the mala mcmc method

    Parameters
    ----------
    sample : pandas.Series
        Sample parameters (lat,lon,mag,dll,dlw,depth_offset)
    
    Returns
    -------
    gradient : numpy array
        The computed gradient
    """
    if mode == 'height':
        # This if statement is for debugging only, can be deleted
        if neg_llh_grad is None or len(neg_llh_grad) == 0:
            raise ValueError('neg_llh_grad is empty list')

        return neg_l_prior(sample.values) + neg_llh_grad(sample.values)
    elif mode == 'both':
        # TODO implement arrival time gradient
        raise NotImplementedError()
    else:
        raise ValueError('Invalid mode, try \'naive\'')