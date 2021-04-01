import numpy as np
import sympy as sy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.elementary.exponential import log
import re
from ast import literal_eval
import argparse
from gauge import build_gauges

# Earth's radius, in meters
R = 6.3781e6
def convert_rectangular(lat, lon, depth=0):
    """Converts site coordinates from latitude, longitude, and depth into
    rectangular coordinates.

    Parameters
    ----------
    lat : float
        Point's latitude in degrees.
    lon : float
        Point's longitude in degrees.
    depth : float
        The distance below the surface of the earth (km). Default is 0.

    Returns
    -------
    rectangular_coordiantes : ((3,) ndarray)
        The computed rectangular coordinates of points
    """
    #spherical coordinates
    phi = np.radians(lon)
    theta = np.radians(90-lat)
    r = R - depth

    # convert to rectangular
    return r*np.array([np.sin(theta)*np.cos(phi),
                       np.sin(theta)*np.sin(phi),
                       np.cos(theta)])

def haversine(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between sets of lat-lon coordinates on a sphere
    with radius R. 
    
    Parameters
    ----------
    lat1 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the initial latitude. 
    lon1 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the initial longitude.
    lat2 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the terminal latitude. 
    lon2 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the terminal latitude. 

    Returns
    -------
    distance : float -or- ndarray of floats
        The computed distance (in meters) between the given point(s). 
        Returns object of the same dimension as the lat/lon parameters.
    """
    phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
    term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
    return 2*R*np.arcsin(np.sqrt(term))

def bearing(lat1, lon1, lat2, lon2):
    """Compute the bearing between two points.
    All of the following parameters must have the same dimension.
    
    Parameters
    ----------
    lat1 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the initial latitude. 
    lon1 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the initial longitude.
    lat2 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the terminal latitude. 
    lon1 : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the terminal latitude.
    
    Returns
    -------
    bearing : float -or- ndarray of floats
        The computed bearing (in degrees) between the given point(s). 
        Returns object of the same dimension as the lat/lon parameters.
    """
    lat1,lon1,lat2,lon2 = np.deg2rad([lat1,lon1,lat2,lon2])
    x = np.cos(lat2)*np.sin(lon2-lon1)
    y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    return np.degrees(np.arctan2(x,y))%360

neg_llh_grad = None
neg_lprior_grad = None
global neg_llh_grad, neg_lprior_grad

def centered_difference(depth_map, lat, lon, step):
    """Compute an approximation to the gradient at (x,y) on a discretized domain

    Paramters
    ---------
    depth_map : map (lat,lon) -> depth
    
    Returns
    -------
    gradient_approx : [d depth_map / d lat, d depth_map / d_lat]
    """
    lat_deriv = (depth_map(lat - step, lon) - 2*depth_map(lat, lon) + depth_map(lat + step, lon)) / step
    lon_deriv = (depth_map(lat, lon - step) - 2*depth_map(lat, lon) + depth_map(lat, lon + step)) / step

    return np.array([lat_deriv, lon_deriv])


def gradient_setup(dip_map, depth_map):
    """Use the simplified tsunami formula to compute the gradient
    
    Parameters
    ----------
    dip_map : 
        Map from lat,lon to dip angle
    depth_map : 
        Map from lat,lon to depth
    
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

        loc = gauge.loc
        scale = gauge.scale
    
        dh_dlat = (A(lat,lon,mag,delta_logl,delta_logw) - loc) / scale**2 * sy.diff(A, lat)
        dh_dlon = (A(lat,lon,mag,delta_logl,delta_logw) - loc) / scale**2 * sy.diff(A, lon)
        dh_dmag = (A(lat,lon,mag,delta_logl,delta_logw) - loc) / scale**2 * sy.diff(A, mag)
        dh_ddll = (A(lat,lon,mag,delta_logl,delta_logw) - loc) / scale**2 * sy.diff(A, delta_logl)
        dh_ddlw = (A(lat,lon,mag,delta_logl,delta_logw) - loc) / scale**2 * sy.diff(A, delta_logw)

        if i == 0:
            # set derivative of depth offset to 0
            dh_ddo = lambda x: 0
            neg_llh_grad = [dh_dlat, dh_dlon, dh_dmag, dh_ddll, dh_ddlw, dh_ddo]
        else:
            neg_llh_grad[0] *= dh_dlat
            neg_llh_grad[1] *= dh_dlon
            neg_llh_grad[2] *= dh_dmag
            neg_llh_grad[3] *= dh_ddll
            neg_llh_grad[4] *= dh_ddlw

    for i in range(len(neg_llh_grad)):
        neg_llh_grad[i] = sy.Lambda((lat, lon, mag, delta_logl, delta_logw), neg_llh_grad[i])

    neg_llh_grad = lambda sample: [neg_lprior_grad[i](*sample) for i in range(len(neg_llh_grad))]


    # Build the gradient of the negative log prior
    # Prior parameters for depth, delta_log_length/width (dll/dlw) and depth_offset (do)
    depth_mu = config.prior['depth_mu']
    depth_std = config.prior['depth_std']
    dll_std = config.prior['delta_logl_std']
    dlw_std = config.prior['delta_lowl_std']
    do_std = config.prior['depth_offset_std']
    d_depth_dlatlon = centered_difference(depth_map, lat, lon, step)

    # Precomputed derivative values based on prior distributions in main.py, and prior.py
    d_lat_prior = lambda lat, lon, do: (depth_mu - (depth_map(lat,lon) + 1000*do)) / depth_std * d_depth_dlatlon[0]
    d_lon_prior = lambda lat, lon, do: (depth_mu - (depth_map(lat,lon + 1000*do))) / depth_std * d_depth_dlatlon[1]
    d_mag_prior = 1
    d_dll_prior = lambda dll: dll / dll_std**2
    d_dlw_prior = lambda dlw: dlw / dlw_std**2
    d_do_prior = lambda lat, lon, do: (depth_mu - (depth_map(lat,lon) + 1000*do)) / depth_std * 1000 + do / do_std**2

    neg_lprior_grad = lambda sample: d_lat_prior(sample[0],sample[1],sample[5]) + d_lon_prior(sample[0],sample[1],sample[5]) +\
                        d_mag_prior + d_dll_prior(sample[3]) + d_dlw_prior(sample[4]) + d_do_prior(sample[0],sample[1],sample[5])

def dU(sample,mode='naive'):
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
    if mode == 'naive':
        
        # This if statement is for debugging only, can be deleted
        if len(neg_llh_grad) == 0:
            raise ValueError('neg_llh_grad is empty list')

        return neg_l_prior(sample.values) + neg_llh_grad(sample)
    else:
        raise ValueError('Invalid mode, try \'naive\'')

def displace(lat, lon, bearing, distance):
    """Compute the lat-lon coordinates of a point on a sphere after the displacement of a given point
    along a specified bearing and distance. R = radius of the earth.
    All of the following parameters must have the same dimension.
    
    Parameters
    ----------
    lat : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the initial latitude. 
    lon : float -or- array_like of floats
        The coordinate or ndarray of coordinates associated with the initial longitude.
    bearing : float -or- array_like of floats
        The orientation(s) of the desired displacement in (degrees).
        Must be either a single float value, or an ndarray with the same dimension as lat, and lon.
    distance : float
        The distance (in meteres) of displacement from the initial point.

    Returns
    -------
    displacement lat, lon : float -or- array_like of floats
        The float value or ndarray of the new latitude coordiantes after displacement (in degrees),
        followed by the float value or ndarray of the new longitude coordiantes.
    """
    lat,lon,bearing = np.deg2rad(lat),np.deg2rad(lon),np.deg2rad(bearing)
    delta = distance/R
    lat2 = np.arcsin(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(bearing))
    lon2 = lon+np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat),
                          np.cos(delta)-np.sin(lat)*np.sin(lat2))
    return np.degrees(lat2),np.degrees(lon2)

def calc_length(magnitude, delta_logl):
    """Computes the rupture length of the fault based on an earthquake's moment magnitude
    using a regression formula.
    
    Parameters FIXME: IS this is meters or KM?
    ----------
    magnitude : float
        The moment magnitude of the earthquake event. 
    delta_logl : float
        An offset factor for the log of the rupture length.

    Returns
    -------
    length : float
        The rupter length in (meters? km?) 
    """
    a = 0.5233956445903871     # slope
    b = 1.0974498706605313     # intercept

    mu_logl = a*magnitude + b
    logl = mu_logl + delta_logl
    return 10**logl

def calc_width(magnitude, delta_logw):
    """Computes the rupture width of the fault based on an earthquake's moment magnitude
    using a regression formula.
    
    Parameters FIXME: IS this is meters or KM?
    ----------
    magnitude : float
        The moment magnitude of the earthquake event. 
    delta_logw : float
        An offset factor for the log of the rupture width.

    Returns
    -------
    length : float
        The rupter width in (meters? km?) 
    """
    m = 0.29922483873212863   # slope
    c = 2.608734705074858     # y intercept

    mu_logw = m*magnitude + c
    logw = mu_logw + delta_logw
    return 10**logw

def calc_slip(magnitude, length, width, mu=4e10):
    """Computes the slip (or displacement) of the fault from the earthquake's magnitude
    and the rupture area using a regression forula.

    Parameters FIXME: IS this is meters or KM?
    ----------
    magnitude : float
        The moment magnitude of the earthquake event.
    length : float
        The length of the fault rupture
    width : float
        The width of the fault rupture
    mu : float
        A scaling factor for the area of the fault rupture. 
   
    Returns
    -------
    slip : float
        The total displacement of the fault plates after rupture event.  
    """
    return 10**(1.5*magnitude+9.05-np.log10(mu*length*width))

def calc_corners(subfault_params):
    """Compute the corners of the Okada rectangles specified in the paramters of a subfault.
    
    Parameters
    ----------
    subfault_params : pandas DataFrame
        The 2-d DataFrame whose columns are (ndarrays) of the Okada parameters
        and whose rows contain the associated data (float values) for each subfault.

    Returns
    -------
    corners : ndarray
        A single stacked array containing the coordinate pairs (float values in degrees)
        for the four corners for the given Okada rectangles.
    """
    lats = np.array(subfault_params['latitude'])
    lons = np.array(subfault_params['longitude'])
    strikes = np.array(subfault_params['strike'])
    lengths = np.array(subfault_params['length'])
    widths = np.array(subfault_params['width'])
    edge1 = displace(lats,lons,strikes,lengths/2)
    edge2 = displace(lats,lons,strikes-180,lengths/2)
    corner1 = displace(edge1[0],edge1[1],strikes+90,widths/2)
    corner2 = displace(edge1[0],edge1[1],strikes-90,widths/2)
    corner3 = displace(edge2[0],edge2[1],strikes+90,widths/2)
    corner4 = displace(edge2[0],edge2[1],strikes-90,widths/2)
    return np.hstack((corner1,corner2,corner3,corner4))

def out_of_bounds(subfault_params, bounds):
    """Returns true if any subfault lies outside of the bounds, or intersects with
    the surface.
    
    Parameters
    ----------
    subfault_params : pandas DataFrame
        The 2-d DataFrame whose columns are (ndarrays) of the Okada parameters
        and whose rows contain the associated data (float values) for each subfault.
    bounds : dict
        The dictionary of the upper and lower limits for latitude/longitude.
        Contains keys: lat_min, lon_min, lat_max, lon_max with associated (float) values.

    Returns
    -------
    out_of_bounds : bool
        Returns True if the subfaults surpass or intersect the bounds/surface.
        Returns False, if the subfaults satisfies the given bounds.
    """
    # check if subfaults are outside bounds
    corners = calc_corners(subfault_params)
    if np.any(corners[0] < bounds['lat_min']) or np.any(corners[0] > bounds['lat_max']):
        return True
    if np.any(corners[1] < bounds['lon_min']) or np.any(corners[1] > bounds['lon_max']):
        return True

    # check if subfaults intersect surface
    for _,subfault in subfault_params.iterrows():
        if subfault['depth'] < .5*subfault['width']*np.sin(np.deg2rad(subfault['dip'])):
            return True

    return False

class Config:
    """Class for configuration file parsing. Configuration files follow this format:

    # inline comments follow normal python conventions
    # sections are on their own line, enclosed in square brackets
    [section]

    # following each section heading are lines of parameters
    # each parameter line is "name = <valid python object>"
    path = 'path/to/file'
    value = 0.5

    [section 2]
    path2 = 'path/to/other/file'
    value = 1e10
    x0 = [1,2,3,4]
    """
    def __init__(self):
        """Initializes a dictionary object for the class."""
        self.dict = dict()

    def __getitem__(self, key):
        """Returns the value associated with a key in the object's dictionary"""
        return self.dict[key]

    def read(self, file):
        """Reads a config file's sections and parameters, then adds the config file data to
        several dictionary objects.
        
        Parameters
        ----------
        file : .cfg file
            The file containing the default value information and the paths to other data files.
        """
        section_pattern = re.compile(r"^\[(\w*)\]")
        param_pattern = re.compile(r"^([^\s]*)\s*=\s*([^\s].*)")
        with open(file,'r') as f:
            lines = f.readlines()
        for line in lines:
            sec = section_pattern.findall(line)
            if sec:
                if sec[0] not in self.dict.keys():
                    section = dict()
                    self.dict[sec[0]] = section
                    setattr(self,sec[0],section)
                else:
                    section = self.dict[sec[0]]

            param = param_pattern.findall(line)
            if param:
                section[param[0][0]] = literal_eval(param[0][1])

parser = argparse.ArgumentParser(description='Run a tsunamibayes {} scenario.')
parser.add_argument('--cfg', dest='config_path', help="config file path", type=str)
parser.add_argument('--outdir', dest='output_dir', default="output/", type=str,
                    help="output directory")
parser.add_argument('-r', dest='resume', action='store_true',
                    help="flag for resuming an in-progress chain")
parser.add_argument('--savf', dest='save_freq', type=int, default=10,
                    help="frequency for saving output")
parser.add_argument('-v', dest='verbose', action='store_true',
                    help="flag for verbose logging")
parser.add_argument('n_samples', help="number of samples to draw", type=int)
