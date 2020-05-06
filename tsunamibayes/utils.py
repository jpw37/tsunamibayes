import numpy as np
import re
from ast import literal_eval
import argparse

# Earth's radius, in meters
R = 6.3781e6

def haversine(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between lat-lon coordinates on a sphere
    with radius R"""
    phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
    term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
    return 2*R*np.arcsin(np.sqrt(term))

def bearing(lat1, lon1, lat2, lon2):
    """Compute the bearing between two points"""
    lat1,lon1,lat2,lon2 = np.deg2rad([lat1,lon1,lat2,lon2])
    x = np.cos(lat2)*np.sin(lon2-lon1)
    y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    return np.degrees(np.arctan2(x,y))%360

def displace(lat, lon, bearing, distance):
    """Compute the lat-lon coordinates of a point given another point, a
    bearing, and a distance. R = radius of the earth."""
    lat,lon,bearing = np.deg2rad(lat),np.deg2rad(lon),np.deg2rad(bearing)
    delta = distance/R
    lat2 = np.arcsin(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(bearing))
    lon2 = lon+np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat),
                          np.cos(delta)-np.sin(lat)*np.sin(lat2))
    return np.degrees(lat2),np.degrees(lon2)

def calc_length(magnitude, delta_logl):
    a = 0.5233956445903871     # slope
    b = 1.0974498706605313     # intercept

    mu_logl = a*magnitude + b
    logl = mu_logl + delta_logl
    return 10**logl

def calc_width(magnitude, delta_logw):
    m = 0.29922483873212863   # slope
    c = 2.608734705074858     # y intercept

    mu_logw = m*magnitude + c
    logw = mu_logw + delta_logw
    return 10**logw

def calc_slip(magnitude, length, width, mu=4e10):
    return 10**(1.5*magnitude+9.05-np.log10(mu*length*width))

def out_of_bounds(subfault_params, bounds):
    """Returns true if any subfault lies outside of the bounds, or intersects with
    the surface"""

    # check if subfaults are outside bounds
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
    corners = np.hstack((corner1,corner2,corner3,corner4))
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
        self.dict = dict()

    def __getitem__(self, key):
        return self.dict[key]

    def read(self, file):
        """Reads a config file, overwriting any currently set values."""
        section_pattern = re.compile(r"^\[(\w*)\]")
        param_pattern = re.compile(r"^([^\s]*)\s*=\s*([^\s]*)$")
        with open(file,'r') as f:
            lines = f.readlines()
        for line in lines:
            sec = section_pattern.findall(line)
            if sec:
                if sec[0] not in self.dict.keys():
                    section = dict()
                    self.dict[sec[0]] = section
                else:
                    section = self.dict[sec[0]]

            param = param_pattern.findall(line)
            if param:
                section[param[0][0]] = literal_eval(param[0][1])

core_parser = argparse.ArgumentParser(description='Run a tsunamibayes {} scenario.')
core_parser.add_argument('--cfg', dest='config', help="config file path")
core_parser.add_argument('--outdir', dest='output_dir', default="output/",
                         help="output directory")
core_parser.add_argument('-r', dest='resume', action='store_true',
                         help="flag for resuming an in-progress chain")
core_parser.add_argument('nsamp', dest='n_samples', help="number of samples to draw")
