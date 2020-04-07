import os
import json
import numpy as np
from clawpack.geoclaw import dtopotools

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW enviornment variable")

def make_dtopo(subfault_params,minlat,maxlat,minlon,maxlon,dtopo_path,verbose=False):
    """ Create dtopo data file for deformation of sea floor due to earthquake.
    Uses the Okada model with fault parameters and mesh specified below.

    Parameters
    ----------
    subfault_params : pandas DataFrame
        DataFrame containing the 9 Okada parameters for each subfault
    minlat : float
        Minimum latitude (for dtopo region)
    maxlat : float
        Maximum latitude
    minlon : float
        Minimum longitude
    maxlon : float
        Maximum longitude
    dtopo_path : string
        Path for writing dtopo file
    verbose : bool, optional
        Flag for verbose output
    """

    subfaults = []
    for _,params in subfault_params.iterrows():
        subfault = dtopotools.SubFault()
        subfault.coordinate_specification = "centroid"
        subfault.latitude = params['Latitude']
        subfault.longitude = params['Longitude']
        subfault.strike = params['Strike']
        subfault.length = params['Length']
        subfault.width = params['Width']
        subfault.slip = params['Slip']
        subfault.depth = params['Depth']
        subfault.dip = params['Dip']
        subfault.rake = params['Rake']
        subfaults.append(subfault)

    fault = dtopotools.Fault()
    fault.subfaults = subfaults
    if verbose: print(fault.subfaults)
    if verbose: print("Mw = ",fault.Mw())
    if verbose: print("Mo = ",fault.Mo())

    times = [1.]
    points_per_degree = 60 # 1 arcminute resolution
    dx = 1/points_per_degree
    n = int((maxlon - minlon)/dx + 1)
    maxlon = minlon + (n-1)*dx
    m = int((maxlat - minlat)/dx + 1)
    maxlat = minlat + (m-1)*dx
    lon = numpy.linspace(minlon, maxlon, n)
    lat = numpy.linspace(minlat, maxlat, m)

    fault.create_dtopography(lon,lat,times,verbose=verbose)
    fault.dtopo.write(dtopo_path, dtopo_type=3)
