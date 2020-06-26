import numpy as np
from clawpack.geoclaw import dtopotools

# !!! Check if geoclaw interface needs to be rebuilt !!!

def make_fault_dtopo(subfault_params,bounds,verbose=False):
    """Create GeoClaw fault object and dtopo for deformation of sea floor due to earthquake.
    Uses the Okada model with fault parameters and mesh specified below.

    Parameters
    ----------
    subfault_params : pandas DataFrame
        DataFrame containing the 9 Okada parameters for each subfault
    bounds : dict
        Dictionary containing the model bounds. Keys are 'lon_min','lon_max',
        'lat_min', 'lat_max'
    verbose : bool, optional
        Flag for verbose output

    Returns
    -------
    fault : GeoClaw BaseFault Object
        A GeoClaw fault object describing the topography changes and subfaults.
    """

    subfaults = []
    for _,params in subfault_params.iterrows():
        subfault = dtopotools.SubFault()
        subfault.coordinate_specification = "centroid"
        subfault.latitude = params['latitude']
        subfault.longitude = params['longitude']
        subfault.strike = params['strike']
        subfault.length = params['length']
        subfault.width = params['width']
        subfault.slip = params['slip']
        subfault.depth = params['depth']
        subfault.dip = params['dip']
        subfault.rake = params['rake']
        subfaults.append(subfault)

    fault = dtopotools.Fault()
    fault.subfaults = subfaults
    if verbose: print(fault.subfaults)
    if verbose: print("Mw = ",fault.Mw())
    if verbose: print("Mo = ",fault.Mo())

    times = [1.]
    points_per_degree = 60 # 1 arcminute resolution
    dx = 1/points_per_degree
    lon_min,lon_max = bounds['lon_min'],bounds['lon_max']
    lat_min,lat_max = bounds['lat_min'],bounds['lat_max']
    n = int((lon_max - lon_min)/dx + 1)
    lon_max = lon_min + (n-1)*dx
    m = int((lat_max - lat_min)/dx + 1)
    lat_max = lat_min + (m-1)*dx
    lon = np.linspace(lon_min, lon_max, n)
    lat = np.linspace(lat_min, lat_max, m)

    fault.create_dtopography(lon,lat,times,verbose=verbose)
    return fault

def write_dtopo(subfault_params,bounds,dtopo_path,verbose=False):
    """Executes make_fault_dtopo and writes the dtopo object to the specified path location.
    
    Parameters
    ----------
    subfault_params : pandas DataFrame
        DataFrame containing the 9 Okada parameters for each subfault
    bounds : dict
        Dictionary containing the model bounds. Keys are 'lon_min','lon_max',
        'lat_min', 'lat_max'
    dtopo_path : string
        Path for writing dtopo file
    verbose : bool, optional
        Flag for verbose output
    """
    fault = make_fault_dtopo(subfault_params,bounds,verbose=False)
    fault.dtopo.write(dtopo_path)
