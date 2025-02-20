import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .maketopo import make_fault_dtopo

def compute_dtopos(fault,model_params,verbose=False):
    """Computes the topography changes along the fault for a
    given model of the earthquake event. Returns the values of these changes
    with their associated locations along the fault.

    Parameters
    ----------
    fault : GeoClaw BaseFault Object
        A GeoClaw fault object describing the topography changes and subfaults.
    model_params : dict
        The dictionary containing a sample's information and whose keys are the
        okada parameters: 'latitude', 'longitude', 'depth_offset', 'strike',
        'length','width','slip','depth','dip','rake', and whose associated
        values are floats.
    verbose : bool
        Flag for verbose output, optional. Default is False.
        If set to true, the function will print the index's iterated within the
        model_params.

    Returns
    -------
    dtopos : array_like of floats
        The ndarray containing the values of the seafloor deformation, or the
        changes of depth in the seafloor. This array must have dimensions:
        (len(x), len(y)).
    x : array_like of floats
        The ndarray containing the x coordinates (longitude) for each point
        along the subfault.
    y : array_like of floats
        The ndarray containing the y coordinates (latitude) for each point
        along the subfault.
    """
    model_params = model_params.reset_index()
    for idx,row in model_params.iterrows():
        subfault_params = fault.subfault_split(
            row['latitude'],
            row['longitude'],
            row['length'],
            row['width'],
            row['slip'],
            row['depth_offset'],
            row['rake']
        )

        if verbose: print(idx,flush=True)
        clawfault = make_fault_dtopo(subfault_params,fault.bounds)
        dtopo = clawfault.dtopo.dZ[0].copy()
        if idx == 0:
            dtopos = np.empty((len(model_params),*dtopo.shape))
            x,y = clawfault.dtopo.x.copy(),clawfault.dtopo.y.copy()
        dtopos[idx] = dtopo
    return dtopos,x,y
