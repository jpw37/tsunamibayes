import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .maketopo import make_fault_dtopo

def compute_dtopos(fault,model_params,verbose=False):
    model_params = model_params.reset_index()
    for idx,row in model_params.iterrows():
        subfault_params = fault.subfault_split(row['latitude'],
                                               row['longitude'],
                                               row['length'],
                                               row['width'],
                                               row['slip'],
                                               row['depth_offset'],
                                               row['rake'])
       
        if verbose: print(idx,flush=True)
        clawfault = make_fault_dtopo(subfault_params,fault.bounds)
        dtopo = clawfault.dtopo.dZ[0].copy()
        if idx == 0:
            dtopos = np.empty((len(model_params),*dtopo.shape))
            x,y = clawfault.dtopo.x.copy(),clawfault.dtopo.y.copy()
        dtopos[idx] = dtopo
    return dtopos,x,y
