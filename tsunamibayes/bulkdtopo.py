import numpy as np
import matplotlib.pyplot as plt
import tsunamibayes as tb
import pandas as pd

def load_model_params(path):
    return pd.read_csv(path,index_col=0)

def compute_subfault_params(fault,model_params,verbose=False):
    subfault_params = list()
    for idx,row in model_params.iterrows():
        if verbose: print(idx,flush=True)
        subfault_params.append(fault.subfault_split(row['latitude'],
                                                    row['longitude'],
                                                    row['length'],
                                                    row['width'],
                                                    row['slip'],
                                                    row['depth_offset'],
                                                    row['rake']))
    return subfault_params

def make_faults(subfault_params,bounds,verbose=False):
    faults = list()
    for idx,df in enumerate(subfault_params):
        if verbose: print(idx,flush=True)
        faults.append(tb.maketopo.make_fault_dtopo(df,bounds))

    return faults

def get_dtopos(faults,verbose=False):
    x,y = faults[0].dtopo.x,faults[0].dtopo.y
    dtopos = list()
    for idx,fault in enumerate(faults):
        if verbose: print(idx,flush=True)
        dtopos.append(fault.dtopo.dZ[0])
    return np.array(dtopos),x,y
