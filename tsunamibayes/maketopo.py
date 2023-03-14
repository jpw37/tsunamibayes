import numpy as np
import landslide_functions as lsf
from clawpack.geoclaw import topotools as topo
import os
try:
    from clawpack.geoclaw import dtopotools
except:
    pass
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy
from itertools import product
from clawpack import geoclaw
from clawpack.geoclaw import topotools as topo
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy
from itertools import product

#landslide model parameters
p_w = 997 #(kg/m^3)
p_s = 1220 #(kg/m^3)
g = 9.8 #(m/s)
f = 0.005
C_F = .009     
step_seconds = 1
simpsons_n = 100
max_iter = 1000
num_files = 5 

#read in the topography file
topo_file = topo.Topography()
topo_file.read('./data/topo/base.tt3', topo_type=3)
dtopo_fname = os.path.join('./', "dtopo.tt3")

def make_dtopo(slide_params, start_coordinates, step_seconds, initial_vel, simpsons_n, max_iter, dpath, num_files = 5):
    """
    Create dtopo data file for deformation of sea floor due to a submarine landslide.
    Uses the simple landslide model.
    """
    #extract parameters that will govern the motion of the slide
    p_w,p_s,g,w,l,d,f,C_F = slide_params
    #get center of mass movement for the slide
    points, grid_vel, slope_vel, thetas = lsf.center_mass_path(start_coordinates, topo_file, step_seconds, max_iter, initial_vel, simpsons_n,slide_params)
    #we look at the slide in the first 10 minutes
    num_points = 600 // step_seconds
    points = points[:num_points]
    #place boxes around discrete center of mass movement
    files, centers_degrees, step = lsf.landslide_boxes(points,d, w, l, num_files, topo_file)
    #create dtopo file for geoclaw
    dtopo = dtopotools.DTopography()
    dtopo.x = topo_file.x
    dtopo.y = topo_file.y
    dtopo.X = topo_file.X
    dtopo.Y = topo_file.Y
    time_step = step * step_seconds
    dtopo_times = np.array([0,time_step*1,time_step*2,time_step*3,time_step*4])
    dtopo.times = dtopo_times
    x_shape, y_shape = files[0].shape
    dZ = np.zeros((len(files),x_shape,y_shape))
    for i in range(len(files)):
        dZ[i] = files[-i]
    dtopo.dZ = dZ
    dtopo.write(dpath, dtopo_type=3)


def write_dtopo(dtopo_path,model_params):
    """Executes the function to create the fault's dtopo object and then
    writes the dtopo object to the specified path location.

    Parameters
    ----------
    subfault_params : pandas DataFrame
        DataFrame containing the 9 Okada parameters for each subfault
    bounds : dict
        Dictionary containing the model bounds. Keys are 'lon_min','lon_max',
        'lat_min', 'lat_max'
    dtopo_path : string
        Path for writing dtopo file
    verbose : bool
        Flag for verbose output, optional. Default is False.
    """

    A = model_params['volume']/model_params['thickness'] # m
    w = np.sqrt(A * model_params['aspect_ratio'])
    l = np.sqrt(A / model_params['aspect_ratio']) # m
    d = model_params['thickness'] # m
    V = model_params['volume']#w*l*d #(m^3)
    slide_params = [p_w,p_s,g,w,l,d,f,C_F]
    start_coordinates = np.array([model_params['latitude'],model_params['longitude']])
    initial_vel = model_params['initial_velocity']

    make_dtopo(slide_params, start_coordinates, step_seconds, initial_vel, simpsons_n, max_iter, dtopo_path, num_files)

    # fault = make_fault_dtopo(subfault_params,bounds,verbose=False)
    # fault.dtopo.write(dtopo_path)


