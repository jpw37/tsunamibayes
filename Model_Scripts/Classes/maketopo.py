"""
Create topo and dtopo files needed for this example:
    etopo10min120W60W60S0S.asc        download from GeoClaw topo repository
    dtopo_usgs100227.tt3              create using Okada model
Prior to Clawpack 5.2.1, the fault parameters we specified in a .cfg file,
but now they are explicit below.

Call functions with makeplots==True to create plots of topo, slip, and dtopo.
"""

import os
import json
import numpy as np

import clawpack.clawutil.data
from clawpack.geoclaw import dtopotools

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW enviornment variable")

# Scratch directory for storing topo and dtopo files:
scratch_dir = os.path.join(CLAW, 'geoclaw', 'scratch')

def get_topo(makeplots=False):
    """
    Retrieve the topo file from the GeoClaw repository.
    """
    from clawpack.geoclaw import topotools
    topo_fname = 'etopo10min120W60W60S0S.asc'
    #url = 'http://www.geoclaw.org/topo/etopo/' + topo_fname
    #clawpack.clawutil.data.get_remote_file(url, output_dir=scratch_dir,
    #            file_name=topo_fname, verbose=True)

    if makeplots:
        from matplotlib import pyplot as plt
        topo = topotools.Topography(os.path.join(scratch_dir,topo_fname), topo_type=3)
        topo.plot()
        fname = os.path.splitext(topo_fname)[0] + '.png'
        plt.savefig(fname)
        print("Created ",fname)



def make_dtopo(params, makeplots=False):
    """
    Create dtopo data file for deformation of sea floor due to earthquake.
    Uses the Okada model with fault parameters and mesh specified below.
    """

    dtopo_fname = os.path.join('./InputData/', "dtopo.tt3")

    # number of cols = number of rectangles * number of changing params + number of constant params
    n = (len(params) - 4) // 5

    # Specify subfault parameters for this simple fault model consisting
    # of a single subfault:

    subfaults = []
    for i in range(n):
        usgs_subfault = dtopotools.SubFault()
        usgs_subfault.strike = params['Strike' + str(i+1)]
        usgs_subfault.length = params['Sublength']
        usgs_subfault.width = params['Subwidth']
        usgs_subfault.depth = params['Depth'+ str(i+1)]
        usgs_subfault.slip = params['Slip']
        usgs_subfault.rake = params['Rake']
        usgs_subfault.dip = params['Dip'+ str(i+1)]
        usgs_subfault.longitude = params['Longitude' + str(i+1)]
        usgs_subfault.latitude = params['Latitude' + str(i+1)]
        usgs_subfault.coordinate_specification = "centroid"
        subfaults.append(usgs_subfault)

    fault = dtopotools.Fault()
    fault.subfaults = subfaults
    print(fault.subfaults)

    print("Mw = ",fault.Mw())
    print("Mo = ",fault.Mo())

    if os.path.exists(dtopo_fname):
        print("*** Not regenerating dtopo file (already exists): %s" \
                    % dtopo_fname)
    else:
        print("Using Okada model to create dtopo file")

        #x = numpy.linspace(-77, -67, 100)
        #y = numpy.linspace(-40, -30, 100)
        times = [1.]

        with open('./PreRun/InputData/model_bounds.txt') as json_file:
            model_bounds = json.load(json_file)

        xlower = model_bounds['xlower']
        xupper = model_bounds['xupper']
        ylower = model_bounds['ylower']
        yupper = model_bounds['yupper']

        # dtopo parameters

        points_per_degree = 60 # 1 minute resolution
        dx = 1./points_per_degree
        mx = int((xupper - xlower)/dx + 1)
        xupper = xlower + (mx-1)*dx
        my = int((yupper - ylower)/dx + 1)
        yupper = ylower + (my-1)*dx
        print("New upper bounds:\n")
        print("latitude:",yupper)
        print("longitude:",xupper)
        x = np.linspace(xlower, xupper, mx)
        y = np.linspace(ylower, yupper, my)

        fault.create_dtopography(x,y,times,verbose=True)
        dtopo = fault.dtopo
        dtopo.write(dtopo_fname, dtopo_type=3)

    if makeplots:
        from matplotlib import pyplot as plt
        if fault.dtopo is None:
            # read in the pre-existing file:
            print("Reading in dtopo file...")
            dtopo = dtopotools.DTopography()
            dtopo.read(dtopo_fname, dtopo_type=3)
            x = dtopo.x
            y = dtopo.y
        plt.figure(figsize=(12,7))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        fault.plot_subfaults(axes=ax1,slip_color=True)
        ax1.set_xlim(x.min(),x.max())
        ax1.set_ylim(y.min(),y.max())
        dtopo.plot_dZ_colors(1.,axes=ax2)
        fname = os.path.splitext(os.path.split(dtopo_fname)[-1])[0] + '.png'
        plt.savefig(fname)
        print("Created ",fname)

if __name__=='__main__':
    get_topo(False)
    make_dtopo(params, False)
