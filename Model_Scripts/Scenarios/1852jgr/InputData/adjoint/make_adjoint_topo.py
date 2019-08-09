"""
    Download topo and dtopo files needed for this example.

    Call functions with makeplots==True to create plots of topo, slip, and dtopo.
"""

from __future__ import print_function
import os,sys
import clawpack.clawutil.data
from clawpack.geoclaw import topotools
from numpy import *

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW enviornment variable")

# Scratch directory for storing topo and dtopo files:
scratch_dir = os.path.join(CLAW, 'geoclaw', 'scratch')


# Initial data for adjoint is Gaussian hump around the gauge locations:
# Pulled from fgmax_grid.txt


def get_topo(makeplots=False):
    """
    Retrieve the topo file from the GeoClaw repository.
    """
    from clawpack.geoclaw import topotools
    topo_fname = 'etopo10min120W60W60S0S.asc'
    url = 'http://depts.washington.edu/clawpack/geoclaw/topo/etopo/' + topo_fname
    clawpack.clawutil.data.get_remote_file(url, output_dir=scratch_dir,
            file_name=topo_fname, verbose=True)

    if makeplots:
        from matplotlib import pyplot as plt
        topo = topotools.Topography(os.path.join(scratch_dir,topo_fname),
                                    topo_type=2)
        topo.plot()
        fname = os.path.splitext(topo_fname)[0] + '.png'
        plt.savefig(fname)
        print("Created ",fname)


def makeqinit():
    """
        Create qinit data file
    """

    with open("../../PreRun/InputData/fgmax_grid.txt") as f:
        temp = f.readlines()
    temp = temp[7:]
    data = zeros((len(temp),2))
    for i in range(len(temp)):
        lon, lat = temp[i].split()
        data[i][0] = lon
        data[i][1] = lat
    xcenter = data[:,0]
    ycenter = data[:,1]

    xlower = min(xcenter)-1
    xupper = max(xcenter)+1
    ylower = min(ycenter)-1
    yupper = max(ycenter)+1

    nxpoints = int((xupper-xlower)*60)+1
    nypoints = int((yupper-ylower)*60)+1

    print("nxpoints etc")
    print(nxpoints)
    print(nypoints)

    def qinit(x,y,scale=150):
        from numpy import where
        from clawpack.geoclaw.util import haversine

        z = 0

    # Gaussian using distance in meters:
        for i in range(len(xcenter)):
            r = haversine(x,y,xcenter[i],ycenter[i])
        #    z = exp(-(r/20e3)**2)
            z = z+exp(-(r/scale)**2)  #This is highly debatable...does this decay to quickly to register on the rough grid we are using?  If we have the decay slower, are we going to have the problem of multiple points stacking up and biasing the solver?
        return z



    outfile= "hump.xyz"
    topotools.topo1writer(outfile,qinit,xlower,xupper,ylower,yupper,nxpoints,nypoints)

if __name__=='__main__':
    get_topo(False)
    makeqinit()
