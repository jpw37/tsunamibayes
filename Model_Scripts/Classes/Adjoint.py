"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""

import numpy as np
#from maketopo import get_topo, make_dtopo
from scipy import stats
import os, sys
from scipy.interpolate import interp1d


class Adjoint:
    """
    This Class sets up and runs the linearized adjoint solver, thus setting up the AMR parameters that are needed for a more accurate and adaptive mesh refinement
    """
    def __init__(self):

        pass

    def run_geo_claw(self):
        """
        Runs the adjoint Geoclaw
        This needs to set up the correct directory for the adjoint...this is the part I don't quite know what to do.  To be specific we need to have access to the fixed_grid.txt file to create the proper topography file.
        """

        from InputData.adjoint.make_adjoint_topo import makeqinit
        os.chdir('./InputData/adjoint')

        print('Making adjoint topography')
        makeqinit()

        print('Running adjoint')
        # os.system('make clean')
        # os.system('make clobber')
        os.system('rm .output')
        os.system('make .output')

        os.chdir('../..')

        return
