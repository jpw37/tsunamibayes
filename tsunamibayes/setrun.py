import os
import glob
from clawpack.clawutil import data

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW enviornment variable")

# Scratch directory for storing topo and dtopo files:
scratch_dir = os.path.join(CLAW, 'geoclaw', 'scratch')

def make_setrun(config):

    def setrun(claw_pkg='geoclaw'):
        assert claw_pkg.lower() == 'geoclaw',  "Expected claw_pkg = 'geoclaw'"

        num_dim = 2
        rundata = data.ClawRunData(claw_pkg, num_dim)

        try:
            geo_data = rundata.geo_data
        except:
            print("*** Error, this rundata has no geo_data attribute")
            raise AttributeError("Missing geo_data attribute")

        # == Physics ==
        geo_data.gravity = 9.81
        geo_data.coordinate_system = 2
        geo_data.earth_radius = 6367.5e3

        # == Forcing Options
        geo_data.coriolis_forcing = False

        # == Algorithm and Initial Conditions ==
        geo_data.sea_level = 0.0
        geo_data.dry_tolerance = 1.e-3
        geo_data.friction_forcing = True
        geo_data.manning_coefficient =.025
        geo_data.friction_depth = 1e6

        # Refinement settings
        refinement_data = rundata.refinement_data
        refinement_data.variable_dt_refinement_ratios = True
        refinement_data.wave_tolerance = 1.e-1
        refinement_data.deep_depth = 1e2
        refinement_data.max_level_deep = 3

        # index of max AMR level
        maxlevel = len(config.geoclaw["refinement_ratios"])+1

        # load all topo files from topo_dir
        topo_data = rundata.topo_data
        topo_dir = config.geoclaw['topo_dir']
        topo_files = glob.glob(topo_dir+"*.tt3")
        for file in topo_files:
            topo_data.topofiles.append([3,1,maxlevel,0.,1.e10, file])

        dtopo_data = rundata.dtopo_data
        dtopo_data.dtopofiles.append([3,maxlevel,maxlevel,config.geoclaw['dtopo_path']])
        dtopo_data.dt_max_dtopo = 0.2

        #------------------------------------------------------------------
        # Standard Clawpack parameters to be written to claw.data:
        #   (or to amr2ez.data for AMR)
        #------------------------------------------------------------------
        clawdata = rundata.clawdata  # initialized when rundata instantiated

        # Number of space dimensions:
        clawdata.num_dim = num_dim

        # Lower and upper edge of computational domain:
        clawdata.lower[0] = config.model_bounds['lon_min']      # west longitude
        clawdata.upper[0] = config.model_bounds['lon_max']       # east longitude

        clawdata.lower[1] = config.model_bounds['lat_min']       # south latitude
        clawdata.upper[1] = config.model_bounds['lat_max']         # north latitude

        # Number of grid cells: Coarsest grid
        clawdata.num_cells[0] = config.geoclaw['xcoarse_grid']
        clawdata.num_cells[1] = config.geoclaw['ycoarse_grid']

        # Number of equations in the system:
        clawdata.num_eqn = 3

        # Number of auxiliary variables in the aux array (initialized in setaux)
        clawdata.num_aux = 3

        # Index of aux array corresponding to capacity function, if there is one:
        clawdata.capa_index = 2

        # Initial time:
        clawdata.t0 = 0.0

        clawdata.output_style = 1
        # Output nout frames at equally spaced times up to tfinal:
        clawdata.num_output_times = 3
        clawdata.tfinal = config.geoclaw['run_time']
        clawdata.output_t0 = True  # output at initial (or restart) time?

        clawdata.output_format = 'ascii'      # 'ascii' or 'netcdf'

        clawdata.output_q_components = 'all'   # need all
        clawdata.output_aux_components = 'none'  # eta=h+B is in q
        clawdata.output_aux_onlyonce = False    # output aux arrays each frame

        clawdata.verbosity = config.geoclaw['verbosity']

        # --------------
        # Time stepping:
        # --------------

        # if dt_variable==1: variable time steps used based on cfl_desired,
        # if dt_variable==0: fixed time steps dt = dt_initial will always be used.
        clawdata.dt_variable = True

        # Initial time step for variable dt.
        # If dt_variable==0 then dt=dt_initial for all steps:
        clawdata.dt_initial = 0.2

        # Max time step to be allowed if variable dt used:
        clawdata.dt_max = 1e+99

        # Desired Courant number if variable dt used, and max to allow without
        # retaking step with a smaller dt:
        clawdata.cfl_desired = 0.75
        clawdata.cfl_max = 1.0

        # Maximum number of time steps to allow between output times:
        clawdata.steps_max = 5000

        # ------------------
        # Method to be used:
        # ------------------

        # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
        clawdata.order = 2

        # Use dimensional splitting? (not yet available for AMR)
        clawdata.dimensional_split = 'unsplit'

        # For unsplit method, transverse_waves can be
        #  0 or 'none'      ==> donor cell (only normal solver used)
        #  1 or 'increment' ==> corner transport of waves
        #  2 or 'all'       ==> corner transport of 2nd order corrections too
        clawdata.transverse_waves = 2

        # Number of waves in the Riemann solution:
        clawdata.num_waves = 3

        # List of limiters to use for each wave family:
        # Required:  len(limiter) == num_waves
        # Some options:
        #   0 or 'none'     ==> no limiter (Lax-Wendroff)
        #   1 or 'minmod'   ==> minmod
        #   2 or 'superbee' ==> superbee
        #   3 or 'mc'       ==> MC limiter
        #   4 or 'vanleer'  ==> van Leer
        clawdata.limiter = ['mc', 'mc', 'mc']

        clawdata.use_fwaves = True    # True ==> use f-wave version of algorithms

        # Source terms splitting:
        #   src_split == 0 or 'none'    ==> no source term (src routine never called)
        #   src_split == 1 or 'godunov' ==> Godunov (1st order) splitting used,
        #   src_split == 2 or 'strang'  ==> Strang (2nd order) splitting used,  not recommended.
        clawdata.source_split = 'godunov'

        # --------------------
        # Boundary conditions:
        # --------------------

        # Number of ghost cells (usually 2)
        clawdata.num_ghost = 2

        # Choice of BCs at xlower and xupper:
        #   0 => user specified (must modify bcN.f to use this option)
        #   1 => extrapolation (non-reflecting outflow)
        #   2 => periodic (must specify this at both boundaries)
        #   3 => solid wall for systems where q(2) is normal velocity

        clawdata.bc_lower[0] = 'extrap'
        clawdata.bc_upper[0] = 'extrap'

        clawdata.bc_lower[1] = 'extrap'
        clawdata.bc_upper[1] = 'extrap'

        # ---------------
        # AMR parameters:
        # ---------------
        amrdata = rundata.amrdata

        # max number of refinement levels:
        amrdata.amr_levels_max = len(config.geoclaw['refinement_ratios'])+1  #JW: I'm not sure if this is going to work...check this :)

        # List of refinement ratios at each level (length at least mxnest-1)
        amrdata.refinement_ratios_x = config.geoclaw['refinement_ratios']
        amrdata.refinement_ratios_y = config.geoclaw['refinement_ratios']
        amrdata.refinement_ratios_t = config.geoclaw['refinement_ratios']

        # Specify type of each aux variable in amrdata.auxtype.
        # This must be a list of length maux, each element of which is one of:
        #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).
        amrdata.aux_type = ['center','capacity','yleft']

        # Flag using refinement routine flag2refine rather than richardson error
        amrdata.flag_richardson = False    # use Richardson?
        amrdata.flag2refine = True

        # steps to take on each level L between regriddings of level L+1:
        amrdata.regrid_interval = 3

        # width of buffer zone around flagged points:
        # (typically the same as regrid_interval so waves don't escape):
        amrdata.regrid_buffer_width  = 2

        # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
        # (closer to 1.0 => more small grids may be needed to cover flagged cells)
        amrdata.clustering_cutoff = 0.700000

        # print info about each regridding up to this level:
        amrdata.verbosity_regrid = 0

        #  ----- For developers -----
        # Toggle debugging print statements:
        amrdata.dprint = False      # print domain flags
        amrdata.eprint = False      # print err est flags
        amrdata.edebug = False      # even more err est flags
        amrdata.gprint = False      # grid bisection/clustering
        amrdata.nprint = False      # proper nesting output
        amrdata.pprint = False      # proj. of tagged points
        amrdata.rprint = False      # print regridding summary
        amrdata.sprint = False      # space/memory output
        amrdata.tprint = True       # time step reporting each level
        amrdata.uprint = False      # update/upbnd reporting

        # More AMR parameters can be set -- see the defaults in pyclaw/data.py

        # ---------------
        # Regions:
        # ---------------
        rundata.regiondata.regions = []
        # to specify regions of refinement append lines of the form
        #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]
        for key,region in config.regions.items():
            rundata.regiondata.regions.append([maxlevel,maxlevel]+region)

        # ---------------
        # FGMax:
        # ---------------
        # == fgmax.data values ==
        fgmax_files = rundata.fgmax_data.fgmax_files
        # for fixed grids append to this list names of any fgmax input files
        fgmax_files.append(config.fgmax['fgmax_grid_path'])
        rundata.fgmax_data.num_fgmax_val = 1

        #------------------------------------------------------------------
        # Adjoint specific data:
        #------------------------------------------------------------------
        # Also need to set flagging method and appropriate tolerances above

        adjointdata = rundata.adjointdata
        adjointdata.use_adjoint = True

        # location of adjoint solution, must first be created:
        # make symlink for adjoint
        adjointdata.adjoint_outdir = config.geoclaw['adjoint_outdir']

        # time period of interest:
        adjointdata.t1 = rundata.clawdata.t0
        adjointdata.t2 = rundata.clawdata.tfinal

        if adjointdata.use_adjoint:
            # need an additional aux variable for inner product:
            rundata.amrdata.aux_type.append('center')
            rundata.clawdata.num_aux = len(rundata.amrdata.aux_type)
            adjointdata.innerprod_index = len(rundata.amrdata.aux_type)

        return rundata

    return setrun

def write_setrun(config_path=None):
    with open('setrun.py','w') as f:
        f.write("from tsunamibayes.setrun import make_setrun\n")
        f.write("from tsunamibayes.utils import Config\n\n")
        f.write("config = Config()\n")
        f.write("config.read('defaults.cfg')\n")
        if config_path:
            f.write("config.read('{}')\n".format(config_path))
        f.write("setrun = make_setrun(config)\n\n")
        f.write("if __name__ == '__main__':\n")
        f.write("   rundata = setrun()\n")
        f.write("   rundata.write()")
