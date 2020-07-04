import numpy as np
import scipy.stats as stats
import json
import tsunamibayes as tb
from gauges import build_gauges

def setup(config):

    # Banda Arc fault object
    arrays = np.load(config.fault['grid_data_path'])
    fault = tb.GridFault(bounds=config.model_bounds,**arrays)

    # load gauges
    gauges = build_gauges()

    # Forward model
    config.fgmax['min_level_check'] = len(config.geoclaw['refinement_ratios'])
    return tb.GeoClawForwardModel(gauges,fault,config.fgmax,
                                           config.geoclaw['dtopo_path'])

if __name__ == "__main__":
    import os
    import pandas as pd
    from tsunamibayes.utils import parser, Config
    from tsunamibayes.setrun import write_setrun

    # parse command line arguments
    args = parser.parse_args()

    # break if both resume and sequential reinit flags are set
    if args.resume_dir and args.seq_reinit_dir:
        raise ValueError("flags '-r' and '-s' cannot both be set")

    # load defaults and config file
    if args.verbose: print("Reading defaults.cfg")
    config = Config()
    config.read('defaults.cfg')

    # # write setrun.py file
    # if args.verbose: print("Writing setrun.py")
    # write_setrun(args.config_path)

    # copy Makefile
    if args.verbose: print("Copying Makefile")
    makefile_path = tb.__file__[:-11]+'Makefile'
    os.system("cp {} Makefile".format(makefile_path))

    # build forward model
    forward_model = setup(config)
    model_params = pd.Series(dtype='float64')
    model_params['latitude'] = -6.249
    model_params['longitude'] = 131.14
    model_params['length'] = 640000.0
    model_params['width'] = 200000.0
    model_params['slip'] = 2.0
    model_params['depth_offset'] = 0.0
    model_params['rake'] = 90.0
    model_output = forward_model.run(model_params)
    llh = forward_model.llh(model_output,verbose=True)
    print(llh)
