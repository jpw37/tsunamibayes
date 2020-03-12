
import json
from prior import BasePrior
from fault import Fault
from scenario import BaseScenario
from forward import ForwardGeoClaw
from gauge import load_gauges

#Load in global parameters
with open('parameters.txt') as json_file:
    global_params = json.load(json_file)


name = 'DEFAULT'

prior_obj = BasePrior(global_params['depth_mu'],
                      global_params['depth_std'],
                      global_params['mindepth'],
                      global_params['maxdepth'],
                      global_params['minlon'],
                      global_params['mag_b'],
                      global_params['mag_loc'],
                      global_params['deltalogl'],
                      global_params['deltalogw'],
                      global_params['deltadepth'] )
fault_obj = Fault(global_params['depth_mu'], name)
gauges = load_gauges(global_params['gauge_file_path'])
model = ForwardGeoClaw(gauges)

scenario_obj = BaseScenario(name,prior_obj,model,gauges, fault_obj)
scenario_obj.initialize_chain(u0=None,method=None,**kwargs)
scenario_obj.sample(global_params['nsamples'])
