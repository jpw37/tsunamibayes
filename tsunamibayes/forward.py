import json
import numpy as np
import pandas as pd
from .fault import BaseFault
from .maketopo import write_dtopo

class BaseForwardModel:
    def __init__(self,gauges):
        self.gauges = gauges
        self.model_output_cols = [gauge.name + " " + obstype
                                  for gauge in gauges
                                  for obstype in gauge.obstypes
                                  if obstype in self.obstypes]

    def run(self,model_params,verbose=False):
        raise NotImplementedError("run() must be implemented in classes \
                                  inheriting from BaseForwardModel")

    def llh(self,model_output,verbose=False):
        raise NotImplementedError("run() must be implemented in classes \
                                  inheriting from BaseForwardModel")

class CompositeForwardModel(BaseForwardModel):
    def __init__(self,submodels,gauges):
        self.submodels = submodels
        self.obstypes = [obstype for submodel in submodels
                         for obstype in submodel.obstypes]
        super().__init__(gauges)

    def run(self,model_params,verbose=False):
        model_output = list()
        for submodel in submodels:
            model_output.append(submodel.run(model_params,verbose))
        return pd.concat(model_output)

    def llh(self,model_output,verbose=False):
        llh = 0
        for submodel in submodels:
            llh += submodel.llh(model_output,verbose)
        return llh

class GeoClawForwardModel(BaseForwardModel):
    obstypes = ['arrival','height','inundation']
    def __init__(self,gauges,fault,dtopo_path,fgmax_grid_path,fgmax_out_path,bathy_path):
        if not isinstance(fault,BaseFault):
            raise TypeError("fault must be an instance of BaseFault or an \
                            inherited class.")
        super.__init__(gauges)
        self.fault = fault
        self.dtopo_path = dtopo_path
        self.fgmax_grid_path = fgmax_grid_path
        self.fgmax_out_path = fgmax_out_path
        self.bathy_path = bathy_path
        self.write_fgmax_grid(self.gauges,self.fgmax_grid_path)

    def run(self,model_params,verbose=False):
        """
        Run  Model(model_params)
        Read gauges
        Return observations (arrival times, Wave heights)
        """
        subfault_params = self.fault.subfault_split(model_params['latitude'],
                                                    model_params['longitude'],
                                                    model_params['length'],
                                                    model_params['width'],
                                                    model_params['slip'],
                                                    model_params['depth_offset'])

        write_dtopo(subfault_params,self.fault.bounds,self.dtopo_path,verbose)
        os.system('rm .output')
        os.system('make .output')

        fgmax_data = np.loadtxt(self.fgmax_out_path)
        bath_data  = np.loadtxt(self.bathy_path)

        # this is the arrival time of the first wave, not the maximum wave
        # converting from seconds to minutes
        arrival_times = fgmax_data[:, -1] /60

        max_heights = fgmax_data[:, 3]
        bath_depth = bath_data[:, -1]

        # these are locations where the wave never reached the gauge...
        max_heights[max_heights < 1e-15] = -9999
        max_heights[np.abs(max_heights) > 1e15] = -9999

        bath_depth[max_heights == 0] = 0
        wave_heights = max_heights + bath_depth

        model_output = pd.Series(dtype='float64')
        for i,gauge in enumerate(gauges):
            if 'arrival' in gauge.obstypes:
                model_output[gauge.name+' arrival'] = arrival_times[i]
            if 'height' in gauge.obstypes:
                model_output[gauge.name+' height'] = wave_heights[i]
            if 'inundation' in gauge.obstypes:
                # put inundation model here
                pass

        return model_output

    def llh(self,model_output,verbose=False):
        llh = 0
        if verbose: print("Gauge Log\n---------")
        for gauge in gauges:
            if verbose: print(gauge.name)
            if 'arrival' in gauge.obstypes:
                arrival_time = model_output[gauge.name+' arrival']
                log_p = gauge.dists['arrival'].logpdf(arrival_time)
                llh += log_p
                if verbose: print("arrival: {:.3f}, llh: {:.3e}".format(arrival_time,log_p))

            if 'height' in gauge.obstypes:
                wave_height = model_output[gauge.name+' height']
                if np.abs(wave_height) > 999999999: log_p = np.NINF
                else: log_p = gauge.dists['height'].logpdf(wave_height)
                llh += log_p
                if verbose: print("height: {:.3f}, llh: {:.3e}".format(wave_height,log_p))

            if 'inundation' in gauge.obstypes:
                inundation = model_output[gauge.name+' inundation']
                log_p = gauge.dists['inundation'].logpdf(inundation)
                llh += log_p
                if verbose: print("inundation: {:.3f}, llh: {:.3e}".format(inundation,log_p))
        return llh

    def write_fgmax_grid(self,gauges,fgmax_grid_path):
        pass


class TestForwardModel(BaseForwardModel):
    obstypes = ['power']
    def run(self,model_params,verbose=False):
        d = {}
        for gauge in self.gauges:
            if 'power' in gauge.obstypes:
                d[gauge.name+' power'] = np.log(model_params['length']*model_params['width'])
        return d

    def llh(self,model_output,verbose=False):
        llh = 0
        for gauge in self.gauges:
            if 'power' in gauge.obstypes:
                llh += gauge.dists['power'].logpdf(model_output[gauge.name+' power'])
        return llh
