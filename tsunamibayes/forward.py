import os
import json
import numpy as np
import pandas as pd
from .fault import BaseFault
from .maketopo import write_dtopo
from . import models

class BaseForwardModel:
    """A parent class giving the outline for other subclasses to run the forward model."""
    def __init__(self,gauges):
        """Initializes the necessary variables for the base class.
        
        Parameters
        ----------
        gauges : (list) of Gauge objects
            The list of gauge objects containing locations data distribution functions for the 
            arrival, height, and inundation for each gauge location.
        """
        self.gauges = gauges
        self.model_output_cols = [gauge.name + " " + obstype
                                  for gauge in gauges
                                  for obstype in gauge.obstypes
                                  if obstype in self.obstypes]

    def run(self,model_params,verbose=False):
        raise NotImplementedError("run() must be implemented in classes "
                                  "inheriting from BaseForwardModel")

    def llh(self,model_output,verbose=False):
        raise NotImplementedError("run() must be implemented in classes "
                                  "inheriting from BaseForwardModel")

class CompositeForwardModel(BaseForwardModel):
    def __init__(self,submodels):
        """Extracts all the gauge objects found within each element of the submodels
        lists and pases those gauges into the constructor of the parent class.
        
        Parameters
        ----------
        submodels : (list) of ForwardModel objects
            A list of GeoClawForwardModel or TestForwardModel objects that each 
            contain their own respective gauges data member.
        """
        self.submodels = submodels
        self.obstypes = [obstype for submodel in submodels
                         for obstype in submodel.obstypes]
        gauges = list()
        for submodel in submodels:
            gauges.extend(gauge for gauge in submodel.gauges if gauge not in gauges)
        super().__init__(gauges)

    def run(self,model_params,verbose=False):
        """Runs each ForwardModel object in their respective subclass, then combines the
        submodels' outputs into a single series.

        Parameters
        ----------
        model_params : dict
            The dictionary whose keys are the okada parameters: 'latitude', 'longitude', 
            'depth_offset', 'strike','length','width','slip','depth','dip','rake', 
            and whose associated values are floats. 
        verbose : bool
            Flag for verbose output, optional. Default is False.

        Returns
        -------
        all_model_output : pandas Series
            The combined/concatenated series containing the model's output for all the submodels.
        """
        model_output = list()
        for submodel in submodels:
            model_output.append(submodel.run(model_params,verbose))
        return pd.concat(model_output)

    def llh(self,model_output,verbose=False):
        """Returns the total loglikelihood for all the submodels based on the model output.
        
        Parameters
        ----------
        model_output : pandas Series
            A pandas series whose axes labels are the cominations of the scenario's gauges 
            names plus 'arrivals', 'height', or 'inundation'. The associated values are floats. 
        verbose : bool
            Flag for verbose output, optional. Default is False.

        Returns
        -------
        llh : float
            The loglikelihood of the given sample's model output. A measure of how likely
            that model output is when compared to the actual observation data at each gauge.
        """
        llh = 0
        for submodel in submodels:
            llh += submodel.llh(model_output,verbose)
        return llh

class GeoClawForwardModel(BaseForwardModel):
    obstypes = ['arrival','height','inundation']
    def __init__(self,gauges,fault,fgmax_params,dtopo_path):
        """Initializes all the necessary variables for the GeoClawForwardModel subclass.
        
        Parameters
        ----------
        gauges : (list) of Gauge objects
            The list of gauge objects containing location data and distribution functions for 
            the arrival, height, and inundation parameters for each gauge location.
        fault : Fault object
            Ususally a previously constructed GridFault object. 
        fgmax_params : dict
            The dictionary containing information for the fixed grid maximums (see GeoClaw pack) 
            with keys 'min_level_check', 'fgmax_grid_path', 'valuemax_path',
            'aux1_path',  as well as others specified in the scenario's defaults.cfg file. 
            Associated values are a variety of strings for file paths & floats. 
        dtopo_path : string
            The name of the .tt3 file location containing the GeoClaw information of dtopo.
            Used later on to create and write dtopo file.
        """
        super().__init__(gauges)
        self.fault = fault
        self.dtopo_path = dtopo_path
        self.fgmax_params = fgmax_params
        self.fgmax_grid_path = fgmax_params['fgmax_grid_path']
        self.valuemax_path = fgmax_params['valuemax_path']
        self.aux1_path = fgmax_params['aux1_path']
        self.write_fgmax_grid(self.gauges,self.fgmax_params)

        # clean up directory
        os.system('make clean')

    def run(self,model_params,verbose=False):
        """Runs the forward model for a specified sample's model parameters, 
        then returns the computed arrival times, wave heights, and inundation distances
        for each gauge location, as appropriate. 
        Essentially, this function works 'backwards' from a sample model of the fault rupture's 
        parameters to compute what the gauge's observations would have been like for that given 
        earthquake event sample.
        
        Parameters
        ----------
        model_params : dict
            The sample's model parameters. The dictionary whose keys are the okada parameters: 
            'latitude', 'longitude', 'depth_offset', 'strike','length','width','slip','depth',
            'dip','rake', and whose associated values are floats.
        verbose : bool
            Flag for verbose output, optional. Default is False.

        Returns
        -------
        model_output : pandas Series
            A pandas series whose axes labels are the cominations of the scenario's gauges 
            names plus 'arrivals', 'height', or 'inundation'. The associated values are floats. 
        """
        # split fault into subfaults aligning to fault zone geometry
        subfault_params = self.fault.subfault_split(model_params['latitude'],
                                                    model_params['longitude'],
                                                    model_params['length'],
                                                    model_params['width'],
                                                    model_params['slip'],
                                                    model_params['depth_offset'],
                                                    model_params['rake'])
        if verbose : print("Parameters of the subfaults when running the forward model : "); print(subfault_params)

        # create and write dtopo file
        write_dtopo(subfault_params,self.fault.bounds,self.dtopo_path,verbose)

        # clear .output
        os.system('rm .output')

        # run GeoClaw
        os.system('make .output')

        # load fgmax and bathymetry data
        if verbose : print("Loading fgmax and bathymetry data from files {} and {}".format(self.valuemax_path,self.aux1_path))
        fgmax_data = np.loadtxt(self.valuemax_path)
        bath_data  = np.loadtxt(self.aux1_path)

        # this is the arrival time of the first wave, not the maximum wave
        # converting from seconds to minutes
        arrival_times = fgmax_data[:, -1] /60

        max_heights = fgmax_data[:, 3]
        bath_depth = bath_data[:, -1]

        # these are locations where the wave never reached the gauge...
        #FIXME add a flag here to show these locations
        max_heights[max_heights < 1e-15] = -9999
        max_heights[np.abs(max_heights) > 1e15] = -9999
        if verbose : 
            print("The wave never reached the gauge in the following location(s)...")
            print(np.abs(max_heights) > 1e15)


        bath_depth[max_heights == 0] = 0
        wave_heights = max_heights + bath_depth

        model_output = pd.Series(dtype='float64')
        for i,gauge in enumerate(self.gauges):
            if 'arrival' in gauge.obstypes:
                model_output[gauge.name+' arrival'] = arrival_times[i]
            if 'height' in gauge.obstypes:
                model_output[gauge.name+' height'] = wave_heights[i]
            if 'inundation' in gauge.obstypes:
                model_output[gauge.name+' inundation'] = models.inundation(wave_heights[i],
                                                                           gauge.beta,
                                                                           gauge.n)

        return model_output

    def llh(self,model_output,verbose=False):
        """Comptues the loglikelihood of the forward model's ouput. 
        Compares the model outputs with the acutal gauge observation distributions for 
        arrival time, wave height, and inundation distance, and calculates the log of 
        the probability distribution function at each point of the model's output for the gagues.
        Finally, sums together all the log values for each observation point and type.

        Parameters
        ----------
        model_output : pandas Series
            A pandas series whose axes labels are the cominations of the scenario's gauges 
            names plus 'arrivals', 'height', or 'inundation'. The associated values are floats. 
        verbose : bool
            Flag for verbose output, optional. Default is False.
            If set to true, will output a Gauge Log with each gauges computed arrival time,
            wave height, and inundation distance with the associted loglikelihood.

        Returns
        -------
        llh : float
            The loglikelihood of the given sample's model output. A measure of how likely
            that model output is when compared to the actual observation data at each gauge.
        """
        llh = 0
        if verbose: print("Gauge Log\n---------{Location, model output, loglikelihood}\n------------")
        for gauge in self.gauges:
            if verbose: print(gauge.name)
            if 'arrival' in gauge.obstypes:
                arrival_time = model_output[gauge.name+' arrival']
                log_p = gauge.dists['arrival'].logpdf(arrival_time)
                llh += log_p
                if verbose: print("arrival:    {:.3f}\tllh: {:.3e}".format(arrival_time,log_p))

            if 'height' in gauge.obstypes:
                wave_height = model_output[gauge.name+' height']
                if np.abs(wave_height) > 999999999: log_p = np.NINF
                else: log_p = gauge.dists['height'].logpdf(wave_height)
                llh += log_p
                if verbose: print("height:     {:.3f}\tllh: {:.3e}".format(wave_height,log_p))

            if 'inundation' in gauge.obstypes:
                inundation = model_output[gauge.name+' inundation']
                log_p = gauge.dists['inundation'].logpdf(inundation)
                llh += log_p
                if verbose: print("inundation: {:.3f}\tllh: {:.3e}".format(inundation,log_p))
        return llh

    def write_fgmax_grid(self,gauges,fgmax_params):
        """Writes a file to store a specific scenario's parameters for the 
        fixed grid maximum monitoring feature in GeoClaw.
        After it is written, the file at the specific path will contain values for each
        fixed grid parameter, the total number of observations from the gauges, and the 
        gauge location information.
        
        Parameters
        ----------
        gauges : (list) of Gauge objects
            The list of gauge objects containing locations data distribution functions for the 
            arrival, height, and inundation for each gauge location.  
        fgmax_params : dict
            The dictionary containing information for the fixed grid maximums (see GeoClaw pack) 
            with keys 'min_level_check', 'fgmax_grid_path', 'valuemax_path',
            'aux1_path', as well as others specified in the scenario's defaults.cfg file. 
            Associated values are a variety of strings for file paths & floats. 
        """
        npts = sum(1 for gauge in gauges if
                   any(obstype in self.obstypes for obstype in gauge.obstypes))

        with open(fgmax_params['fgmax_grid_path'],'w') as f:
            f.write(str(fgmax_params['tstart_max'])+'\t# tstart_max\n')
            f.write(str(fgmax_params['tend_max'])+'\t# tend_max\n')
            f.write(str(fgmax_params['dt_check'])+'\t# dt_check\n')
            f.write(str(fgmax_params['min_level_check'])+'\t# min_level_check\n')
            f.write(str(fgmax_params['arrival_tol'])+'\t# arrival_tol\n')
            f.write('0'+'\t# point_style\n')
            f.write(str(npts)+'\t# n_pts\n')
            for gauge in gauges:
                if any(obstype in self.obstypes for obstype in gauge.obstypes):
                    f.write(str(gauge.lon)+' '+str(gauge.lat))
                    f.write('\t# '+gauge.name+'\n')

class TestForwardModel(BaseForwardModel):
    obstypes = ['power']
    def run(self,model_params,verbose=False):
        """Runs a test for the forward model. For each gauge with an obervation type
        'power', the function computes the log of the sample's fault rupture area.

        Parameters
        ----------
        model_params : dict
            The dictionary containing the sample's parameters and  whose keys are 
            the okada parameters, and whose associated values are floats. 
            Here, only the 'length' and 'width' parameters are used to calculate fault rupture area.
        verbose : bool
            Flag for verbose output, optional. Default is False.
        
        Returns
        -------
        d : dict
            The dictionary whose keys are the 'gauge's name' + 'power', and whose
            values are the computed powers from the model's length and width.
        """
        d = {}
        for gauge in self.gauges:
            if 'power' in gauge.obstypes:
                d[gauge.name+' power'] = np.log(model_params['length']*model_params['width'])
        return d

    def llh(self,model_output,verbose=False):
        """Computes the loglikelihood of the output results from running the Test forward model.
        
        Parameters
        ----------
        model_params : dict
            The dictionary containing the sample's parameters and  whose keys are 
            the okada parameters, and whose associated values are floats. 
            Here, only the 'length' and 'width' parameters are used to calculate fault rupture area.
        verbose : bool
            Flag for verbose output, optional. Default is False.

        Returns
        -------
        llh : float
            The loglikelihood of the given sample's model output. A measure of how likely
            that model output is when compared to the actual observation data at each gauge.
        """
        llh = 0
        for gauge in self.gauges:
            if 'power' in gauge.obstypes:
                llh += gauge.dists['power'].logpdf(model_output[gauge.name+' power'])
        return llh
