import numpy as np
import pandas as pd
from tsunamibayes import BasePrior
from tsunamibayes.utils import calc_length, calc_width, out_of_bounds


class SulawesiPrior(BasePrior):
    """The child class of Base Prior that creates a prior distribution,
    specifically for the Sulawesi 1820 event.
    """
    def __init__(
        self,
        f_latlon,
        f_mag,
        f_delta_logl,
        f_delta_logw,
        f_depth_offset,
        f_dip_offset,
        f_strike_offset,
        f_rake_offset,
        w_latlon,
        w_mag,
        w_delta_logl,
        w_delta_logw,
        w_depth_offset,
        w_dip_offset,
        w_strike_offset,
        w_rake_offset
    ):
        """Initializes all the necessary variables for the subclass.

        Parameters
        ----------
        latlon : LatLonPrior Object
            Contains attirbutes fault and depth_dist, with methods logpdf,
            pdf, and rvs.
        mag : scipy.stats rv_frozen object
            A truncated continous random variable describing the sample's
            magnitude with fixed parameters shape, location and scale.
        delta_logl : scipy.stats rv_frozen object
            The continous random variable describing the sample's standard
            deviation for the log of the length, also with fixed parameters.
        delta_logw : scipy.stats rv_frozen object
            The continous random variable describing the sample's standard
            deviation for the log of the width, also with fixed parameters.
        depth_offset : scipy.stats rv_frozen object
            The continous random variable describing the sample's depth offset,
            also with fixed parameters.
        """
        self.f_latlon = f_latlon
        self.f_mag = f_mag
        self.f_delta_logl = f_delta_logl
        self.f_delta_logw = f_delta_logw
        self.f_depth_offset = f_depth_offset
        self.f_dip_offset = f_dip_offset
        self.f_rake_offset = f_rake_offset
        self.f_strike_offset = f_strike_offset

        self.w_latlon = w_latlon
        self.w_mag = w_mag
        self.w_delta_logl = w_delta_logl
        self.w_delta_logw = w_delta_logw
        self.w_depth_offset = w_depth_offset
        self.w_dip_offset = w_dip_offset
        self.w_rake_offset = w_rake_offset
        self.w_strike_offset = w_strike_offset


    def logpdf(self,sample):
        """Computes the log of the probability density function. Adds
        together the logs of all the probability density functions for each
        of sample's attributes.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float
            values.

        Returns
        -------
        lpdf : float
            The log of the probability density function for the sample.
        """
        f_lat = sample["flores_latitude"]
        w_lat = sample["walanae_latitude"]
        f_lon = sample["flores_longitude"]
        w_lon = sample["walanae_latitude"]
        f_mag = sample["flores_magnitude"]
        w_mag = sample["walanae_magnitude"]
        f_delta_logl = sample["flores_delta_logl"]
        w_delta_logl = sample["walanae_delta_logl"]
        f_delta_logw = sample["flores_delta_logw"]
        w_delta_logw = sample["walanae_delta_logw"]
        f_depth_offset = sample["flores_depth_offset"]
        w_depth_offset = sample["walanae_depth_offset"]
        f_dip_offset = sample['flores_dip_offset']
        w_dip_offset = sample["walanae_dip_offset"]
        f_rake_offset = sample['flores_rake_offset']
        w_rake_offset = sample['walanae_rake_offset']
        f_strike_offset = sample['flores_strike_offset']
        w_strike_offset = sample['walanae_strike_offset']

        print('\nCalculating the LOGPDF\n--------------\n')
        print('sample:',sample)
        lpdf = self.f_latlon.logpdf(sample)
        lpdf += self.w_latlon.logpdf(sample)
        print('lpdf latlon:', lpdf)
        lpdf += self.f_mag.logpdf(f_mag)
        lpdf += self.w_mag.logpdf(w_mag)
        print('lpdf mag:', lpdf)
        lpdf += self.f_delta_logl.logpdf(f_delta_logl)
        lpdf += self.w_delta_logl.logpdf(w_delta_logl)
        print('lpdf delta_logl:', lpdf)
        lpdf += self.f_delta_logw.logpdf(f_delta_logw)
        lpdf += self.w_delta_logw.logpdf(w_delta_logw)
        print('lpdf delta_logw:', lpdf)
        lpdf += self.f_depth_offset.logpdf(f_depth_offset)
        lpdf += self.w_depth_offset.logpdf(w_depth_offset)
        print('lpdf depth_offset:', lpdf)
        lpdf += self.f_dip_offset.logpdf(f_dip_offset)
        lpdf += self.w_dip_offset.logpdf(w_dip_offset)
        print('lpdf dip_offset:', lpdf)
        lpdf += self.f_rake_offset.logpdf(f_rake_offset)
        lpdf += self.w_rake_offset.logpdf(w_rake_offset)
        print('lpdf rake_offset:', lpdf)
        lpdf += self.f_strike_offset.logpdf(f_strike_offset)
        lpdf += self.w_strike_offset.logpdf(w_strike_offset)
        print('lpdf strike_offset:', lpdf)

        return lpdf

    def rvs(self):
        """Computes random variates for each of Sulawesi Prior's data members,
        then returns the organized set of random variates for each.

        Returns
        -------
        rvs : pandas Series
            A series containing axis labels for each of banda prior's
            variables, with the associated random variates (float values) for
            each parameter.
        """
        latlon = self.latlon.rvs()
        mag = self.mag.rvs()
        delta_logl = self.delta_logl.rvs()
        delta_logw = self.delta_logw.rvs()
        depth_offset = self.depth_offset.rvs()
        params = np.array(latlon+[mag,delta_logl,delta_logw,depth_offset])
        columns = [
            'latitude',
            'longitude',
            'magnitude',
            'delta_logl',
            'delta_logw',
            'depth_offset',
            'dip_offset',
            'strike_offset',
            'rake_offset'
        ]
        return pd.Series(params, columns)

class LatLonPrior(BasePrior):
    def __init__(self,fault,depth_dist,fault_id):
        """Initializes all the necessary variables for the subclass.

        Parameters
        ----------
        fault :  GridFault Object
            From the tsunamibayes module in fault.py
        depth_dist : scipy.stats rv_frozen object
            The truncated continous random variable describing the depth
            with fixed shape, location and scale parameters.
        """
        self.fault = fault
        self.depth_dist = depth_dist
        self.fault_id = fault_id

    def logpdf(self,sample):
        """Checks to insure that the sample's subfaults are not out of bounds,
        then computes the log of the depth distribution's probability density
        function evaluated at the sample's depth.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float
            values.

        Returns
        -------
        NINF -or- logpdf : float
            Returns negative inifity if out-of-bounds,
            otherwise returns the log of the probability density function
            for the depth distribution evaluated at the sample's depth.
        """
        
        # compute the logpdf for each fault seperately based on the fault id
        if self.fault_id == 0:
            length = calc_length(sample['flores_magnitude'],sample['flores_delta_logl'])
            width = calc_width(sample['flores_magnitude'],sample['flores_delta_logw'])
            
            subfault_params = self.fault.subfault_split_RefCurve(
                lat = sample['flores_latitude'],
                lon = sample['flores_longitude'],
                length = length,
                width = width,
                slip = 1,
                depth_offset = sample['flores_depth_offset'],
                dip_offset = sample['flores_dip_offset'],
                rake_offset = sample['flores_rake_offset']
            )


        if self.fault_id == 1:
            length = calc_length(sample['walanae_magnitude'],sample['walanae_delta_logl'])
            width = calc_width(sample['walanae_magnitude'],sample['walanae_delta_logw'])

            subfault_params = self.fault.subfault_split_RefCurve(
                lat = sample['walanae_latitude'],
                lon = sample['walanae_longitude'],
                length = length,
                width = width,
                slip = 1,
                depth_offset = sample['walanae_depth_offset'],
                dip_offset = sample['walanae_dip_offset'],
                rake_offset = sample['walanae_rake_offset']
            )
            

        # compute subfaults (for out-of-bounds calculation)
        # flores_length = calc_length(sample['flores_magnitude'],sample['flores_delta_logl'])
        # walanae_length = calc_length(sample['walanae_magnitude'],sample['walanae_delta_logl'])
        # flores_width = calc_width(sample['flores_magnitude'],sample['flores_delta_logw'])
        # walanae_width = calc_width(sample['walanae_magnitude'],sample['walanae_delta_logl'])
        
        # find subfault params for each seperate fault and then hstack them into a single thing
        
        #flores_subfault_params = self.fault.subfault_split_RefCurve(
        #    lat = sample['flores_latitude'],
        #    lon = sample['flores_longitude'],
        #    length = flores_length,
        #    width = flores_width,
        #    slip = 1,
        #    depth_offset = sample['flores_depth_offset'],
        #    dip_offset = sample['flores_dip_offset'],
        #    rake_offset = sample['flores_rake_offset']
        #)
        #walanae_subfault_params = self.fault.subfault_split_RefCurve(
        #    lat = sample['walanae_latitude'],
        #    lon = sample['walanae_longitude'],
        #    length = walanae_length,
        #    width = walanae_width,
        #    slip = 1,
        #    depth_offset = sample['walanae_depth_offset'],
        #    dip_offset = sample['walanae_dip_offset'],
        #    rake_offset = sample['walanae_dip_offset']
        #)

        # print("this is flores_subfault_params")
        # print(flores_subfault_params)
        
        # print("this is walanae_subfault_params")
        # print(walanae_subfault_params)

        # Hstack Flores and Walanae subfault params
        #subfault_params = pd.concat([flores_subfault_params, walanae_subfault_params], axis=1)

        if subfault_params.isnull().values.any():
            return np.NINF

        if out_of_bounds(subfault_params,self.fault.model_bounds):
            return np.NINF
        else:
            if self.fault_id == 0:
                depth = (self.fault.depth_map(sample['flores_latitude'], sample['flores_longitude'])
                         + 1000*sample['flores_depth_offset'])
            if self.fault_id == 1:
                depth = (self.fault.depth_map(sample['walanae_latitude'], sample['walanae_longitude'])
                         + 1000*sample['walanae_depth_offset'])
            
            return self.depth_dist.logpdf(depth)

    def pdf(self,sample):
        """Checks to insure that the sample's subfaults are not out of bounds,
        then evaluates the depth distribution's probability density function
        at the sample's depth.

        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float
            values.

        Returns
        -------
        pdf : float
            The value of the probability density function for the depth
            distribution evaluated at the depth of the sample.
        """
        # compute subfaults (for out-of-bounds calculation)
        length = calc_length(sample['magnitude'],sample['delta_logl'])
        width = calc_width(sample['magnitude'],sample['delta_logw'])

        subfault_params = self.fault.subfault_split_RefCurve(
            lat = sample['latitude'],
            lon = sample['longitude'],
            length = length,
            width = width,
            slip = 1,
            depth_offset = sample['depth_offset'],
            dip_offset = sample['dip_offset'],
            rake_offset = sample['rake_offset']
        )

        if subfault_params.isnull().values.any():
            return 0
        if out_of_bounds(subfault_params,self.fault.model_bounds):
            return 0
        else:
            depth = (
                self.fault.depth_map(sample['latitude'],sample['longitude'])
                + 1000*sample['depth_offset']
            )
            return self.depth_dist.pdf(depth)

    def rvs(self):
        """Produces two random variate values for latitude and longitude
        based on a random variate of the depth distribution.

        Returns
        -------
        lat, lon : (list) of floats
            The random variates for latitude and longitude within the fault's
            bounds.
        """
        d = self.depth_dist.rvs()
        I,J = np.nonzero(
            (d - 500 < self.fault.depth)
            & (self.fault.depth < d + 500)
        )
        idx = np.random.randint(len(I))
        return [self.fault.lat[I[idx]],self.fault.lon[J[idx]]]
