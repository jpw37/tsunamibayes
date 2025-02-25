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
        latlon,
        mag,
        delta_logl,
        delta_logw,
        depth_offset,
        dip_offset,
        strike_offset,
        rake_offset
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
        self.latlon = latlon
        self.mag = mag
        self.delta_logl = delta_logl
        self.delta_logw = delta_logw
        self.depth_offset = depth_offset
        self.dip_offset = dip_offset
        self.rake_offset = rake_offset
        self.strike_offset = strike_offset

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
        lat = sample["latitude"]
        lon = sample["longitude"]
        mag = sample["magnitude"]
        delta_logl = sample["delta_logl"]
        delta_logw = sample["delta_logw"]
        depth_offset = sample["depth_offset"]
        dip_offset = sample['dip_offset']
        rake_offset = sample['rake_offset']
        strike_offset = sample['strike_offset']

        print('\nCalculating the LOGPDF\n--------------\n')
        print('sample:',sample)
        lpdf = self.latlon.logpdf(sample)
        print('lpdf latlon:', lpdf)
        lpdf += self.mag.logpdf(mag)
        print('lpdf mag:', lpdf)
        lpdf += self.delta_logl.logpdf(delta_logl)
        print('lpdf delta_logl:', lpdf)
        lpdf += self.delta_logw.logpdf(delta_logw)
        print('lpdf delta_logw:', lpdf)
        lpdf += self.depth_offset.logpdf(depth_offset)
        print('lpdf depth_offset:', lpdf)
        lpdf += self.dip_offset.logpdf(dip_offset)
        print('lpdf dip_offset:', lpdf)
        lpdf += self.rake_offset.logpdf(rake_offset)
        print('lpdf rake_offset:', lpdf)
        lpdf += self.strike_offset.logpdf(strike_offset)
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
    def __init__(self,fault,depth_dist):
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
            return np.NINF
        if out_of_bounds(subfault_params,self.fault.model_bounds):
            return np.NINF
        else:
            depth = (
                self.fault.depth_map(sample['latitude'],sample['longitude'])
                + 1000*sample['depth_offset']
            )
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
