import numpy as np
from tsunamibayes import BasePrior
from tsunamibayes.utils import calc_length, calc_width, out_of_bounds

class BandaPrior(BasePrior):
    """The child class of Base Prior that creates a prior distribution,
    specifically for the Banda 1852 event."""
    def __init__(self,latlon,mag,delta_logl,delta_logw,depth_offset):
        """Initializes all the necessary variables for the subclass.
        
        Parameters
        ----------
        latlon : LatLonPrior Object
            Contains attirbutes fault and depth_dist, with methods logpdf, pdf, and rvs.
        mag : scipy.stats rv_frozen object
            A truncated continous random variable describing the sample's 
            magnitude with fixed parameters shape, location and scale.
        delta_logl : scipy.stats rv_frozen object
            The continous random variable describing the sample's standard deviation for
            the log of the length, also with fixed parameters. 
        delta_logw : scipy.stats rv_frozen object
            The continous random variable describing the sample's standard deviation for
            the log of the width, also with fixed parameters. 
        depth_offset : scipy.stats rv_frozen object
            The continous random variable describing the sample's depth offset, 
            also with fixed parameters. 
        """
        self.latlon = latlon
        self.mag = mag
        self.delta_logl = delta_logl
        self.delta_logw = delta_logw
        self.depth_offset = depth_offset

    def logpdf(self,sample):
        """Computes the log of the probability density function. Adds
        together the logs of all the probability denisty functions for each
        of sample's attributes.
        
        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values. 
        
        Returns
        -------
        lpdf : float
            The log of the probability density function for the sample.
        """ 
        lat    = sample["latitude"]
        lon    = sample["longitude"]
        mag    = sample["magnitude"]
        delta_logl = sample["delta_logl"]
        delta_logw = sample["delta_logw"]
        depth_offset = sample["depth_offset"]

        lpdf = self.latlon.logpdf(sample)
        lpdf += self.mag.logpdf(mag)
        lpdf += self.delta_logl.logpdf(delta_logl)
        lpdf += self.delta_logw.logpdf(delta_logw)
        lpdf += self.depth_offset.logpdf(depth_offset)

        return lpdf

    def rvs(self):
        """Computes random variates for each of Banda Prior's data members,
        then returns the organized set of random variates for each.
        
        Returns
        -------
        rvs : pandas Series
            A series containing axis labels for each of banda prior's variables,
            with the associated random variates (float values) for each parameter. 
        """
        latlon = self.latlon.rvs()
        mag = self.mag.rvs()
        delta_logl = self.delta_logl.rvs()
        delta_logw = self.delta_logw.rvs()
        depth_offset = self.depth_offset.rvs()
        params = np.array(latlon+[mag,delta_logl,delta_logw,depth_offset])
        return pd.Series(params,["latitude",
                                 "longitude",
                                 "magnitude",
                                 "delta_logl",
                                 "delta_logw",
                                 "depth_offset"])

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
        then computes the log of the depth distribution's probability density function 
        evaluated at the sample's depth. 
        
        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values.

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
        subfault_params = self.fault.subfault_split(sample['latitude'],
                                                    sample['longitude'],
                                                    length,
                                                    width,
                                                    1,
                                                    sample['depth_offset'])

        if out_of_bounds(subfault_params,self.fault.bounds):
            return np.NINF
        else:
            depth = self.fault.depth_map(sample['latitude'],sample['longitude']) + 1000*sample['depth_offset']
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
            'delta_logw', and 'depth_offset' with their associated float values.

        Returns
        -------
        pdf : float
            The value of the probability density function for the depth distribution
            evaluated at the depth of the sample. 
        """
        # compute subfaults (for out-of-bounds calculation)
        length = calc_length(sample['magnitude'],sample['delta_logl'])
        width = calc_width(sample['magnitude'],sample['delta_logw'])
        subfault_params = self.fault.subfault_split(sample['latitude'],
                                                    sample['longitude'],
                                                    length,
                                                    width,
                                                    1,
                                                    sample['depth_offset'])

        if out_of_bounds(subfault_params,self.fault.bounds):
            return 0
        else:
            depth = self.fault.depth_map(sample['latitude'],sample['longitude']) + 1000*sample['depth_offset']
            return self.depth_dist.pdf(depth)

    def rvs(self):
        """Produces two random variate values for latitude and longitude
        based on a random variate of the depth distribution.
        
        Returns
        -------
        lat, lon : (list) of floats
            The random variates for latitude and longitude within the fault's bounds.
        """
        d = self.depth_dist.rvs()
        I,J = np.nonzero((d - 500 < self.fault.depth)&(self.fault.depth < d + 500))
        idx = np.random.randint(len(I))
        return [self.fault.lat[I[idx]],self.fault.lon[J[idx]]]
