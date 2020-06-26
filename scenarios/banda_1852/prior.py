import numpy as np
from tsunamibayes import BasePrior
from tsunamibayes.utils import calc_length, calc_width, out_of_bounds

class BandaPrior(BasePrior):
    """The child class of Base Prior that creates a prior distribution,
    specifically for the Banda 1852 event."""
    def __init__(self,latlon,mag,delta_logl,delta_logw,depth_offset):
        self.latlon = latlon
        self.mag = mag
        self.delta_logl = delta_logl
        self.delta_logw = delta_logw
        self.depth_offset = depth_offset

    def logpdf(self,sample):
        """Compute the log of the probability density function
        
        Parameters
        ----------
        sample : dict
            The dictionary containing the arrays of information for a sample's
            lat, lon, mag, depth, and FIXME: it's delta info
        
        Returns
        -------
        lpdf : float
            The log of the probability density function.
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
        then returns a one-dimensional ndarray of the random variates
        with their repsecive axis-labels """
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
        self.fault = fault
        self.depth_dist = depth_dist

    def logpdf(self,sample):
        """Computes the log of the probability density function for the subfaults,
        with consideration for out-of-bounds calculation. Returns negative inifity if out-of-bounds,
        otherwise it computes the normal logpdf."""
        # compute subfaults (for out-of-bounds calculation)
        length = calc_length(sample['magnitude'],sample['delta_logl'])
        width = calc_width(sample['magnitude'],sample['delta_logw'])
        subfault_params = self.fault.subfault_split(sample['latitude'],
                                                    sample['longitude'],
                                                    length,
                                                    width,
                                                    1,
                                                    sample['depth_offset'])
        
        if subfault_params.isnull().values.any():
            return np.NINF
        if out_of_bounds(subfault_params,self.fault.bounds):
            return np.NINF
        else:
            depth = self.fault.depth_map(sample['latitude'],sample['longitude']) + 1000*sample['depth_offset']
            return self.depth_dist.logpdf(depth)

    def pdf(self,sample):
        """Similar to logpdf, this function computes the probability density function 
        for the subfaults, with consideration for out-of-bounds calculation. Returns 0 if out-of-bounds,
        otherwise it computes the normal pdf."""
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
        d = self.depth_dist.rvs()
        I,J = np.nonzero((d - 500 < self.fault.depth)&(self.fault.depth < d + 500))
        idx = np.random.randint(len(I))
        return [self.fault.lat[I[idx]],self.fault.lon[J[idx]]]
