from tsunamibayes import BasePrior
from tsuamibayes.utils import calc_length, calc_width, out_of_bounds

class BandaPrior(BasePrior):
    def __init__(self,latlon,mag,deltalogl,deltalogw,depth_offset):
        self.latlon = latlon
        self.mag = mag
        self.deltalogl = deltalogl
        self.deltalogw = deltalogw
        self.depth_offset = depth_offset

    def logpdf(self,sample):
        lat    = sample["latitude"]
        lon    = sample["longitude"]
        mag    = sample["magnitude"]
        deltalogl = sample["deltalogl"]
        deltalogw = sample["deltalogw"]
        depth_offset = sample["depth_offset"]

        lpdf = self.latlon.logpdf(sample)
        lpdf += self.mag.logpdf(mag)
        lpdf += self.deltalogl.logpdf(deltalogl)
        lpdf += self.deltalogw.logpdf(deltalogw)
        lpdf += self.depth_offset.logpdf(depth_offset)

        return lpdf

    def rvs(self):
        latlon = self.latlon.rvs()
        mag = self.mag.rvs()
        deltalogl = self.deltalogl.rvs()
        deltalogw = self.deltalogw.rvs()
        depth_offset = self.depth_offset.rvs()
        params = np.array(latlon+[mag,deltalogl,deltalogw,depth_offset])
        return pd.Series(params,["latitude",
                                 "longitude",
                                 "magnitude",
                                 "deltalogl",
                                 "deltalogw",
                                 "depth_offset"])

class LatLonPrior(BasePrior):
    def __init__(self,fault,depth_dist):
        self.fault = fault
        self.depth_dist = depth_dist

    def logpdf(self,sample):
        # compute subfaults (for out-of-bounds calculation)
        length = calc_length(sample['magnitude'],sample['delta_logl'])
        width = calc_width(sample['magnitude'],sample['delta_logw'])
        subfault_params = self.fault.subfault_split(sample['latitude'],
                                                    sample['longitude'],
                                                    length,
                                                    width,
                                                    slip=1,
                                                    sample['depth_offset'])

        if out_of_bounds(subfault_params,self.fault.bounds):
            return np.NINF
        else:
            depth = self.fault.depth_map(sample['latitude'],sample['longitude']) + 1000*sample['depth_offset']
            return self.depth_dist.logpdf(depth)

    def pdf(self,sample):
        # compute subfaults (for out-of-bounds calculation)
        length = calc_length(sample['magnitude'],sample['delta_logl'])
        width = calc_width(sample['magnitude'],sample['delta_logw'])
        subfault_params = self.fault.subfault_split(sample['latitude'],
                                                    sample['longitude'],
                                                    length,
                                                    width,
                                                    slip=1,
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
