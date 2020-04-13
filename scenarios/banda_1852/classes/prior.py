from tsunamibayes import BasePrior

class BandaPrior(BasePrior):
    def __init__(self,fault):


    def logpdf(self,sample):
        raise NotImplementedError

    def rvs(self):
        raise NotImplementedError

class LatLonPrior(BasePrior):
    def __init__(self,fault,depth_dist):
        
