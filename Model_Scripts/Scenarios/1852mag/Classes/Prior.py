"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

class LatLonPrior:
    """A class for a distance-based latitude/longitude prior. NOTE: this creates
    an unnormalized density function, not a true pdf"""

    def __init__(self,fault,mu,sigma,mindepth,maxdepth,minlon):
        """
        Parameters
        ----------
        fault : instance of Fault
            Fault object
        dist : scipy.rv_continuouis
            Distribution on depth
        """
        self.fault = fault
        self.mu = mu
        self.sigma = sigma
        self.mindepth = mindepth
        self.maxdepth = maxdepth
        self.minlon = minlon

    def logpdf(self,lat,lon,width,deltadepth):
        """Evaluates the logpdf of the prior"""
        if lon < self.minlon: return np.NINF
        try:
            depth = self.fault.depth_from_lat_lon(lat,lon)[0] + 1000*deltadepth #deltadepth im km to avoid singular covariance matrix
        except ValueError:
            return np.NINF
        mindepth = max(self.mindepth,.5*width*np.sin(np.deg2rad(self.fault.dip_from_lat_lon(lat,lon))))
        a, b = (mindepth - self.mu) / self.sigma, (self.maxdepth - self.mu) / self.sigma
        return stats.truncnorm.logpdf(depth,a,b,loc=self.mu,scale=self.sigma)

    def pdf(self,lat,lon,width,deltadepth):
        """Evaluates the pdf of the prior"""
        if lon < self.minlon: return np.NINF
        try:
            depth = self.fault.depth_from_lat_lon(lat,lon)[0] + 1000*deltadepth #deltadepth im km to avoid singular covariance matrix
        except ValueError:
            return np.NINF
        mindepth = max(self.mindepth,.5*width*np.sin(np.deg2rad(self.fault.dip_from_lat_lon(lat,lon))))
        a, b = (mindepth - self.mu) / self.sigma, (self.maxdepth - self.mu) / self.sigma
        return stats.truncnorm.pdf(depth,a,b,loc=self.mu,scale=self.sigma)

    def rvs(self):
        """Return a random point on the fault"""
        # idx = np.random.randint(len(self.fault.lonpts))
        # return [self.fault.latpts[idx],self.fault.lonpts[idx]]
        raise NotImplementedError

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self,latlon,mag,deltalogl,deltalogw,deltadepth):
        """
        Initialize the class with priors

        Parameters
        ----------
        latlon : instance of LatLonPrior
            Prior distribution on lat/lon

        mag : instance of scipy.stats.pareto
            Prior distribution on magnitude
        """
        self.priors = {"latlon":latlon,
                       "mag":mag,
                       "deltalogl":deltalogl,
                       "deltalogw":deltalogw,
                       "deltadepth":deltadepth}

    def logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        lat    = sample["Latitude"]
        lon    = sample["Longitude"]
        mag    = sample["Magnitude"]
        deltalogl = sample["DeltaLogL"]
        deltalogw = sample["DeltaLogW"]
        deltadepth = sample["DeltaDepth"]
        width = sample["Width"]

        # prior for longitude, latitude
        lpdf = self.priors["latlon"].logpdf(lat,lon,width,deltadepth)

        # Pareto prior on magnitude
        lpdf += self.priors["mag"].logpdf(mag)

        # Normal prior on DeltaLogL
        lpdf += self.priors["deltalogl"].logpdf(deltalogl)

        # Normal prior on DeltaLogW
        lpdf += self.priors["deltalogw"].logpdf(deltalogw)

        # Normal prior on DeltaDepth
        lpdf += self.priors["deltadepth"].logpdf(deltadepth)

        return lpdf

    def rvs(self):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        raise NotImplementedError
        # CHECK ORDER OF PRODUCED RESULTS
        latlon = self.priors["latlon"].rvs()
        mag = self.priors["mag"].rvs()
        deltalogl = self.priors["deltalogl"].rvs()
        deltalogw = self.priors["deltalogw"].rvs()
        deltadepth = self.priors["deltadepth"].rvs()
        params = np.array(latlon+[mag,deltalogl,deltalogw,deltadepth])
        return pd.Series(params,["Latitude",
                                 "Longitude",
                                 "Magnitude",
                                 "DeltaLogL",
                                 "DeltaLogW",
                                 "DeltaDepth"])
