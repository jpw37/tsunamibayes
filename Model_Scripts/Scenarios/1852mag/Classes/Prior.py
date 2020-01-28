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

    def __init__(self,fault,sigma):
        """
        Parameters
        ----------
        fault : instance of Fault
            Fault object
        sigma : float
            Standard deviation of the half-normal distribution on distance from
            the fault
        """
        self.fault = fault
        self.dist = stats.halfnorm(scale=sigma)

    def logpdf(self,lat,lon):
        """Evaluates the logpdf of the prior"""
        distance = self.fault.distance(lat,lon)
        return self.dist.logpdf(distance)

    def pdf(self,lat,lon):
        """Evaluates the pdf of the prior"""
        distance = self.fault.distance(lat,lon)
        return self.dist.pdf(distance)

    def rvs(self):
        """Return a random point on the fault"""
        idx = np.random.randint(len(self.fault.lonpts))
        return [self.fault.latpts[idx],self.fault.lonpts[idx]]

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self,latlon,mag,deltalogl,deltalogw):
        """
        Initialize the class with priors

        Parameters
        ----------
        latlon : instance of LatLonPrior
            Prior distribution on lat/lon

        mag : instance of scipy.stats.pareto
            Prior distribution on magnitude
        """
        self.priors = {"latlon":latlon,"mag":mag,"deltalogl":deltalogl,"deltalogw":deltalogw}

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

        if mag < 0:
            lpdf = np.NINF
        else:
            # prior for longitude, latitude
            lpdf = self.priors["latlon"].logpdf(lat,lon)

            # Pareto prior on magnitude
            lpdf += self.priors["mag"].logpdf(mag)

            # Normal prior on DeltaLogL
            lpdf += self.priors["deltalogl"].logpdf(deltalogl)

            # Normal prior on DeltaLogW
            lpdf += self.priors["deltalogw"].logpdf(deltalogw)

        return lpdf

    def rvs(self):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        # CHECK ORDER OF PRODUCED RESULTS
        latlon = self.priors["latlon"].rvs()
        mag = self.priors["mag"].rvs()
        deltalogl = self.priors["deltalogl"].rvs()
        deltalogw = self.priors["deltalogw"].rvs()
        params = np.array(latlon+[mag,deltalogl,deltalogw])
        return pd.Series(params,["Latitude","Longitude","Magnitude","DeltaLogL","DeltaLogW"])
