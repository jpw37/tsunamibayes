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
        self.latlon = stats.halfnorm(scale=sigma)

    def logpdf(self,lat,lon):
        """Evaluates the logpdf of the prior"""
        distance = self.fault.distance(lat,lon)
        return self.latlon.logpdf(distance)

    def pdf(self,lat,lon):
        """Evaluates the pdf of the prior"""
        distance = self.fault.distance(lat,lon)
        return self.latlon.pdf(distance)

    def rvs(self):
        """Return a random point on the fault"""
        idx = np.random.randint(len(self.fault.lonpts))
        return [self.fault.lonpts[idx],self.fault.latpts[idx]]

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self,latlon,mag):
        """
        Initialize the class with priors

        Parameters
        ----------
        latlonstrike : instance of LatLonPrior
            Prior distribution on lat/lon

        mag : instance of scipy.stats.pareto
            Prior distribution on magnitude
        """
        #rv_continuous.__init__(self)
        self.priors = {"latlon":latlon,"mag":mag}

    def logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        lat    = sample["Latitude"]
        lon    = sample["Longitude"]
        mag    = sample["Magnitude"]

        if mag < 0:
            lpdf = np.NINF
        else:
            #prior for longitude, latitude
            lpdf = self.priors["latlon"].logpdf(lat,lon)

            #Garret and spencer wrote this 18 June 2019
            #If mag is not constructed with gaussian KDE then this will not work
            lpdf += self.priors["mag"].logpdf(mag)

        return lpdf

    def rvs(self):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        # CHECK ORDER OF PRODUCED RESULTS
        lonlatstrike = self.priors["latlon"].rvs()
        mag = self.priors["mag"].rvs()
        params = np.array(lonlat+[mag])
        return pd.Series(params,["Longitude","Latitude","Magnitude"])
