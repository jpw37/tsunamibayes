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
        return [self.fault.latpts[idx],self.fault.lonpts[idx]]

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self,latlon,mag,lengthwidth):
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
        length = sample["Length"]
        width = sample["Width"]

        if mag < 0:
            lpdf = np.NINF
        else:
            # prior for longitude, latitude
            lpdf = self.priors["latlon"].logpdf(lat,lon)

            # Pareto prior on magnitude
            lpdf += self.priors["mag"].logpdf(mag)

            # conditional prior for length and width
            lpdf += self.lwlogpdf(length,width,mag)

        return lpdf

    def rvs(self):
        """
        Pick a random set of parameters out of the prior
        :return:
        """
        # CHECK ORDER OF PRODUCED RESULTS
        latlon = self.priors["latlon"].rvs()
        mag = self.priors["mag"].rvs()
        params = np.array(latlon+[mag]+self.lwrvs())
        return pd.Series(params,["Latitude","Longitude","Magnitude","Length",'Width'])

    def lwlogpdf(self,length,width,mag):
        # length
        m1 = 0.6423327398       # slope
        c1 = 2.1357387698       # y intercept
        e1 = 0.4073300731874614 # Error bar
        a = mag * m1 + c1 - e1
        b = mag * m1 + c1 + e1
        p = stats.truncnorm.logpdf(np.log10(100*length),a,b)

        # width
        m2 = 0.4832185193       # slope
        c2 = 3.1179508532       # y intercept
        e2 = 0.4093407095518345 # error bar
        a = mag * m2 + c2 - e2
        b = mag * m2 + c2 + e2
        p  += stats.truncnorm.logpdf(np.log10(100*width),a,b)

        return p

    def lwrvs(self):
        # length
        m1 = 0.6423327398       # slope
        c1 = 2.1357387698       # y intercept
        e1 = 0.4073300731874614 # Error bar
        a = mag * m1 + c1 - e1
        b = mag * m1 + c1 + e1
        length = (10**stats.truncnorm.rvs(a,b,size=1)[0])/100

        # width
        m2 = 0.4832185193       # slope
        c2 = 3.1179508532       # y intercept
        e2 = 0.4093407095518345 # error bar
        a = mag * m2 + c2 - e2
        b = mag * m2 + c2 + e2
        width  = (10**stats.truncnorm.rvs(a,b,size=1)[0])/100

        return [length,width]
