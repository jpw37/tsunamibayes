"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

class LatLonStrikePrior:
    """A class for a distance-based latitude/longitude/strike prior. NOTE: this creates
    an unnormalized density function, not a true pdf"""

    def __init__(self,fault,sigma_d,sigma_s):
        """
        Parameters
        ----------
        fault : instance of Fault
            Fault object
        sigma_d : float
            Standard deviation of the half-normal distribution on distance from
            the fault
        sigma_s : float
            Standard deviation of the half-normal distribution on the difference between
            the proposal strike angle and the mean strike angle of the N nearest
            points on the fault
        """
        self.fault = fault
        self.latlon = stats.halfnorm(scale=sigma_d)
        self.strike = stats.halfnorm(scale=sigma_s)

    def logpdf(self,lat,lon,strike):
        """Evaluates the logpdf of the prior"""
        distance,mean_strike = self.fault.distance_strike(lat,lon)
        strikediff = strike - mean_strike
        strikediff = np.abs((strikediff + 180) % 360 - 180)
        return self.latlon.logpdf(distance) + self.strike.logpdf(strikediff)

    def pdf(self,lat,lon,strike):
        """Evaluates the pdf of the prior"""
        distance,mean_strike = self.fault.distance_strike(lat,lon)
        strikediff = strike - mean_strike
        strikediff = np.abs((strikediff + 180) % 360 - 180)
        return self.latlon.pdf(distance)*self.strike.pdf(strikediff)

    def rvs(self):
        """Return a random point on the fault"""
        idx = np.random.randint(len(self.fault.lonpts))
        return [self.fault.lonpts[idx],self.fault.latpts[idx],self.fault.strikepts[idx]]

class Prior:
    """
    This class handles the logpdf calculation for the priors given from the custom class
    """
    def __init__(self,latlonstrike,mag):
        """
        Initialize the class with priors

        Parameters
        ----------
        latlonstrike : instance of LatLonStrikePrior
            Prior distribution on lat/lon/strike

        mag : instance of scipy.stats.pareto
            Prior distribution on magnitude
        """
        #rv_continuous.__init__(self)
        self.priors = {"latlonstrike":latlonstrike,"mag":mag}

    def logpdf(self, sample):
        """
        Calculate the prior log likelihood
        :param sample:
        :return:
        """
        lat    = sample["Latitude"]
        lon    = sample["Longitude"]
        strike = sample["Strike"]
        mag    = sample["Magnitude"]

        if mag < 0:
            lpdf = np.NINF
        else:
            #prior for longitude, latitude
            lpdf = self.priors["latlonstrike"].logpdf(lat,lon,strike)

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
        lonlatstrike = self.priors["latlonstrike"].rvs()
        mag = self.priors["mag"].rvs()
        params = np.array(lonlatstrike+[mag])
        return pd.Series(params,["Longitude","Latitude","Strike", "Magnitude"])
