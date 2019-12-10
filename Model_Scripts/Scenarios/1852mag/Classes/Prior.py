"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

def circmean(angles,weights):
    x,y = np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))
    return np.degrees(np.arctan2(weights@y,weights@x))

class LatLonStrikePrior:
    """A class for a distance-based latitude/longitude/strike prior. NOTE: this creates
    an unnormalized density function, not a true pdf"""

    def __init__(self,latpts,lonpts,strikepts,sigma_d,sigma_s):
        """
        Parameters
        ----------
        latpts : (N,) ndarray
            Array containing the latitude coordinates of the points on the fault
        lonpts : (N,) ndarray
            Array containing the longitude coordinates of the points on the fault
        sigma_d : float
            Standard deviation of the half-normal distribution on distance from
            the fault
        sigma_s : float
            Standard deviation of the half-normal distribution on the difference between
            the proposal strike angle and the mean strike angle of the N nearest
            points on the fault
        """
        self.latpts = latpts
        self.lonpts = lonpts
        self.strikepts = strikepts
        self.latlon = stats.halfnorm(scale=sigma_d)
        self.strike = stats.halfnorm(scale=sigma_s)

    @staticmethod
    def haversine(lat1,lon1,lat2,lon2):
        """Computes great-circle distance between lat-lon coordinates, in kilometers"""
        R = 6371
        phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
        term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
        return 2*R*np.arcsin(np.sqrt(term))

    def distance(self,lat,lon):
        """Computes the distance from a given lat/lon coordinate to the fault"""
        distances = LatLonStrikePrior.haversine(lat,lon,self.latpts,self.lonpts)
        return distances.min()

    def strike_from_lat_lon(self,lat,lon):
        """Computes the weighted mean strike angle"""
        distances = LatLonStrikePrior.haversine(lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/50)
        weights /= weights.sum()
        return circmean(self.strikepts,weights)%360
        # idx = np.argsort(distances)
        # return stats.circmean(self.strikepts[idx[:N]],high=360)

    def distance_strike(self,lat,lon):
        """Computes both the distance from the fault, and the weighted mean strike angle"""
        distances = LatLonStrikePrior.haversine(lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/50)
        weights /= weights.sum()
        return distances.min(),circmean(self.strikepts,weights)%360#,self.strikepts[np.argmin(distances)]
        # idx = np.argsort(distances)
        # return distances[idx[0]],stats.circmean(self.strikepts[idx[:N]],high=360)

    def logpdf(self,lat,lon,strike):
        """Evaluates the logpdf of the prior"""
        distance,mean_strike = self.distance_strike(lat,lon)
        return self.latlon.logpdf(distance) + self.strike.logpdf((strike-mean_strike)%360)

    def pdf(self,lat,lon,strike):
        """Evaluates the pdf of the prior"""
        distance,mean_strike = self.distance_strike(lat,lon)
        return self.latlon.pdf(distance)*self.strike.pdf((strike-mean_strike)%360)

    def rvs(self):
        """Return a random point on the fault"""
        idx = np.random.randint(len(self.lonpts))
        return [self.lonpts[idx],self.latpts[idx],self.strikepts[idx]]

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
