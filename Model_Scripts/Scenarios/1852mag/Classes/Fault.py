import numpy as np

class Fault:
    """A class for data relating to the fault"""
    def __init__(self,latpts,lonpts,strikepts,name):
        self.latpts = latpts
        self.lonpts = lonpts
        self.strikepts = strikepts
        self.name = name

    @staticmethod
    def haversine(lat1,lon1,lat2,lon2):
        """Computes great-circle distance between lat-lon coordinates, in kilometers"""
        R = 6371
        phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
        term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
        return 2*R*np.arcsin(np.sqrt(term))

    @staticmethod
    def circmean(angles,weights):
        x,y = np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))
        return np.degrees(np.arctan2(weights@y,weights@x))

    def distance(self,lat,lon):
        """Computes the distance from a given lat/lon coordinate to the fault"""
        distances = Fault.haversine(lat,lon,self.latpts,self.lonpts)
        return distances.min()

    def strike_from_lat_lon(self,lat,lon):
        """Computes the weighted mean strike angle"""
        distances = Fault.haversine(lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/50)
        weights /= weights.sum()
        return circmean(self.strikepts,weights)%360

    def distance_strike(self,lat,lon):
        """Computes both the distance from the fault, and the weighted mean strike angle"""
        distances = Fault.haversine(lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/50)
        weights /= weights.sum()
        return distances.min(),Fault.circmean(self.strikepts,weights)%360
