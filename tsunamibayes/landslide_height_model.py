import math
import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline


class LandslideHeightModel:
    def __init__(self, center_mass_depth, thickness, landslide_speed, volume, aspect_ratio, lat1, lon1, lat2, lon2, lref=None):
        self.center_mass_depth = center_mass_depth
        self.thickness = thickness
        self.landslide_speed = landslide_speed
        self.volume = volume
        self.aspect_ratio = aspect_ratio
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        self.convert_size() # Uses volume, thickness and aspect_ratio to find length and width
        if lref is not None:
            self.length = lref

        # print(self.length, "length")
        # print(self.width, "width")
        # print(self.thickness, "thickness")

    def haversine(self):
        """Computes great-circle distance between sets of lat-lon coordinates on a
        sphere with radius R.

        Parameters
        ----------
        lat1 : float -or- array_like of floats
            The coordinate or ndarray of coordinates associated with the initial
            latitude.
        lon1 : float -or- array_like of floats
            The coordinate or ndarray of coordinates associated with the initial
            longitude.
        lat2 : float -or- array_like of floats
            The coordinate or ndarray of coordinates associated with the terminal
            latitude.
        lon2 : float -or- array_like of floats
            The coordinate or ndarray of coordinates associated with the terminal
            latitude.

        Returns
        -------
        distance : float -or- ndarray of floats
            The computed distance (in meters) between the given point(s).
            Returns object of the same dimension as the lat/lon parameters.
        """
        # Earth's radius, in meters
        R = 6.3781e6

        phi1, phi2 = np.deg2rad(self.lat1), np.deg2rad(self.lat2)
        lam1, lam2 = np.deg2rad(self.lon1), np.deg2rad(self.lon2)
        term = (np.sin(.5 * (phi2 - phi1)) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(.5 * (lam2 - lam1)) ** 2)
        distance_in_meters = 2 * R * np.arcsin(np.sqrt(term))
        return distance_in_meters/1000

    def tsunami_speed(self):
        return (9.8 * self.center_mass_depth) ** .5

    def speeds(self):
        return -3.74 * (((self.landslide_speed - self.tsunami_speed())/self.tsunami_speed()) ** 2)

    def beaching_factor(self):
        return (0.7345 * self.thickness)*(self.width/self.length)\
            *((self.length/self.center_mass_depth)**0.36)*(math.exp(self.speeds()))\
            *((self.length/self.haversine())**0.69)

    def convert_size(self):
        self.length = ((self.volume/self.thickness) * self.aspect_ratio) **.5
        self.width = (self.volume/self.thickness)/self.length / 1000

    def max_flow_depth(self):
        return (self.beaching_factor() ** .8) * (self.center_mass_depth ** .2)







if __name__ == '__main__':
    center_mass_depth = 2800  # meters
    thickness = 50  # meters
    width = 8  # kilometers
    length = 1  # meters
    initial_velocity = 20  # meters per second
    volume = 3000000
    volume = 30000000000 # 30,000,000,000
    aspect_ratio = .375
    lat1 = -6.2
    lon1 = 130
    lat2 = -4.5248
    lon2 = 129.8965

    landslideheightmodel = LandslideHeightModel(center_mass_depth, thickness, initial_velocity, volume, aspect_ratio, lat1, lon1, lat2, lon2)

    print(landslideheightmodel.max_flow_depth())