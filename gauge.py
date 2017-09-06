# A file containing the Gauge class
import numpy as np

class Gauge:
    """A gauge object class. Has a name and means and standard deviations
    for wave heights and arrival times.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        arrival_mean (float): the mean value for the arrival time.
        arrival_std (float): the standard deviation for arrival time.
        height_mean (float): the mean for wave height.
        height_std (float): the standard deviation for wave height.
    """
    def __init__(self, name, arrival_mean, arrival_std, height_mean, height_std):
        self.name = name
        self.arrival_mean = arrival_mean
        self.arrival_std = arrival_std
        self.height_mean = height_mean
        self.height_std = height_std
