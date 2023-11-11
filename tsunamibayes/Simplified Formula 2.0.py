import math
import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline

class HeightModel:
    def __init__(self, slip, length, width, rake, dip, lat1, lon1, lat2, lon2, shortest_time_path):
        # Initialize parameters as attributes
        self.slip = slip
        self.length = length
        self.width = width
        self.rake = rake
        self.dip = dip
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        self.shortest_time_path = shortest_time_path

        # Call the method to read the matrix once during initialization
        # How are we going to read in the bathymetry data?? I don't know how to format this lol
        self.matrix = self.read_file_into_matrix(sys.argv[1])

    def create_x(self):
        # Generate an array representing longitude coordinates in order to interpolate the depth at a specific latlon coordinate
        start_value = 124.991666666667
        step = 0.016666666667
        array_length = 571
        my_array = np.linspace(start_value, start_value + step * (array_length - 1), array_length)
        return my_array

    def create_y(self):
        # Generate an array representing longitude coordinates in order to interpolate the depth at a specific latlon coordinate
        start_value = -9.508333333333
        step = 0.016666666667
        array_length = 421
        my_array = np.linspace(start_value, start_value + step * (array_length - 1), array_length)
        return my_array

    def make_matrix(self, lines):
        # Convert a list of strings into a matrix of integers
        matrix = []
        for line in lines:
            line = line.split()
            new_line = [int(num) for num in line]
            matrix.append(new_line)
        return matrix[::-1]

    def readlines(self, filename):
        with open(filename) as file:
            return file.readlines()

    def writelines(self, filename, content):
        with open(filename, "w") as file:
            file.writelines(content)

    def read_file_into_matrix(self, infile):
        lines = self.readlines(infile)
        matrix = self.make_matrix(lines)
        return np.array(matrix).T

    def get_depth(self, lon, lat):
        # Interpolate depth at a given longitude and latitude using the bathymetry matrix
        x = self.create_x()
        y = self.create_y()
        z = np.array(self.matrix)
        interp_function = RectBivariateSpline(x, y, z)
        interp_value = interp_function(lon, lat)
        return interp_value[0][0]

    def depth_at_source(self):
        # Calculate the depth at the source coordinates(lat1, lon1)
        depth = self.get_depth(lon1, lat1)
        return abs(depth)

    def alpha(self):
        a = (1 - (dip/180)) * math.sin(np.deg2rad(dip)) * abs(math.sin(np.deg2rad(rake)))
        return a

    def initial_tsunami_amplitude(self):
        a_naught = (self.alpha() * self.slip) / (math.cosh(np.deg2rad((4 * math.pi * self.depth_at_source())/(self.width + self.length))))
        return a_naught

    def mean_depth(self):
        # Compute the mean depth along the shortest time path
        depth_lst = [self.get_depth(lat_lon[1], lat_lon[0]) for lat_lon in self.shortest_time_path]
        return abs(sum(depth_lst) / len(depth_lst))

    def psi(self):
        p = .5 + (0.575 * math.exp(-.0175 * (self.length / self.mean_depth())))
        return p

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

        phi1,phi2 = np.deg2rad(self.lat1),np.deg2rad(self.lat2)
        lam1,lam2 = np.deg2rad(self.lon1),np.deg2rad(self.lon2)
        term = (np.sin(.5 * (phi2 - phi1))**2 + np.cos(phi1) * np.cos(phi2) * np.sin(.5 * (lam2 - lam1))**2)
        return 2*R*np.arcsin(np.sqrt(term))

    def propagation_loss(self):
        p = (1 + ((2 * self.haversine()) / self.length)) ** self.psi()
        return p

    def shoaling_correction(self):
        s_l = (self.depth_at_source()/self.mean_depth()) ** .25
        return s_l

    def wave_height(self):
        a_s_r = self.initial_tsunami_amplitude() * self.propagation_loss() * self.shoaling_correction()
        return a_s_r


if __name__ == '__main__':
    slip = 10.175
    length = 512087
    width = 145077
    rake = 90
    dip = 10.993
    lat1 = -4.678971296
    lon1 = 132.1308854
    lat2 = -4.5175
    lon2 = 129.775
    shortest_time_path = [(131.125000000123, -4.674999999903001), (131.108333333456, -4.674999999903001),
           (131.091666666789, -4.674999999903001), (131.075000000122, -4.674999999903001),
           (131.058333333455, -4.674999999903001), (131.041666666788, -4.674999999903001),
           (131.025000000121, -4.674999999903001), (131.008333333454, -4.674999999903001),
           (130.991666666787, -4.674999999903001), (130.97500000012, -4.674999999903001),
           (130.958333333453, -4.674999999903001), (130.941666666786, -4.674999999903001),
           (130.925000000119, -4.674999999903001), (130.908333333452, -4.674999999903001),
           (130.891666666785, -4.674999999903001), (130.875000000118, -4.674999999903001),
           (130.85833333345101, -4.674999999903001), (130.841666666784, -4.674999999903001),
           (130.825000000117, -4.674999999903001), (130.80833333345, -4.674999999903001),
           (130.791666666783, -4.674999999903001), (130.775000000116, -4.674999999903001),
           (130.758333333449, -4.674999999903001), (130.741666666782, -4.674999999903001),
           (130.725000000115, -4.674999999903001), (130.708333333448, -4.674999999903001),
           (130.691666666781, -4.674999999903001), (130.675000000114, -4.674999999903001),
           (130.658333333447, -4.674999999903001), (130.64166666678, -4.674999999903001),
           (130.625000000113, -4.674999999903001), (130.608333333446, -4.674999999903001),
           (130.591666666779, -4.674999999903001), (130.575000000112, -4.674999999903001),
           (130.558333333445, -4.674999999903001), (130.558333333445, -4.658333333236),
           (130.541666666778, -4.658333333236), (130.541666666778, -4.641666666569001),
           (130.525000000111, -4.641666666569001), (130.525000000111, -4.624999999902),
           (130.525000000111, -4.6083333332350005), (130.525000000111, -4.591666666568001),
           (130.525000000111, -4.574999999901), (130.525000000111, -4.558333333234001),
           (130.525000000111, -4.541666666567001), (130.525000000111, -4.5249999999),
           (130.525000000111, -4.508333333233001), (130.525000000111, -4.491666666566),
           (130.525000000111, -4.474999999899), (130.525000000111, -4.458333333232001),
           (130.508333333444, -4.458333333232001), (130.508333333444, -4.441666666565),
           (130.491666666777, -4.441666666565), (130.47500000011001, -4.441666666565),
           (130.47500000011001, -4.4249999998980005), (130.458333333443, -4.4249999998980005),
           (130.441666666776, -4.4249999998980005), (130.425000000109, -4.4249999998980005),
           (130.408333333442, -4.4249999998980005), (130.391666666775, -4.4249999998980005),
           (130.375000000108, -4.4249999998980005), (130.358333333441, -4.4249999998980005),
           (130.341666666774, -4.4249999998980005), (130.325000000107, -4.4249999998980005),
           (130.30833333344, -4.4249999998980005), (130.291666666773, -4.4249999998980005),
           (130.275000000106, -4.4249999998980005), (130.258333333439, -4.4249999998980005),
           (130.241666666772, -4.4249999998980005), (130.225000000105, -4.4249999998980005),
           (130.208333333438, -4.4249999998980005), (130.191666666771, -4.4249999998980005),
           (130.175000000104, -4.4249999998980005), (130.158333333437, -4.4249999998980005),
           (130.14166666677, -4.4249999998980005), (130.125000000103, -4.4249999998980005),
           (130.108333333436, -4.4249999998980005), (130.09166666676902, -4.4249999998980005),
           (130.075000000102, -4.4249999998980005), (130.058333333435, -4.4249999998980005),
           (130.041666666768, -4.4249999998980005), (130.025000000101, -4.4249999998980005),
           (130.008333333434, -4.4249999998980005), (129.991666666767, -4.4249999998980005),
           (129.9750000001, -4.4249999998980005), (129.958333333433, -4.4249999998980005),
           (129.941666666766, -4.4249999998980005), (129.925000000099, -4.4249999998980005),
           (129.908333333432, -4.4249999998980005), (129.891666666765, -4.4249999998980005),
           (129.875000000098, -4.4249999998980005), (129.858333333431, -4.4249999998980005),
           (129.841666666764, -4.4249999998980005), (129.825000000097, -4.4249999998980005),
           (129.80833333343, -4.4249999998980005), (129.791666666763, -4.4249999998980005),
           (129.77500000009601, -4.4249999998980005), (129.77500000009601, -4.441666666565),
           (129.77500000009601, -4.458333333232001), (129.77500000009601, -4.474999999899),
           (129.77500000009601, -4.491666666566), (129.77500000009601, -4.508333333233001),
           (129.77500000009601, -4.5249999999)]
    heightmodel = HeightModel(slip, length, width, rake, dip, lat1, lon1, lat2, lon2, shortest_time_path)

    print(heightmodel.wave_height())

