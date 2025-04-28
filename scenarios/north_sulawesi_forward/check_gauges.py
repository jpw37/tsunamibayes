import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tsunamibayes import Gauge
import scipy.stats as stats

# The purpose of this file is to check that the observation location points are 
# close to the shore but still have negative values.

# in order to run this file on a windows computer, comment out the mudpy import in the forward file

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.
    
    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Balikpapan
    name = 'Balikpapan'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = [-1.2968463, -1.2902488, -1.2772992]
    gauge.lon = [116.8047846, 116.8590154, 116.9160517]
    gauge.loc = 3
    gauge.scale = 0.8
    gauges.append(gauge)

    # Bontang
    name = 'Bontang'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = [0.0936413, 0.0885611, 0.1068972]
    gauge.lon = [117.5044100, 117.5261494, 117.5809318]
    gauge.loc = 3
    gauge.scale = 0.8
    gauges.append(gauge)

    # Tandjoengbatoe
    name = 'Tandjoengbatoe'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8,scale=0.4)
    gauge = Gauge(name,dists)
    gauge.lat = [2.2671474, 2.2723291, 2.2803883]
    gauge.lon = [118.0996852, 118.1060936, 118.1070125]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tarakan
    name = 'Tarakan'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [3.3010757, 3.2844641, 3.2685585]
    gauge.lon = [117.5575058, 117.5739720, 117.5966022]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tawau
    name = 'Tawau'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [4.2526254, 4.2309717, 4.2315381]
    gauge.lon = [117.8368558, 117.8794758, 117.9244083]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Semporna
    name = 'Semporna'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [4.5058546, 4.4942023, 4.4858872]
    gauge.lon = [118.6074533, 118.6122207, 118.6162575]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Bongao Island
    name = 'Bongao Island'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [5.0082928, 5.0163516, 5.0260919]
    gauge.lon = [119.7686109, 119.7776486, 119.7823453]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Zamboanga
    name = 'Zamboanga'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [6.9015741, 6.8814426, 6.8746361]
    gauge.lon = [122.0322138, 122.0784756, 122.1042892]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Pagadian
    name = 'Pagadian'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [7.7815198, 7.7954067, 7.8057244]
    gauge.lon = [123.4436997, 123.4489195, 123.4682773]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # General Santos
    name = 'General Santos'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [6.0884893, 6.0960155, 6.0961621]
    gauge.lon = [125.1643202, 125.1736935, 125.1865727]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tahuna
    name = 'Tahuna'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [3.6045022, 3.6003375, 3.6035812]
    gauge.lon = [125.4702396, 125.4773798, 125.4877711]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Manado
    name = 'Manado'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [1.5188348, 1.4928330, 1.4766450]
    gauge.lon = [124.8178960, 124.8271245, 124.8045957]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Boroko
    name = 'Boroko'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [0.9261088, 0.9190513, 0.9177993]
    gauge.lon = [123.2759763, 123.2892930, 123.3082661]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Genluma
    name = 'Genluma'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [0.9370277, 0.9437447, 0.9413242]
    gauge.lon = [123.0320480, 123.0332300, 123.0417845]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Boeol
    name = 'Boeol'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [1.1797719, 1.1718516, 1.1660355]
    gauge.lon = [121.4355451, 121.4426615, 121.4496740]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tolotoli
    name = 'Tolotoli'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [1.0283203, 1.0365876, 1.0455639]
    gauge.lon = [120.7673113, 120.7894084, 120.7984559]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Palu
    name = 'Palu'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = [-0.8477276, -0.8707821, -0.8275325]
    gauge.lon = [119.8336537, 119.8548193, 119.8650675]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)


    return gauges

def make_matrix(lines):
    # Split each line into a list
    split_lines = [line.split() for line in lines]

    # Convert to a NumPy array of objects (initially all strings)
    arr = np.array(split_lines, dtype=object)

    # Try to convert entries to integers, set invalid ones to None
    def safe_int(x):
        try:
            return int(x)
        except ValueError:
            return None

    vectorized_safe_int = np.vectorize(safe_int)
    matrix = vectorized_safe_int(arr)

    return matrix

import numpy as np

def generate_lat_long_grid(xllcorner, yllcorner, ncols, nrows, cellsize):
    """
    Generates a grid of latitude and longitude coordinates using NumPy.

    Returns:
    - grid (np.ndarray): 2D array of (longitude, latitude) coordinate tuples.
    """
    start_lon = xllcorner
    start_lat = yllcorner + cellsize * (nrows - 1)  # Start at top-left

    # Create arrays of longitudes and latitudes
    lons = start_lon + np.arange(ncols) * cellsize
    lats = start_lat - np.arange(nrows) * cellsize

    # Create 2D grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Stack into (lon, lat) tuples
    grid = np.dstack((lon_grid, lat_grid))

    return grid


def get_bathymetry_for_gauge_points(bathy_grid, lat_lon_grid, gauges):
    """
    For each gauge, get bathymetry at its lat/lon points.

    Parameters:
    - bathy (np.ndarray): 2D array of bathymetry values.
    - latlon_grid (np.ndarray): 3D array of (lon, lat) points.
    - gauges (list): List of Gauge objects with .lat and .lon attributes.

    Returns:
    - bathy_values (dict): Mapping from gauge name to list of bathymetry values.
    """
    nrows, ncols, _ = lat_lon_grid.shape
    flat_latlon = lat_lon_grid.reshape(-1, 2)  # (nrows * ncols, 2)
    
    bathy_values = dict()

    for gauge in gauges:
        values = []
        for lat, lon in zip(gauge.lat, gauge.lon):
            # Find the closest grid point
            dists = np.linalg.norm(flat_latlon - np.array([lon, lat]), axis=1)
            idx = np.argmin(dists)

            # Convert flat idx back to 2D (row, col)
            row, col = divmod(idx, ncols)
            
            # Get bathymetry value
            bathy_value = int(bathy_grid[row, col])
            values.append(bathy_value)

        bathy_values[gauge.name] = values

    return bathy_values


def plot_bathymetry_with_gauges(bathy_grid, lat_lon_grid, gauges):
    """
    Plots the bathymetry grid and overlays gauge locations.

    Parameters:
    - bathy_grid (np.ndarray): 2D array of bathymetry values.
    - lat_lon_grid (np.ndarray): 3D array of (longitude, latitude) coordinate pairs.
    - gauges (list): List of Gauge objects with .lat, .lon, and .name attributes.
    """
    # Extract longitude and latitude 2D arrays
    lon_grid = lat_lon_grid[:, :, 0]
    lat_grid = lat_lon_grid[:, :, 1]

    plt.figure(figsize=(12, 10))

    # Plot bathymetry background
    pcol =plt.pcolormesh(lon_grid, lat_grid, bathy_grid, shading='auto')

    # Plot each gauge as a red dot
    for gauge in gauges:
        plt.scatter(gauge.lon, gauge.lat, color='red', s=40, edgecolor='black', label=gauge.name)

    plt.colorbar(pcol, label='Bathymetry (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Bathymetry Map with Gauge Locations')
    plt.gca().set_aspect('equal')
    plt.grid(True)

    # Only one label per name (avoid duplicate labels in legend)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())

    plt.show()






if __name__ == '__main__':
    gauges = build_gauges()

    # Read the bathymetry data from a file
    with open(r'C:\Users\ashle\Documents\GitHub\tsunamibayes\scenarios\north_sulawesi_forward\data\GEBCO_28_Apr_2025_5086e97df180\gebco_2024_n10.0_s-2.0_w116.0_e129.0.asc', 'r') as file:
        lines = file.readlines()
    
    ind = 1
    # Extract grid properties (ncols, nrows, etc.) from the read lines
    ncols = int(lines[0].split()[ind])  # Extract the number of columns
    nrows = int(lines[1].split()[ind])  # Extract the number of rows
    xllcorner = float(lines[2].split()[ind])  # Extract x-coordinate of lower left corner
    yllcorner = float(lines[3].split()[ind])  # Extract y-coordinate of lower left corner

    cellsize = float(lines[4].split()[ind])  # Extract cell size
    # Create a matrix/grid using data starting from line 7
    bathy_grid = make_matrix(lines[6:])
    lat_lon_grid = generate_lat_long_grid(xllcorner, yllcorner, ncols, nrows, cellsize)
    depths = get_bathymetry_for_gauge_points(bathy_grid, lat_lon_grid, gauges)
    for key, value in depths.items():
        print(f"{key}: {value}")
    # plot_bathymetry_with_gauges(bathy_grid, lat_lon_grid, gauges)