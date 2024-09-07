import numpy as np
from time_model_class import TimeModel
import os


def generate_lat_long_grid():
        """
        Generates a grid of latitude and longitude coordinates.

        Returns:
        - grid (list): Grid of latitude and longitude coordinates.
        """
        start_value = (xllcorner, yllcorner)  # Starting latitude and longitude to line up with the bathymetry data

        grid = []
        for row in range(nrows):
            current_row = []
            for col in range(ncols):
                value = (start_value[0] + col * cellsize, start_value[1] + row * cellsize)
                current_row.append(value)
            grid.append(current_row)

        return np.array(grid[::-1])

def make_matrix(lines):
    # Converts the bathymetry data into a grid
    matrix = []
    for line in lines:
        line = line.split()
        new_line = [int(num) for num in line]
        matrix.append(new_line)
    return np.array(matrix)

def readlines(filename):
    # Reads in the bathymetry data
    with open(filename) as file:
        return file.readlines()


if __name__== "__main__":
    ncols = 571
    nrows = 421
    xllcorner = 124.991666666667
    yllcorner = -9.508333333333
    cellsize = 0.016666666667
    file_path = r"C:\Users\ashle\Documents\Whitehead Research\Research 2023\1852\etopo.tt3"
    lines = readlines(file_path)
    matrix = make_matrix(lines[6:])
    mesh_width = 20
    sampled_matrix = matrix[::mesh_width, ::mesh_width]
    lat_long_matrix = generate_lat_long_grid()
    sampled_lat_long = lat_long_matrix[::mesh_width, ::mesh_width]
    print(sampled_lat_long)
    start_lat_long = (132.13, -4.68)
    start_x, start_y = 0, 1

    saved_file1 = 'times8.npy'  #times9.npy
    saved_file2 = 'derivatives8.npy' #derivatives9.npy
    if os.path.exists(saved_file1) and os.path.exists(saved_file2):
        total_time_array = np.load(saved_file1)
        derivative_matrix = np.load(saved_file2)
    else:
        total_time_matrix = []
        derivative_matrix = []
        found = False
        for row in sampled_lat_long:
            row_total_time = []
            row_derivative = []
            for lat_long in row:
                print(lat_long)
                time_model_instance = TimeModel(start_lat_long, lat_long, file_path)
                if not found:
                    start_x, start_y = time_model_instance.find_start_point(start_lat_long)
                    print(start_x, start_y)
                    start_x /= mesh_width
                    start_y /= mesh_width
                    found = True
                path, total_time = time_model_instance.dijkstras_algorithm()
                row_total_time.append(total_time)
                derivative = time_model_instance.compute_derivative()
                row_derivative.append(derivative)
            total_time_matrix.append(row_total_time)
            derivative_matrix.append(row_derivative)

        total_time_array = np.array(total_time_matrix)
        derivative_matrix = np.array(derivative_matrix)
        np.save(saved_file1, total_time_array)
        np.save(saved_file2, derivative_matrix)


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# Create a figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap and vector field on the first subplot
axs[0].imshow(np.flipud(total_time_array), origin='lower', cmap='hot', interpolation="nearest")
axs[0].set_title('Tsunami Arrival Time Mesh Grid')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
x = [round(row[0]) for row in sampled_lat_long[0]]
y = [round(row[0][1]) for row in sampled_lat_long]
axs[0].set_yticks(range(len(y)))
axs[0].set_yticklabels(sorted(y))
axs[0].set_xticks(range(len(x)))
axs[0].set_xticklabels(x)
axs[0].grid(True)
point_x_idx = x.index(132)
point_y_idx = len(y) - 1 - y.index(-5)
axs[0].scatter(point_x_idx, point_y_idx, color='green', label='Earthquake Center')

# Define grid coordinates for the vector field
derivative_matrix = np.flipud(derivative_matrix)
x = np.arange(derivative_matrix.shape[1])
y = np.arange(derivative_matrix.shape[0])
X, Y = np.meshgrid(x, y)

# Extract x and y components of vectors
U = derivative_matrix[:, :, 0]
V = derivative_matrix[:, :, 1]

# Plot vector field on the first subplot
axs[0].quiver(X, Y, U, V, scale=10)

# Plot coastlines on the second subplot
axs[1] = plt.subplot(122, projection=ccrs.PlateCarree(), aspect='equal')
axs[1].coastlines(resolution='10m')

# Customize the coastlines subplot
x = [round(row[0], 1) for row in sampled_lat_long[0]]
y = [round(row[0][1], 1) for row in sampled_lat_long]
axs[1].set_extent((x[0], x[-1], y[0], y[-1]))
y_step = round((y[-1] - y[0]) / len(y), 1)
axs[1].set_xticks(np.arange(x[0], x[-1]))
axs[1].set_yticks(np.arange(round(y[0]), round(y[-1]) + y_step, step=y_step))

plt.show()



