import numpy as np
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple
import json

def get_grids(filename):
    data = []
    f = open(filename, 'r')  # 'r' = read
    lines = f.read().split()
    f.close()

    grid_dict = {}
    info_dict = {}
    max_grid = 100
    grid_num_list = []

    for _ in range(200):
        grid_num = int(lines[0])
        grid_num_list.append(grid_num)
        if grid_num > max_grid:
            max_grid = grid_num

        # The corresponding parameters are the second entry in each of first lines
        AMR_level = int(lines[2])
        mx = int(lines[4])
        my = int(lines[6])
        xlow = float(lines[8])
        ylow = float(lines[10])
        dx = float(lines[12])
        dy = float(lines[14])
        grid = []

        info_dict[grid_num] = (mx, my, xlow, ylow, dx, dy)

        j = 16
        for k in range(my):
            grid_temp = np.zeros((mx, 4))
            for i in range(mx):
                grid_temp[i, :] = [lines[j + i] for i in range(4)]
                j += 4
            grid.append(grid_temp.tolist())
        lines = lines[16 + 4 * mx * my:]
        if lines[:3] == []:
            break
        grid_dict[grid_num] = grid
    return grid_dict, info_dict


def useful_grids(grid, info, long_min, long_max, lat_min, lat_max):
    """
    Filters grids based on whether any of their four corners fall within the specified longitude and latitude bounds.
    
    Args:
        grid (dict): Dictionary containing grid data.
        info (dict): Dictionary containing grid info where:
                     info[i][2] = bottom-left longitude
                     info[i][3] = bottom-left latitude
                     info[i][0] = mx (number of points in x direction)
                     info[i][1] = my (number of points in y direction)
                     info[i][4] = dx (grid spacing in x direction)
                     info[i][5] = dy (grid spacing in y direction)
        long_min (float): Minimum longitude boundary.
        long_max (float): Maximum longitude boundary.
        lat_min (float): Minimum latitude boundary.
        lat_max (float): Maximum latitude boundary.

    Returns:
        dict, dict: Filtered grid and info dictionaries containing only grids whose corners fall within the specified bounds.
    """

    desired_grid = {}
    desired_info = {}

    for i in range(1, 129):
        if i in grid.keys():
            # Extract grid information
            bottom_left_long = info[i][2]
            bottom_left_lat = info[i][3]
            mx = info[i][0]
            my = info[i][1]
            dx = info[i][4]
            dy = info[i][5]

            # Calculate the coordinates of the four corners
            bottom_right_long = bottom_left_long
            bottom_right_lat = bottom_left_lat + dy * my
            
            top_left_long = bottom_left_long + dx * mx
            top_left_lat = bottom_left_lat
            
            top_right_long = bottom_left_long + dx * mx
            top_right_lat = bottom_left_lat + dy * my

            # Check if any of the four corners are within the specified bounds
            # if (
            #     (long_min <= bottom_left_long <= long_max and lat_min <= bottom_left_lat <= lat_max) or
            #     (long_min <= bottom_right_long <= long_max and lat_min <= bottom_right_lat <= lat_max) or
            #     (long_min <= top_left_long <= long_max and lat_min <= top_left_lat <= lat_max) or
            #     (long_min <= top_right_long <= long_max and lat_min <= top_right_lat <= lat_max)
            # ):
            # 66951461 is only in grid, 466 is lefts within grid, 485 is rights within grid, 486 is both within grid
            
            
            if (
                ((long_min <= bottom_left_long <= long_max and lat_min <= bottom_left_lat <= lat_max) and
                (long_min <= top_left_long <= long_max and lat_min <= top_left_lat <= lat_max)) or 
                ((long_min <= bottom_right_long <= long_max and lat_min <= bottom_right_lat <= lat_max) and
                (long_min <= top_right_long <= long_max and lat_min <= top_right_lat <= lat_max))
            ):
                desired_grid[i] = grid[i]
                desired_info[i] = info[i]

    return desired_grid, desired_info


def condensed_grids(desired_grid):
    condensed_grid = {}
    for i in desired_grid.keys():
        grid = np.array(desired_grid[i])
        x, y, z = grid.shape
        temp_grid = np.zeros((y, x))
        for j in range(x):
            temp_grid[:, j] = grid[j][:, 3]
        condensed_grid[i] = temp_grid.tolist()
    return condensed_grid, list(desired_grid.keys())


def parse_args(argv: Optional[List[str]] = None) -> Namespace:
    """Parse and validate arguments.

    Parameters
    __________

    argv: Optional[List[str]]
        Program argument vector
    """

    argp: ArgumentParser = ArgumentParser()

    argp.add_argument("filename", help="name of file", type=str)
    argp.add_argument("outfile", help="name of output file", type=str)
    argp.add_argument("long_min", help="estimate of lower bound of longitude of subgrid", type=float)
    argp.add_argument("long_max", help="estimate of upper bound of longitude of subgrid", type=float)
    argp.add_argument("lat_min", help="estimate of lower bound of latitude of subgrid", type=float)
    argp.add_argument("lat_max", help="estimate of upper bound of latitude of subgrid", type=float)
    args: Namespace = argp.parse_args()

    return args


def main(args: Optional[Namespace] = None) -> None:
    """Executes main program.

    Parameters
    __________

    args : Optional[Namespace]
        Program arguments
    """

    if args is None:
        args = parse_args()

    grid_dict, info_dict = get_grids(args.filename)
    desired_grid, desired_info = useful_grids(grid_dict, info_dict, args.long_min, args.long_max, args.lat_min, args.lat_max)
    condensed_grid = condensed_grids(desired_grid)


if __name__ == "__main__":
    _args: Namespace = parse_args()
    main(_args)
