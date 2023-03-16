import numpy as np
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple
import json

def get_grids(filename):
    data = []
    f = open(filename, 'r') # 'r' = read
    lines = f.read().split()
    f.close()

    grid_dict = {}
    info_dict = {}
    max_grid = 100
    grid_num_list = []

    for i in range(200):
        grid_num = int(lines[0])
        grid_num_list.append(grid_num)
        if grid_num > max_grid:
            max_grid = grid_num

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
                grid_temp[i,:] = [lines[j+i] for i in range(4)]
                j+=4
            grid.append(grid_temp.tolist())
        lines = lines[16+4*mx*my:]
        if lines[:3] == []:
            break
        grid_dict[grid_num] = grid
    return grid_dict, info_dict

def useful_grids(grid, info, lat_min, lat_max, long_min, long_max):
    """with open(grid) as f:
        dat = f.read()
    data = json.loads(dat)
    grid = {}
    for key in list(data.keys()):
        grid[key] = np.array(data[key])

    with open(info) as f:
        dat = f.read()
    data = json.loads(dat)
    info = {}
    for key in list(data.keys()):
        info[key] = np.array(data[key])"""

    desired_grid = {}
    desired_info = {}
    for i in range(1, 129):
        #i = str(i)
        if info[i][2]>lat_min and info[i][2]<lat_max and info[i][3]>long_min and info[i][3]<long_max:
            desired_grid[i] = grid[i]
            desired_info[i] = info[i]
    return desired_grid, desired_info

def condensed_grids(desired_grid):
    condensed_grid = {}
    for i in desired_grid.keys():
        grid = np.array(desired_grid[i])
        x,y,z = grid.shape
        temp_grid = np.zeros((y,x))
        for j in range(x):
            temp_grid[:,j] = grid[j][:,3]
        condensed_grid[i] = temp_grid.tolist()
    return condensed_grid
    """with open(outfile, 'w') as f:
        f.write(json.dumps(condensed_grid))"""

"""def write(grid, info_grid, outfile):
    #file = open(outfile, 'w+')
    #content = str(grid)
    #file.write(content)
    #file.close()
    #new_dict = {}
    #for key, value in grid.items():
    #    new_dict[key] = value

    print(np.array(grid[2]).shape)
    with open(outfile, 'w') as f:
        #for key, value in grid.items():
        #    f.write('%s:%s/n' % (key, value))
        f.write(json.dumps(grid))
    with open('info.txt', 'w') as f:
        f.write(json.dumps(info_grid))"""

def parse_args(argv: Optional[List[str]] = None) -> Namespace:
    """Parse and validate arguments.

    Parameters
    __________

    argv: Optional[List[str]]
        Program argument vector
    """

    argp: ArgumentParser = ArgumentParser()

    argp.add_argument("filename", help = "name of file", type=str)
    argp.add_argument("outfile", help = "name of output file", type=str)
    argp.add_argument("lat_min", help = "estimate of lower bound of lattitude of subgrid", type=float)
    argp.add_argument("lat_max", help = "estimate of upper bound of lattitude of subgrid", type=float)
    argp.add_argument("long_min", help = "estimate of lower bound of longitude of subgrid", type=float)
    argp.add_argument("long_max", help = "estimate of upper bound of longitude of subgrid", type=float)
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

    #if not args.output_dir.exists():
    #    args.output_dir.mkdir()

    grid_dict, info_dict = get_grids(args.filename)
    desired_grid, desired_info = useful_grids(grid_dict, info_dict, args.lat_min, args.lat_max, args.long_min, args.long_max)
    condensed_grid = condensed_grids(desired_grid)


if __name__ == "__main__":
    _args: Namespace = parse_args()
    main(_args)


#print('grid list', grid_num_list)
#print('number of grids', len(grid_num_list))

#print(grid.shape)
#print('grid len:', len(grid_dict[128]))
#print('grid:', grid_dict[128][0])
#print(lines[16+4*mx*my-4:16+4*mx*my+10])
#print(lines[16+4*mx])
#print(lines[16+4*mx+1])

#print(grid[0,0].typeof())
#f.close()
#for i in range(10):
#    print(lines[i])
#print(lines[0])
#print(len(lines[0]))
#print(lines[0][3:6])
