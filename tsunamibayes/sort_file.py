import numpy as np
import json

def useful_grids(grid, info):
    with open(grid) as f:
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
        info[key] = np.array(data[key])
    
    desired_grid = {}
    for i in range(1, 129):
        i = str(i)
        if info[i][2]>130 and info[i][2]<132.7 and info[i][3]>-6.5 and info[i][3]<-3.5:
            desired_grid[i] = grid[i]
    print(desired_grid.keys())
    return desired_grid

def condensed_grids(desired_grid, outfile):
    condensed_grid = {}
    print(desired_grid.keys())
    for i in desired_grid.keys():
        grid = desired_grid[i]
        x,y,z = grid.shape
        temp_grid = np.zeros((y,x))
        for j in range(x):
            temp_grid[:,j] = grid[j][:,3]
        condensed_grid[i] = temp_grid.tolist()
    with open(outfile, 'w') as f:
        f.write(json.dumps(condensed_grid))

if __name__ == "__main__":    
    desired_grid = useful_grids('output.txt', 'info.txt')
    condensed_grids(desired_grid, 'consolidated_grid')
