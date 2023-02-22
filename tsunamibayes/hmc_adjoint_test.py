import numpy as np
from sort_file import useful_grids
from sort_file import condensed_grids
import json

desired_grid = useful_grids('output.txt', 'info.txt')
condensed_grids(desired_grid, 'consolidated_grid.txt')
with open('consolidated_grid.txt') as f:
    dat = f.read()
data = json.loads(dat)
info = {}
for key in list(data.keys()):
    info[key] = np.array(data[key])
print(info['76'].shape)
#condensed_grid = np.load('consolidated_grid.txt', allow_pickle=True)

X = np.linspace(130,131,44)
Y = np.linspace(-4.5,-3.5,38)

from okada_autograd import jacobian
from okada_autograd import okada_derivative
from time import time

# get the jacobian with respect to each variable
d_okada_length = jacobian(okada_derivative, 0)
d_okada_width = jacobian(okada_derivative, 1)
d_okada_depth = jacobian(okada_derivative, 2)
d_okada_latitude = jacobian(okada_derivative, 3)
d_okada_longitude = jacobian(okada_derivative, 4)
d_okada_strike = jacobian(okada_derivative, 5)
d_okada_slip = jacobian(okada_derivative, 6)
d_okada_dip = jacobian(okada_derivative, 7)
d_okada_rake = jacobian(okada_derivative, 8)
#Parallel(n_jobs=4)(print(delayed(jacobian)(okada_derivative, i) for i in range(9)))

#list_functions = [print(d_okada_length(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_width(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_depth(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_latitude(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_longitude(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_strike(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_slip(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_dip(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)), print(d_okada_rake(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y))]
#Parallel(n_jobs=4)(delayed(list_functions)(list_functions[i] for i in range(9)))
# plug in parameter values to evaluate the derivative, these values are arbitrary
start = time()
len = d_okada_length(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
print(len.shape)
#length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y
width = d_okada_width(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
depth = d_okada_depth(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
lat = d_okada_latitude(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
long = d_okada_longitude(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
strike = d_okada_strike(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
slip = d_okada_slip(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
dip = d_okada_dip(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
rake = d_okada_rake(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y)
print(time()-start)

start = time()
print(info['76']*len)
print(time()-start)
