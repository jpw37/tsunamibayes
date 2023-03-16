from jax import numpy as np
from jax import jacobian
import multiprocessing
import jax
from joblib import Parallel, delayed

#TODO update the function docstrings and variable names
#TODO check that dZ grid is same as other Okada model
@jax.jit
def okada_derivative(length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y):
    ''' Returns the numeric partial derivatives of surface deformation with respect to the Okada parameters.

    Parameters:
        length:
        width:
        depth:
        latitude:
        longitude:
        strike:
        slip:
        dip:
        rake:
        X:
        Y:

    Returns:

    '''

    x0 = X[0]
    y0 = Y[0]
    x_length = len(X)
    y_length = len(Y)

    radians = np.pi/180 # radians conversion
    earth_radius = 6.378e6 # radius of the earth

    dip *= radians# convert dip to radians
    rake *= radians # convert rake to radians
    strike *= radians # convert strike to radians
    half_length = length * 0.5 # get the half length

    # calculate focal depth used for Okada's model (adjust depth according to angle of dip)
    depth = depth + width * np.sin(dip)

    # STILL DON'T KNOW WHAT IS GOING ON HERE
    dx = width * np.cos(dip) * np.cos(strike)
    dy = width * np.cos(dip) * np.sin(strike)
    yl = earth_radius * (latitude-y0) * radians - dy


    ds = slip * np.cos(rake) # displacement of strike
    dd = slip * np.sin(rake) # displacement of dip
    sn = np.sin(dip) # save these frequent values for easier use later
    cs = np.cos(dip)

    dZ = np.eye(y_length, x_length)

    x,y = np.meshgrid(X,Y)

    xl = earth_radius * np.cos(radians*y0) * (longitude-x0) * radians + dx
    yy = earth_radius * (y-y0*np.ones_like(y)) * radians
    xx = earth_radius * np.cos(radians*y0) * (x-x0*np.ones_like(x)) * radians

    # use trigonometry to figure out how much distance was moved along the strike and dip
    x1 = (xx-xl*np.ones_like(xx)) * np.sin(strike) + (yy-yl*np.ones_like(yy)) * np.cos(strike)
    x2 = (xx-xl*np.ones_like(xx)) * np.cos(strike) - (yy-yl*np.ones_like(yy)) * np.sin(strike)
    #TODO: note that x1, x2 are (5,10) matrices, so I have to treat them as such in dip_slip and strike_slip

    # In Okada's paper, x2 is distance up the fault plane, not down dip:
    # the distance along dip should be negative (down)
    x2 = -x2
    #x3 = 0.0

    p = x2 * cs + depth * sn

    # ADD COMMENTS
    f1 = strike_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, np.ones_like(x1)*depth)
    f2 = strike_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*(p-width), dip, np.ones_like(x1)*depth)
    f3 = strike_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, np.ones_like(x1)*depth)
    f4 = strike_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*p-width, dip, np.ones_like(x1)*depth)
    g1 = dip_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, np.ones_like(x1)*depth)
    g2 = dip_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*p-width, dip, np.ones_like(x1)*depth)
    g3 = dip_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, np.ones_like(x1)*depth)
    g4 = dip_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*p-width, dip, np.ones_like(x1)*depth)

    us = (f1-f2-f3+f4) * ds
    ud = (g1-g2-g3+g4) * dd

    # displacement grid
    dZ = (us+ud)

    return dZ

def strike_slip(x2, y1, y2, dip_angle, depth):
    '''Compute strike-slip (helper function for okada)

    Parameters:
            x2: distance down the fault plane (nxm matrix)
            y1: surface deformation in the x direction (nxm matrix)
            y2: surface deformation in the y direction (nxm matrix)
            dip_angle: dip angle
            depth: depth (nxm matrix)

    Returns:
            f: function of strike slip'''

    cs = np.cos(dip_angle) # constant
    sn = np.sin(dip_angle) # constant

    # define new variables, according to equations 30 and 28 in the Okada paper (a4 is I4)
    q = x2*sn - depth*cs # nxm matrix
    d_bar = y2*sn - q*cs # nxm matrix

    #r = np.zeros_like(y1)
    #a4 = np.zeros_like(y1)
    #f = np.zeros_like(y1)

    #r = np.sqrt(np.linalg.matrix_power(y1,2) + np.linalg.matrix_power(y2,2) + np.linalg.matrix_power(q,2))
    r = np.sqrt(y1**2 + y2**2 + q**2)
    a4 = 0.5*(np.log(r+d_bar) - sn*np.log(r+y2))/cs
    f = -1*(d_bar*q/(r*(r+y2)) + q*sn/(r+y2) + a4*sn) / (2*np.pi)
    #for i in range(y1.shape[0]):
        #for j in range(y1.shape[1]):
            #r = r.at[i,j].set(np.sqrt(y1[i,j]*y1[i,j] + y2[i,j]*y2[i,j] + q[i,j]*q[i,j]))
            #a4 = a4.at[i,j].set(0.5*(np.log(r[i,j]+d_bar[i,j]) - sn*np.log(r[i,j]+y2[i,j]))/cs)

            # u_z from Okada paper equation 25, letting u1 be 1
            #f = f.at[i,j].set(-1*(d_bar[i,j]*q[i,j]/(r[i,j]*(r[i,j]+y2[i,j])) + q[i,j]*sn/(r[i,j]+y2[i,j]) + a4[i,j]*sn) / (2*np.pi))

    return  f

def dip_slip(x2, y1, y2, dip_angle, depth):
    '''Compute dip-slip (helper function for Okada)

    Parameters:
            x2: distance down the fault plane (nxm matrix)
            y1: surface deformation in the x direction (nxm matrix)
            y2: surface deformation in the y direction (nxm matrix)
            dip_angle: dip angle
            depth: depth (nxm matrix)

    Returns:
            g: function of dip slip'''

    cs = np.cos(dip_angle)
    sn = np.sin(dip_angle)

    # define new variables, according to equations 28 and 30 in the Okada paper, with a5 = I5
    q = x2*sn - depth*cs # nxm matrix
    d_bar = y2*sn - q*cs #nxm matrix

    #r = np.zeros_like(y1)
    #xx = np.zeros_like(y1)
    #a5 = np.zeros_like(y1)
    #g = np.zeros_like(y1)

    #r = np.sqrt(np.linalg.matrix_power(y1,2) + np.linalg.matrix_power(y2,2) + np.linalg.matrix_power(q,2))
    #xx = np.sqrt(np.linalg.matrix_power(y1,2) + np.linalg.matrix_power(q,2))
    r = np.sqrt(y1**2 + y2**2 + q**2)
    xx = np.sqrt(y1**2 + q**2)
    a5 = 1/cs*np.arctan((y2*(xx+q*cs)+xx*(r+xx)*sn) / (y1*(r+xx)*cs))
    g = -1*(d_bar*q/(r*(r+y1)) + sn*np.arctan(y1*y2/(q*r)) - a5*sn*cs) / (2*np.pi)

    #for i in range(y1.shape[0]):
        #for j in range(y1.shape[1]):
            #r = r.at[i,j].set(np.sqrt(y1[i,j]*y1[i,j] + y2[i,j]*y2[i,j] + q[i,j]*q[i,j]))  #nxm matrix
            #xx = xx.at[i,j].set(np.sqrt(y1[i,j]*y1[i,j] + q[i,j]*q[i,j])) # nxm matrix
            #a5 = a5.at[i,j].set(1/cs*np.arctan((y2[i,j]*(xx[i,j]+q[i,j]*cs)+xx[i,j]*(r[i,j]+xx[i,j])*sn) / (y1[i,j]*(r[i,j]+xx[i,j])*cs)))

            # u_z from Okada paper equation 26, letting u2 be 1
            #g = g.at[i,j].set(-1*(d_bar[i,j]*q[i,j]/(r[i,j]*(r[i,j]+y1[i,j])) + sn*np.arctan(y1[i,j]*y2[i,j]/(q[i,j]*r[i,j])) - a5[i,j]*sn*cs) / (2*np.pi))

    return g

if __name__ == "__main__":
    #print(dip_slip())
    #X = np.array([1.,2.,3.,4.,5.])
    #Y = np.array([3.,5.,7.,9.,11.])
    X = np.linspace(130,131,38)
    Y = np.linspace(-4.5,-3.5,46)


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
    print(d_okada_length(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    #length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y
    print(d_okada_width(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_depth(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_latitude(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_longitude(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_strike(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_slip(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_dip(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    print(d_okada_rake(410489.82154734,81172.520770329,10035.247895443,-3.80457187273708,131.405862952901,121.106320856361,15.6458838693299,9.1367026992073,90.0, X, Y))
    '''
    pool = multiprocessing.Pool()
    with multiprocessing.Pool() as pool:
        items = [(45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y), (45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y), (45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y), (45.,45.,45.,45.,45.,45.,45.,45.,45., X, Y)]
        for result in pool.map(d_okada_rake, items):
            print(result)
    pool.close()
    '''
    # geoclaw sample values: strike 16, length 450*10^3, width 100*10^3, depth 35*10^3, slip 15, rake 104, dip 14,
    # longitude -72.668, latitude -35.826
    # X = np.linspace(-77, -67, 100), Y= np.linspace(-40,-30,100)
