#from jax import numpy as np
from jax import jacobian
import jax.numpy as np
from jax.config import config

# tell jax to use float64 for increased accuracy
config.update("jax_enable_x64", True)

#TODO: clean up code, make it prettier and make more sense!
def okada_derivative(length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y):
    ''' Returns the numeric partial derivatives of surface deformation with respect to the Okada parameters.

    Parameters:
        length: horizontal measurement of the fault plane
        width: vertical measurement of the fault plane
        depth: distance from epicenter to hypocenter
        latitude: latitude of epicenter
        longitude: longitude of epicenter
        strike: horizontal motion of fault plane during earthquake
        slip: vertical motion of fault plane during earthquake
        dip: angle between surface and fault plane
        rake: angle between strike and slip
        X: horizontal spatial grid values
        Y: vertical spatial grid values

    Returns:
        dZ: grid of seafloor deformation of each coordinate on the X x Y spatial grid
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

    dx = width * np.cos(dip) * np.cos(strike)
    dy = width * np.cos(dip) * np.sin(strike)
    yl = earth_radius * (latitude-y0) * radians - dy


    ds = slip * np.cos(rake) # displacement of strike
    dd = slip * np.sin(rake) # displacement of dip
    sn = np.sin(dip) # save these frequent values for easier use later
    cs = np.cos(dip)

    dZ = np.eye(y_length, x_length)

    x,y = np.meshgrid(X,Y)

    # compare to okada.py xl calculation
    # = earth_radius * np.cos(radians*y0) * (longitude-x0) * radians + dx
    xl = earth_radius * np.cos(radians*y) * (longitude-x0) * radians + dx
    yy = earth_radius * (y-y0*np.ones_like(y)) * radians
    #xx = earth_radius * np.cos(radians*y0) * (x-x0*np.ones_like(x)) * radians
    xx = earth_radius * np.cos(radians*y) * (x-x0*np.ones_like(x)) * radians

    # use trigonometry to figure out how much distance was moved along the strike and dip
    x1 = (xx-xl*np.ones_like(xx)) * np.sin(strike) + (yy-yl*np.ones_like(yy)) * np.cos(strike)
    x2 = (xx-xl*np.ones_like(xx)) * np.cos(strike) - (yy-yl*np.ones_like(yy)) * np.sin(strike)

    # In Okada's paper, x2 is distance up the fault plane, not down dip:
    # the distance along dip should be negative (down)
    x2 = -x2
    #x3 = 0.0

    p = x2 * cs + depth * sn

    # ADD COMMENTS
    f1 = strike_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, depth)
    f2 = strike_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*(p-width), dip, depth)
    f3 = strike_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, depth)
    f4 = strike_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*(p-width), dip, depth)
    g1 = dip_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, depth)
    g2 = dip_slip(x2, x1+np.ones_like(x1)*half_length, np.ones_like(x1)*(p-width), dip, depth)
    g3 = dip_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*p, dip, depth)
    g4 = dip_slip(x2, x1-np.ones_like(x1)*half_length, np.ones_like(x1)*(p-width), dip, depth)

    us = (f1-f2-f3+f4) * ds
    ud = (g1-g2-g3+g4) * dd

    # displacement grid
    dZ = (us+ud)

    return dZ

def strike_slip(x2, y1, y2, dip_angle, depth):
    '''Compute strike-slip (helper function for Okada)

    Parameters:
            x2: distance down the fault plane (nxm matrix)
            y1: surface deformation in the x direction (nxm matrix)
            y2: surface deformation in the y direction (nxm matrix)
            dip_angle: dip angle
            depth: depth (nxm matrix)

    Returns:
            f: function of strike slip'''

    sn = np.sin(dip_angle) # constant
    cs = np.cos(dip_angle) # constant

    # define new variables, according to equations 30 and 28 in the Okada paper (a4 is I4)
    q = x2*sn - depth*cs # nxm matrix
    d_bar = y2*sn - q*cs # nxm matrix

    r = np.sqrt(y1**2 + y2**2 + q**2)
    a4 = 0.5*1/cs*(np.log(r+d_bar) - sn*np.log(r+y2))
    f = -(d_bar*q/r/(r+y2) + q*sn/(r+y2) + a4*sn)/(2.0*np.pi)


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

    sn = np.sin(dip_angle)
    cs = np.cos(dip_angle)

    # define new variables, according to equations 28 and 30 in the Okada paper, with a5 = I5
    q = x2*sn - depth*cs # nxm matrix
    d_bar = y2*sn - q*cs #nxm matrix

    r = np.sqrt(y1**2 + y2**2 + q**2)
    xx = np.sqrt(y1**2 + q**2)
    a5 = 0.5*2/cs*np.arctan((y2*(xx+q*cs)+xx*(r+xx)*sn)/y1/(r+xx)/cs)
    g = -(d_bar*q/r/(r+y1) + sn*np.arctan(y1*y2/q/r) - a5*sn*cs)/(2.0*np.pi)

    return g

def get_derivatives():
    d_okada_length = jacobian(okada_derivative, 0)
    d_okada_width = jacobian(okada_derivative, 1)
    d_okada_depth = jacobian(okada_derivative, 2)
    d_okada_latitude = jacobian(okada_derivative, 3)
    d_okada_longitude = jacobian(okada_derivative, 4)
    d_okada_strike = jacobian(okada_derivative, 5)
    d_okada_slip = jacobian(okada_derivative, 6)
    d_okada_dip = jacobian(okada_derivative, 7)
    d_okada_rake = jacobian(okada_derivative, 8)
    return [d_okada_length, d_okada_width, d_okada_depth, d_okada_latitude, d_okada_longitude, d_okada_strike, d_okada_slip, d_okada_dip, d_okada_rake]

def get_okada_deriv(derivative, length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y):
    return derivative(length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y)

if __name__ == "__main__":
    #print(dip_slip())
    #X = np.array([44.5, 44.6, 44.7, 44.8, 44.9, 45., 45.1, 45.2, 45.3, 45.4])
    #Y = np.array([44.8, 44.9, 45., 45.1, 45.2])
    #print(okada_derivative(40000.,45000.,45.,45.,45.,45.,45.,45.,45.,X,Y))

    X = np.linspace(-77, -67, 38)
    Y= np.linspace(-40,-30,46)
    #length, width, depth, latitude, longitude, strike, slip, dip, rake, X, Y
    #print(okada_derivative(450000, 100000, 35000, -35.826, -72.668, 16, 15, 14, 104, X, Y))

    # geoclaw sample values: strike 16, length 450*10^3, width 100*10^3, depth 35*10^3, slip 15, rake 104, dip 14,
    # longitude -72.668, latitude -35.826

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

    # plug in parameter values to evaluate the derivative, these values are arbitrary
    print(d_okada_length(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_width(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_depth(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_latitude(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_longitude(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_strike(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_slip(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_dip(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))
    print(d_okada_rake(450000., 100000., 35000., -35.826, -72.668, 16., 15., 14., 104., X, Y))


    # geoclaw sample values: strike 16, length 450*10^3, width 100*10^3, depth 35*10^3, slip 15, rake 104, dip 14,
    # longitude -72.668, latitude -35.826
    # X = np.linspace(-77, -67, 100), Y= np.linspace(-40,-30,100)