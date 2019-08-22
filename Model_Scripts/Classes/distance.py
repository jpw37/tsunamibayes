import numpy as np
import scipy.linalg as la
#from cvxopt import solvers, matrix

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / np.sqrt(axis @ axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def convert_rectangular(lat, lon, depth=0):
    """Converts site coordinates from latitude, longitude, and depth into
    rectangular coordinates.

    Parameters:
        lat (float): latitude in degrees
        lon (float): longitude in degrees
        depth (float): distance below the surface of the earth (km)

    Returns:
        ((3,) ndarray): rectangular coordinates of points
    """
    R_earth = 6371 # radius of the earth in km

    #earthquake hypocenter coordinates (spherical, radians)
    phi = np.radians(lon)
    theta = np.radians(90-lat)
    r = R_earth - depth

    # convert to rectangular
    loc = r * np.array([np.sin(theta)*np.cos(phi),
                        np.sin(theta)*np.sin(phi),
                        np.cos(theta)])

    return loc


def basis_change(len_vec, width_vec):
    """Find change of basis matrix for converting fault plane to the unit square in the x-y plane.

    Parameters:
        len_vec (ndarray): gives top edge of rectangle
        width_vec (ndarray): gives side edge of rectangle
    """

    # find normal vector
    normal_vec = np.cross(len_vec, width_vec)

    # change of basis matrix for std basis -> len_vec, width_vec, normal_vec
    S = np.hstack((len_vec.reshape(-1, 1), width_vec.reshape(-1, 1), normal_vec.reshape(-1, 1)))

    # inverse: change of basis matrix for len, width, normal -> std basis
    S_inv = np.linalg.inv(S)

    return S_inv

def fault_plane(hypocenter, lat, lon, length, width, strike, dip):
    """ Finds the length and width vectors describing the fault plane as a
    rectangle centered at the hypocenter with given parameters.

    Parameters:
        lat(deg) - Latitude of the epicenter (centroid) of the earthquake
        long(deg) - Longitude of the epicenter (centroid) of the earthquke
        length(km) - Length of top edge of the fault rectangle
        width(km) - Length of inclined edge of fault rectange
        strike(radians) - Orientation of the top edge, measured in radians
            clockwise form North. The fault plane dips downward to the right
            when moving along the top edge in the strike direction.
        dip(radians) - Angle at which the plane dips downward from the top edge
            (a positive angle between 0 and pi/2 radians)

    Returns:
        length_vec ((3,) ndarray): the vector between the hypocenter of the
            fault plane and the center of the short edges
        width_vec ((3,) ndarray): the vector between the hypocenter of the fault
            plane and the center of the long edges
        hypocenter ((3,) ndarray): location of the hypocenter in rectangular coordinates
    """
    hypocenter_dir = hypocenter / la.norm(hypocenter)

    phi = np.radians(lon)
    theta = np.radians(90-lat)

    # length vector starts out pointing north from hypocenter
    north_unit = np.array([-np.cos(theta)*np.cos(phi),
                            -np.cos(theta)*np.sin(phi),
                            np.sin(theta)])

    # rotate length vector by strike
    len_vec = (rotation_matrix(hypocenter_dir, -strike) @ north_unit) * length/2

    # rotate length vector by 90deg + strike to get width vector
    horizontal = rotation_matrix(hypocenter_dir, np.pi/2 - strike) @ north_unit

    # rotate width vector by dip
    width_vec = (rotation_matrix(len_vec, dip) @ horizontal) * width / 2

    return len_vec, width_vec

def distance(site_lat, site_lon, length, width, strike, dip, depth, lat, lon):
    """Computes the shortest distance between a point on a spherical globe and
    a rectangle (Okada) in 3D space.


    Parameters:
    ----------
    site_lat (float) - Latitude of the observation site (degrees)
    site_lon (float) - Longitude of the observation site (d)
    length(km) - Length of top edge of the fault rectangle
    width(km) - Length of inclined edge of fault rectange
    strike(deg) - Orientation of the top edge, measured in degrees clockwise form North. The
        fault plane dips downward to the right when moving along the top edge in the strike direction.
    dip(deg) - Angle at which the plane dips downward from the top edge
        a positive angle between 0 and 90 degrees.
    depth(km) - Depth below sea level (positive) of the hypocenter of the earthquake
    lat(deg) - Latitude of the epicenter (centroid) of the earthquake
    Long(deg) - Longitude of the epicenter (centroid) of the earthquke

    Returns:
    ----------
    distance(float) - The shortest distance between site and earthquake
    """
    #convert strike and dip to radians
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    
    # convert site and hypocenter to rectangular coordinates
    site = convert_rectangular(site_lat, site_lon)
    hypocenter = convert_rectangular(lat, lon, depth)

    # find length and width vectors for fault plane
    len_vec, width_vec = fault_plane(hypocenter, lat, lon, length, width, strike_rad, dip_rad)

    # change to a basis where the fault plane is a unit square on the x-y plane
    # makes it easier to find closest point on fault plane to site
    S = basis_change(len_vec, width_vec)
    site_changed = S @ site
    hypocenter_changed = S @ hypocenter

    # find closest point on fault plane to site
    closest_pt = hypocenter_changed

    #changing site x coordinates to be within bounds
    if site_changed[0] < hypocenter_changed[0] - 1:
        closest_pt[0] -= 1
    elif site_changed[0] > hypocenter_changed[0] + 1:
        closest_pt[0] += 1
    else:
         closest_pt[0] = site_changed[0]

    #changing site y coordinates to be within bounds
    if site_changed[1] < hypocenter_changed[1] - 1:
        closest_pt[1] -= 1
    elif site_changed[1] > hypocenter_changed[1] + 1:
        closest_pt[1] += 1
    else:
         closest_pt[1] = site_changed[1]

    # reverse change of basis
    closest_pt = la.inv(S) @ closest_pt

    return np.linalg.norm(site - closest_pt)

'''
# convex optimization solution
def minimize_cvx(site_lat, site_lon, length, width, strike, dip, depth, lat, lon):

    len_vec, wid_vec, hypocenter = fault_plane(lat, lon, depth, length, width, strike, dip)

    site_loc = convert_rectangular(site_lat, site_lon)

    # set up objective function: ||x-site_loc||^2
    P = matrix(np.eye(3)*2)
    q = matrix(-2.*site_loc)

    # set up constraints
    # closest point must be on plane defined by length and width vectors
    A = np.cross(len_vec, wid_vec).reshape(1, 3) # normal vector
    b = matrix(np.array([A @ hypocenter]))
    A = matrix(A)

    g1 = wid_vec
    h1 = wid_vec @ (hypocenter+wid_vec)
    g2 = -wid_vec
    h2 = -wid_vec @ (hypocenter-wid_vec)

    # top edges (short edges)
    g3 = len_vec
    h3 = len_vec @ (hypocenter+len_vec)
    g4 = -len_vec
    h4 = -len_vec @ (hypocenter-len_vec)

    G = matrix(np.vstack((g1, g2, g3, g4)))
    h = matrix(np.array([h1, h2, h3, h4]))

    # minimize
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    min_dist = np.sqrt(sol['primal objective'] + site_loc @ site_loc)

    return min_dist
'''