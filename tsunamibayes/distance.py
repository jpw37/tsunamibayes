import numpy as np


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
        width_vec ((3,) ndarray): the vector between the hypocenter of the
            fault plane and the center of the long edges
        hypocenter ((3,) ndarray): location of the hypocenter in rectangular
            coordinates
    """
    hypocenter_dir = hypocenter / np.linalg.norm(hypocenter)

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


def distance(site_lat, site_lon, length, width, strike, dip, depth, hypo_lat,
            hypo_lon, grid_size=100):
    """Computes the shortest distance between a point on a spherical globe and
    a rectangle (Okada) in 3D space.

    Parameters:
    ----------
    site_lat (float) - Latitude of the observation site (degrees)
    site_lon (float) - Longitude of the observation site (d)
    strike(deg) - Orientation of the top edge, measured in degrees clockwise
        from North. The fault plane dips downward to the right when moving
        along the top edge in the strike direction.
    dip(deg) - Angle at which the plane dips downward from the top edge -- a
        positive angle between 0 and 90 degrees.
    depth(km) - Depth below sea level (positive) of the hypocenter of the
        earthquake
    length(km) - Length of top edge of the fault rectangle
    width(km) - Length of inclined edge of fault rectange
    hypo_lat (float) - Latitude of epicenter of earthquake (degrees)
    hypo_lon (float) - Longitude of epicenter of earthquake (degrees)

    Returns:
    ----------
    distance(float) - The shortest distance between site and the Okada
        rectangle
    """

    hypocenter = convert_rectangular(hypo_lat, hypo_lon, depth)
    site = convert_rectangular(site_lat, site_lon)

    strike = np.deg2rad(strike)
    dip = np.deg2rad(dip)

    len_vec, wid_vec = fault_plane(
        hypocenter, hypo_lat, hypo_lon, length, width, strike, dip
    )

    t = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(t, t)
    steps = np.vstack((X.flatten(), Y.flatten()))

    A = np.zeros((3, 2))
    A[:, 0] = len_vec
    A[:, 1] = wid_vec

    points = hypocenter.reshape(-1, 1) + A @ steps
    distances = np.linalg.norm(points - site.reshape(-1, 1), axis=0)

    return np.min(distances)
