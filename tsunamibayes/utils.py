import numpy as np

# Earth's radius, in meters
R = 6.3781e6

def haversine(lat1,lon1,lat2,lon2):
    """Computes great-circle distance between lat-lon coordinates on a sphere
    with radius R"""
    phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
    term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
    return 2*R*np.arcsin(np.sqrt(term))

def bearing(lat1,lon1,lat2,lon2):
    """Compute the bearing between two points"""
    lat1,lon1,lat2,lon2 = np.deg2rad([lat1,lon1,lat2,lon2])
    x = np.cos(lat2)*np.sin(lon2-lon1)
    y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    return np.degrees(np.arctan2(x,y))%360

def step(lat,lon,bearing,distance):
    """Compute the lat-lon coordinates of a point given another point, a
    bearing, and a distance. R = radius of the earth."""
    lat,lon,bearing = np.deg2rad(lat),np.deg2rad(lon),np.deg2rad(bearing)
    delta = distance/R
    lat2 = np.arcsin(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(bearing))
    lon2 = lon+np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat),
                          np.cos(delta)-np.sin(lat)*np.sin(lat2))
    return np.degrees(lat2),np.degrees(lon2)

def calc_length(magnitude,delta_logl):
    a = 0.5233956445903871     # slope
    b = 1.0974498706605313     # intercept

    mu_logl = a*magnitude + b
    logl = mu_logl + delta_logl
    return 10**logl

def calc_width(magnitude,delta_logw):
    m = 0.29922483873212863   # slope
    c = 2.608734705074858     # y intercept

    mu_logw = m*magnitude + c
    logw = mu_logw + delta_logw
    return 10**logw

def calc_slip(magnitude,length,width,mu=4e10):
    return 10**(1.5*magnitude+9.05-np.log10(mu*length*width))

def out_of_bounds(subfault_params,bounds):
    """Returns true if any subfault lies outside of the bounds, or intersects with
    the surface"""

    # check if subfaults are outside bounds
    lats = np.array(subfault_params['latitude'])
    lons = np.array(subfault_params['longitude'])
    edge1 = step(lats,lons,strike,length/2,R)
    edge2 = step(lats,lons,strike-180,length/2,R)
    corner1 = step(edge1[0],edge1[1],strike+90,width/2)
    corner2 = step(edge1[0],edge1[1],strike-90,width/2)
    corner3 = step(edge2[0],edge2[1],strike+90,width/2)
    corner4 = step(edge2[0],edge2[1],strike-90,width/2)
    corners = np.hstack((corner1,corner2,corner3,corner4))
    if np.any(corners[0] < bounds['lat_min']) or np.any(corners[0] > bounds['lat_max']):
        return True
    if np.any(corners[1] < bounds['lon_min']) or np.any(corners[1] > bounds['lon_max']):
        return True

    # check if subfaults intersect surface
    for _,subfault in subfault_params.iterrows():
        if subfault['depth'] < .5*subfault['width']*np.sin(np.deg2rad(subfault['dip']])):
            return True

    return False
