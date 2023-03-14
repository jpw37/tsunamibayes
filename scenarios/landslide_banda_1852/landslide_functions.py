from clawpack import geoclaw
from clawpack.geoclaw import topotools as topo
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy
from itertools import product
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Read in topography file
topo_file = topo.Topography()
topo_file.read('./data/topo/base.tt3', topo_type=3)

#create a function to interpolate bathymetry 
bathy = interpolate.RectBivariateSpline(topo_file.x, topo_file.y, topo_file.Z.transpose())

def deg_to_km(deg):
    return deg / .008

def km_to_deg(km):
    return km * .008

def m_to_deg(meters):
    return meters / 111000

def deg_to_m(deg):
    return deg * 111000

  
def u(t,theta,v_0,down,slide_params):
    """
    Find the velocity of landslide, with initial velocity v_0, moving down a slope with angle theta. 
    Parameters:
    theta (float): angle of slope in radians
    t (float): number of seconds landslide moves before velocity is measured
    v_0 (float): initial velocity
    """
    p_w,p_s,g,w,l,d,f,C_F = slide_params
    V = w*l*d
    V_w = w*l*d 
    m_w = p_w*V_w 
    m_s = p_s*V 
    C_m = m_w/m_s
    # define a, b and c according to our original pde
    a = p_s*V
    b = (p_s - p_w)*g*V*(np.sin(theta) - f*np.cos(theta))
    c = p_w*((C_F*l*d)/2)
    np.sqrt(np.abs(b) / np.sqrt(c))
    # if we are moving in the opposite direction of the slope we need to ensure our friction force is in the opposite direction of movement
    if down == False:
        b = (p_s - p_w)*g*V*(-np.sin(theta) - f*np.cos(theta))

    # our pde breaks into the following cases
    if b < 0 and v_0 >= 0:
        vel = - np.sqrt(-b)*np.tan(((np.sqrt(-b)*np.sqrt(c)*t)/a) 
                                    - np.arctan(np.sqrt(c)*v_0 / np.sqrt(-b))) / np.sqrt(c)
    if b > 0:
        if v_0 < np.sqrt(b) / np.sqrt(c):
            vel = np.sqrt(b)*np.tanh(((np.sqrt(b)*np.sqrt(c)*t)/a) 
                                    + np.arctanh(np.sqrt(c)*v_0 / np.sqrt(b))) / np.sqrt(c)
        if v_0 > np.sqrt(b) / np.sqrt(c):
            vel = np.sqrt(b)*(1/np.tanh(((np.sqrt(b)*np.sqrt(c)*t)/a) 
                                     + np.arctanh(np.sqrt(b) / (np.sqrt(c)*v_0)))) / np.sqrt(c)

    if isinstance(vel, float):
        if vel >=0:    
            return vel
        else:
            return 0
    else:
        vel[vel<0] = 0
        return vel


def simpsons(a,b,n,u,theta,v_0,down,slide_params):
    """
    Find the area under our velocity function u. 
    Parameters:
    a (float): beginning number of seconds
    b (float): end number of seconds
    n (float): number of intervals in approximation
    theta (float): angle of slope, parameter for u
    v_0 (float): initial velocity, parameter for u
    """
    
    delta = (b - a) / n
    endpoints = [a]
    c = a
    for i in range(n):
        c = c + delta
        endpoints.append(c)
    inside_coeff = [4,2]*n
    coeff = [1] + inside_coeff[:(n-1)] + [1]

    return (delta / 3) * np.sum(u(np.array(endpoints), theta, v_0,down,slide_params) * coeff)


def center_mass_path(coordinates, topo_file, seconds, max_iters, initial_vel, simpsons_n,slide_params):
    """
    Find the path of centers of mass with starting position and initial velocity.
    Parameters:
    coordinates (list, [.,.]): starting coordinated in lon/lat
    topo_file: topography file 
    seconds (int): number of seconds for each step
    max_iters (int): maximum number of steps to take
    initial_vel (list, [.,.]): initial velocity of slide
    simpsons_n (int): number of intervals to use in simpsons approximation of the position
    """
    path = []
    grid_vel = []
    # find delta from original grid (less likely results will come from the interpolation)
    delta = np.max([topo_file.delta[0],topo_file.delta[0]])
    print('Delta', delta)
    long_min, long_max, lat_min, lat_max = topo_file.extent
#     v_lon_s, v_lat_s = [initial_vel[0],initial_vel[1]]
    thetas = []
    slope_vel = []
    # note we will change signs[0] to be in the downslope direction, this may not be [1,1]
    signs_slope = []
    
    elev1 = bathy(coordinates[0] + delta, coordinates[1])[0][0]
    elev2 = bathy(coordinates[0] - delta, coordinates[1])[0][0]
    elev3 = bathy(coordinates[0], coordinates[1] + delta)[0][0]
    elev4 = bathy(coordinates[0], coordinates[1] - delta)[0][0]
    print('Coordinates')
    print(coordinates[0] + delta, coordinates[1])
    print(coordinates[0] - delta, coordinates[1])
    print(coordinates[0], coordinates[1] + delta)
    print(coordinates[0], coordinates[1] - delta)
    print('Elevation',elev1,elev2,elev3,elev4)
    # determine the direction of the slope (positive is up and right respectively)
    sign_lon = (elev2 - elev1) / np.abs(elev2 - elev1)
    sign_lat = (elev4 - elev3) / np.abs(elev4 - elev3)
        
    # find the angle of the slope determined by elevation offsets
    theta_lon = np.arctan(np.abs((elev1 - elev2)) / deg_to_m(2*delta))
    theta_lat = np.arctan(np.abs((elev3 - elev4)) / deg_to_m(2*delta))
    ratio = theta_lon / theta_lat
    v_lat_s = initial_vel / (ratio + 1) 
    v_lon_s = ratio * v_lat_s
    print(v_lat_s)
    print(v_lon_s)
    signs_vel = [sign_lon,sign_lat]

    for i in range(max_iters):
        
        # get the elevation at our current position
        elevation = bathy(coordinates[0],coordinates[1])[0][0]
 
        # get elevations at offsets in each dimension
        elev1 = bathy(coordinates[0] + delta, coordinates[1])[0][0]
        elev2 = bathy(coordinates[0] - delta, coordinates[1])[0][0]
        elev3 = bathy(coordinates[0], coordinates[1] + delta)[0][0]
        elev4 = bathy(coordinates[0], coordinates[1] - delta)[0][0]
        # store path
        path.append([elevation, coordinates[0], coordinates[1]])

        # determine the direction of the slope (positive is up and right respectively)
        sign_lon = (elev2 - elev1) / np.abs(elev2 - elev1)
        sign_lat = (elev4 - elev3) / np.abs(elev4 - elev3)
        signs_slope.append([sign_lon,sign_lat])
        
        # find the angle of the slope determined by elevation offsets
        theta_lon = np.arctan(np.abs((elev1 - elev2)) / deg_to_m(2*delta))
        theta_lat = np.arctan(np.abs((elev3 - elev4)) / deg_to_m(2*delta))
        thetas.append(np.array([theta_lon,theta_lat]))

    
        # if we are changing the direction we are moving in longitude
        if signs_slope[-1][0] != signs_vel[0]:
            v_lon_s = u(seconds,theta_lon,np.abs(v_lon_s), False,slide_params)
            meters_lon_s = simpsons(0,seconds,simpsons_n,u,theta_lon,np.abs(v_lon_s), False,slide_params)

        if signs_slope[-1][1] != signs_vel[1]:
            v_lat_s = u(seconds,theta_lat,np.abs(v_lat_s), False,slide_params)
            meters_lat_s = simpsons(0,seconds,simpsons_n,u,theta_lat,np.abs(v_lat_s), False,slide_params)
                
        # moving the same direction as before in longitude
        if signs_slope[-1][0] == signs_vel[0]:
            v_lon_s = u(seconds,theta_lon, np.abs(v_lon_s),  True,slide_params)
            meters_lon_s = simpsons(0,seconds,simpsons_n,u,theta_lon,np.abs(v_lon_s),True,slide_params)

        if signs_slope[-1][1] == signs_vel[1]:
            v_lat_s = u(seconds,theta_lat, np.abs(v_lat_s), True,slide_params)
            meters_lat_s = simpsons(0,seconds,simpsons_n,u,theta_lat,np.abs(v_lat_s), True,slide_params)
       
        # store our velocity on the slope
        v_lon_s = np.abs(v_lon_s) * signs_vel[0]
        v_lat_s = np.abs(v_lat_s) * signs_vel[1]
        
        meters_lon_s = np.abs(meters_lon_s) * signs_vel[0]
        meters_lat_s = np.abs(meters_lat_s) * signs_vel[1]
        
        slope_vel.append([v_lon_s,v_lat_s])
        
        # find the velocity on the grid
        v_lat_g = np.cos(theta_lat)*v_lat_s
        v_lon_g = np.cos(theta_lon)*v_lon_s
        vel_g = np.array([v_lon_g,v_lat_g])
        grid_vel.append(vel_g)
        
        # find the distance moved on the grid
        meters_lat_g = np.cos(theta_lat)*meters_lat_s
        meters_lon_g = np.cos(theta_lon)*meters_lon_s
        meters_g = np.array([meters_lon_g,meters_lat_g])
        
        # move on grid
        coordinates = coordinates + m_to_deg(meters_g)
        
        # break statement using velocity
        if i >=30:
            if np.linalg.norm(slope_vel[-1]) < 1:
                break
        # out of bounds
        if coordinates[0] - delta < long_min or coordinates[0] + delta > long_max:
            print(f"out of longitude range at iteration {i+1}")
            break
        if coordinates[1] - delta < lat_min or coordinates[1] + delta > lat_max:
            print(f"out of lattitude range at iteration {i+1}")
            break
            
    path = np.array(path)
    points = np.array([path.T[1],path.T[2]]).T

    return points, grid_vel, slope_vel, thetas




def F(A, theta):
    """Rotate the points in A about the origin by theta radians.
    Parameters:
    A ((2,n) ndarray): Array containing points in R2 stored as columns.
    theta (float): The rotation angle in radians.
    """
    B = np.array([[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    C = B @ A
    return C

def getfillPoints(corners, X):
    """
    Return the points from X inside a rectangle.
    Parameters:
    corners ((n,2) ndarray): Array containing points corresponding to the corners of our rectangle stored as rows.
    Must be in cyclical order.
    X ((n,2) ndarray): Array containing points, stored as rows, to check.
    """
    polygon = Polygon(corners)
    fillPoints = []
    for x in X:
        point = Point(x)
        if polygon.contains(point):
            fillPoints.append(x)
    return fillPoints

def landslide_boxes(points,thickness,width,length,intermediate_steps,topo_file):
    #append topography files
    files = []
    #append previous angles in case future angle is 0
    angles = [0]
    #step size based on the number of files wanted
    step = len(points) // intermediate_steps
    #list of discrete center of mass (lat/lon)
    centers_degrees = []
    #list of discrete center of mass (index in topo file)
    centers_position = []
    #distance between grid points on topography file
    delta_x = topo_file.delta[0]
    delta_y = topo_file.delta[1]
    #sign of angle rotation
    signs = []
    #dimesions of the box in grid steps taken
    x_step = np.max(np.array([round((m_to_deg(width) / delta_x) / 2),1]))
    y_step = np.max(np.array([round((m_to_deg(length) / delta_y) / 2),1]))
    #print information about length and width
    print(f"Width used was {deg_to_m(x_step*2 * delta_x)} m")
    print(f"Length used was {deg_to_m(y_step*2 * delta_y)} m")
    
    for i in np.arange(0,len(points),step):
        #get lists of the indices as well as lon/lat coordinates of our centers of mass
        center = [points[i][0],points[i][1]]        
        x_center = np.argmin(np.abs(topo_file.x - center[0]))
        y_center = np.argmin(np.abs(topo_file.y - center[1]))
        center = [x_center,y_center]
        centers_degrees.append([topo_file.x[x_center],topo_file.y[y_center]])
        centers_position.append(center)
    
    for j,i in enumerate(np.arange(0,len(points),step)):
        # current center of mass
        center = [points[i][0],points[i][1]]        
        x_center = np.argmin(np.abs(topo_file.x - center[0]))
        y_center = np.argmin(np.abs(topo_file.y - center[1]))
        center = [x_center,y_center]
          
        # find angle betwee current and next angle of mass (unless we are at the last point where we use the current and previous)
        if j == len(centers_degrees) - 1:
            point2 = np.array(centers_position[-2]) - np.array(center)
            #case where little distance is moved and multiple points may snap to the same point on the grid
            if np.any(point2) == 0:
                angle = angles[-1]
                angles.append(angle)
            else:
                #the last point uses the opposite sign angle as our last step (since it is current and previous)
                angle = -signs[-1]*np.arccos(point2[0]/np.sqrt(point2[0]**2 + point2[1]**2))
                angles.append(angle)
            
        else:
            point2 = np.array(centers_position[j + 1]) - np.array(center)
            #the sign for our angle of rotation depends on if we are in an upper or lower quadrant
            if np.sign(point2[1]) >= 0:
                sign = -1
            else:
                sign = 1
            signs.append(sign)
            #case where little distance is moved and multiple points may snap to the same point on the grid
            if np.any(point2) == 0:
                angle = angles[-1]
                angles.append(angle)
            else:
                angle = sign*np.arccos(point2[0]/np.sqrt(point2[0]**2 + point2[1]**2))
                angles.append(angle)
                
                
        # read in topography file and set elevation to 0
        topo_file_next = topo.Topography()
        topo_file_next.read('./data/topo/base.tt3', topo_type=3)
        topo_file_next.Z = np.zeros_like(topo_file_next.Z)
        
        # get the coordinates of our box shifted centered at the origin
        box_coordinates_x =  np.arange(center[0] - x_step,center[0] + x_step + 1,1) - center[0]
        box_coordinates_y = np.arange(center[1] - y_step, center[1] + y_step + 1, 1) - center[1]
        box_grid = np.meshgrid(box_coordinates_x,box_coordinates_y)
        box_cord_list = np.append(box_grid[0].reshape(-1,1),box_grid[1].reshape(-1,1),axis=1).T
        
        # get the corners of our box
        x_max = np.max(box_cord_list[0])
        x_min = np.min(box_cord_list[0])
        y_max = np.max(box_cord_list[1])
        y_min = np.min(box_cord_list[1])
        # get the corners of our rotated box
        corners = np.array([[x_min, y_min],[x_min, y_max],[x_max,y_max],[x_max, y_min]])
        corners_rot = F(corners.T,angle).T

        # get gridpoints to check when we fill our polygon
        max_ = np.max(box_cord_list.T)
        x = np.arange(-max_,max_)
        X = np.meshgrid(x,x)
        grid_points = np.append(X[0].reshape(-1,1),X[1].reshape(-1,1),axis=1).T
        # now we fill missing points in the rectangle
        rect_filled = getfillPoints(corners_rot,grid_points.T)
        # rotate our original rectangle
        rotate = F(box_cord_list, angle)
        # snap to grid points
        rotate = np.rint(rotate).astype(int)
        # combine with filled rectangle
        end_rect = np.vstack((rotate.T,rect_filled))
        end_rect = np.unique(end_rect, axis=0)
        # shift back to center
        end_rect = end_rect.T
        end_rect[0] += center[0]
        end_rect[1] += center[1]
        end_rect = end_rect.T
        end_rect = np.array([end_rect.T[1],end_rect.T[0]])
        #add mass to topography file
        topo_file_next.Z[tuple(end_rect.astype(int))] += thickness 
        files.append(topo_file_next.Z)
    
    return files,centers_degrees, step


