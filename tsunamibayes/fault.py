from .gaussian_process_regressor import GPR
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import multivariate_normal
from .utils import displace, haversine, bearing


class BaseFault:
    """A class for data relating to the fault"""
    def __init__(self,bounds):
        """Creates and initializes the BaseFault object with specified lat/lon bounds.
        However, one must use a subclass to call to the functions of this parent class.

        Parameters
        ----------
        bounds : dict
            The dictionary of the upper and lower limits for latitude/longitude for the model.
            Keys are 'lon_min','lon_max','lat_min', 'lat_max', with associated (float) values.
        """
        self.bounds = bounds

    def depth_map(self,lat,lon):
        raise NotImplementedError("depth_map must be implemented in classes "
                                  "inheriting from BaseFault")

    def strike_map(self,lat,lon):
        raise NotImplementedError("strike_map must be implemented in classes "
                                  "inheriting from BaseFault")

    def dip_map(self,lat,lon):
        raise NotImplementedError("dip_map must be implemented in classes "
                                  "inheriting from BaseFault")

    def subfault_split(self,lat,lon,length,width,slip,depth_offset=0,rake=90,n=11,m=3):
        """Splits a given Okada rectangle into a collection of subfaults fit
        to the geometry of the fault.

        Parameters
        ----------
        lat : float
            Latitude coordinate for the sample (degrees)
        lon : float
            Longitude coordinate for the sample (degrees)
        length : float
            Length of the original Okada rectangle (meters)
        width : float
            Width of Okada rectangle (meters)
        slip : float
            The slip of the fault (meters). Total displacement of fault.
        depth_offset : float
            Offset for depth (meters), Optional. Defaults to 0.
        rake : float
            Rake parameter (degrees), Optional. Defaults to 90.
        n : int
            Number of splits along length, Optional. Defaults to 11. MUST BE AN ODD NUMBER.
        m : int
            Number of splits along width, Optional. Defaults to 3. MUST BE AN ODD NUMBER.

        Returns:
        --------
        subfault_params : pandas DataFrame
            The 2-d DataFrame whose columns are (ndarrays) of the Okada parameters
            and whose rows contain the associated data (float values)  for each subfault.
        """
        n_steps = 8                         #What is this n_steps used for?
        length_step = length/(n*n_steps)
        width_step = width/(m*n_steps)
        sublength = length/n
        subwidth = width/m

        lats = np.empty(n)
        lons = np.empty(n)
        lats[(n - 1)//2] = lat      #index [(n - 1)//2] is the middle of the rectangle split. For default values, this is index 5.
        lons[(n - 1)//2] = lon      #These assign the lat/lon values to the middle index of the lats/lons arrays.

        # add strikeward and anti-strikeward centers
        bearing1 = self.strike_map(lat,lon)         #Calculates the strike of the initial lat/lon point. Returns just a scalar value!
        bearing2 = (bearing1-180)%360               #Calculates the strike angle in the opposite direction, 180 difference from bearing1.
        lat1,lon1 = lat,lon                         #Creates lat1, lon1 variables to hold each iterative value as we displace the coordinates and add to the lats/lons np.arrays.
        lat2,lon2 = lat,lon
        for i in range(1,(n - 1)//2+1):             #For splitting rectangles in the 'n-axis', which is the strike-ward axis.
            for j in range(n_steps):                #Finds the values of the adjacent subfault split rectangle using iterative steps.
                lat1,lon1 = displace(lat1,lon1,bearing1,length_step)    #Takes a small step along bearing1.
                lat2,lon2 = displace(lat2,lon2,bearing2,length_step)    #Takes a small step along bearing2.
                bearing1 = self.strike_map(lat1, lon1)                  #Calculates new strike angle for bearing1 based on lat/lon of step. Returns just a scalar!
                bearing2 = (self.strike_map(lat2, lon2)-180)%360        #Calculates new strike angle for bearing2 based on lat/lon of step.
            lats[(n-1)//2+i] = lat1         #After the loop has taken n_steps, add the calculated lat coordinate to the index for the above split.
            lats[(n-1)//2-i] = lat2         #Same as above, adds the lat coordinate to the index of the split below the center point.
            lons[(n-1)//2+i] = lon1         #Same as above, just for lon. Notice how we are layering around the center point as we move away.
            lons[(n-1)//2-i] = lon2         #THESE ARRAYS ARE ALL size (n,)

        strikes = self.strike_map(lats,lons)        #Computes the strikes for all of the subfault splits alogn the strikeward (n) axis. Has dimension (n,)
        dips = self.dip_map(lats,lons)              #The dip will all be the same for this axis??
        dipward = (strikes+90)%360                  #Creates an array of angles perpendicular (+90) to the strike angels for each point. Has dimension (n,)

        Lats = np.empty((m,n))                      #WE SHOULD REALLY RENAME THESE VECTORS, THIS MAKES IT CONFUSING
        Lons = np.empty((m,n))
        Strikes = np.empty((m,n))                   #YEAH LET'S NOT JUST CAPATILZE THIS STUFF...
        Dips = np.empty((m,n))
        Lats[(m-1)//2] = lats                       #Assigns array of lats from the strikeward (n-axis) calculations to middle row of Lats.
        Lons[(m-1)//2] = lons                       #Same as above, same action for Lats, Lons, Strikes, and Dips.
        Strikes[(m-1)//2] = strikes                 #These np.arrays are size (n,)
        Dips[(m-1)//2] = dips

        # add dipward and antidipward centers

        #WHY DO WE CALCULATE THE DIP HERE, BUT NOT FOR THE N-AXIS?
        templats1,templons1 = lats.copy(),lons.copy()   #Copies the lats, lons, and dips as placeholder variables as we expand in the m-axis using our step size.
        templats2,templons2 = lats.copy(),lons.copy()
        tempdips1,tempdips2 = dips.copy(),dips.copy()
        for i in range(1,(m - 1)//2+1):                 #Iterate over the dipward/antidipward axis (m-axis)
            for j in range(n_steps):                    #Calculate the dips, strikes and displacements over these little steps.
                templats1,templons1 = displace(templats1,templons1,dipward,width_step*np.cos(np.deg2rad(tempdips1)))        #Diplaces each set of points along the dipward angle. These are arrays of size (n,)
                templats2,templons2 = displace(templats2,templons2,dipward,-width_step*np.cos(np.deg2rad(tempdips2)))       #Displaces all the points along the antidipward angle.
                tempdips1 = self.dip_map(templats1,templons1)       #Computes the dip for the set of points, returns array of size (n,)
                tempdips2 = self.dip_map(templats2,templons2)
            Lats[(m-1)//2+i] = templats1    #Stores the latest lats and lons after taking n_steps along the dipward/antidipward angle.
            Lats[(m-1)//2-i] = templats2    #Adds these arrays to the next outer index in the Lats matrix, adds them along the m-axis.
            Lons[(m-1)//2+i] = templons1
            Lons[(m-1)//2-i] = templons2
            Strikes[(m-1)//2+i] = self.strike_map(templats1,templons1)      #Computes strike of the templats/templons, returns array of size (n,)
            Strikes[(m-1)//2-i] = self.strike_map(templats2,templons2)
            Dips[(m-1)//2+i] = tempdips1
            Dips[(m-1)//2-i] = tempdips2
            #The above matrices are now full of the correct values of for all the splits along the n and m axes.

        Depths = self.depth_map(Lats.flatten(),Lons.flatten()) + depth_offset   #Calculates the depths for the entire matrix of Lats/Lons. Returns a flattened array (m*n,)
        data = [Lats,Lons,Strikes,Dips,Depths]      #Store all of our data in a list.
        data = [arr.flatten() for arr in data]      #Iterate through all of the data, flatten the lat/lon/strike/dip matrices
        subfault_params = pd.DataFrame(np.array(data).T,columns=['latitude','longitude','strike','dip','depth'])    #Transform data into a pandas dataframe.
        subfault_params['length'] = sublength       #sublength = length/n, the same value is stored for each subfault. Size(m*n,)
        subfault_params['width'] = subwidth         #subwidth = width/m.
        subfault_params['slip'] = slip              #Usually these are the default values passed in, will generally all be the same for each subfault for both slip and rake.
        subfault_params['rake'] = rake

        return subfault_params

    def subfault_split2(self,lat,lon,length,width,slip,depth_offset=0,rake=90,m=11,n=3,rake_type='uniform',slip_dist='uniform'):
        """Splits a given Okada rectangle into a collection of subfaults fit
        to the geometry of the fault.
        Takes into account the type of rake and slip distrubution present in the fault data.

        Parameters
        ----------
        lat : float
            Latitude coordinate (degrees)
        lon : float
            Longitude coordinate (degrees)
        length : float
            Length of rectangle (meters)
        width : float
            Width of rectangle (meters)
        slip : float
            Slip parameter (meters) Total displacement of fault
        depth_offset : float
            Offset for depth (meters), optional. Defaults to 0.
        m : int
            Number of splits along length, optional. Defaults to 11.
        n : int
            Number of splits along width, optional. Defaults to 3.
        rake : string
            The type of orientation of block movement during a fault rupture, optional. Defaults to 'uniform'.
        slip_dist : string
            The shape of the slip distribution, optional. Defaults to 'uniform'.

        Returns:
        --------
        subfault_params : pandas DataFrame
            The 2-d DataFrame whose columns are (ndarrays) of the Okada parameters
            and whose rows contain the associated data (float values) for each subfault.
        """
        n_steps = 8
        length_step = length/(m*n_steps)
        width_step = width/(n*n_steps)
        sublength = length/m
        subwidth = width/n

        lats = np.empty(m)
        lons = np.empty(m)

        mid_l = int((m-1)/2)
        mid_w = int((n-1)/2)
        lats[mid_l] = lat
        lons[mid_l] = lon

        # add strikeward and anti-strikeward centers
        bearing1 = self.strike_map(lat,lon)
        bearing2 = (bearing1-180)%360
        lat1,lon1 = lat,lon
        lat2,lon2 = lat,lon
        for i in range(1,mid_l+1):
            for j in range(n_steps):
                lat1,lon1 = displace(lat1,lon1,bearing1,length_step)
                lat2,lon2 = displace(lat2,lon2,bearing2,length_step)
                bearing1 = self.strike_map(lat1, lon1)
                bearing2 = (self.strike_map(lat2, lon2)-180)%360
            lats[mid_l+i],lons[mid_l+i] = lat1,lon1
            lats[mid_l-i],lons[mid_l-i] = lat2,lon2

        strikes = self.strike_map(lats,lons)
        dips = self.dip_map(lats,lons)
        dipward = (strikes+90)%360

        Lats = np.empty((m,n))
        Lons = np.empty((m,n))
        Strikes = np.empty((m,n))
        Dips = np.empty((m,n))
        Lats[:,mid_w] = lats
        Lons[:,mid_w] = lons
        Strikes[:,mid_w] = strikes
        Dips[:,mid_w] = dips

        # add dipward and antidipward centers
        templats1,templons1 = lats.copy(),lons.copy()
        templats2,templons2 = lats.copy(),lons.copy()
        tempdips1,tempdips2 = dips.copy(),dips.copy()
        for i in range(1,mid_w+1):
            for j in range(n_steps):
                templats1,templons1 = displace(templats1,templons1,dipward,width_step*np.cos(np.deg2rad(tempdips1)))
                templats2,templons2 = displace(templats2,templons2,dipward,-width_step*np.cos(np.deg2rad(tempdips2)))
                tempdips1 = self.dip_map(templats1,templons1)
                tempdips2 = self.dip_map(templats2,templons2)
            Lats[:,mid_w+i],Lons[:,mid_w+i] = templats1,templons1
            Lats[:,mid_w-i],Lons[:,mid_w-i] = templats2,templons2
            Strikes[:,mid_w+i] = self.strike_map(templats1,templons1)
            Strikes[:,mid_w-i] = self.strike_map(templats2,templons2)
            Dips[:,mid_w+i] = tempdips1
            Dips[:,mid_w-i] = tempdips2

        if slip_dist == 'elliptical':
            if mid_w == 0: mid_w = 1
            dist = multivariate_normal(mean=np.array([mid_l,mid_w]),cov=np.array([[(mid_l/1.8)**2,0],[0,(mid_w/1.5)**2]]))
            X,Y = np.meshgrid(np.arange(m),np.arange(n),indexing='ij')
            Slips = dist.pdf(np.array([X,Y]).T).T
            Slips *= slip/Slips.mean()
        else:
            Slips = slip*np.ones_like(Lats)

        Depths = self.depth_map(Lats.flatten(),Lons.flatten()) + depth_offset
        data = [Lats,Lons,Strikes,Dips,Depths,Slips]
        data = [arr.flatten() for arr in data]
        subfault_params = pd.DataFrame(np.array(data).T,columns=['latitude','longitude','strike','dip','depth','slip'])
        subfault_params['length'] = sublength
        subfault_params['width'] = subwidth

        #central strike
        if rake_type == 'parallel':
            central_strike = self.strike_map(lat,lon)
            subfault_params['rake'] = rake-central_strike+Strikes.flatten()
        else:
            subfault_params['rake'] = rake

        return subfault_params

    def subfault_split_RefCurve(self,lat,lon,length,width,slip,depth_offset=0,dip_offset = 0,rake_offset = 0,rake=90,n=11,m=3):
        """Splits a given Okada rectangle into a collection of subfaults fit
        to the geometry of the fault.

        Parameters
        ----------
        lat : float
            Latitude coordinate for the sample (degrees)
        lon : float
            Longitude coordinate for the sample (degrees)
        length : float
            Length of the original Okada rectangle (meters)
        width : float
            Width of Okada rectangle (meters)
        slip : float
            The slip of the fault (meters). Total displacement of fault.
        depth_offset : float
            Offset for depth (meters), Optional. Defaults to 0.
        rake : float
            Rake parameter (degrees), Optional. Defaults to 90.
        n : int
            Number of splits along length, Optional. Defaults to 11. MUST BE AN ODD NUMBER.
        m : int
            Number of splits along width, Optional. Defaults to 3. MUST BE AN ODD NUMBER.

        Returns:
        --------
        subfault_params : pandas DataFrame
            The 2-d DataFrame whose columns are (ndarrays) of the Okada parameters
            and whose rows contain the associated data (float values)  for each subfault.
        """
        n_steps = 8
        length_step = length/(n*n_steps)
        width_step = width/(m*n_steps)
        sublength = length/n
        subwidth = width/m

        lats = np.empty((n,1))              #Added dimension (n,1) to lats/long curves to interface with distance function axes.
        lons = np.empty((n,1))
        lats[(n - 1)//2] = lat
        lons[(n - 1)//2] = lon

        #THE BELOW PART SEEMS TO BE DOING FINE SINCE WE DEAL ONLY WITH SCALARS HERE...
        # add strikeward and anti-strikeward centers
        bearing1 = self.strike_map(lat,lon) #Calculates the strike just for one point, returns a scalar.
        bearing2 = (bearing1-180)%360       #Adds 180 to the strike angle to compute antistrikeward angle.
        lat1,lon1 = lat,lon
        lat2,lon2 = lat,lon
        for i in range(1,(n - 1)//2+1):     #Computes the lats/lons for the subfaults along the n-axis.
            for j in range(n_steps):        #Takes n_steps to compute each adjacent subfault.
                lat1,lon1 = displace(lat1,lon1,bearing1,length_step)    #Displaces each lat1 lon1 by the small step.
                lat2,lon2 = displace(lat2,lon2,bearing2,length_step)
                bearing1 = self.strike_map(lat1, lon1)                  #Computes the new bearing from the strike angle of that point.
                bearing2 = (self.strike_map(lat2, lon2)-180)%360
            lats[(n-1)//2+i] = lat1
            lats[(n-1)//2-i] = lat2
            lons[(n-1)//2+i] = lon1
            lons[(n-1)//2-i] = lon2
            #The onlye problem here is that maybe these are size (n,1) and they should be (n,)

        #Interesting...Even though lats lons have size (n,1), the following return arrays of size (n,)
        strikes = self.strike_map(lats,lons)        #We pass in lats lons as arrays of size (n,1), this returns an array of size (n)
        dips = self.dip_map(lats,lons)
        dipward = (strikes+90)%360                  #Dipward is an array of size (n,) with all of the strike angles + 90 degrees.

        Lats = np.empty((m,n))
        Lons = np.empty((m,n))
        Strikes = np.empty((m,n))
        Dips = np.empty((m,n))
        Lats[(m-1)//2] = lats.flatten()
        Lons[(m-1)//2] = lons.flatten()
        Strikes[(m-1)//2] = strikes
        Dips[(m-1)//2] = dips

        # add dipward and antidipward centers
        templats1,templons1 = lats.copy(),lons.copy()
        templats2,templons2 = lats.copy(),lons.copy()
        tempdips1,tempdips2 = dips.copy(),dips.copy()
        for i in range(1,(m - 1)//2+1):
            for j in range(n_steps):
                templats1,templons1 = displace(templats1.flatten(),templons1.flatten(),dipward,width_step*np.cos(np.deg2rad(tempdips1)))        #To pass into displace, we needed to flatten templats and templons.
                templats2,templons2 = displace(templats2.flatten(),templons2.flatten(),dipward,-width_step*np.cos(np.deg2rad(tempdips2)))       #These return arrays of size (n,)
                templats1, templons1 = templats1[:,np.newaxis], templons1[:,np.newaxis]
                templats2, templons2 = templats2[:,np.newaxis], templons2[:,np.newaxis]
                tempdips1 = self.dip_map(templats1,templons1)   #In order to interface with dip_map, we need to add back a new axis so that we pass in arrays of size (n,1)
                tempdips2 = self.dip_map(templats2,templons2)   #After the dips are computed, this returns an array of size (n,)
            Lats[(m-1)//2+i] = templats1.flatten()
            Lats[(m-1)//2-i] = templats2.flatten()
            Lons[(m-1)//2+i] = templons1.flatten()
            Lons[(m-1)//2-i] = templons2.flatten()
            Strikes[(m-1)//2+i] = self.strike_map(templats1,templons1)
            Strikes[(m-1)//2-i] = self.strike_map(templats2,templons2)
            Dips[(m-1)//2+i] = tempdips1
            Dips[(m-1)//2-i] = tempdips2
            #From all this we learn that, strike_map, dip_map, and depth_map must take in arrays of size (n,1), but they return arrays of size (n,)

        Depths = self.depth_map(Lats.flatten()[:,np.newaxis],Lons.flatten()[:,np.newaxis]) + depth_offset   #Calculates the depths for the entire matrix of Lats/Lons. Returns a flattened array (m*n,)
        Dips = self.dip_map(Lats.flatten()[:,np.newaxis],Lons.flatten()[:,np.newaxis]) + dip_offset   #Calculates the dips for the entire matrix of Lats/Lons. Returns a flattened array (m*n,)
        data = [Lats,Lons,Strikes,Dips,Depths]
        data = [arr.flatten() for arr in data]
        subfault_params = pd.DataFrame(np.array(data).T,columns=['latitude','longitude','strike','dip','depth'])
        subfault_params['length'] = sublength   #The length of each subfault, should be the same for all splits. sublength = length/n
        subfault_params['width'] = subwidth     #subwidth = width/m, same for each subfault.
        subfault_params['slip'] = slip
        subfault_params['rake'] = (rake + rake_offset)

        return subfault_params

class GridFault(BaseFault):
    """A subclass that inherits from BaseFault.  """
    def __init__(self,lat,lon,depth,dip,strike,bounds):
        """Initializes all the correct variables for the GridFault subclass.
        Creates interpolations for depth, dip, and strike
        which will be used later to determine values at specific points.

        Parameters
        ----------
        lat : array_like of floats
            An ndarray of (floats) containing the latitude coordinates along the fault line. (degrees)
        lon : array_like of floats
            An ndarray of (floats) containinng the longitude coordinates along the fault line. (degrees)
        depth : array_like of floats
            An ndarray of (floats) containinng data for the depth along the fault line. (meters)
        dip : array_like of floats
            An ndarray of (floats) containinng data for the dip along the fault line. (degrees)
            The information for the angles at which the plane dips downward from the top edge
            (a positive angle between 0 and 90)
        strike : array_like of floats
            An ndarray of (floats) containinng data for the strike orientation along the fault. (degrees)
        bounds : dict
            The dictionary of the upper and lower limits for latitude/longitude.
            Contains keys: lat_min, lon_min, lat_max, lon_max with associated (float) values.
        """
        super().__init__(bounds)
        self.depth_interp = RegularGridInterpolator((lat,lon),depth,bounds_error=False)
        self.dip_interp = RegularGridInterpolator((lat,lon),dip,bounds_error=False)
        self.strike_interp = RegularGridInterpolator((lat,lon),strike,bounds_error=False)
        self.lat = lat
        self.lon = lon
        self.depth = np.nan_to_num(depth)
        self.dip = dip
        self.strike = strike

    @classmethod
    def from_slab2(cls,depth_file,dip_file,strike_file,bounds):
        """This provides the aternate constructor for Gridfault that accepts data files,
        reads those data files, and returns a constructor that accepts keyword arguments
        (lat, lon, depth, dip, strike).

        Parameters
        ----------
        depth_file : text file of floats
            The file containing the depth (in meters) readings along the fault.
        dip_file : text file of floats
            The file containing the angle measurements of dip all along the fault (degrees).
        strike_file : text file of floats
            The file containing the strike orientations in radians along the fault. (degrees)
        bounds : dict
            The dictionary of the upper and lower limits for latitude/longitude.
            Contains keys: lat_min, lon_min, lat_max, lon_max with associated (float) values.

        Returns
        -------
        gridfault : Gridfault
            Alternate object constructor.
        """
        arrays = load_slab2_data(depth_file,dip_file,strike_file,bounds)
        return cls(bounds=bounds,**arrays)

    def depth_map(self,lat,lon):
        """Interpolates the depth from a specified geographic point.

        Parameters
        ----------
        lat : float, array_like of floats
            The array of latitude values along the fault, or can be single-valued float. (degrees)

        lon : float, array_like of floats
            The array of longitude values along the fault, or can be single-valued float. (degrees)

        Returns
        -------
        arr : ndarray of floats
            The array of interpolated depths (meters) associated to the pairs of coordinates passed-in.
        -or-
        arr[0] : float
            The single value interpolated depth, when only a simple coordinate is passed in for lat and lon.
        """
        arr = self.depth_interp(np.array([lat,lon]).T)
        if isinstance(lat,float) or isinstance(lat,int):
            return arr[0]
        else:
            return arr

    def dip_map(self,lat,lon):
        """Interpolates the dip from a specified geographic point.

        Parameters
        ----------
        lat : float, array_like of floats
            The array of latitude values along the fault, or can be single-coordinate float. (degrees)

        lon : float, array_like of floats
            The array of longitude values along the fault, or can be single-coordinate float. (degrees)

        Returns
        -------
        arr : ndarray of floats
            The array of interpolated dip measurements (degrees) associated to the pairs of coordinates passed-in.
        -or-
        arr[0] : float
            The single value interpolated dip, when only a simple coordinate is passed in for lat and lon.
        """
        arr = self.dip_interp(np.array([lat,lon]).T)
        if isinstance(lat,float) or isinstance(lat,int):
            return arr[0]
        else:
            return arr

    def strike_map(self,lat,lon):
        """Interpolates the strike from a specified geographic point.

        Parameters
        ----------
        lat : float, array_like of floats
            The array of latitude values along the fault, or can be single-coordinate float. (degrees)

        lon : float, array_like of floats
            The array of longitude values along the fault, or can be single-coordinate float. (degrees)

        Returns
        -------
        arr : ndarray of floats
            The array of interpolated strike (degrees) associated to the pairs of coordinates passed-in.
        -or-
        arr[0] : float
            The single value interpolated strike, when only a simple coordinate is passed in for lat and lon.
        """
        arr = self.strike_interp(np.array([lat,lon]).T)
        if isinstance(lat,float) or isinstance(lat,int):
            return arr[0]
        else:
            return arr

def load_slab2_data(depth_file,dip_file,strike_file,bounds):
    """Loads the depth, dip, and strike data for the fault and returns a dictionary of arrays
    that contain the 'slices' of this data between a specified set of latitude and longitude bounds.

    Parameters
    ----------
    depth_file : text file of floats
        The file containing the depth (in meters) readings along the fault.
    dip_file : text file of floats
        The file containing the angle measurements of dip all along the fault (degrees).
    strike_file : text file of floats
        The file containing the strike orientations in radians along the fault. (degrees)
    bounds : dict
            The dictionary of the upper and lower limits for latitude/longitude.
            Contains keys: lat_min, lon_min, lat_max, lon_max with associated (float) values.

    Returns
    -------
    arrays : dict
        A dictionary containing with keys: lat, lon, depth, dip, strike and their associated
        ndarrays of (float) values within the upper and lower geographical bounds of the fault.
    """
    # load depth file, extract lat/lon grid, make latitude array in increasing order
    depth = np.loadtxt(depth_file,delimiter=',')
    lat = np.unique(depth[:,1])
    lon = np.unique(depth[:,0])

    # make depth positive and in meters, load strike and dip, reshape arrays
    depth = -1000*depth[:,2]
    dip = np.loadtxt(dip_file,delimiter=',')[:,2]
    strike = np.loadtxt(strike_file,delimiter=',')[:,2]
    depth = depth.reshape((lat.shape[0],lon.shape[0]))[::-1]
    dip = dip.reshape((lat.shape[0],lon.shape[0]))[::-1]
    strike = strike.reshape((lat.shape[0],lon.shape[0]))[::-1]

    # compute indices corresponding to the bounds
    i,j = np.nonzero((lat >= bounds['lat_min'])&(lat <= bounds['lat_max']))[0][[0,-1]]
    k,l = np.nonzero((lon >= bounds['lon_min'])&(lon <= bounds['lon_max']))[0][[0,-1]]

    # pass data to the GridFault constructor
    arrays = {'lat':lat[i:j+1],
              'lon':lon[k:l+1],
              'depth':depth[i:j+1,k:l+1],
              'dip':dip[i:j+1,k:l+1],
              'strike':strike[i:j+1,k:l+1]}

    return arrays

def save_slab2_npz(depth_file,dip_file,strike_file,bounds,save_path):
    """Saves the dictionary of arrays for lat, lon, depth, dip, strike to a .npz file.

    Parameters
    ----------
    depth_file : text file of floats
        The file containing the depth (in meters) readings along the fault.
    dip_file : text file of floats
        The file containing the angle measurements of dip all along the fault (degrees).
    strike_file : text file of floats
        The file containing the strike orientations in radians along the fault. (degrees)
    bounds : dict
        The dictionary of the upper and lower limits for latitude/longitude.
        Contains keys: lat_min, lon_min, lat_max, lon_max with associated (float) values.
    save_path : string or file
        The location or path where the data is to be saved.
    """
    arrays = load_slab2_data(depth_file,dip_file,strike_file,bounds)
    np.savez(save_path,**arrays)


class ReferenceCurveFault(BaseFault):
    """A class for data relating to the fault"""
    def __init__(self,latpts,lonpts,strikepts,depth_curve,dip_curve,bounds,smoothing=50000):
        """Initializes all the necessary variables for the subclass.

        Parameters
        ----------
        latpts : (N,) ndarray
            Array containing latitudes of reference points on the fault
        lonpts : (N,) ndarray
            Array containing longitudes of reference points on the fault
        strikepts : (N,) ndarray
            Array containing the strike angles at reference points on the fault
        depth_curve : callable
            Function giving the depth along perpendicular transects of the fault.
            The argument to the function is assumed to be the signed distance from
            the reference fault points, with the positive direction being dipward
        dip_curve : callable
            Function giving the dip angle along perpendicular transects of the fault.
            The argument to the function is assumed to be the signed distance from
            the reference fault points, with the positive direction being dipward
        bounds : dict
            Dictionary containing the model bounds. Keys are 'lon_min','lon_max',
            'lat_min', 'lat_max'
        smoothing : int
            The smoothing coefficient used later in computing the weighted mean
            strike angle.
            Default is set to 50000.
        """
        super().__init__(bounds)
        self.latpts = latpts
        self.lonpts = lonpts
        self.strikepts = strikepts
        self.depth_curve = depth_curve
        self.dip_curve = dip_curve
        self.smoothing = smoothing


    @staticmethod #Probably obsolete, we shoudn't need this too much...
    def quad_interp(x,y):
        """Computes a quadratic curve for depth passing through three points.

        Parameters
        ----------
        x : (3,) ndarray
            The array of 3 x-coordinates (floats) that the curve must pass through.
        y : (3,) ndarray
            The array of 3 y-coordinates (floats) that the curve must pass through.

        Returns
        -------
        curve : (3,) ndarray
            The array of floats who values are the coefficients [a,b,c] for the
            quadratic equation ax^2+bx+c that passes through the three points
            specified in the x and y ndarrays.
        """
        A = np.ones((3,3))
        A[:,0] = x**2
        A[:,1] = x
        return np.linalg.solve(A,y)

    @staticmethod
    def depth_dip_curves(x,y,surf_dist):
        """Returns callable functions for the depth and dip curves passing through
        three points.

        Parameters
        ----------
        x : (3,) ndarray
            Distances from fault refernce points, in meters
        y : (3,) ndarray
            Depth values
        surf_dist : float
            Distance from fault reference points to the fault's intersection with
            the Earth's surface

        Returns
        -------
        depth_curve : (function)
            The callable function that accepts one positional arugment (ie. x)
            and returns the quadratic depth curve passing through the 3 specified points.
        dip_curve : (function)
            The callable function that accepts one positional arugment (ie. x)
            and returns the quadratic dip curve passing through the 3 specified points.
        """
        c = ReferenceCurveFault.quad_interp(x,y)
        depth_curve = lambda x: (c[0]*x**2 + c[1]*x + c[2])*(x > -np.abs(surf_dist))
        dip_curve = lambda x: np.degrees(np.arctan(2*c[0]*x + c[1]))*(x > -np.abs(surf_dist))
        return depth_curve, dip_curve

    @staticmethod
    def circmean(angles,weights):
        """Computes the weighted mean of angles.

        Parameters
        ----------
        angles : array_like
            The ndarray of angles (in degrees) for which the mean is to be computed.
        weights : array_like
            The ndarray of weights (floats) associated with the given angles, and
            hence must have the same dimsion as 'angles'.

        Returns
        -------
        mean : float
            The computed weighted mean.
        """
        x,y = np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))
        mean = np.degrees(np.arctan2(weights@y,weights@x))
        return mean

    @staticmethod
    def side(lat,lon,fault_lat,fault_lon,strike):
        """Computes on which side of the fault that a given point lies, given
        the closet point on the fault and the strike angle there. This is done
        by comparing the latitudes/longitudes, depending on the strike angle.

        Parameters
        ----------
        lat : float
            The latitude coordiante (degrees) of the point on eithe side of the fault.
        lon : float
            The longitude coordiante (degrees) of the point on eithe side of the fault.
        fault_lat : float
            The latitude of the closest point (degrees) on the fault relative to the given
            coordinate passed in.
        fault_lon : float
            The longitude of the closest point (degrees) on the fault relative to the given
            coordinate passed in.
        strike : float
            The strike angle of the closest point (degrees) on the fault relative to the given
            coordinate passed in.

        Returns
        -------
        1 -or- -1 : int
            1 if down-dip of the fault, -1 if up-dip.
        """
        # Scalar option.
        if np.isscalar(strike):
            if 0 <= (strike+45)%360 < 90:
                return np.sign((lon-fault_lon+180)%360-180)
            elif 90 <= (strike+45)%360 < 180:
                return -np.sign(lat-fault_lat)
            elif 180 <= (strike+45)%360 < 270:
                return -np.sign((lon-fault_lon+180)%360-180)
            else:
                return np.sign(lat-fault_lat)

        # Vectorized option.
        fault_lat = fault_lat.reshape(lat.shape)
        fault_lon = fault_lon.reshape(lon.shape)
        strike = strike.reshape(lat.shape)
        sides = np.empty(lat.shape)

        mask1 = (0 <= (strike+45)%360) & ((strike+45)%360 < 90)
        sides[mask1] = np.sign((lon[mask1]-fault_lon[mask1]+180)%360-180)

        mask2 = (90 <= (strike+45)%360) & ((strike+45)%360 < 180)
        sides[mask2] = -np.sign(lat[mask2]-fault_lat[mask2])

        mask3 = (180 <= (strike+45)%360) & ((strike+45)%360 < 270)
        sides[mask3] = -np.sign((lon[mask3]-fault_lon[mask3]+180)%360-180)

        mask4 = ~(mask1 | mask2 | mask3)
        sides[mask4] = np.sign(lat[mask4]-fault_lat[mask4])

        return sides


    def distance(self,lat,lon,retclose=False):
        """Computes the distance from a given lat/lon coordinate to the fault.
        Optionally return the index of the closest point.

        Parameters
        ----------
        lat : float
            The latitude coordinate (degrees) near the fault for which the distance
            is to be calculated.
        lon : float
            The latitude coordinate (degrees) near the fault for which the distance
            is to be calculated.
        retclose : bool
            The boolean flag that indicates whether or not to return the index of the
            closest point. Default is false.

        Returns
        -------
        min_distance : float
            The computed minimum distance (in meters) from the point to the fault.
        index : int
            (Optionally) returns the index of the closest point on the fault.
        """
        distances = haversine(
            lat,
            lon,
            self.latpts,
            self.lonpts
        )
        #I think we need an if statement here for when the distances array doesn't have more than 1 dimension.
        if retclose:
            return distances.min(axis=1), distances.argmin(axis=1)
        else:
            return distances.min(axis=1)


    def strike_map(self,lat,lon):
        """Computes the weighted mean strike angle.

        Parameters
        ----------
        lat : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        mean : float or np.array(n,)
            The computed weighted mean for the strike angles. (degrees)
        """
        distances = haversine(
            lat,
            lon,
            self.latpts,
            self.lonpts
        )
        weights = np.exp(-distances/self.smoothing)
        strikes = (ReferenceCurveFault.circmean(self.strikepts,weights)%360)
        return strikes


    def depth_map(self,lat,lon,retside=False):
        """Computes the depth for a given lat-lon coordinate.

        Parameters
        ----------
        lat : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
        retside : bool
            A boolean flag that determines whether the function also returns
            the side of the given point (dipward or antidipward) when set to
            True.
            Default is False.

        Returns
        -------
        depth : float or np.array (n,)
            The interpolated depth (in meters) for the given coordinate.
        side : signed int or np.array (n,)
            (Optionally) Returns 1 if the given point is dipward of the fault,
            -1 if antidipward.
        """
        distance,idx = self.distance(lat,lon,retclose=True)

        side = ReferenceCurveFault.side(
            lat,
            lon,
            self.latpts[idx],
            self.lonpts[idx],
            self.strikepts[idx]
        )

        # Change the distance anywhere idx == 0 or idx == len(self.latpts)-1
        mask = (idx == 0) | (idx == (len(self.latpts)-1))
        bear = bearing(
            self.latpts[idx[mask]],
            self.lonpts[idx[mask]],
            np.squeeze(lat[idx[mask]]),
            np.squeeze(lon[idx[mask]])
        )
        distance[idx[mask]] = distance[idx[mask]]*np.sin(
            np.deg2rad(self.strikepts[idx[mask]]-bear)
        )
        side[idx[mask]] = -np.sign(distance[idx[mask]][:,np.newaxis])
        distance[idx[mask]] = np.abs(distance[idx[mask]])

        depth = self.depth_curve(np.squeeze(side)*distance)

        return ((depth,side) if retside else depth)


    def dip_map(self,lat,lon):
        """Computes the dip for a given lat-lon coordinate.

        Parameters
        ----------
        lat : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        dip : float or np.array (n,)
            The interpolated dip (in degrees) for the given coordinate.
        """
        distance,idx = self.distance(lat,lon,retclose=True)

<<<<<<< HEAD
        side = ReferenceCurveFault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        print(np.shape(idx))
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        return self.dip_curve(side*distance)
=======
        side = ReferenceCurveFault.side(
            lat,
            lon,
            self.latpts[idx],
            self.lonpts[idx],
            self.strikepts[idx]
        )

        mask = (idx == 0) | (idx == (len(self.latpts)-1))
        bear = bearing(
            self.latpts[idx[mask]],
            self.lonpts[idx[mask]],
            np.squeeze(lat[idx[mask]]),
            np.squeeze(lon[idx[mask]])
        )
        distance[idx[mask]] = distance[idx[mask]]*np.sin(
            np.deg2rad(self.strikepts[idx[mask]]-bear)
        )
        side[idx[mask]] = -np.sign(distance[idx[mask]][:,np.newaxis])
        distance[idx[mask]] = np.abs(distance[idx[mask]])

        return self.dip_curve(np.squeeze(side)*distance)
>>>>>>> 9b4df4a... Vectorize ReferenceCurveFault and finish MultiFault


    def depth_dip(self,lat,lon):
        """Computes both the depth and dip for a given lat-lon coordinate.

        Parameters
        ----------
        lat : float
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        depth : float
            The interpolated depth (in meters) for the given coordinate.
        dip : float
            The interpolated dip (in degrees) for the given coordinate.
        """
        distance,idx = self.distance(lat,lon,retclose=True)

        side = ReferenceCurveFault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        return self.depth_curve(side*distance),self.dip_curve(side*distance)


    def distance_strike(self,lat,lon):
        """Computes both the distance from the fault, and the weighted mean strike angle.

        Parameters
        ----------
        lat : float
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        min_distance : float
            The computed minimum distance (in meters) from the point to the fault.
        mean : float
            The computed weighted mean for the strike angles. (degrees)
        """
        distances = haversine(lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/self.smoothing)
        #weights /= weights.sum()
        return distances.min(), ReferenceCurveFault.circmean(self.strikepts,weights)%360


class GaussianProcessFault(BaseFault):
    """A class for fault-related data, where depth_map, dip_map, and
    strike_map are found by training a Gaussian process on sample depth,
    dip, and strike data.
    """
    def __init__(
        self, lats, lons, depths, dips, strikes, bounds, kers, noise=None
        ):
        """Initializes all the necessary variables for the subclass.

        Parameters
        ----------
        lats : (N,) ndarray
            Array containing latitudes of points on the fault.
        lons : (N,) ndarray
            Array containing longitudes of points on the fault.
        depths : (N,) ndarray
            Array containing depth value at each latitude-longitude pair.
        dips : (N,) ndarray
            Array containing dip value at each latitude-longitude pair.
        strikes : (N,) ndarray
            Array containing strike value at each latitude-longitude pair.
        bounds : dict
            Dictionary containing the model bounds. Keys are 'lon_min',
            'lon_max', 'lat_min', 'lat_max'.
        kernels : dict
            Dictionary containing the kernel functions for each of the
            three Gaussian processes to fit. Keys are 'depth', 'dip', and
            'strike'.
        noise_levels : dict
            Dictionary containing the noise level for each of the three
            Gaussian processes. Keys are 'depth', 'dip', and 'strike'.
            If
        """
        super().__init__(bounds)

        if noise is None:
            noise = {'depth': 1, 'dip': 1, 'strike': 1}

        # Initialize the GPRs.
        self.depth_gpr = GPR(
            kernel=kers['depth'],
            noise_level=noise['depth']
        )
        self.dip_gpr = GPR(
            kernel=kers['dip'],
            noise_level=noise['dip']
        )
        self.strike_gpr = GPR(
            kernel=kers['strike'],
            noise_level=noise['strike']
        )

        # Train each of the GPRs.
        X = np.vstack([lats, lons]).T
        self.depth_gpr.fit(X,depths)
        self.dip_gpr.fit(X,dips)
        self.strike_gpr.fit(X,strikes)


    @classmethod
    def from_file(cls, filename, bounds, kers, noise):
        """Alternate constructor for the GaussianProcessFault class.

        Parameters
        ----------
        filename : string
            String containing the filepath to the .npz file containing the
            necessary data.
            The file is expected to be an .npz file with the following
            attributes:
                lats : (N,) ndarray
                lons : (N,) ndarray
                depths : (N,) ndarray
                dips : (N,) ndarray
                strikes : (N,) ndarray
        """
        arrays = np.load(filename)
        return cls(**arrays, bounds=bounds, kers=kers, noise=noise)


    def strike_map(self,lat,lon,return_std=False):
        """Computes the weighted mean strike angle.

        Parameters
        ----------
        lat : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        mean : float or np.array(n,)
            The computed weighted mean for the strike angles. (degrees)
        """
        latlon = np.vstack([lat, lon]).T
        return self.strike_gpr.predict(latlon,return_std=return_std)


    def depth_map(self,lat,lon,return_std=False):
        """Computes the depth for a given lat-lon coordinate.

        Parameters
        ----------
        lat : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        depth : float or np.array (n,)
            The interpolated depth (in meters) for the given coordinate.
        side : signed int or np.array (n,)
            (Optionally) Returns 1 if the given point is dipward of the
            fault, -1 if antidipward.
        """
        latlon = np.vstack([lat, lon]).T
        return self.depth_gpr.predict(latlon,return_std=return_std)


    def dip_map(self,lat,lon,return_std=False):
        """Computes the dip for a given lat-lon coordinate.

        Parameters
        ----------
        lat : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.
            is to be calculated.
        lon : float or np.array (n,1)
            The latitude coordinate (degrees) near the fault.

        Returns
        -------
        dip : float or np.array (n,)
            The interpolated dip (in degrees) for the given coordinate.
        """
        latlon = np.vstack([lat, lon]).T
        return self.dip_gpr.predict(latlon,return_std=return_std)
