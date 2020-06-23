import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import multivariate_normal
from .utils import displace

class BaseFault:
    """A class for data relating to the fault"""
    def __init__(self,bounds):
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

        Parameters:
        -----------
        lat : float
            Latitude coordinate (degrees)
        lon : float
            Longitude coordinate (degrees)
        length : float
            Length of rectangle (meters)
        width : float
            Width of rectangle (meters)
        depth_offset : float, optional
            Offset for depth (meters). Defaults to 0.
        rake : float
            Rake parameter (degrees). Defaults to 90.
        n : int, optional
            Number of splits along length. Defaults to 11.
        m : int, optional
            Number of splits along width. Defaults to 3.

        Returns:
        --------
        subfault_params : pandas DataFrame
            DataFrame containing the Okada parameters for each subfault
        """
        n_steps = 8
        length_step = length/(n*n_steps)
        width_step = width/(m*n_steps)
        sublength = length/n
        subwidth = width/m

        lats = np.empty(n)
        lons = np.empty(n)
        lats[(n - 1)//2] = lat
        lons[(n - 1)//2] = lon

        # add strikeward and anti-strikeward centers
        bearing1 = self.strike_map(lat,lon)
        bearing2 = (bearing1-180)%360
        lat1,lon1 = lat,lon
        lat2,lon2 = lat,lon
        for i in range(1,(n - 1)//2+1):
            for j in range(n_steps):
                lat1,lon1 = displace(lat1,lon1,bearing1,length_step)
                lat2,lon2 = displace(lat2,lon2,bearing2,length_step)
                bearing1 = self.strike_map(lat1, lon1)
                bearing2 = (self.strike_map(lat2, lon2)-180)%360
            lats[(n-1)//2+i] = lat1
            lats[(n-1)//2-i] = lat2
            lons[(n-1)//2+i] = lon1
            lons[(n-1)//2-i] = lon2

        strikes = self.strike_map(lats,lons)
        dips = self.dip_map(lats,lons)
        dipward = (strikes+90)%360

        Lats = np.empty((m,n))
        Lons = np.empty((m,n))
        Strikes = np.empty((m,n))
        Dips = np.empty((m,n))
        Lats[(m-1)//2] = lats
        Lons[(m-1)//2] = lons
        Strikes[(m-1)//2] = strikes
        Dips[(m-1)//2] = dips

        # add dipward and antidipward centers
        templats1,templons1 = lats.copy(),lons.copy()
        templats2,templons2 = lats.copy(),lons.copy()
        tempdips1,tempdips2 = dips.copy(),dips.copy()
        for i in range(1,(m - 1)//2+1):
            for j in range(n_steps):
                templats1,templons1 = displace(templats1,templons1,dipward,width_step*np.cos(np.deg2rad(tempdips1)))
                templats2,templons2 = displace(templats2,templons2,dipward,-width_step*np.cos(np.deg2rad(tempdips2)))
                tempdips1 = self.dip_map(templats1,templons1)
                tempdips2 = self.dip_map(templats2,templons2)
            Lats[(m-1)//2+i] = templats1
            Lats[(m-1)//2-i] = templats2
            Lons[(m-1)//2+i] = templons1
            Lons[(m-1)//2-i] = templons2
            Strikes[(m-1)//2+i] = self.strike_map(templats1,templons1)
            Strikes[(m-1)//2-i] = self.strike_map(templats2,templons2)
            Dips[(m-1)//2+i] = tempdips1
            Dips[(m-1)//2-i] = tempdips2

        Depths = self.depth_map(Lats.flatten(),Lons.flatten()) + depth_offset
        data = [Lats,Lons,Strikes,Dips,Depths]
        data = [arr.flatten() for arr in data]
        subfault_params = pd.DataFrame(np.array(data).T,columns=['latitude','longitude','strike','dip','depth'])
        subfault_params['length'] = sublength
        subfault_params['width'] = subwidth
        subfault_params['slip'] = slip
        subfault_params['rake'] = rake

        return subfault_params

    def subfault_split2(self,lat,lon,length,width,slip,depth_offset=0,rake=90,m=11,n=3,rake_type='uniform',slip_dist='uniform'):
        """Splits a given Okada rectangle into a collection of subfaults fit
        to the geometry of the fault.

        Parameters:
        -----------
        lat : float
            Latitude coordinate (degrees)
        lon : float
            Longitude coordinate (degrees)
        length : float
            Length of rectangle (meters)
        width : float
            Width of rectangle (meters)
        depth_offset : float, optional
            Offset for depth (meters). Defaults to 0.
        rake : float
            Rake parameter (degrees). Defaults to 90.
        m : int, optional
            Number of splits along length. Defaults to 11.
        n : int, optional
            Number of splits along width. Defaults to 3.

        Returns:
        --------
        subfault_params : pandas DataFrame
            DataFrame containing the Okada parameters for each subfault
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

class GridFault(BaseFault):
    def __init__(self,lat,lon,depth,dip,strike,bounds):
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
        arrays = load_slab2_data(depth_file,dip_file,strike_file,bounds)
        return cls(bounds=bounds,**arrays)

    def depth_map(self,lat,lon):
        arr = self.depth_interp(np.array([lat,lon]).T)
        if isinstance(lat,float) or isinstance(lat,int):
            return arr[0]
        else:
            return arr

    def dip_map(self,lat,lon):
        arr = self.dip_interp(np.array([lat,lon]).T)
        if isinstance(lat,float) or isinstance(lat,int):
            return arr[0]
        else:
            return arr

    def strike_map(self,lat,lon):
        arr = self.strike_interp(np.array([lat,lon]).T)
        if isinstance(lat,float) or isinstance(lat,int):
            return arr[0]
        else:
            return arr

def load_slab2_data(depth_file,dip_file,strike_file,bounds):
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
    arrays = load_slab2_data(depth_file,dip_file,strike_file,bounds)
    np.savez(save_path,**arrays)

class ReferenceCurveFault(BaseFault):
    """A class for data relating to the fault"""
    def __init__(self,latpts,lonpts,strikepts,depth_curve,dip_curve,smoothing=50000,bounds):
        """
        Paramters
        ---------
        latpts : (N,) ndarray
            Array containing latitudes of reference points on the fault
        lonpts : (N,) ndarray
            Array containing longitudes of reference points on the fault
        strikepts : (N,) ndarray
            Array containing the strike angles at reference points on the fault
        R : float
            Local radius of the earth, in meters
        depth_curve : callable
            Function giving the depth along perpendicular transects of the fault.
            The argument to the function is assumed to be the signed distance from
            the reference fault points, with the positive direction being dipward
        dip_curve : callable
            Function giving the dip angle along perpendicular transects of the fault.
            The argument to the function is assumed to be the signed distance from
            the reference fault points, with the positive direction being dipward
        name : string
            Name of the fault

        """
        super.__init__(bounds)
        self.latpts = latpts
        self.lonpts = lonpts
        self.strikepts = strikepts
        self.depth_curve = depth_curve
        self.dip_curve = dip_curve
        self.smoothing = smoothing

    @staticmethod
    def quad_interp(x,y):
        """Computes a quadratic curve for depth passing through three points."""
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
        surf_dist = float
            Distance from fault reference points to the fault's intersection with
            the Earth's surface
        """
        c = Fault.quad_interp(x,y)
        depth_curve = lambda x: (c[0]*x**2 + c[1]*x + c[2])*(x > -np.abs(surf_dist))
        dip_curve = lambda x: np.degrees(np.arctan(2*c[0]*x + c[1]))*(x > -np.abs(surf_dist))
        return depth_curve, dip_curve

    @staticmethod
    def circmean(angles,weights):
        """Computes the weighted mean of angles"""
        x,y = np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))
        return np.degrees(np.arctan2(weights@y,weights@x))

    @staticmethod
    def side(lat,lon,fault_lat,fault_lon,strike):
        """Computes on which side of the fault that a given point lies, given
        the closet point on the fault and the strike angle there. This is done
        by comparing the latitudes/longitudes, depending on the strike angle.
        Returns 1 if dipward of the fault, -1 if antidipward.
        """
        if 0 <= (strike+45)%360 < 90:
            return np.sign((lon-fault_lon+180)%360-180)
        elif 90 <= (strike+45)%360 < 180:
            return -np.sign(lat-fault_lat)
        elif 180 <= (strike+45)%360 < 270:
            return -np.sign((lon-fault_lon+180)%360-180)
        else:
            return np.sign(lat-fault_lat)

    def distance(self,lat,lon,retclose=False):
        """Computes the distance from a given lat/lon coordinate to the fault.
        Optionally return the index of the closest point."""
        distances = Fault.haversine(self.R,lat,lon,self.latpts,self.lonpts)
        if retclose:
            return distances.min(), distances.argmin()
        else:
            return distances.min()

    def strike_from_lat_lon(self,lat,lon):
        """Computes the weighted mean strike angle"""
        distances = Fault.haversine(self.R,lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/self.smoothing)
        #weights /= weights.sum()
        return ReferenceCurveFault.circmean(self.strikepts,weights)%360

    def distance_strike(self,lat,lon):
        """Computes both the distance from the fault, and the weighted mean strike angle"""
        distances = Fault.haversine(self.R,lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/self.smoothing)
        #weights /= weights.sum()
        return distances.min(), ReferenceCurveFault.circmean(self.strikepts,weights)%360

    def depth_from_lat_lon(self,lat,lon,retside=False):
        """Computes the depth for a given lat-lon coordinate"""
        distance,idx = self.distance(lat,lon,retclose=True)

        side = ReferenceCurveFault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = Fault.bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        depth = self.depth_curve(side*distance)

        if retside: return depth,side
        else: return depth

    def dip_from_lat_lon(self,lat,lon):
        distance,idx = self.distance(lat,lon,retclose=True)

        side = ReferenceCurveFault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = Fault.bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        return self.dip_curve(side*distance)

    def depth_dip(self,lat,lon):
        distance,idx = self.distance(lat,lon,retclose=True)

        side = ReferenceCurveFault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = Fault.bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        return self.depth_curve(side*distance),self.dip_curve(side*distance)
