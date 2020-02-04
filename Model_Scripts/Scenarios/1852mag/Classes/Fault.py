import numpy as np

class Fault:
    """A class for data relating to the fault"""
    def __init__(self,latpts,lonpts,strikepts,depth_curve,dip_curve,R,name):
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
        self.latpts = latpts
        self.lonpts = lonpts
        self.strikepts = strikepts
        self.depth_curve = depth_curve
        self.dip_curve = dip_curve
        self.R = R
        self.name = name
        self.smoothing = 50000

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
    def haversine(R,lat1,lon1,lat2,lon2):
        """Computes great-circle distance between lat-lon coordinates on a sphere
        with radius R"""
        phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
        term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
        return 2*R*np.arcsin(np.sqrt(term))

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

    @staticmethod
    def bearing(lat1,lon1,lat2,lon2):
        """Compute the bearing between two points"""
        lat1,lon1,lat2,lon2 = np.deg2rad([lat1,lon1,lat2,lon2])
        x = np.cos(lat2)*np.sin(lon2-lon1)
        y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
        return np.degrees(np.arctan2(x,y))%360

    @staticmethod
    def step(lat,lon,bearing,distance,R):
        """Compute the lat-lon coordinates of a point given another point, a
        bearing, and a distance. R = radius of the earth."""
        lat,lon,bearing = np.deg2rad(lat),np.deg2rad(lon),np.deg2rad(bearing)
        delta = distance/R
        lat2 = np.arcsin(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(bearing))
        lon2 = lon+np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat),np.cos(delta)-np.sin(lat)*np.sin(lat2))
        return np.degrees(lat2),np.degrees(lon2)

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
        return Fault.circmean(self.strikepts,weights)%360

    def distance_strike(self,lat,lon):
        """Computes both the distance from the fault, and the weighted mean strike angle"""
        distances = Fault.haversine(self.R,lat,lon,self.latpts,self.lonpts)
        weights = np.exp(-distances/self.smoothing)
        #weights /= weights.sum()
        return distances.min(),Fault.circmean(self.strikepts,weights)%360

    def depth_from_lat_lon(self,lat,lon,retside=False):
        """Computes the depth for a given lat-lon coordinate"""
        distance,idx = self.distance(lat,lon,retclose=True)

        side = Fault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = Fault.bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        depth = self.depth_curve(side*distance)

        # if depth < 0: depth = 0

        if retside: return depth,side
        else: return depth

    def dip_from_lat_lon(self,lat,lon):
        distance,idx = self.distance(lat,lon,retclose=True)

        side = Fault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = Fault.bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        return self.dip_curve(side*distance)

    def depth_dip(self,lat,lon):
        distance,idx = self.distance(lat,lon,retclose=True)

        side = Fault.side(lat,lon,self.latpts[idx],self.lonpts[idx],self.strikepts[idx])
        if idx == 0 or idx == len(self.latpts)-1:
            bearing = Fault.bearing(self.latpts[idx],self.lonpts[idx],lat,lon)
            distance = distance*np.sin(np.deg2rad(self.strikepts[idx]-bearing))
            side = -np.sign(distance)
            distance = np.abs(distance)
        return self.depth_curve(side*distance),self.dip_curve(side*distance)
