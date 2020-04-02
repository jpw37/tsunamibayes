import numpy as np

class Fault:
    """A class for data relating to the fault"""
    def __init__(self,R,name):
        """
        Paramters
        ---------
        name : string
            Name of the fault
        """
        self.R = R
        self.name = name

    @staticmethod
    def haversine(R,lat1,lon1,lat2,lon2):
        """Computes great-circle distance between lat-lon coordinates on a sphere
        with radius R"""
        phi1,phi2,lam1,lam2 = np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon1),np.deg2rad(lon2)
        term = np.sin(.5*(phi2-phi1))**2+np.cos(phi1)*np.cos(phi2)*np.sin(.5*(lam2-lam1))**2
        return 2*R*np.arcsin(np.sqrt(term))

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

    def strike_from_lat_lon(self,lat,lon):
        pass

    def depth_from_lat_lon(self,lat,lon):
        pass

    def dip_from_lat_lon(self,lat,lon):
        pass

    def depth_map(self,lat,lon):
        pass

    def strike_map(self,lat,lon):
        pass

    def dip_map(self,lat,lon):
        pass

    def split_rect(self,lat,lon,length,width,depth_offset=0,n=11,m=3):
        """
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
        depth_offset : float
            Noise parameter for sampling depth (meters)
        n : int
            Number of splits along length
        m : int
            Number of splits along width

        Returns:
        --------
        rectangles : ndarray 5x(n*m) matrix
            Lats,Lons,Strikes,Dips,Depths of each subrectangle
        sublength : float
            The length of each subrectangle
        subwidth : float
            The width of each subrectangle
        """
        R = self.R
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
        bearing1 = self.strike_from_lat_lon(lat,lon)
        bearing2 = (bearing1-180)%360
        lat1,lon1 = lat,lon
        lat2,lon2 = lat,lon
        for i in range(1,(n - 1)//2+1):
            for j in range(n_steps):
                lat1,lon1 = self.step(lat1,lon1,bearing1,length_step,R)
                lat2,lon2 = self.step(lat2,lon2,bearing2,length_step,R)
                bearing1 = self.strike_from_lat_lon(lat1, lon1)
                bearing2 = (self.strike_from_lat_lon(lat2, lon2)-180)%360
            lats[(n-1)//2+i] = lat1
            lats[(n-1)//2-i] = lat2
            lons[(n-1)//2+i] = lon1
            lons[(n-1)//2-i] = lon2

        strikes = self.strike_map(np.vstack((lats,lons)).T)
        dips = self.dip_map(np.vstack((lats,lons)).T)
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
                templats1,templons1 = self.step(templats1,templons1,dipward,width_step*np.cos(np.deg2rad(tempdips1)),R)
                templats2,templons2 = self.step(templats2,templons2,dipward,-width_step*np.cos(np.deg2rad(tempdips2)),R)
                tempdips1 = self.dip_map(np.vstack((templats1,templons1)).T)
                tempdips2 = self.dip_map(np.vstack((templats2,templons2)).T)
            Lats[(m-1)//2+i] = templats1
            Lats[(m-1)//2-i] = templats2
            Lons[(m-1)//2+i] = templons1
            Lons[(m-1)//2-i] = templons2
            Strikes[(m-1)//2+i] = self.strike_map(np.vstack((templats1,templons1)).T)
            Strikes[(m-1)//2-i] = self.strike_map(np.vstack((templats2,templons2)).T)
            Dips[(m-1)//2+i] = tempdips1
            Dips[(m-1)//2-i] = tempdips2

        Depths = self.depth_map(np.vstack((Lats.flatten(),Lons.flatten())).T) + depth_offset
        data = [Lats,Lons,Strikes,Dips,Depths]
        data = [arr.flatten() for arr in data]
        rectangles = np.array(data).T
        return rectangles, sublength, subwidth
