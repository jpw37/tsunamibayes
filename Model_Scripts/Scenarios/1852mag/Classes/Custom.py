"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
from scipy import stats

from MCMC import MCMC
from Prior import Prior,LatLonPrior
from Fault import Fault, GridFault

from scipy.stats import truncnorm

class Custom(MCMC):
    """
    Use this class to create a custom prior and custom earthquake parameters MCMC draws
    When the Variable for use_custom is set to true, this class will be used as the main MCMC class for the Scenario
    """
    def __init__(self):
        MCMC.__init__(self)
        self.sample_cols = ['Longitude', 'Latitude', 'Magnitude','DeltaLogL','DeltaLogW','DeltaDepth']
        self.proposal_cols = ['P-Longitude', 'P-Latitude', 'P-Magnitude','P-DeltaLogL','P-DeltaLogW','P-DeltaDepth']
        self.observation_cols = ['Mw', 'gauge 0 arrival', 'gauge 0 height', 'gauge 1 arrival', 'gauge 1 height', 'gauge 2 arrival', 'gauge 2 height', 'gauge 3 arrival', 'gauge 3 height', 'gauge 4 arrival', 'gauge 4 height', 'gauge 5 arrival', 'gauge 5 height', 'gauge 6 arrival', 'gauge 6 height', 'gauge 7 arrival', 'gauge 7 height']
        self.mw = 0
        self.length_split = 11
        self.width_split = 3
        self.num_rectangles = self.length_split * self.width_split
        cols = []
        for i in range(self.num_rectangles):
            cols += ['Latitude' + str(i+1)]
            cols += ['Longitude' + str(i+1)]
            cols += ['Strike' + str(i+1)]
            cols += ['Dip' + str(i+1)]
            cols += ['Depth' + str(i+1)]
        cols += [ 'Sublength', 'Subwidth', 'Slip', 'Rake']
        self.okada_cols = cols
        self.fault = self.build_fault()
        self.prior = self.build_priors()

    def build_fault(self):
        data = np.load("./InputData/bandadata.npz")
        R = 6377905
        return GridFault(data['lat'],data['lon'],data['depth'],data['depth_unc'],data['dip'],data['strike'],R,"Banda Arc")

    # def split_rect(self, lat, lon, strike, leng, num=3, method="Step"):
    #     """Split a given rectangle into 3 of equal length that more closely follow the
    #     curve of the fault.
    #
    #     Parameters:
    #         lat (float): latitude of center
    #         lon (float): longitude of center
    #         leng (float): length of the long edge (km)
    #
    #     Return:
    #         list rectangles represented by a list of parameters: [lat,long,strike,leng]
    #     """
    #     if num < 3 or num % 2!=1:
    #         raise ValueError("'num' must be an odd integer of at least 3!")
    #
    #
    #     """Spencer & Garret:
    #         The code now constructs a Fault object, which is just a container for
    #         the fault data, and has methods to compute distance and the strike map.
    #         We can also extend the class to manage the other Okada parameter maps.
    #         Until then, the important method is Fault.strike_from_lat_lon().
    #         Custom now initializes with a Fault instance, self.fault. So you can
    #         compute the strike map using self.fault.strike_from_lat_lon(lat,lon)
    #     """
    #
    #     # DEPRICATED
    #     # #Pulling prior of lon/lat information to contruct best fit approximaiton of strike
    #     # prior_lat = self.latlongstrikeprior[:,0]
    #     # prior_lon = self.latlongstrikeprior[:,1]
    #     # prior_strike = self.latlongstrikeprior[:,2]
    #     #
    #     # #Constructing best fit
    #     # A = np.vstack([np.ones(len(prior_lat)), prior_lat, prior_lon, prior_lat*prior_lon, prior_lat**2, prior_lon**2, prior_lat**2*prior_lon, prior_lon**2*prior_lat, prior_lat**3, prior_lon**3]).T
    #     # lat_long_bestfit = np.linalg.lstsq(A, prior_strike, rcond=None)[0]
    #     #
    #     # #strike/latitude linear regression
    #     # def strike_from_lat_long(lat, lon):
    #     #     temp_array = np.array([1, lat, lon, lat*lon, lat**2, lon**2, lat**2*lon, lon**2*lat, lat**3, lon**3])
    #     #     return temp_array @ lat_long_bestfit
    #         #line of best fit for strike given latitude
    #     # strike_from_lat = np.poly1d([-4.69107194e-01, -1.31232324e+01, -1.44327025e+02,
    #     #                            -7.82503768e+02, -2.13007839e+03, -2.40708004e+03])
    #     strike_from_lat_lon = self.fault.strike_from_lat_lon
    #     step = self.fault.step
    #     nleng= leng/num
    #     rects = []
    #     rects.append([lat, lon, strike, nleng])
    #
    #     if method == "Avg": # NOT IMPLEMENTED
    #         lat_temp=lat
    #         long_temp=lon
    #         for i in range((num - 1)//2):
    #             #Find the edge of the center rect and the strike at that point
    #             edge1_lat = lat_temp + nleng/222*np.cos(np.radians(strike))
    #             edge1_long = long_temp + nleng/222*np.sin(np.radians(strike))
    #             strike1 = strike_from_lat_long(edge1_lat,edge1_long)
    #             #Find the far egde of the adjacent rectangle
    #             end1_lat = edge1_lat + nleng/111*np.cos(np.radians(strike1))
    #             end1_long = edge1_long + nleng/111*np.sin(np.radians(strike1))
    #             #average the strike of the two points
    #             strike1 = (strike1 + strike_from_lat_long(end1_lat, end1_long))/2
    #             #find the coordinates of the rectangle using the new strike
    #             rect1_lat = edge1_lat + nleng/222*np.cos(np.radians(strike1))
    #             rect1_long = edge1_long + nleng/222*np.sin(np.radians(strike1))
    #
    #             rects.append([rect1_lat, rect1_long, strike1, nleng])
    #             lat_temp=rect1_lat
    #             long_temp=rect1_long
    #
    #         lat_temp=lat
    #         long_temp=lon
    #         for i in range((num - 1)//2):
    #             #Find the edge of the center rect and the strike at that point
    #             edge2_lat = lat_temp - nleng/222*np.cos(np.radians(strike))
    #             edge2_long = long_temp - nleng/222*np.sin(np.radians(strike))
    #             strike2 = strike_from_lat_long(edge2_lat,edge1_long)
    #             #Find the far egde of the adjacent rectangle
    #             end2_lat = edge1_lat - nleng/111*np.cos(np.radians(strike2))
    #             end2_long = edge1_long - nleng/111*np.sin(np.radians(strike2))
    #             #average the strike of the two points
    #             strike2 = (strike2 + strike_from_lat_long(end2_lat, end2_long))/2
    #             #find the coordinates of the rectangle using the new strike
    #             rect2_lat = edge2_lat - nleng/222*np.cos(np.radians(strike2))
    #             rect2_long = edge2_long - nleng/222*np.sin(np.radians(strike2))
    #
    #             rects.append([rect2_lat, rect2_long, strike2, nleng])
    #             lat_temp=rect2_lat
    #             long_temp=rect2_long
    #
    #     elif method == "Center": # NOT IMPLEMENTED
    #         lat_temp=lat
    #         long_temp=lon
    #         for i in range((num - 1)//2):
    #             edge1_lat = lat_temp + nleng/222*np.cos(np.radians(strike))
    #             edge1_long = long_temp + nleng/222*np.sin(np.radians(strike))
    #             strike1 = strike_from_lat_long(edge1_lat,edge1_long)
    #             mid1_lat = edge1_lat + nleng/222*np.cos(np.radians(strike1))
    #             mid1_long = edge1_long + nleng/222*np.sin(np.radians(strike1))
    #             strike1 = strike_from_lat_long(mid1_lat, mid1_long)
    #             rect1_lat = edge1_lat + nleng/222*np.cos(np.radians(strike1))
    #             rect1_long = edge1_long + nleng/222*np.sin(np.radians(strike1))
    #
    #             rects.append([rect1_lat, rect1_long, strike1, nleng])
    #             lat_temp=rect1_lat
    #             long_temp=rect1_long
    #
    #         lat_temp=lat
    #         long_temp=lon
    #         for i in range((num - 1)//2):
    #             edge2_lat = lat_temp - nleng/222*np.cos(np.radians(strike))
    #             edge2_long = long_temp - nleng/222*np.sin(np.radians(strike))
    #             strike2 = strike_from_lat_long(edge2_lat,edge1_long)
    #             mid2_lat = edge1_lat - nleng/222*np.cos(np.radians(strike2))
    #             mid2_long = edge1_long - nleng/222*np.sin(np.radians(strike2))
    #             strike2 = strike_from_lat_long(mid2_lat, mid2_long)
    #             rect2_lat = edge2_lat - nleng/222*np.cos(np.radians(strike2))
    #             rect2_long = edge2_long - nleng/222*np.sin(np.radians(strike2))
    #
    #             rects.append([rect2_lat, rect2_long, strike2, nleng])
    #             lat_temp=rect2_lat
    #             long_temp=rect2_long
    #
    #     elif method == "Step":
    #         #define step length
    #         num_steps = 8
    #         step_len = nleng/num_steps
    #
    #         #add rectangles in direction of positive strike
    #         bearing = strike
    #         step_lat = lat
    #         step_lon = lon
    #         for i in range((num - 1)//2):
    #             for i in range(num_steps):
    #                 step_lat,step_lon = step(step_lat,step_lon,bearing,step_len,self.fault.R)
    #                 bearing = strike_from_lat_lon(step_lat, step_lon)
    #             rects.append([step_lat, step_lon, strike_from_lat_lon(step_lat, step_lon), nleng])
    #
    #         #add rectangles in direction of negative strike
    #         bearing = (strike-180)%360
    #         step_lat = lat
    #         step_lon = lon
    #         for i in range((num - 1)//2):
    #             for i in range(num_steps):
    #                 step_lat,step_lon = step(step_lat,step_lon,bearing,step_len,self.fault.R)
    #                 bearing = (strike_from_lat_lon(step_lat, step_lon)-180)%360
    #             rects.append([step_lat, step_lon, strike_from_lat_lon(step_lat, step_lon), nleng])
    #
    #         return rects
    #
    #     else:
    #         raise ValueError("'method' must be either 'Avg', 'Center', or 'Step'")
    #
    #     rects.append([rect1_lat, rect1_long, strike1, nleng])
    #     rects.append([rect2_lat, rect2_long, strike2, nleng])
    #
    #     return rects

    def split_rect(self,fault,lat,lon,length,width,deltadepth,n=11,m=3):
        R = fault.R
        # n = int(length/15000)
        # m = int(width/15000)
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
        bearing1 = fault.strike_from_lat_lon(lat,lon)
        bearing2 = (bearing1-180)%360
        lat1,lon1 = lat,lon
        lat2,lon2 = lat,lon
        for i in range(1,(n - 1)//2+1):
            for j in range(n_steps):
                lat1,lon1 = Fault.step(lat1,lon1,bearing1,length_step,R)
                lat2,lon2 = Fault.step(lat2,lon2,bearing2,length_step,R)
                bearing1 = fault.strike_from_lat_lon(lat1, lon1)
                bearing2 = (fault.strike_from_lat_lon(lat2, lon2)-180)%360
            lats[(n-1)//2+i] = lat1
            lats[(n-1)//2-i] = lat2
            lons[(n-1)//2+i] = lon1
            lons[(n-1)//2-i] = lon2

        strikes = fault.strike_map(np.vstack((lats,lons)).T)
        dips = fault.dip_map(np.vstack((lats,lons)).T)
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
                templats1,templons1 = Fault.step(templats1,templons1,dipward,width_step*np.cos(np.deg2rad(tempdips1)),R)
                templats2,templons2 = Fault.step(templats2,templons2,dipward,-width_step*np.cos(np.deg2rad(tempdips2)),R)
                tempdips1 = fault.dip_map(np.vstack((templats1,templons1)).T)
                tempdips2 = fault.dip_map(np.vstack((templats2,templons2)).T)
            Lats[(m-1)//2+i] = templats1
            Lats[(m-1)//2-i] = templats2
            Lons[(m-1)//2+i] = templons1
            Lons[(m-1)//2-i] = templons2
            Strikes[(m-1)//2+i] = fault.strike_map(np.vstack((templats1,templons1)).T)
            Strikes[(m-1)//2-i] = fault.strike_map(np.vstack((templats2,templons2)).T)
            Dips[(m-1)//2+i] = tempdips1
            Dips[(m-1)//2-i] = tempdips2

        Depths = fault.depth_map(np.vstack((Lats.flatten(),Lons.flatten())).T) + deltadepth
        data = [Lats,Lons,Strikes,Dips,Depths]
        data = [arr.flatten() for arr in data]
        return np.array(data).T, sublength, subwidth

    def get_length(self, deltalogl, mag):
        """ Length is sampled from a truncated normal distribution that
        is centered at the linear regression of log10(length_cm) and magnitude.
        Linear regression was calculated from wellscoppersmith data.

        Parameters:
        mag (float): the magnitude of the earthquake

        Returns:
        length (float): Length in meters. a sample from the normal distribution centered on the regression
        """

        m = 0.5233956445903871       # slope
        c = 1.0974498706605313     # y intercept

        mu_logl = m*mag + c
        logl = mu_logl + deltalogl
        return 10**logl

    def get_width(self, deltalogw, mag):
        """
        Parameters:
        mag (float): the magnitude of the earthquake

        Returns:
        width (float): width in meters
        """
        m = 0.29922483873212863       # slope
        c = 2.608734705074858     # y intercept

        mu_logw = m*mag + c
        logw = mu_logw + deltalogw
        return 10**logw

    def get_slip(self, length, width, mag):
        """Calculated from magnitude and rupture area, Ron Harris gave us the equation
            Parameters:
            Length (float): m
            Width (float): m
            mag (float): moment magnitude

            Return:
            slip (float): meters
            """
        #Dr. Harris' rigidity constant : 3.2e11 dynes/cm^2
        mu_dyn_cm2 = 3.e11
        mu = mu_dyn_cm2 * 1e-5 * 1e4 #convert to N/m^2
        slip = 10**(3/2 * ( mag + 6.06 )) / (mu * length * width)
        print("this is calculated slip:",slip,"m")
        #print(slip)
        return slip

    def acceptance_prob(self, sample_params, proposal_params, cur_prior_lpdf, prop_prior_lpdf):
        """
        Calculate the acceptance probability given the lpdf for the current and proposed parameters

        :param prop_prior_lpdf: proposed parameters likelihood
        :param cur_prior_lpdf: current parameters likelihood
        :return:
        """
        change_llh = self.change_llh_calc()

        # Log-Likelihood
        change_prior_lpdf = prop_prior_lpdf - cur_prior_lpdf

        print("prop_prior_lpdf is:")
        print(prop_prior_lpdf)
        print("cur_prior_lpdf is:")
        print(cur_prior_lpdf)
        #print("proposal kernel asymmetry q(sample|proposal)-q(proposalsample):")
        #print(logqs-logqp)
        # Note we use np.exp(new - old) because it's the log-likelihood
        return min(1, np.exp(change_llh+change_prior_lpdf))

    def draw(self, prev_draw):
        """
        Draw with the random walk sampling method, using a multivariate_normal
        distribution with the following specified std deviations to
        get the distribution of the step size.

        Returns:
            draws (array): An array of the 9 parameter draws.
        """
        # deep copy of prev_draw
        new_draw = prev_draw.copy()

        # Random walk draw lat/lon/strike
        longitude_std = 0.075
        latitude_std = 0.075
        magnitude_std = 0.05
        deltalogl_std = 0.005
        deltalogw_std = 0.005
        deltadepth_std = .5 #in km to avoid singular covariance matrix

        # square for std => cov
        cov = np.diag(np.square([longitude_std,
                                 latitude_std,
                                 magnitude_std,
                                 deltalogl_std,
                                 deltalogw_std,
                                 deltadepth_std]))
        mean = np.zeros(6)

        # random draw from normal distribution
        e = stats.multivariate_normal(mean, cov).rvs()
        new_draw[['Longitude','Latitude','Magnitude','DeltaLogL','DeltaLogW','DeltaDepth']] += e

        return new_draw

    def build_priors(self):
        """
        Builds the priors
        :return:
        """
        # samplingMult = 50
        # bandwidthScalar = 2.0
        # # build longitude, latitude and strike prior
        # raw_data = pd.read_excel('./InputData/Fixed92kmFaultOffset50kmgapPts.xlsx')
        # self.latlongstrikeprior = np.array(raw_data[['POINT_X', 'POINT_Y', 'Strike']])
        # distrb0 = gaussian_kde(self.latlongstrikeprior.T)
        #
        # #Garret and spencer chose this 18 June 2019
        # data2 = stats.norm.rvs(size = 1000,loc = np.log(8), scale = 0.05)
        # distrb1 = gaussian_kde(data2)
        #
        # dists = [distrb0, distrb1]

        depth_mu = 30000
        depth_std = 5000
        mindepth = 2500
        maxdepth = 50000
        minlon = 126
        latlon = LatLonPrior(self.fault,depth_mu,depth_std,mindepth,maxdepth)
        mag = stats.truncexpon(b=3,loc=6.5)
        deltalogl = stats.norm(scale=0.18842320591492676) # sample standard deviation from data
        deltalogw = stats.norm(scale=0.17186788334444705) # sample standard deviation from data
        deltadepth = stats.norm(scale=5) # in km to avoid numerically singular covariance matrix
        return Prior(latlon,mag,deltalogl,deltalogw,deltadepth)

    def map_to_okada(self, draws):
        """
        TODO: JARED AND JUSTIN map to okada space
        :param draws:
        :return: okada_params
        """
        lon    = draws["Longitude"] #These need to be scalars
        lat    = draws["Latitude"]
        self.mw = draws["Magnitude"]
        deltalogl = draws["DeltaLogL"]
        deltalogw = draws["DeltaLogW"]
        deltadepth = draws["DeltaDepth"]

        #get Length,Width,Slip from fitted line
        width = self.get_width(deltalogw,self.mw)
        length = self.get_length(deltalogl,self.mw)
        slip = self.get_slip(length, width, self.mw)

        #deterministic okada parameters
        rake = 90
        dip = self.fault.dip_from_lat_lon(lat,lon)
        depth = self.fault.depth_from_lat_lon(lat,lon)[0] + 1000*deltadepth #deltadepth in km to avoid singular covariance matrix
        strike = self.fault.strike_from_lat_lon(lat,lon)

        #original_rectangle = np.array([strike, length, width, depth, slip, rake, dip, lon, lat])
        rectangles, sublength, subwidth = self.split_rect(self.fault, lat, lon, length, width, 1000*deltadepth, n = self.length_split, m = self.width_split)
        temp = []
        for i, rect in enumerate(rectangles):
            temp_lat = rect[0]
            temp_lon = rect[1]
            temp_strike = rect[2]
            temp_dip = rect[3]
            temp_depth = rect[4]
            temp.append(temp_lat)
            temp.append(temp_lon)
            temp.append(temp_strike)
            temp.append(temp_dip)
            temp.append(temp_depth)
        temp += [sublength, subwidth, slip, rake]
        return pd.Series(temp, self.okada_cols)

    def make_observations(self, params, arrivals, heights):
        """
        Computes the observations to save to the observations file based off the earthqauke parameters
        and the wave heights and arrival times at each gauge.  The default setting is to save the
        earthquake magnitude, and the arrival times and wave heights at each gauge.
        :param params: numpy array of Okada parameters
        :param arrivals: list of arrival times for the specified gauges
        :param heights: list of Geoclaw produced wave heights at each gauge
        :return: a list that provides the observations in the correct ordering
        """
        obvs = []
        obvs.append(self.mw)
        for ii in range(len(arrivals)): #alternate arrival times with wave heights
            obvs.append(arrivals[ii])
            obvs.append(heights[ii])

        return obvs

    def init_guesses(self, init):
        """
        Initialize the sample parameters
        :param init: String: (manual, random or restart)
        :return:
        """
        guesses = None
        if init == "manual":
            # initial guesses taken from sample 49148 of 20190320_chains/007
            #strike =  1.90000013e+02
            #length =  1.33325981e+05
            #width  =  8.45009646e+04
            #depth  =  5.43529311e+04
            #slip   =  2.18309283e+01
            #rake   =  9.00000000e+01
            #dip    =  1.30000000e+01
            #long   =  1.30850829e+02
            #lat    = -5.45571375e+00

            # DEPRICATED
            #guesses = np.array([strike, length, width, depth, slip, rake, dip,
            #  long, lat])
            # strike =  1.90000013e+02
            #length =  1.33325981e+05
            #width  =  8.45009646e+04

            lon    =  1.32e+02
            lat    = -5.0e+00
            mag = 8.5
            dellogl = 0
            dellogw = 0
            deltadepth = 0
            #guesses = np.array([strike, length, width, slip, long, lat])
            vals = np.array([lon, lat, mag, dellogl, dellogw, deltadepth])
            guesses = pd.Series(vals, self.sample_cols)

        elif init == "random":
            guesses = self.prior.rvs()
            #guesses = pd.DataFrame(columns=self.sample_cols)
            #guesses.loc[0] = vals
            #raise Exception('random initialization not currently tested')

        elif init == "restart":
            #guesses = np.load('../samples.npy')[0][:9]

            ## np.save("guesses.npy", self.guesses)
            #print("initial sample is:")
            #print(guesses)
            #raise Exception('restart initialization not currently implemented')
            guesses = None

        return guesses

    def prior_logpdf(self,sample):
        length = self.get_length(sample['DeltaLogL'],sample['Magnitude'])
        width = self.get_width(sample['DeltaLogW'],sample['Magnitude'])
        rects,sublength,subwidth = self.split_rect(self.fault,sample['Latitude'],sample['Longitude'],length,width,sample['DeltaDepth'],n = self.length_split,m = self.width_split)
        if np.any(np.isnan(rects)):
            return -np.NINF
        return self.prior.logpdf(sample,rects,subwidth)
