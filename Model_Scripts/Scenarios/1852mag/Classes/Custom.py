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
from Prior import Prior,LatLonStrikePrior
from Fault import Fault

from scipy.stats import truncnorm

class Custom(MCMC):
    """
    Use this class to create a custom prior and custom earthquake parameters MCMC draws
    When the Variable for use_custom is set to true, this class will be used as the main MCMC class for the Scenario
    """
    def __init__(self):
        MCMC.__init__(self)
        self.sample_cols = ['Strike','Longitude', 'Latitude', 'Magnitude']
        self.proposal_cols = ['P-Strike','P-Logitude', 'P-Latitude', 'P-Magnitude']
        self.observation_cols = ['Mw', 'gauge 1 arrival', 'gauge 1 height', 'gauge 2 arrival', 'gauge 2 height', 'gauge 3 arrival', 'gauge 3 height', 'gauge 4 arrival', 'gauge 4 height', 'gauge 5 arrival', 'gauge 5 height', 'gauge 6 arrival', 'gauge 6 height']
        self.mw = 0
        self.num_rectangles = 3
        cols = []
        for i in range(self.num_rectangles):
            cols += ['Strike' + str(i+1)]
            cols += ['Length' + str(i+1)]
            cols += ['Longitude' + str(i+1)]
            cols += ['Latitude' + str(i+1)]
        cols += [ 'Width', 'Depth', 'Slip', 'Rake', 'Dip']
        self.okada_cols = cols
        self.fault = self.build_fault()

    def build_fault(self):
        data = pd.read_csv("./InputData/BandaHypoSegments.csv")
        latpts = np.array(data["Lat"])
        lonpts = np.array(data["Long"])
        strikepts = np.array(data["Strike"])%360
        return Fault(latpts,lonpts,strikepts,"Banda Arc")

    def split_rect(self, lat, lon, strike, leng, num=3, method="Step"):
        """Split a given rectangle into 3 of equal length that more closely follow the
		curve of the fault.

		Parameters:
			lat (float): latitude of center
			lon (float): longitude of center
			strike (float): orientation of the long edge, measured in degrees
					clockwise from north
			leng (float): length of the long edge (km)

		Return:
			list rectangles represented by a list of parameters: [lat,long,strike,leng]
		"""
        if num < 3 or num % 2!=1:
            raise ValueError("'num' must be an odd integer of at least 3!")


        """Spencer & Garret:
            The code now constructs a Fault object, which is just a container for
            the fault data, and has methods to compute distance and the strike map.
            We can also extend the class to manage the other Okada parameter maps.
            Until then, the important method is Fault.strike_from_lat_lon().
            Custom now initializes with a Fault instance, self.fault. So you can
            compute the strike map using self.fault.strike_from_lat_lon(lat,lon)
        """

        # DEPRICATED
        # #Pulling prior of lon/lat information to contruct best fit approximaiton of strike
        # prior_lat = self.latlongstrikeprior[:,0]
        # prior_lon = self.latlongstrikeprior[:,1]
        # prior_strike = self.latlongstrikeprior[:,2]
        #
        # #Constructing best fit
        # A = np.vstack([np.ones(len(prior_lat)), prior_lat, prior_lon, prior_lat*prior_lon, prior_lat**2, prior_lon**2, prior_lat**2*prior_lon, prior_lon**2*prior_lat, prior_lat**3, prior_lon**3]).T
        # lat_long_bestfit = np.linalg.lstsq(A, prior_strike, rcond=None)[0]
        #
		# #strike/latitude linear regression
        # def strike_from_lat_long(lat, lon):
        #     temp_array = np.array([1, lat, lon, lat*lon, lat**2, lon**2, lat**2*lon, lon**2*lat, lat**3, lon**3])
        #     return temp_array @ lat_long_bestfit
            #line of best fit for strike given latitude
        strike_from_lat = np.poly1d([-4.69107194e-01, -1.31232324e+01, -1.44327025e+02,
                                    -7.82503768e+02, -2.13007839e+03, -2.40708004e+03])
        nleng= leng/num
        rects = []
        rects.append([lat, lon, strike, nleng])

        if method == "Avg":
            lat_temp=lat
            long_temp=lon
            for i in range((num - 1)//2):
				#Find the edge of the center rect and the strike at that point
                edge1_lat = lat_temp + nleng/222*np.cos(np.radians(strike))
                edge1_long = long_temp + nleng/222*np.sin(np.radians(strike))
                strike1 = strike_from_lat_long(edge1_lat,edge1_long)
				#Find the far egde of the adjacent rectangle
                end1_lat = edge1_lat + nleng/111*np.cos(np.radians(strike1))
                end1_long = edge1_long + nleng/111*np.sin(np.radians(strike1))
				#average the strike of the two points
                strike1 = (strike1 + strike_from_lat_long(end1_lat, end1_long))/2
				#find the coordinates of the rectangle using the new strike
                rect1_lat = edge1_lat + nleng/222*np.cos(np.radians(strike1))
                rect1_long = edge1_long + nleng/222*np.sin(np.radians(strike1))

                rects.append([rect1_lat, rect1_long, strike1, nleng])
                lat_temp=rect1_lat
                long_temp=rect1_long

            lat_temp=lat
            long_temp=lon
            for i in range((num - 1)//2):
				#Find the edge of the center rect and the strike at that point
                edge2_lat = lat_temp - nleng/222*np.cos(np.radians(strike))
                edge2_long = long_temp - nleng/222*np.sin(np.radians(strike))
                strike2 = strike_from_lat_long(edge2_lat,edge1_long)
				#Find the far egde of the adjacent rectangle
                end2_lat = edge1_lat - nleng/111*np.cos(np.radians(strike2))
                end2_long = edge1_long - nleng/111*np.sin(np.radians(strike2))
				#average the strike of the two points
                strike2 = (strike2 + strike_from_lat_long(end2_lat, end2_long))/2
				#find the coordinates of the rectangle using the new strike
                rect2_lat = edge2_lat - nleng/222*np.cos(np.radians(strike2))
                rect2_long = edge2_long - nleng/222*np.sin(np.radians(strike2))

                rects.append([rect2_lat, rect2_long, strike2, nleng])
                lat_temp=rect2_lat
                long_temp=rect2_long

        elif method == "Center":
            lat_temp=lat
            long_temp=lon
            for i in range((num - 1)//2):
                edge1_lat = lat_temp + nleng/222*np.cos(np.radians(strike))
                edge1_long = long_temp + nleng/222*np.sin(np.radians(strike))
                strike1 = strike_from_lat_long(edge1_lat,edge1_long)
                mid1_lat = edge1_lat + nleng/222*np.cos(np.radians(strike1))
                mid1_long = edge1_long + nleng/222*np.sin(np.radians(strike1))
                strike1 = strike_from_lat_long(mid1_lat, mid1_long)
                rect1_lat = edge1_lat + nleng/222*np.cos(np.radians(strike1))
                rect1_long = edge1_long + nleng/222*np.sin(np.radians(strike1))

                rects.append([rect1_lat, rect1_long, strike1, nleng])
                lat_temp=rect1_lat
                long_temp=rect1_long

            lat_temp=lat
            long_temp=lon
            for i in range((num - 1)//2):
                edge2_lat = lat_temp - nleng/222*np.cos(np.radians(strike))
                edge2_long = long_temp - nleng/222*np.sin(np.radians(strike))
                strike2 = strike_from_lat_long(edge2_lat,edge1_long)
                mid2_lat = edge1_lat - nleng/222*np.cos(np.radians(strike2))
                mid2_long = edge1_long - nleng/222*np.sin(np.radians(strike2))
                strike2 = strike_from_lat_long(mid2_lat, mid2_long)
                rect2_lat = edge2_lat - nleng/222*np.cos(np.radians(strike2))
                rect2_long = edge2_long - nleng/222*np.sin(np.radians(strike2))

                rects.append([rect2_lat, rect2_long, strike2, nleng])
                lat_temp=rect2_lat
                long_temp=rect2_long


        elif method == "Step":
            #define step length
            num_steps = 8
            step_len = nleng/num_steps/111 #convert from kilometers to degrees (only good near equator)

            #add rectangles in direction of positive strike
            step_strike = strike
            step_lat = lat
            step_lon = lon
            for i in range((num - 1)//2):
                for i in range(num_steps):
                    step_lat += step_len*np.cos(np.radians(step_strike))
                    step_lon += step_len*np.sin(np.radians(step_strike))
                    step_strike = strike_from_lat(step_lat)
                rects.append([step_lat, step_lon, step_strike, nleng])

            #add rectangles in direction of negative strike
            step_strike = strike
            step_lat = lat
            step_lon = lon
            for i in range((num - 1)//2):
                for i in range(num_steps):
                    step_lat -= step_len*np.cos(np.radians(step_strike))
                    step_lon -= step_len*np.sin(np.radians(step_strike))
                    step_strike = strike_from_lat(step_lat)
                rects.append([step_lat, step_lon, step_strike, nleng])

            return rects

        else:
            raise ValueError("'method' must be either 'Avg', 'Center', or 'Step'")

        rects.append([rect1_lat, rect1_long, strike1, nleng])
        rects.append([rect2_lat, rect2_long, strike2, nleng])

        return rects

    def get_length(self, mag):
        """ Length is sampled from a truncated normal distribution that
		is centered at the linear regression of log10(length_cm) and magnitude.
        Linear regression was calculated from wellscoppersmith data.

		Parameters:
		mag (float): the magnitude of the earthquake

		Returns:
		length (float): Length in meters. a sample from the normal distribution centered on the regression
	    """
        #m1 = 0.6423327398       # slope
        #c1 = 0.1357387698       # y intercept
        #e1 = 0.4073300731874614 # Error bar

        m1 = 0.6423327398       # slope
        c1 = 2.1357387698       # y intercept
        e1 = 0.4073300731874614 # Error bar

        #Calculate bounds on error distribution
        a = mag * m1 + c1 - e1
        b = mag * m1 + c1 + e1
        #print("calculated length")
        #print(10**truncnorm.rvs(a,b,size=1)[0])
        #return 10**truncnorm.rvs(a,b,size=1)[0] #regression was done on log10(length_cm)
        l  = 10**truncnorm.rvs(a,b,size=1)[0]
        l /= 100. #convert to m
        print("calculated length:",l,"m")
        return l

    def get_width(self, mag):
        """ Width is sampled from a truncated normal distribution that
        is centered at the linear regression of log10(width_cm) and magnitude
        Linear regression was calculated from wellscoppersmith data.

        Parameters:
        mag (float): the magnitude of the earthquake

        Returns:
        width (float): width in meters. a sample from the normal distribution centered on the regression
        """
        m2 = 0.4832185193       # slope
        c2 = 3.1179508532       # y intercept
        e2 = 0.4093407095518345 # error bar

        #m2 = 0.4832185193       # slope
        #c2 = 0.1179508532       # y intercept
        #e2 = 0.4093407095518345 # error bar

	      #Calculate bounds on error distribution
        a = mag * m2 + c2 - e2
        b = mag * m2 + c2 + e2
        #print("calculated width")
        #print(10**truncnorm.rvs(a,b,size=1)[0])
        #return 10**truncnorm.rvs(a, b, size=1)[0] #regression was done on log10(width_cm)
        w  = 10**truncnorm.rvs(a,b,size=1)[0]
        w /= 100. #convert to m
        print("calculated width:",w,"m")
        return w #regression was done on log10(width_cm)

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

    def acceptance_prob(self, cur_prior_lpdf, prop_prior_lpdf):
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

        strike_std = 5.
        longitude_std = 0.15
        latitude_std = 0.15
        magnitude_std = 0.1 #garret arbitraily chose this

        # square for std => cov
        cov = np.diag(np.square([strike_std, longitude_std, latitude_std, magnitude_std]))
        mean = np.zeros(4)
        cov *= 0.25

        # random draw from normal distribution
        e = stats.multivariate_normal(mean, cov).rvs()

        # does sample update normally
        vals = prev_draw.values + e
        new_draw = pd.DataFrame(columns=self.sample_cols)
        new_draw.loc[0] = vals
        print("Random walk difference:", e)
        print("New draw:", new_draw)

        return new_draw.loc[0]

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

        latlonstrike = LatLonStrikePrior(self.fault,sigma_d=100,sigma_s=3)
        mag = stats.pareto(b=1,loc=7,scale=0.4)

        return Prior(latlonstrike,mag)

    def map_to_okada(self, draws):
        """
        TODO: JARED AND JUSTIN map to okada space
        :param draws:
        :return: okada_params
        """
        lon    = draws["Longitude"] #These need to be scalars
        lat    = draws["Latitude"]
        strike = draws["Strike"]
        self.mw = draws["Magnitude"]

        #get Length,Width,Slip from fitted line
        length = self.get_length(self.mw) #* 1e-2
        width = self.get_width(self.mw) #* 1e-2
        slip = self.get_slip(length, width, self.mw)

        print("length")
        print(length)
        print("width")
        print(width)
        print("slip")
        print(slip)
        print('Mw', self.mw)

        #deterministic okada parameters
        rake = 90
        dip = 13
        depth = self.doctored_depth_1852_adhoc(lon, lat, dip)
        #original_rectangle = np.array([strike, length, width, depth, slip, rake, dip, lon, lat])

        rectangles = self.split_rect(lat, lon, strike, length, num = self.num_rectangles)
        temp = []
        for i, rect in enumerate(rectangles):
            temp_lat = rect[0]
            temp_lon = rect[1]
            temp_strike = rect[2]
            temp_length = rect[3]
            temp.append(temp_strike)
            temp.append( temp_length)
            temp.append( temp_lon)
            temp.append( temp_lat )
        temp += [width, depth, slip, rake, dip]
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
        #obvs[0] = self.compute_mw(params[1], params[2], params[4]) #first the magnitude
        #obvs.append(self.compute_mw(params["Length"], params["Width"], params["Slip"])) #first the magnitude
        obvs.append(self.mw)
        for ii in range(len(arrivals)): #alternate arrival times with wave heights
            obvs.append(arrivals[ii])
            obvs.append(heights[ii])

        return obvs


    def compute_mw(self, L, W, slip, mu=30.e9):
        """
        Computes the Magnitude for a set of porposal parameters for saving
        :param L: float: Length of Earthquake
        :param W: float: Width of Earthquake
        :param slip: float: Slip of Earthquake
        :param mu:
        :return: Magnitude of Earthquake
        """
        unitConv = 1e7  # convert from Nm to 1e-7 Nm
        Mw = (2 / 3) * np.log10(L * W * slip * mu * unitConv) - 10.7
        return Mw

    def haversine_distance(self, p1, p2):
        """
        This function  is set up separately because the haversine distance
        likely will still be useful after we're done with this adhoc approach.

        Note, this does not account for the oblateness of the Earth. Not sure if
        this will cause a problem.
        """
        r = 6371000

        # Setting up haversine terms of distance expansion
        hav_1 = np.power(np.sin((p2[1] - p1[1]) / 2 * np.pi / 180), 2.0)
        hav_2 = np.cos(p2[1] * np.pi / 180) * np.cos(p1[1] * np.pi / 180) * np.power(
            np.sin((p2[0] - p1[0]) / 2 * np.pi / 180), 2.0)

        # taking the arcsine of the root of the sum of the haversine terms
        root = np.sqrt(hav_1 + hav_2)
        arc = np.arcsin(root)

        # return final distance between the two points
        return 2 * r * arc

    def doctored_depth_1852_adhoc(self, longitude, latitude, dip):
        """
        This is a function written specifically for our 1852 depth fix.
        We make use of the fault points used in generating our prior as
        jumping off point for fixing the depth of an event. We use a
        simple trig correction based on a 20degree dip angle and the haversine distance
        to get the depth of the earthquake in question.

        Note, this will do the dip correction regardless of which side
        of the trench our sample is on. Recognizing when the sample is
        on the wrong side seems nontrivial, so we have not implemented
        a check for this here.
        """
        ## set up sample point and fault array
        #p1 = np.array([longitude, latitude])
        #fault_file = './InputData/fault_array.npy'
        #fault_array = np.load(fault_file)
        ## will store haversine distances for comparison
        #dist_array = np.zeros(len(fault_array)//2)
        #for i in range(len(dist_array)):
        #    x = fault_array[2 * i]
        #    y = fault_array[2 * i + 1]
        #    p2 = np.array([x, y])
        #    dist_array[i] = self.haversine_distance(p1, p2)

        #dist = np.amin(dist_array)

        ## need to add trig correction
        #return (20000 + dist * np.tan(20 * np.pi / 180))
        #set up sample point and fault array
        p1 = np.array([longitude,latitude])
        fault_file = './InputData/fault_array.npy'
        fault_array = np.load(fault_file)
        #will store haversine distances for comparison
        dist_array = np.zeros(len(fault_array)//2)
        for i in range(len(dist_array)):
            x = fault_array[2*i]
            y = fault_array[2*i + 1]
            p2 = np.array([x,y])
            dist_array[i] = self.haversine_distance(p1, p2)

        dist    = np.amin(dist_array)
        distind = np.argmin(dist_array)
        arcpt = np.array([fault_array[2*distind],fault_array[2*distind + 1]])

        #arcmidpt = np.array([129.,-6.])
        #dist_array = np.zeros(len(fault_array)//2)
        #for i in range(len(dist_array)):
        #    x = fault_array[2*i]
        #    y = fault_array[2*i + 1]
        #    p2 = np.array([x,y])
        #    dist_array[i] = self.haversine_distance(arcmidpt, p2)
        #midptdist = np.amin(dist_array)

        #midpoint of prior arc
        arcmidpt = np.array([129.,-6.])
        #distance between midpoint and point
        distpt = self.haversine_distance(p1, arcmidpt)
        #distance between midpoint and arc
        distarc = self.haversine_distance(arcpt, arcmidpt)
        slope = 1. if distpt < distarc else -1.
        print("distpt is:")
        print(distpt)
        print("distarc is:")
        print(distarc)

        #need to add trig correction
        return (21316. + slope*dist*np.tan(dip*np.pi/180))

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

            #guesses = np.array([strike, length, width, depth, slip, rake, dip,
            #  long, lat])
            strike =  1.90000013e+02
#            length =  1.33325981e+05
#            width  =  8.45009646e+04
#            slip   =  2.18309283e+01
            lon    =  1.30850829e+02
            lat    = -5.45571375e+00
<<<<<<< HEAD
            mag = 9.0

=======

>>>>>>> 0018b79433826ba4963cdd54aefadde008216c3a
            #guesses = np.array([strike, length, width, slip, long, lat])
            vals = np.array([strike, lon, lat, mag])
            guesses = pd.Series(vals, self.sample_cols)

        elif init == "random":
            prior = self.()
            guesses = prior.rvs(1)
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
