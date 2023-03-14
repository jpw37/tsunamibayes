import numpy as np
from tsunamibayes import BasePrior
from tsunamibayes.utils import calc_length, calc_width, out_of_bounds
import math
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy import special
import scipy

class BandaPrior(BasePrior):
    """The child class of Base Prior that creates a prior distribution,
    specifically for the Banda 1852 event."""
    def __init__(self,latlon,vel,volume,thickness,aspect_ratio):
        """Initializes all the necessary variables for the subclass.
        
        Parameters
        ----------
        latlon : LatLonPrior Object
            Contains attirbutes fault and depth_dist, with methods logpdf, pdf, and rvs.
        mag : scipy.stats rv_frozen object
            A truncated continous random variable describing the sample's 
            magnitude with fixed parameters shape, location and scale.
        delta_logl : scipy.stats rv_frozen object
            The continous random variable describing the sample's standard deviation for
            the log of the length, also with fixed parameters. 
        delta_logw : scipy.stats rv_frozen object
            The continous random variable describing the sample's standard deviation for
            the log of the width, also with fixed parameters. 
        depth_offset : scipy.stats rv_frozen object
            The continous random variable describing the sample's depth offset, 
            also with fixed parameters. 
        """
        self.latlon = latlon
        self.initial_velocity = vel
        self.volume = volume
        self.thickness = thickness
        self.aspect_ratio = aspect_ratio

    def logpdf(self,sample):
        """Computes the log of the probability density function. Adds
        together the logs of all the probability denisty functions for each
        of sample's attributes.
        
        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values. 
        
        Returns
        -------
        lpdf : float
            The log of the probability density function for the sample.
        """ 
        lat = sample["latitude"]
        lon = sample["longitude"]
        initial_velocity = sample["initial_velocity"]
        volume = sample["volume"]
        thickness = sample["thickness"]
        aspect_ratio = sample['aspect_ratio']
        print("PRIOR VALUES")
        lpdf = self.latlon.logpdf(sample)
        print('longitude/latitude val')
        print(self.latlon.logpdf(sample))
        lpdf += self.initial_velocity.logpdf(initial_velocity)
        print('velocity val')
        print(self.initial_velocity.logpdf(initial_velocity))
        lpdf += self.volume.logpdf(sample)
        print('volume val')
        print(self.volume.logpdf(sample))
        lpdf += self.thickness.logpdf(thickness)
        print('thickness val')
        lpdf += self.aspect_ratio.logpdf(aspect_ratio)
        return lpdf

    def rvs(self):
        """Computes random variates for each of Banda Prior's data members,
        then returns the organized set of random variates for each.
        
        Returns
        -------
        rvs : pandas Series
            A series containing axis labels for each of banda prior's variables,
            with the associated random variates (float values) for each parameter. 
        """
        latlon = self.latlon.rvs()
        initial_velocity = self.initial_velocity.rvs()
        volume = self.volume.rvs()
        thickness = self.thickness.rvs()
        params = np.array(latlon+[initial_velocity,volume,thickness])
        return pd.Series(params,["latitude",
                                 "longitude",
                                 "initial_velocity",
                                 "volume",
                                 "thickness"])

class LatLonPrior(BasePrior):
    def __init__(self,topo,depth_dist):
        """Initializes all the necessary variables for the subclass.

        Parameters
        ----------
        fault :  GridFault Object
            From the tsunamibayes module in fault.py 
        depth_dist : scipy.stats rv_frozen object
            The truncated continous random variable describing the depth 
            with fixed shape, location and scale parameters.
        """
        self.topo = topo
        self.depth_dist = depth_dist

    def logpdf(self,sample):
        """Checks to insure that the sample's subfaults are not out of bounds,
        then computes the log of the depth distribution's probability density function 
        evaluated at the sample's depth. 
        
        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values.

        Returns
        -------
        NINF -or- logpdf : float
            Returns negative inifity if out-of-bounds,
            otherwise returns the log of the probability density function 
            for the depth distribution evaluated at the sample's depth. 
        """
        #check if coordinates are out of bounds
        if sample['latitude'] < np.min(self.topo.x) or sample['latitude'] > np.max(self.topo.x):
            return np.NINF
        if sample['longitude'] < np.min(self.topo.y) or sample['longitude'] > np.max(self.topo.y):
            return np.NINF
        in 
        #get index of coordinates
        y_index = np.argmin(np.abs(self.topo.y - sample['longitude']))
        x_index = np.argmin(np.abs(self.topo.x - sample['latitude']))
        #we check 4 corners of the box
        delta_x = topo_file.delta[0]
        delta_y = topo_file.delta[1]
        x_step = np.max(np.array([round((m_to_deg(l) / delta_x) / 2),1]))
        y_step = np.max(np.array([round((m_to_deg(l) / delta_y) / 2),1]))
        #coordinates cannot start above water
        if self.topo.Z[y_index,x_index] >=0 or self.topo.Z[y_index + x_step, x_index] >=0 or self.topo.Z[y_index - x_step,x_index] >=0 or self.topo.Z[y_index, x_index + y_step] >=0 or self.topo.Z[y_index - x_step, x_index - y_step] >=0:
            return np.NINF
        #find depth at coordinates and return logpdf
        depth = -self.topo.Z[y_index,x_index]
        return self.depth_dist.logpdf(depth)

    def pdf(self,sample):
        """Checks to insure that the sample's subfaults are not out of bounds,
        then evaluates the depth distribution's probability density function 
        at the sample's depth. 
        
        Parameters
        ----------
        sample : pandas Series of floats
            The series containing the arrays of information for a sample.
            Contains keys 'latitude', 'longitude', 'magnitude', 'delta_logl',
            'delta_logw', and 'depth_offset' with their associated float values.

        Returns
        -------
        pdf : float
            The value of the probability density function for the depth distribution
            evaluated at the depth of the sample. 
        """
        #check if coordinates are out of bounds
        if sample['latitude'] < np.min(self.topo.x) or sample['latitude'] > np.max(self.topo.x):
            return np.NINF
        if sample['longitude'] < np.min(self.topo.y) or sample['longitude'] > np.max(self.topo.y):
            return np.NINF
        #get index of coordinates
        y_index = np.argmin(np.abs(self.topo.y - sample['longitude']))
        x_index = np.argmin(np.abs(self.topo.x - sample['latitude']))
        #coordinates cannot start above water
        if self.topo.Z[y_index,x_index] >=0
            return np.NINF
        #find depth at coordinates and return pdf
        depth = -self.topo.Z[y_index,x_index]
        return self.depth_dist.pdf(depth)

    # def rvs(self):
    #     """Produces two random variate values for latitude and longitude
    #     based on a random variate of the depth distribution.
        
    #     Returns
    #     -------
    #     lat, lon : (list) of floats
    #         The random variates for latitude and longitude within the fault's bounds.
    #     """
    #     d = self.depth_dist.rvs()
    #     I,J = np.nonzero((d - 500 < self.fault.depth)&(self.fault.depth < d + 500))
    #     idx = np.random.randint(len(I))
    #     return [self.fault.lat[I[idx]],self.fault.lon[J[idx]]]


class DepthPrior(BasePrior):
    """
    The child class of Base Prior that creates a prior distribution
    for the depth of a landslide.
    """
    def __init__(
        self,
        chi=2.7,
        max_d=5000,
        min_d=20,
        density=500,
        scale=300
    ):
        """
        DEPTH
        Initializes all the necessary variables for the subclass.
        Parameters
        ----------
        chi : float
            The chi value desired for computing the depth distribution.
            Values less than 3 will result in an improperly shaped distribution
            that weights smaller depths inappropriately high. The larger
            the chi value, the lower confidence smaller depth values will 
            have.
        max_d : float
            The maximum depth value allowed when computing the propability
            of a given depth.
        min_d : float
            The minimum depth value allowed when computing the propability
            of a given depth.
        density : int
            The number of points to sample between zero and max_d.
        scale : float
            The scale desired for computing the depth distribution. Determines
            how wide the distribution is. The larger the scale, the more evenly
            distributed the distribution will be.
        Raises
        ------
        TypeError
            If chi, max_d, and scale are not floats, or if density is not an int.
        ValueError
            If chi, max_d, min_d, density, or scale are negative.
        """                
        # check input
        if not isinstance(chi, (int, float, np.integer, np.float)):
            raise TypeError(f"chi parameter must be a float an int. Received {type(chi)} ({chi}) instead.")
        if not isinstance(max_d, (int, float, np.integer, np.float)):
            raise TypeError(f"max_d parameter must be a float or an int. Received {type(max_d)} ({max_d}) instead.")
        if not isinstance(min_d, (int, float, np.integer, np.float)):
            raise TypeError(f"min_d parameter must be a float or an int. Received {type(min_d)} ({min_d}) instead.")
        if not isinstance(density, (int, np.integer)):
            raise TypeError(f"density parameter must be an int. Received {type(density)} ({density}) instead.")
        if not isinstance(scale, (int, float, np.integer, np.float)):
            raise TypeError(f"scale parameter must be a float or an int. Received {type(scale)} ({scale}) instead.")
        if chi < 0 or max_d < 0 or min_d < 0 or density < 0 or scale < 0:
            raise ValueError(f"Input out of bounds. Expected all positive parameters but received:\n\tChi:\t\t{chi}\n\tMaximum Depth:  {max_d}\n\tMinimum Depth:  {min_d}\n\tDensity:\t{density}\n\tScale:\t\t{scale}")
        
        # save attributes
        self.max = max_d
        self.min = min_d
        self.density = density
        self.scale = scale
        self.chi = chi
        self.distribution = self.gausser()
        
        
    def gausser(self):
        """
        DEPTH
        Compute chi-squared distribution around each point in the data.
        Returns
        -------
        y : nparray
            Returns a 1 dimensional numpy array of probability values.
        """
        # compute x and y
        x = np.linspace(0, self.max, self.density)
        y = 1 * (x/self.scale)**(self.chi/2-1) * np.exp(-(x/self.scale)/2) / 2**(self.chi/2) / special.gamma(self.chi/2)
        
        # find minimum depth index
        shallow = self.min * self.density // self.max
        y[:shallow] = 0
        
        # return the sum of all the gaussian distributions
        return y

    
    def logpdf(self, sample):
        """
        DEPTH
        Checks to ensure that the sample is greater than zero,
        then computes the log of the depth distribution's probability density
        function evaluated at the sample.
        Parameters
        ----------
        sample : float
            A float representating a sample between zero and self.max.
        Raises
        ------
        TypeError
            If the sample is not a float.
        ValueError
            If the sample is greater than the maximum limit.
        Returns
        -------
        NINF -or- logpdf : float
            Returns negative infinity if negative,
            otherwise returns the log of the probability density function
            for the depth distribution evaluated at the sample.
        """
        # check input and bounds on depth sample
        if not isinstance(sample, (int, float, np.integer, np.float)):
            raise TypeError(f"Sample must be a float or an int. Instead received {type(sample)} ({sample}).")
        if sample < 0: 
            return np.NINF
        elif sample > self.max:
            return np.NINF
        elif sample < self.min:
            return np.NINF

        
        # return an array of the log probabilities  of volume at the given slope
        return np.log(self.distribution[int(sample * self.density / self.max)])

    
    def rvs(self, n=1):
        """
        DEPTH
        Produces n random variate values for depth.
        Parameters
        ----------
        n : int
            Number of samples to draw.
        Raises
        ------
        TypeError
            If n is not an int.
        ValueError
            If n is not positive.
        Returns
        -------
        probability : float -or- nparray
            The random variates for depth greater than zero.
            Returns a float if n = 1 or a numpy array if n > 1.
        """
        # check input
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"n must be an integer. Instead received {type(n)} ({n}).")
        if n < 1:
            raise ValueError(f"n must be positive. Instead received {n}")

        # get the depth indices
        prob = np.array([i for i, val in enumerate(self.distribution) for freq in range(int(1000*val))])
        depth = np.random.choice(prob, n)
        
        # find their probabilities and return as a float or array
        sample = [self.distribution[d] for d in depth]
        if n == 1:
            return sample[0]
        return np.array(sample)
    
    def plot(self):
        """
        DEPTH
        Creates and shows a plot of the Depth prior distribution.
        """
        # set domain
        x = np.linspace(0, self.max, self.density)
        
        # plot distribution
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, self.distribution, c='r', lw=2)
        
        # set plot parameters
        plt.title("Depth Prior Distribution")
        plt.xlabel("Depth $(m)$")
        plt.ylabel("Confidence")
        plt.show()

class VolumePrior(BasePrior):
    """
    The child class of Base Prior that creates a prior distribution
    for the volume of a landslide relative to the slope.
    """
    def __init__(
        self,
        topo,
        sigma=4,
        select=2,
        max_slope=10,
        max_volume=40,
        density_slope=100,
        density_volume=400
    ):
        """
        VOLUME
        Initializes all the necessary variables for the subclass.
        Parameters
        ----------
        sigma : float
            The variance desired for computing the volume distribution.
        select : int
            Must be either 2 or 3. 
            2 selects the dataset that contains all samples with at 
            least a slope and a volume measurement.
            3 selects the dataset that contains all samples with slope,
            volume, and thickness measurements.
        max_slope : float
            The maximum slope value allowed when retrieving the propability
            of the landslide slope.
        max_volume : float
            The maximum volume value allowed when retrieving the propability
            of the landslide volume given its slope.
        density_slope : int
            The number of points to sample the slope at between zero and 
            the max_slope value.
        density_volume : int
            The number of points to sample the volume at between zero and 
            the max_volume value.
        Raises
        ------
        TypeError 
            If sigma, max_slope, or max_volume are not floats, or 
            if density_slope or density_volume are not ints. 
        ValueError
            If sigma, max_slope, max_volume, density_slope, or
            density_volume are negative.
        """        
        # check input
        if not isinstance(sigma, (int, float, np.integer, np.float)):
            raise TypeError(f"sigma parameter must be a float or an int. Received {type(sigma)} ({sigma}) instead.")
        if not isinstance(max_slope, (int, float, np.integer, np.float)):
            raise TypeError(f"max_slope parameter must be a float or an int. Received {type(max_slope)} ({max_slope}) instead.")
        if not isinstance(max_volume, (int, float, np.integer, np.float)):
            raise TypeError(f"max_volume parameter must be a float or an int. Received {type(max_volume)} ({max_volume}) instead.")
        if not isinstance(density_slope, (int, np.integer)):
            raise TypeError(f"density_slope parameter must be an int. Received {type(density_slope)} ({density_slope}) instead.")
        if not isinstance(max_slope, (int, np.integer)):
            raise TypeError(f"density_volume parameter must be an int. Received {type(density_volume)} ({density_volume}) instead.")
        if sigma < 0 or max_slope < 0 or max_volume < 0 or density_slope < 0 or density_volume < 0:
            raise ValueError(f"Input out of bounds. Expected all positive parameters but received:\n\tSigma:\t\t{sigma}\n\tMaximum Slope:  {max_slope}\n\tMaximum Volume: {max_volume}\n\tSlope Density:\t{density_slope}\n\tVolume Density:\t{density_volume}")
        
        # import data
        if select == 3:
            df_array = np.asarray(pd.read_csv("./data/Landslide_GParams_ThVmSl.csv"))
            size = 30
        else:
            df_array = np.asarray(pd.read_csv("./data/Landslide_Gauss_Parameters_Volume.csv"))
            size = 60
            
        # save attributes
        self.max_slope = max_slope
        self.max_volume = max_volume
        self.density_slope = density_slope
        self.density_volume = density_volume
        self.volume = np.array([[x for x in df_array[0:size, 3] if not math.isnan(float(x))]])
        self.meanSlope = np.array([[x for x in df_array[0:size, 4] if not math.isnan(float(x))]])        
        self.sigma = sigma
        self.distribution = self.gausser()
        self.topo = topo
        
    def gausser(self):
        """
        VOLUME
        Compute gaussian distribution around each point in the data.
        Returns
        -------
        meshgrid : nparray
            Returns a 2 dimensional numpy array of probability 
            values that map slope to volume.
        """
        # set up meshgrid
        x_init = np.linspace(0, self.max_slope, self.density_slope)
        y_init = np.linspace(0, self.max_volume, self.density_volume)
        x, y = np.meshgrid(x_init, y_init)
        
        # set slopes and volumes based on the data set we are interested in
        total = np.zeros(x.shape)
            
        # for every point in the data, get the gaussian distribution around it
        for i in range(self.meanSlope.size):
            total += np.exp(-((self.meanSlope[:,i] - x)**2 + (self.volume[:,i] - y)**2) / 2 / self.sigma**2)

        # return the sum of all the gaussian distributions
        return total

    
    def logpdf(self,sample):
        """
        VOLUME
        Checks to ensure that the sample's slope and volume are greater than zero,
        then computes the log of the volume distribution's probability density
        function evaluated at the sample's slope and volume.
        Parameters
        ----------
        sample_slope : float
            A float representating a sample slope between zero and max_slope.
        sample_volume : float
            A float representating a sample volume between zero and max_volume.
        Raises
        ------
        TypeError
            If sample is not a float.
        ValueError
            If sample is greater than the maximum limit.
        Returns
        -------
        NINF -or- logpdf : float
            Returns negative infinity if negative,
            otherwise returns the log of the probability density function
            for the volume distribution evaluated at the sample's slope.
        """
        def deg_to_km(deg):
            return deg / .008

        def km_to_deg(km):
            return km * .008

        def m_to_deg(meters):
            return meters / 111000

        def deg_to_m(deg):
            return deg * 111000
        print(sample)
        # convert to km^3 instead of m^3
        sample_volume = sample['volume'] / (1000**3)
        latitude = sample['latitude']
        longitude = sample['longitude']
        bathy = scipy.interpolate.RectBivariateSpline(self.topo.x, self.topo.y, self.topo.Z.transpose())
        elev1 = bathy(latitude + self.topo.delta[0], longitude)
        elev2 = bathy(latitude - self.topo.delta[0], longitude) 
        elev3 = bathy(latitude, longitude - self.topo.delta[0]) 
        elev4 = bathy(latitude, longitude + self.topo.delta[0])
        theta_lon = np.arctan(np.abs((elev1 - elev2)) / deg_to_m(2*self.topo.delta[0]))[0][0]
        theta_lat = np.arctan(np.abs((elev3 - elev4)) / deg_to_m(2*self.topo.delta[0]))[0][0]
        sample_slope = ((theta_lat + theta_lon) / 2)
        # check input and bounds on depth sample
        if not isinstance(sample_slope, (int, float, np.integer, np.float)):
            raise TypeError(f"Sample_slope must be a float or an int. Instead received {type(sample_slope)} ({sample_slope}).")
        if not isinstance(sample_volume, (int, float, np.integer, np.float)):
            raise TypeError(f"Sample_volume must be a float or an int. Instead received {type(sample_volume)} ({sample_volume}).")
        if sample_slope < 0 or sample_volume < 0:
            return np.NINF
        if sample_slope > self.max_slope:
            return np.NINF
        if sample_volume > self.max_volume:
            return np.NINF
            
        # return the log probabilities at the given slope and volume
        slope_index = int(sample_slope * self.density_slope / self.max_slope)
        volume_index = int(sample_volume * self.density_volume / self.max_volume)
        return np.log(self.distribution[volume_index][slope_index])

    
    
    def rvs(self, n=1):
        """
        VOLUME
        Produces n random variate values for volume based on a random 
        variate of the slope distribution.
        **NOTE**
        Randomly selects the slope and volume uniformly, not based on the 
        relative probability of each value.
        Parameters
        ----------
        n : int
            Number of samples to draw.
        Raises
        ------
        TypeError
            If n is not an int.
        ValueError
            If n is not positive.
        Returns
        -------
        probability : float -or- nparray
            The random variates for volume and slope greater than zero.
            Returns a float if n = 1 or a numpy array if n > 1.
        """   
        # check input
        if type(n) is not int:
            raise TypeError(f"n must be an integer. Instead received {type(n)} ({n}).")
        if n < 1:
            raise ValueError(f"n must be positive. Instead received {n}")
        
        # get the slope and volume indices
        slope = np.random.randint(0, self.density_slope, n)
        volume = np.random.randint(0, self.density_volume, n)
        
        # find their probabilities and return as a float or array
        sample = [self.distribution[v][s] for v, s in zip(volume, slope)]
        if n == 1:
            return sample[0]
        return np.array(sample)
    
    def plot(self, showPoints=False):
        """
        VOLUME
        Creates and shows a plot of the Volume vs Slope prior distribution.
        Parameters
        ----------
        showPoints : boolean
            Determines if the plot shows the points that the construct
            the volume and slope probability distribution.
        """
        # implement a new meshgrid
        x = np.linspace(0, self.max_slope, self.density_slope)
        y = np.linspace(0, self.max_volume, self.density_volume)
        X, Y = np.meshgrid(x, y)
        
        # plot the contour based on the distribution attribute
        fig = plt.figure(figsize=(4, 6))
        plt.contourf(X, Y, self.distribution, 50, cmap='ocean_r')
        if type(showPoints) is bool and showPoints:
            plt.scatter(self.meanSlope, self.volume, marker='o', c="1")
            plt.ylim(0, self.max_volume)
        
        # set plot parameters
        plt.colorbar()
        plt.title("Volume Prior Distribution Given Slope")
        plt.xlabel("Slope (Degrees)")
        plt.ylabel("Volume ($km^3$)")
        plt.show()


class ThicknessPrior(BasePrior):
    """
    The child class of Base Prior that creates a prior distribution,
    for the thickness of a landslide.
    """
    def __init__(
        self,
        sigma=10,
        select=2,
        max_t=150,
        density=500,
    ):
        """
        THICKNESS
        Initializes all the necessary variables for the subclass.
        Parameters
        ----------
        sigma : float
            The variance desired for computing the thickness distribution.
        max_t : float
            The maximum thickness value allowed when computing the propability
            of a given thickness.
        density : int
            The number of points to sample the slope between zero and 
            the max_t value.
        Raises
        ------
        TypeError
            If sigma or max_t are not floats, or density is not an int.
        ValueError
            If sigma, max_t, or density are negative.
        """        
        # check input
        if not isinstance(sigma, (int, float, np.integer, np.float)):
            raise TypeError(f"sigma parameter must be a float or an int. Received {type(sigma)} ({sigma}) instead.")
        if not isinstance(max_t, (int, float, np.integer, np.float)):
            raise TypeError(f"max_t parameter must be a float or an int. Received {type(max_t)} ({max_t}) instead.")
        if not isinstance(density, (int, np.integer)):
            raise TypeError(f"density parameter must be an int. Received {type(density)} ({density}) instead.")
        if sigma < 0 or max_t < 0 or density < 0:
            raise ValueError(f"Input out of bounds. Expected all positive parameters but received:\n\tSigma:\t\t{sigma}\n\tMax Thickness:  {max_t}\n\tDensity:\t{density}")
        
        
        # import data
        df_array = np.asarray(pd.read_csv("./data/Landslide_Gauss_Parameters_Thickness.csv"))
            
        # save attributes
        self.max = max_t
        self.density = density
        self.thickness = np.array([[x for x in df_array[0:20, 2] if not math.isnan(float(x))]])
        self.sigma = sigma
        self.distribution = self.gausser()
        
        
    def gausser(self):
        """
        THICKNESS
        Compute gaussian distribution around each point in the data.
        Returns
        -------
        y : nparray
            Returns a 1 dimensional numpy array of thickness probability values.
        """
        # initialize x and y
        x = np.linspace(0, self.max, self.density)
        y = np.zeros(x.shape)
        
        # set thickness confidence based on the data set we are interested in
        for j in range(self.thickness.size):
            y += np.exp(-((self.thickness[:, j] - x) ** 2) / 2 / self.sigma ** 2)

        # return the sum of all the gaussian distributions
        return y

    
    def logpdf(self, sample):
        """
        THICKNESS
        Checks to ensure that the sample is greater than zero,
        then computes the log of the thickness distribution's probability density
        function evaluated at the sample.
        Parameters
        ----------
        sample : float
            A float representating a sample between zero and self.max.
        Raises
        ------
        TypeError
            If sample is not a float.
        ValueError
            If sample is greater than the maximum limit.
        Returns
        -------
        NINF -or- logpdf : float
            Returns negative infinity if negative,
            otherwise returns the log of the probability density function
            for the thickness distribution evaluated at the sample.
        """
        # check input and bounds on thickness sample
        if not isinstance(sample, (int, float, np.integer, np.float)):
            raise TypeError(f"Sample must be a float or an int. Instead received {type(sample)} ({sample}).")
        elif sample < 0:
            return np.NINF
        elif sample > self.max:
            return np.NINF
        
        # return an array of the log probabilities  of volume at the given slope
        return np.log(self.distribution[int(sample * self.density / self.max)])

    
    def rvs(self, n=1):
        """
        THICKNESS
        Produces n random variate values for thickness.
        Parameters
        ----------
        n : int
            Number of samples to draw.
        Raises
        ------
        TypeError
            If n is not an int.
        ValueError
            If n is not positive.
        Returns
        -------
        probability : float -or- nparray
            The random variates for thickness greater than zero.
            Returns a float if n = 1 or a numpy array if n > 1.
        """
        # check input
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"n must be an integer. Instead received {type(n)} ({n}).")
        if n < 1:
            raise ValueError(f"n must be positive. Instead received {n}")
            
        # get the thickness indices
        prob = np.array([i for i, val in enumerate(self.distribution) for freq in range(int(1000*val))])
        thickness = np.random.choice(prob, n)
#         thickness = np.random.randint(0, self.density, n)

        # find their probabilities and return as a float or array
        sample = [self.distribution[t] for t in thickness]
        if n == 1:
            return sample[0]
        return np.array(sample)
    
    def plot(self, showPoints=False):
        """
        THICKNESS
        Creates and shows a plot of the Thickness prior distribution
        with the points that construct the distribution.
        Parameters
        ----------
        showPoints : boolean
            Determines if the plot shows the points that the construct
            the thickness probability distribution.
        """
        # check input
        if not isinstance(showPoints, bool):
            showPoints = False
        
        # create domain
        x = np.linspace(0, self.max, self.density)
        
        # construct plot based on the distribution attribute
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, self.distribution, c='g', lw=4)
        if showPoints:
            plt.scatter(self.thickness, np.zeros((len(self.thickness[0]), 1)), color='k')

        # set plot parameters
        plt.title("Thickness Prior Distribution")
        plt.ylabel("Confidence")
        plt.xlabel("Thickness $(m)$")
        plt.xlim(0, self.max + 1)
        plt.show()