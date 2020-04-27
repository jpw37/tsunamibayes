"""
Created By Cody Kesler
Created 10/19/2018
Property of BYU Mathematics Dept.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# class Gauge:
#     """A gauge object class. Mainly for data storage.
#
#     Attributes:
#         name (int): the name you wish to give the gauge (a 5 digit number).
#         longitude (float): the longitude location of the gauge.
#         latitude (float): the latitude location of the gauge.
#         distance (float): the distance from the gauge to shore. (km)
#         kind (list): the type of distribution to be used for the wave
#             arrival time and height respectively. ex: ['norm', 'chi2']
#         arrival_params (list): list of params for the arrival time distribution
#         height_params (list): list of params for the height distribution
#         inundation_params (list): list of params for the inundation distribution
#         arrival_dist (stats object): distribution of arrival time at gauge
#         height_dist (stats object): distribution of wave height at gauge
#     """
#     def __init__(self, name, longitude, latitude, distance,
#                     kind, arrival_params, height_params, inundation_params, beta, n, city_name):
#         self.name = name
#         self.city_name = city_name
#         self.longitude = longitude
#         self.latitude = latitude
#         self.distance = distance
#         self.kind = kind
#         self.arrival_params = arrival_params
#         self.height_params = height_params
#         self.beta = beta
#         self.n = n
#         self.inundation_params = inundation_params
#         if name is not None: # Allows for None initialized object
#             # Kind[0] is for Wave Arrival Times
#             # kind[1] is for Wave Height
#             # kind[2] is for Inundation
#             if kind[0] == 'norm':
#                 mean = arrival_params[0]
#                 std = arrival_params[1]
#                 self.arrival_dist = stats.norm(mean, std)
#             elif kind[0] == 'chi2':
#                 k = arrival_params[0]
#                 loc = arrival_params[1]
#                 scale = arrival_params[2]
#                 self.arrival_dist = stats.chi2(k, loc=loc, scale=scale)
#             elif kind[0] == 'chi':
#                 k = arrival_params[0]
#                 loc = arrival_params[1]
#                 scale = arrival_params[2]
#                 self.arrival_dist = stats.chi(k, loc=loc, scale=scale)
#             elif kind[0] == 'skewnorm':
#                 skew_param = arrival_params[0]
#                 mean = arrival_params[1]
#                 std = arrival_params[2]
#                 self.arrival_dist = stats.skewnorm(skew_param, mean, std)
#
#             if kind[1] == 'norm':
#                 mean = height_params[0]
#                 std = height_params[1]
#                 self.height_dist = stats.norm(mean, std)
#             elif kind[1] == 'chi2':
#                 k = height_params[0]
#                 loc = height_params[1]
#                 scale = height_params[2]
#                 self.height_dist = stats.chi2(k, loc=loc, scale=scale)
#             elif kind[1] == 'chi':
#                 k = height_params[0]
#                 loc = height_params[1]
#                 scale = height_params[2]
#                 self.height_dist = stats.chi(k, loc=loc, scale=scale)
#             elif kind[1] == 'skewnorm':
#                 skew_param = height_params[0]
#                 mean = height_params[1]
#                 std = height_params[2]
#                 self.height_dist = stats.skewnorm(skew_param, mean, std)
#
#             if kind[2] == 'norm':
#                 mean = inundation_params[0]
#                 std = inundation_params[1]
#                 self.inundation_dist = stats.norm(mean, std)
#             elif kind[2] == 'chi2':
#                 k = inundation_params[0]
#                 loc = inundation_params[1]
#                 scale = inundation_params[2]
#                 self.inundation_dist = stats.chi2(k, loc=loc, scale=scale)
#             elif kind[2] == 'chi':
#                 k = inundation_params[0]
#                 loc = inundation_params[1]
#                 scale = inundation_params[2]
#                 self.inundation_dist = stats.chi(k, loc=loc, scale=scale)
#             elif kind[2] == 'skewnorm':
#                 skew_param = inundation_params[0]
#                 mean = inundation_params[1]
#                 std = inundation_params[2]
#                 self.inundation_dist = stats.skewnorm(skew_param, mean, std)
#
#
#     def to_json(self):
#         """
#         Convert object to dict of attributes for json
#         """
#         d = dict()
#         d['name'] = self.name
#         d['longitude'] = self.longitude
#         d['latitude'] = self.latitude
#         d['distance'] = self.distance
#         d['kind'] = self.kind
#         d['arrival_params'] = self.arrival_params
#         d['height_params'] = self.height_params
#         d['inundation_params'] = self.inundation_params
#         d['beta'] = self.beta
#         d['n'] = self.n
#         d['city_name'] = self.city_name
#         return d
#
#     def plot(self):
#         if self.name is not None: # Allows for None initialized object
#             # Kind[0] is for Wave Arrival Times
#             # kind[1] is for Wave Height
#             # kind[2] is for Inundation
#             if self.kind[0] != None:
#                 f = plt.figure()
#                 domain = np.linspace(0,120,1000)
#                 plt.plot(domain,self.arrival_dist.pdf(domain))
#                 plt.xlabel("Arrival time (minutes)")
#                 plt.title("Arrival time")
#                 #plt.close()
#             if self.kind[1] != None:
#                 f = plt.figure()
#                 domain = np.linspace(0,25,1000)
#                 plt.plot(domain,self.height_dist.pdf(domain))
#                 plt.xlabel("Wave height (meters)")
#                 plt.title("Wave height")
#                 #plt.close()
#             if self.kind[2] != None:
#                 f = plt.figure()
#                 domain = np.linspace(0,1000,1000)
#                 plt.plot(domain,self.inundation_dist.pdf(domain))
#                 plt.xlabel("Inundation length (meters)")
#                 plt.title("Inundation length")
#                 #plt.close()
#
# def from_json(d):
#     """
#     Converts from json file format into gauge object
#     """
#     return Gauge(d['name'], d['longitude'], d['latitude'],d['distance'], d['kind'], d['arrival_params'],
#                     d['height_params'], d['inundation_params'], d['beta'], d['n'], d['city_name'])

class Gauge:
    plot_bounds = {'arrival':(0,120),'height':(0,10),'inundation':(0,500)}
    """Class for managing data related to observations. A Gauge object
    loosely corresponds to an observation location, along with probability
    distributions representing each type of observation associated
    with that location.

    Parameters
    ----------
    name : str
        Name of the observation (for use in output data files)
    dists : dict, optional
        Dictionary of scipy.stats frozen_rv objects. Each distribution's key
        corresponds to which type of observation the distribution is associated
        with
    **kwargs : optional
        Additional arguments to class constructor. Set as class attributes
    """

    def __init__(self,name,longitude,latitude,dists={},beta=None,n=None,city_name=None):
        # check if distributions are scipy.stats rv_frozen objects
        for obstype,dist in dists.items():
            if not isinstance(dist,stats._distn_infrastructure.rv_frozen):
                raise TypeError("dists['{}'] must be a frozen scipy.stats \
                                distribution".format(obstype))

        # core attributes
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.beta = beta
        self.n = n
        self.city_name = city_name
        self.dists = dists
        self.obstypes = self.dists.keys()

        # set instance attributes for keyword arguments
        # for key,value in kwargs.items():
        #     setattr(self,key,value)

    @classmethod
    def from_shapes(cls,name,longitude,latitude,dist_params,beta=None,n=None,city_name=None):
        """Alternate constructor for Gauge objects. Accepts a `params` dictionary
        rather than a `dists` dictionary, where the `params` dictionary contains
        the various parameters associated with the scipy.stats frozen_rv object
        that will be constructed within the class.

        Parameters
        ----------
        name : str
            Name of the observation (for use in output data files)
        dist_params : dict
            Dictionary of distribution parameters. Each key corresponds to
            another dictionary. For example:

            dist_params['arrival'] = {'name':'norm','shapes':{'loc':1,'scale':2}}.

            This dictionary must contain a valid scipy.stats distribution name,
            as well as a dictionary of shape parameters that will be passed to the
            scipy.stats constructor as keyword arguments.
        **kwargs
            Additional arguments to the Gauge class constructor.
        """
        dists = {}
        for obstype,d in dist_params.items():
            if 'name' not in d.keys():
                raise TypeError("Observation type '{}' must have an associated \
                                distribution name")
            elif 'shapes' not in d.keys():
                raise TypeError("Observation type '{}' must have associated \
                                distribution shape parameters")
            dists[obstype] = getattr(stats,d['name'])(**d['shapes'])
        return cls(name,longitude,latitude,dists,beta,n,city_name)

    def to_json(self):
        ignore = ['dists','obstypes']
        d = {key:self.__dict__[key] for key in self.__dict__ if key not in ignore}
        d['dist_params'] = {}
        for key,dist in self.dists.items():
            d['dist_params'][key] = {'name':dist.dist.name,'shapes':dist.kwds}
        return d

    @classmethod
    def from_json(cls,d):
        return cls.from_shapes(**d)

    def plot(self,obstype,ax=None):
        dist = self.dists[obstype]

        if obstype in ['arrival','height','inundation']:
            a,b = Gauge.plot_bounds[obstype]
        else:
            a,b = dist.a,dist.b
            if a == np.NINF:
                a = dist.interval(.999)[0]
            if b == np.inf:
                b = dist.interval(.999)[1]

        if ax is None:
            fig = plt.figure()
            ax = plt.axes()

        x = np.linspace(a,b,1000)
        ax.plot(x,dist.pdf(x))
        ax.set_title(str(self.name) + ' ' + obstype)

    def plot_all(self,fig=None):
        if fig is None:
            fig = plt.figure()

        nobs = len(self.obstypes)
        if nobs == 1:
            n = 1
            m = 1
        else:
            n = 2
            m = np.ceil(nobs/2)
            fig.set_size_inches(10, 5*m)

        for i,obstype in enumerate(self.obstypes):
            ax = fig.add_subplot(m,n,i+1)
            self.plot(obstype,ax)

def dump_gauges(gauges,outfile):
    with open(outfile,'w') as f:
        json.dump([gauge.to_json() for gauge in gauges],f)

def load_gauges(infile):
    with open(infile,'r') as f:
        lst = json.load(f)
    return [Gauge.from_json(d) for d in lst]
