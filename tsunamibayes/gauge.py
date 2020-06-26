import numpy as np
import scipy.stats as stats
import json
import matplotlib.pyplot as plt

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

    def __init__(self,name,dists={},**kwargs):
        # check if distributions are scipy.stats rv_frozen objects
        for obstype,dist in dists.items():
            if not isinstance(dist,stats._distn_infrastructure.rv_frozen):
                raise TypeError("dists['{}'] must be a frozen scipy.stats \
                                distribution".format(obstype))

        # core attributes
        self.name = name
        self.dists = dists
        self.obstypes = self.dists.keys()

        # set instance attributes for keyword arguments
        for key,value in kwargs.items():
            setattr(self,key,value)

    @classmethod
    def from_shapes(cls,name,dist_params,**kwargs):
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
        return cls(name,dists,**kwargs)

    def to_json(self):
        """Encodes the distribution parameters for the gauges in json form

        Returns
        -------
        d : dict 
            Dictionary of distribution parameters in json form.
        """
        ignore = ['dists','obstypes']
        d = {key:self.__dict__[key] for key in self.__dict__ if key not in ignore}
        d['dist_params'] = {}
        for key,dist in self.dists.items():
            d['dist_params'][key] = {'name':dist.dist.name,'shapes':dist.kwds}
        return d

    @classmethod
    def from_json(cls,d):
        """Descodes/deserializes the data from .json form. Opposite function as to_json """
        return cls.from_shapes(**d)

    def plot(self,obstype,ax=None):
        """Plots the specified observation type for gauges in an appropriate interval.

        Parameters
        ----------
        obstype : str
            The name of the attribute to be graphed (ex. 'arrival', 'height', 'inundation')
        ax : Axes object
            Defaults to None. The object used to create axes and axis lables for the plot. 
        """
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
        ax.set_title(self.name + ' ' + obstype)

    def plot_all(self,fig=None):
        """Plots all of the observation data: arrivals, heights, innundations.

        Parameters
        ----------
        fig : Figure object
            Defaults to None. An object of the matplotlib.pyplot class 
            used to create the figure on the plot.
        """
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
    """Opens the 'outfile' and writes/encodes the gauge data to a .json file to serialize the data"""
    with open(outfile,'w') as f:
        json.dump([gauge.to_json() for gauge in gauges],f)

def load_gauges(infile):
    """Opens and reads 'infile' and decodes the data from .json form. Performs the opposite as dump_gauges """
    with open(infile,'r') as f:
        lst = json.load(f)
    return [Gauge.from_json(d) for d in lst]
