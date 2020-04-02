import scipy.stats as stats
import json

class Gauge:
    """Class for managing data related to observations. A Gauge object
    loosely corresponds to an observation location, along with probability distributions
    representing each type of observation associated with that location.

    Parameters
    ----------
    name : str
        Name of the observation (for use in output data files)
    lat : float
        Latitude of observation location
    lon : float
        Longitude of observation location
    dists : dict
        Dictionary of scipy.stats frozen_rv objects. Each distribution's key
        corresponds to which type of observation the distribution is associated
        with
    **kwargs : optional
        Additional arguments to class constructor. Set as class attributes
    """

    def __init__(self,name,lat,lon,dists,**kwargs):
        # core attributes
        self.name = name
        self.lat = lat
        self.lon = lon

        for obstype,dist in dists.items():
            if not isinstance(dist,stats._distn_infrastructure.rv_frozen):
                raise TypeError("dists['{}'] must be a frozen scipy.stats distribution".format(obstype))
        self.dists = dists
        self.obstypes = self.dists.keys()

        for key,value in kwargs.items():
            setattr(self,key,value)

    @classmethod
    def from_shapes(cls,name,lat,lon,dist_params,**kwargs):
        """Alternate constructor for Gauge objects. Accepts a `params` dictionary
        rather than a `dists` dictionary, where the `params` dictionary contains
        the various parameters associated with the scipy.stats frozen_rv object
        that will be constructed within the class.

        Parameters
        ----------
        name : str
            Name of the observation (for use in output data files)
        lat : float
            Latitude of observation location
        lon : float
            Longitude of observation location
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
                raise TypeError("Observation type '{}' must have an associated distribution name")
            elif 'shapes' not in d.keys():
                raise TypeError("Observation type '{}' must have associated distribution shape parameters")
            dists[obstype] = getattr(stats,d['name'])(**d['shapes'])
        return cls(name,lat,lon,dists,**kwargs)

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

def dump_gauges(gauges,outfile):
    with open(outfile,'w') as f:
        json.dump([gauge.to_json() for gauge in gauges],f)

def load_gauges(infile):
    with open(infile,'r') as f:
        lst = json.load(f)
    return [Gauge.from_json(d) for d in lst]
