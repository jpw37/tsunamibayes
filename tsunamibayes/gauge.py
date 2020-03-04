import scipy.stats as stats

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
    vs30 : float (optional)
        Shear-wave velocity in the top 30 meters of soil. Only used for shake
        observations
    """
    tsunami_obstypes = ['arrival','height','inundation']
    shake_obstypes = ['pga']
    obstypes = tsunami_obstypes + shake_obstypes
    def __init__(self,name,lat,lon,dists,vs30=None):
        # core attributes
        self.name = name
        self.lat = lat
        self.lon = lon

        # shake observation error catching
        if vs30 is not None:
            if not any([obstype in dists.keys() for obstype in Gauge.shake_obstypes]):
                raise TypeError("At least one observation type from {} must be specified when 'vs30' is given".format(Gauge.shake_obstypes))

        for obstype,dist in dists.items():
            if obstype not in Gauge.obstypes:
                raise TypeError("'{}' is not a valid observation type (must be one of {})".format(obstype,Gauge.obstypes))
            if not isinstance(dist,stats._distn_infrastructure.rv_frozen):
                raise TypeError("dists['{}'] must be a frozen scipy.stats distribution".format(obstype))
        self.dists = dists
        self.obstypes = self.dists.keys()

    @classmethod
    def from_shapes(cls,name,lat,lon,params,**kwargs):
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
        params : dict
            Dictionary of distribution parameters. Each key corresponds to another dictionary:
            params['arrival'] = {'name':'norm','shapes':{'loc':1,'scale':2}}.
            This dictionary must contain a valid scipy.stats distribution name,
            as well as a dictionary of shape parameters that will be passed to the
            scipy.stats constructor as keyword arguments.
        **kwargs
            Additional arguments to the Gauge class constructor.
        """
        dists = {}
        for obstype,d in params.items():
            if obstype not in Gauge.obstypes:
                raise TypeError("'{}' is not a valid observation type (must be one of {})".format(obstype,Gauge.obstypes))
            if 'name' not in d.keys():
                raise TypeError("Observation type '{}' must have an associated distribution name")
            if 'shapes' not in d.keys():
                raise TypeError("Observation type '{}' must have associated distribution shape parameters")
            dists[obstype] = getattr(stats,d['name'])(**d['shapes'])
        return cls(name,lat,lon,dists,**kwargs)

    def to_json(self):
        ignore = ['dists','obstypes']
        d = {key:self.__dict__[key] for key in self.__dict__ if key not in ignore}
        d['params'] = {}
        for key,dist in self.dists.items():
            d['params'][key] = {'name':dist.dist.name,'shapes':dist.kwds}
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
