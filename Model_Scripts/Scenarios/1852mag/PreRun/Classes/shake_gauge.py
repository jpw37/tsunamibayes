"""
Created By Josh Lapicola
Created 6/12/2019
Property of BYU Mathematics Dept.
"""

from scipy import stats

class shake_gauge:
    """A gauge object class. Mainly for data storage.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        longitude (float): the longitude location of the gauge.
        latitude (float): the latitude location of the gauge.
        kind (string): the type of distribution to be used. ex: 'norm'
        params (list): list of params for the arrival time distribution.
        distribution (stats object): distribution of observed MMI at gauge.
        city_name
    """
    def __init__(self, name, longitude, latitude, VS30, kind, params, city_name):
        self.latitude = latitude
        self.longitude = longitude
        self.VS30 = VS30
        self.kind = kind
        if self.kind == 'norm':
            self.distribution = stats.norm(loc=params[0], scale=params[1])
        elif self.kind == 'skewnorm':
            self.distribution = stats.skewnorm(loc=params[0], scale=params[1], a=params[2])
        elif self.kind == 'uniform':
            self.distribution = stats.uniform(loc=params[0], scale=params[1])
        elif self.kind == 'felt':
            self.distribution = felt_distribution(a=0, b=12)
        self.name = name
        self.city_name = city_name

    def to_json(self):
        """
        Convert object to dict of attributes for json
        """
        d = dict()
        d['name'] = self.name
        d['longitude'] = self.longitude
        d['latitude'] = self.latitude
        d['VS30'] = self.VS30
        d['kind'] = self.kind
        d['params'] = self.params
        d['city_name'] = self.city_name
        return d

def from_json(d):
    """
    Converts from json file format into gauge object
    """
    return Gauge(d['name'], d['longitude'], d['latitude'],d['VS30'], d['kind'], d['params'], d['city_name'])
