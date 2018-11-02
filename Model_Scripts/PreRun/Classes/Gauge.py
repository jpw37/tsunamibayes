# A file containing the Gauge class and gauge related functions
from scipy import stats

class Gauge:
    """A gauge object class. Mainly for data storage.

    Attributes:
        name (int): the name you wish to give the gauge (a 5 digit number).
        longitude (float): the longitude location of the gauge.
        latitude (float): the latitude location of the gauge.
        distance (float): the distance from the gauge to shore. (km)
        kind (list): the type of distribution to be used for the wave
            arrival time and height respectively. ex: ['norm', 'chi2']
        arrival_params (list): list of params for the arrival time distribution
        height_params (list): list of params for the height distribution
        inundation_params (list): list of params for the inundation distribution
        arrival_dist (stats object): distribution of arrival time at gauge
        height_dist (stats object): distribution of wave height at gauge
    """
    def __init__(self, name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name):
        self.name = name
        self.city_name = city_name
        self.longitude = longitude
        self.latitude = latitude
        self.distance = distance
        self.kind = kind
        self.arrival_params = arrival_params
        self.height_params = height_params
        self.beta = beta
        self.n = n
        self.inundation_params = inundation_params
        if name is not None: # Allows for None initialized object
            # Kind[0] is for Wave Arrival Times
            # kind[1] is for Wave Height
            # kind[2] is for Inundation
            if kind[0] == 'norm':
                mean = arrival_params[0]
                std = arrival_params[1]
                self.arrival_dist = stats.norm(mean, std)
            elif kind[0] == 'chi2':
                k = arrival_params[0]
                loc = arrival_params[1]
                self.arrival_dist = stats.chi2(k, loc=loc)
            elif kind[0] == 'skewnorm':
                skew_param = arrival_params[0]
                mean = arrival_params[1]
                std = arrival_params[2]
                self.arrival_dist = stats.skewnorm(skew_param, mean, std)

            if kind[1] == 'norm':
                mean = height_params[0]
                std = height_params[1]
                self.height_dist = stats.norm(mean, std)
            elif kind[1] == 'chi2':
                k = height_params[0]
                loc = height_params[1]
                self.height_dist = stats.chi2(k, loc=loc)
            elif kind[1] == 'skewnorm':
                skew_param = height_params[0]
                mean = height_params[1]
                std = height_params[2]
                self.height_dist = stats.skewnorm(skew_param, mean, std)

            if kind[2] == 'norm':
                mean = inundation_params[0]
                std = inundation_params[1]
                self.inundation_dist = stats.norm(mean, std)
            elif kind[2] == 'chi2':
                k = inundation_params[0]
                loc = inundation_params[1]
                self.inundation_dist = stats.chi2(k, loc=loc)
            elif kind[2] == 'skewnorm':
                skew_param = inundation_params[0]
                mean = inundation_params[1]
                std = inundation_params[2]
                self.inundation_dist = stats.skewnorm(skew_param, mean, std)


    def to_json(self):
        """
        Convert object to dict of attributes for json
        """
        d = dict()
        d['name'] = self.name
        d['longitude'] = self.longitude
        d['latitude'] = self.latitude
        d['distance'] = self.distance
        d['kind'] = self.kind
        d['arrival_params'] = self.arrival_params
        d['height_params'] = self.height_params
        d['inundation_params'] = self.inundation_params
        d['beta'] = self.beta
        d['n'] = self.n
        d['city_name'] = self.city_name
        return d

    def from_json(self, d):
        """
        Converts from json file format into gauge object
        """
        self.__init__(d['name'], d['longitude'], d['latitude'],
                        d['distance'], d['kind'], d['arrival_params'],
                        d['height_params'], d['inundation_params'], d['beta'], d['n'], d['city_name'])
