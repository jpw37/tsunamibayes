from gauge import Gauge, calculate_probability


#This is used to load all the priors from 1852 into the setup.py file

def load_priors():
    gauges = []

#GAUGE 1_____________Water Height______________
    city_name = 'Pulu Ai - Wichmann'
    name = 10000 # Pulu Ai - Wichmann
    longitude = -4.517863
    latitude = 129.7745653
    distance = 2.5 # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm', None]

    # For kind = 'norm'
    arrival_mean = None # in minutes
    arrival_std = None
    arrival_params = [arrival_mean, arrival_std]
    height_mean = 1. # in meters
    height_std =  0.2
    height_params = [height_mean, height_std]
    inundation_std = None
    inundation_mean = None
    inundation_params = [inundation_mean, inundation_std]

    beta = 0
    n = .015 # OR .03

    g = Gauge(name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())


#GAUGE 2_____________Water Height______________
    city_name = 'Ambonia - Wichmann'
    name = 10000
    longitude = -3.693521
    latitude = 128.175538
    distance = 2.5 # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm', None]

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    height_mean = 1.8  # in meters
    height_std = 0.1
    arrival_params = [arrival_mean, arrival_std]
    height_params = [height_mean, height_std]
    inundation_std = None
    inundation_mean = None
    inundation_params = [inundation_mean, inundation_std]
    beta = 0
    n = .015

    g = Gauge(name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())

# GAUGE 3___________Water Height________________
    city_name = 'Banda Neira - Tsunami Catalog'
    name = 10000  #
    longitude = -4.529905
    latitude = 129.897376
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm', None]

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    arrival_params = [arrival_mean, arrival_std]
    height_mean = 4.0  # in meters
    height_std = 1.0
    height_params = [height_mean, height_std]
    inundation_std = None
    inundation_mean = None
    inundation_params = [inundation_mean, inundation_std]
    beta = 0
    n = .015

    g = Gauge(name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())

# GAUGE 4____________Water Height_______________
    city_name = 'PULAU BURU - Wichmann Catalog'
    name = 10000  #
    longitude = -3.848928
    latitude = 126.733678
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'chi2', None]

    # For kind = 'chi2'
    arrival_k = None # chi2 parameter
    arrival_lower_bound = None # in meters
    arrival_params = [arrival_k, arrival_lower_bound]
    height_k = 1.1 # chi2 param
    height_lower_bound = 0 # in meters
    height_params = [height_k, height_lower_bound]
    inundation_k = None
    inundation_lower_bound = None
    inundation_params = [inundation_k, inundation_lower_bound]

    beta = 0
    n = .015

    g = Gauge(name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())

# GAUGE 5___________Water Height________________
    city_name = 'Saparua - Wichmann Catalog'
    name = 10000  #
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm', None]

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    arrival_params = [arrival_mean, arrival_std]
    height_mean = 3.0  # in meters
    height_std = .75
    height_params = [height_mean, height_std]
    inundation_std = None
    inundation_mean = None
    inundation_params = [inundation_mean, inundation_std]
    beta = 0
    n = .03

    g = Gauge(name, longitude, latitude, distance,
                    kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())

# GAUGE 6___________Inundation________________
    city_name = 'Suparua - Wichmann'
    name = 10000  # Suparua - Wichmann
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, None, 'norm']

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    arrival_params = [arrival_mean, arrival_std]
    height_mean = None  # in meters
    height_std = None
    height_params = [height_mean, height_std]
    inundation_std = 123.44
    inundation_mean = .185806
    inundation_params = [inundation_mean, inundation_std]

    beta = 0
    n = .03

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())


# GAUGE 7___________Inundation_________________
    city_name = 'Banda Neira - Wichmann'
    name = 10000  #
    longitude = -4.529905
    latitude = 129.897376
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, None, 'skewnorm']

    # For kind = 'skewnorm'
    arrival_skew_param = None
    arrival_mean = None  # in minutes
    arrival_std = None
    arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    height_skew_param = None
    height_mean = None  # in meters
    height_std = None
    height_params = [height_skew_param, height_mean, height_std]
    inundation_skew_param = 3
    inundation_mean = 231  # in meters
    inundation_std = 85
    inundation_params = [inundation_skew_param, inundation_mean, inundation_std]

    beta = 0
    n = .03

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())

# GAUGE 8___________Inundation_________________
    city_name = 'Lonthor - Wichmann'
    name = 10000  #
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, None, 'skewnorm']


    # For kind = 'skewnorm'
    arrival_skew_param = None
    arrival_mean = None  # in minutes
    arrival_std = None
    arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    height_skew_param = None
    height_mean = None  # in meters
    height_std = None
    height_params = [height_skew_param, height_mean, height_std]
    inundation_skew_param = -5
    inundation_mean = 20  # in meters
    inundation_std = 5
    inundation_params = [inundation_skew_param, inundation_mean, inundation_std]

    beta = 0
    n = .03

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())


# GAUGE 9___________Arrival Time_________________
    city_name = 'Banda Neira  - Wichmann'
    name = 10000
    longitude = -4.529905
    latitude = 129.897376
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['skewnorm', None, None]


    # For kind = 'skewnorm'
    arrival_skew_param = 5
    arrival_mean = 15  # in minutes
    arrival_std = 10
    arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    height_skew_param = None
    height_mean = None  # in meters
    height_std = None
    height_params = [height_skew_param, height_mean, height_std]
    inundation_skew_param = None
    inundation_mean = None  # in meters
    inundation_std = None
    inundation_params = [inundation_skew_param, inundation_mean, inundation_std]

    beta = 0
    n = .03

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())


# GAUGE 10___________Arrival Time________________
    city_name = 'Pulau Saparua  - Wichmann'
    name = 10000
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['skewnorm', None, None]

    # For kind = 'skewnorm'
    arrival_skew_param = 3.5
    arrival_mean = 60  # in minutes
    arrival_std = 10
    arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    height_skew_param = None
    height_mean = None  # in meters
    height_std = None
    height_params = [height_skew_param, height_mean, height_std]
    inundation_mean = None  # in meters
    inundation_std = None
    inundation_params = [inundation_skew_param, inundation_mean, inundation_std]

    beta = 0
    n = .03

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    gauges.append(g.to_json())


# GAUGE 11___________Wave Height________________
    # ---------------------------------------------------------------------
    # THIS IS THE AMBONIA OBSERVATION THAT WE ARE OMITTING FROM THIS RUN.
    # ---------------------------------------------------------------------
    # city_name = 'Pulau Ambon (Ambonia) - Whichmann'
    # name = 10000  #
    # longitude = -4.517863
    # latitude = 129.7745653
    # distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    # kind = [None, 'chi2', None]
    #
    # # For kind = 'chi2'
    # arrival_k = None # chi2 parameter
    # arrival_lower_bound = None # in meters
    # arrival_params = [arrival_k, arrival_lower_bound]
    # height_k = 50 # chi2 param
    # height_lower_bound = 0 # in meters
    # height_params = [height_k, height_lower_bound]
    # inundation_k= None
    # inundation_lower_bound = None
    # inundation_params = [inundation_k, inundation_lower_bound]
    #
    # beta = 0
    # n = .015
    #
    #     g = Gauge(name, longitude, latitude, distance,
    #               kind, arrival_params, height_params, inundation_params, beta, n, city_name)
    #     gauges.append(g.to_json())


# _______________Examples of What Goes In The Gauge____________________
    # kind = ['Arrival, Height, Inundation]

    # For kind = 'norm' also kind[0]
    # arrival_mean = None  # in minutes
    # arrival_std = None
    # arrival_params = [arrival_mean, arrival_std]
    # height_mean = None  # in meters
    # height_std = None
    # height_params = [height_mean, height_std]
    # inundation_std = None
    # inundation_mean = None
    # inundation_params = [inundation_mean, inundation_std]

    #
    # For kind = 'chi2' of kind[1]
    # arrival_k = None  # chi2 parameter
    # arrival_lower_bound = None  # in meters
    # arrival_params = [arrival_k, arrival_lower_bound]
    # height_k = None  # chi2 param
    # height_lower_bound = None  # in meters
    # height_params = [height_k, height_lower_bound]
    # inundation_k = None
    # inundation_lower_bound = None
    # inundation_params = [inundation_k, inundation_lower_bound]
    #
    # For kind = 'skewnorm' OR kind[2]
    # arrival_skew_param = None
    # arrival_mean = None  # in minutes
    # arrival_std = None
    # arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    # height_skew_param = None
    # height_mean = None  # in meters
    # height_std = None
    # height_params = [height_skew_param, height_mean, height_std]
    # inundation_mean = None  # in meters
    # inundation_std = None
    # inundation_params = [inundation_skew_param, inundation_mean, inundation_std]

    return gauges

gauges = load_priors()

# for gauge in gauges:
#     print(gauge)


print(calculate_probability(gauges))
