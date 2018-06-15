from gauge import Gauge


def load_priors():
    gauges = []

#GAUGE 1_____________Water Height______________
    
    name = 10000 # Pulu Ai - Wichmann
    longitude = -4.517863
    latitude = 129.7745653
    distance = 2.5 # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm']

    # For kind = 'norm'
    arrival_mean = None # in minutes
    arrival_std = None
    height_mean = 1. # in meters
    height_std =  0.2
    arrival_params = [arrival_mean, arrival_std]
    height_params = [height_mean, height_std]
    beta = 0
    n = 0

    g = Gauge(name, longitude, latitude, distance,
                    kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())


#GAUGE 2_____________Water Height______________
    name = 10000 # Ambonia - Wichmann
    longitude = -3.693521
    latitude = 128.175538
    distance = 2.5 # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm']

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    height_mean = 1.8  # in meters
    height_std = 0.1
    arrival_params = [arrival_mean, arrival_std]
    height_params = [height_mean, height_std]
    beta = 0
    n = 0

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())

# GAUGE 3___________Water Height________________
    name = 10000  # Banda Neira
    longitude = -4.529905
    latitude = 129.897376
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm']

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    height_mean = 4.0  # in meters
    height_std = 1.0
    arrival_params = [arrival_mean, arrival_std]
    height_params = [height_mean, height_std]

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())

# GAUGE 4____________Water Height_______________
    name = 10000  # PULAU BURU
    longitude = -3.848928
    latitude = 126.733678
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'chi2']

    # For kind = 'chi2'
    arrival_k = None # chi2 parameter
    arrival_lower_bound = None # in meters
    height_k = 1.1 # chi2 param
    height_lower_bound = 0 # in meters
    arrival_params = [arrival_k, arrival_lower_bound]
    height_params = [height_k, height_lower_bound]

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())

# GAUGE 5___________Water Height________________
    name = 10000  # Saparua
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = [None, 'norm']

    # For kind = 'norm'
    arrival_mean = None  # in minutes
    arrival_std = None
    height_mean = 3.0  # in meters
    height_std = .75
    arrival_params = [arrival_mean, arrival_std]
    height_params = [height_mean, height_std]

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())

# GAUGE 6___________Inundation________________
    name = 10000  # Suparua - Wichmann
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['norm', 'norm']

    # mean = 123.44
    # var = .185806  # meters squared

    # TODO Not sure how the inundation is factored in to the guages -no previous examples


    # g = Gauge(name, longitude, latitude, distance,
    #           kind, arrival_params, height_params, beta, n)
    # gauges.append(g.to_json())


# GAUGE 7___________Inundation_________________
    name = 10000  #  Banda Neira - Wichmann
    longitude = -4.529905
    latitude = 129.897376
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['norm', 'norm']

    # mean = 231
    # var = 85
    # skew = 3

    # TODO Not sure how the inundation is factored in to the guages -no previous examples

    # g = Gauge(name, longitude, latitude, distance,
    #           kind, arrival_params, height_params, beta, n)
    # gauges.append(g.to_json())

# GAUGE 8___________Inundation_________________
    name = 10000  # Lonthor - Wichmann
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['norm', 'norm']

    # mean = 20
    # var = 5
    # skew = -5

    # TODO Not sure how the inundation is factored in to the guages -no previous examples


    # g = Gauge(name, longitude, latitude, distance,
    #           kind, arrival_params, height_params, beta, n)
    # gauges.append(g.to_json())


# GAUGE 9___________Arrival Time_________________
    name = 10000  # Banda Neira - Wichmann
    longitude = -4.529905
    latitude = 129.897376
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['skewnorm', None]


    # For kind = 'skewnorm'
    arrival_skew_param = 5
    arrival_mean = 15  # in minutes
    arrival_std = 10
    height_skew_param = None
    height_mean = None  # in meters
    height_std = None
    arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    height_params = [height_skew_param, height_mean, height_std]


    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())


# GAUGE 10___________Arrival Time________________
    name = 10000  # Pulau Saparua - Wichmann
    longitude = -3.576063
    latitude = 128.658715
    distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    kind = ['skewnorm', None]

    # For kind = 'skewnorm'
    arrival_skew_param = 3.5
    arrival_mean = 60  # in minutes
    arrival_std = 10
    height_skew_param = None
    height_mean = None  # in meters
    height_std = None
    arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    height_params = [height_skew_param, height_mean, height_std]

    g = Gauge(name, longitude, latitude, distance,
              kind, arrival_params, height_params, beta, n)
    gauges.append(g.to_json())


# GAUGE 11___________Wave Height________________
    # ---------------------------------------------------------------------
    # THIS IS THE AMBONIA OBSERVATION THAT WE ARE OMITTING FROM THIS RUN.
    # ---------------------------------------------------------------------
    # name = 10000  #  Pulau Ambon (Ambonia) - Whichmann
    # longitude = -4.517863
    # latitude = 129.7745653
    # distance = 2.5  # in kilometers (max 5) //TODO Not sure what this is??
    # kind = ['chi2', 'chi2']
    #
    # # For kind = 'chi2'
    # arrival_k = None # chi2 parameter
    # arrival_lower_bound = None # in meters
    # height_k = 50 # chi2 param
    # height_lower_bound = 0 # in meters
    # arrival_params = [arrival_k, arrival_lower_bound]
    # height_params = [height_k, height_lower_bound]
    #
    # g = Gauge(name, longitude, latitude, distance,
    #           kind, arrival_params, height_params, beta, n)
    # gauges.append(g.to_json())


# _______________Examples of What Goes In The Gauge____________________

    # For kind = 'norm'
    # arrival_mean = 30.  # in minutes
    # arrival_std = 6.
    # height_mean = None  # in meters
    # height_std = None
    # arrival_params = [arrival_mean, arrival_std]
    # # height_params = [height_mean, height_std]
    #
    # # For kind = 'chi2'
    # arrival_k = None # chi2 parameter
    # arrival_lower_bound = None # in meters
    # height_k = None # chi2 param
    # height_lower_bound = None # in meters
    # arrival_params = [arrival_k, arrival_lower_bound]
    # height_params = [height_k, height_lower_bound]
    #
    # # For kind = 'skewnorm'
    # arrival_skew_param = None
    # arrival_mean = None # in minutes
    # arrival_std = None
    # height_skew_param = None
    # height_mean = None # in meters
    # height_std = None
    # # arrival_params = [arrival_skew_param, arrival_mean, arrival_std]
    # # height_params = [height_skew_param, height_mean, height_std]

    return gauges

priors = load_priors()

for prior in priors:
    print(prior)
