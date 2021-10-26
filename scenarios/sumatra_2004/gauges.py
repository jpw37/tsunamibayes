from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.
    
    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Phi Phi
    name = 'Phi Phi'
    dists = dict()
    dists['height'] = skewnorm(loc=3.5, scale=1.5, a=2.5)
    gauge = Gauge(name,dists)
    gauge.lat = 7.746111
    gauge.lon = 98.773889
    gauges.append(gauge)

    # Khao Lak
    name = 'Khao Lak'
    dists = dict()
    dists['arrival'] = stats.skewnorm(loc=10.5, scale=.25, a=0)
    gauge = Gauge(name,dists)
    gauge.lat = 8.733333
    gauge.lon = 98.233333
    gauges.append(gauge)

    # Phuket
    name = 'Phuket'
    dists = dict()
    dists['height'] = skewnorm(loc=10, scale=1.5, a=0)
    gauge = Gauge(name,dists)
    gauge.lat = 8
    gauge.lon = 98.283333
    gauges.append(gauge)

    # Banda Aceh
    name = 'Banda Aceh'
    dists = dict()
    dists['height'] = stats.skewnorm(loc=30, scale=12, a=-10)
    dists['arrival'] = stats.skewnorm(loc=8.8, scale=.5, a=-1)
    gauge = Gauge(name,dists)
    gauge.lat = 5.562639
    gauge.lon = 95.293194
    gauges.append(gauge)

    # Telwatta
    name = 'Telwatta'
    dists = dict()
    dists['inundation'] = skewnorm(loc=225, scale=100, a=5)
    gauge = Gauge(name,dists)
    gauge.lat = 6.17
    gauge.lon = 80.085833
    gauges.append(gauge)
    
    # Trincomalee
    name = 'Trincomalee'
    dists = dict()
    dists['height'] = stats.skewnorm(loc=14, scale=2, a=0)
    dists['inundation'] = stats.skewnorm(loc=1, scale=.05, a=0)
    gauge = Gauge(name,dists)
    gauge.lat = 8.597222
    gauge.lon = 81.222222
    #gauge.beta = 1.1067189507222546
    #gauge.n = 0.06
    gauges.append(gauge)

    # Chittagong
    name = 'Chittagong'
    dists = dict()
    dists['height'] = stats.skewnorm(loc=2, scale=.5, a=-1)
    gauge = Gauge(name,dists)
    gauge.lat = 2.268333
    gauge.lon = 91.772222
    gauges.append(gauge)

    # Puducherry
    name = 'Puducherry'
    dists = dict()
    dists['height'] = stats.skewnorm(loc=12.5, scale=3, a=0)
    gauge = Gauge(name,dists)
    gauge.lat = 11.9275
    gauge.lon = 70.834167
    gauges.append(gauge)

    # Penang
    name = 'Penang'
    dists = dict()
    dists['height'] = stats.skewnorm(loc=100, scale=50, a=10)
    gauge = Gauge(name,dists)
    gauge.lat = 5.429167
    gauge.lon = 100.320833
    gauges.append(gauge)

    return gauges

if __name__=="__main__":
    """Builds the scenario's gauges and stores the data in either a default file,
    or a file specified by the user in the command line."""
    from sys import argv

    if len(argv) == 1:
        gauges_path = 'data/gauges.json'
    else:
        gauges_path = argv[1]

    dump_gauges(build_gauges(),gauges_path)