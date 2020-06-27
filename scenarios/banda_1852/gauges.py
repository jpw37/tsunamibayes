from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.
    
    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Pulu Ai
    name = 'Pulu Ai'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.2)
    gauge = Gauge(name,dists)
    gauge.lat = -4.5166
    gauge.lon = 129.775
    gauges.append(gauge)

    # Ambon
    name = 'Ambon'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8,scale=0.1)
    gauge = Gauge(name,dists)
    gauge.lat = -3.691
    gauge.lon = 128.178
    gauges.append(gauge)

    # Banda Neira
    name = 'Banda Neira'
    dists = dict()
    dists['arrival'] = stats.skewnorm(a=2,loc=15,scale=5)
    dists['height'] = stats.norm(loc=6.5,scale=1)
    dists['inundation'] = stats.skewnorm(a=3,loc=231,scale=85)
    gauge = Gauge(name,dists)
    gauge.lat = -4.5248
    gauge.lon = 129.896
    gauge.beta = 4.253277987952933
    gauge.n = 0.03
    gauges.append(gauge)

    # Buru
    name = 'Buru'
    dists = dict()
    dists['height'] = stats.chi(df=1.01,loc=1.0,scale=1.0)
    gauge = Gauge(name,dists)
    gauge.lat = -3.3815
    gauge.lon = 127.115
    gauges.append(gauge)

    # Saparua
    name = 'Saparua'
    dists = dict()
    dists['arrival'] = stats.norm(loc=45,scale=5)
    dists['height'] = stats.norm(loc=5,scale=.75)
    dists['inundation'] = stats.norm(loc=120,scale=10)
    gauge = Gauge(name,dists)
    gauge.lat = -3.576
    gauge.lon = 128.657
    gauge.beta = 1.1067189507222546
    gauge.n = 0.03
    gauges.append(gauge)

    # Kulur
    name = 'Kulur'
    dists = dict()
    dists['height'] = stats.norm(loc=2.5,scale=0.7)
    gauge = Gauge(name,dists)
    gauge.lat = -3.501
    gauge.lon = 128.562
    gauges.append(gauge)

    # Ameth
    name = 'Ameth'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=1)
    gauge = Gauge(name,dists)
    gauge.lat = -3.6455
    gauge.lon = 128.807
    gauges.append(gauge)

    # Amahai
    name = 'Amahai'
    dists = dict()
    dists['height'] = stats.norm(loc=3.5,scale=1)
    gauge = Gauge(name,dists)
    gauge.lat = -3.338
    gauge.lon = 128.921
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
