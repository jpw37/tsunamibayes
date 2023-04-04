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
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = -4.5175
    gauge.lon = 129.775
    gauge.loc = 3
    gauge.scale = 0.8
    gauges.append(gauge)

    # Ambon
    name = 'Ambon'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8,scale=0.4)
    gauge = Gauge(name,dists)
    gauge.lat = -3.691
    gauge.lon = 128.178
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Banda Neira
    name = 'Banda Neira'
    dists = dict()
    dists['arrival'] = stats.skewnorm(a=2,loc=15,scale=5)
    dists['height'] = stats.norm(loc=6.5,scale=1.5)
    dists['inundation'] = stats.norm(loc=185,scale=65)
    gauge = Gauge(name,dists)
    gauge.lat = -4.5248
    gauge.lon = 129.8965
    gauge.beta = 4.253277987952933
    gauge.n = 0.06
    gauge.loc = 6.5
    gauge.scale = 1.5
    gauges.append(gauge)

    # Buru
    name = 'Buru'
    dists = dict()
    dists['height'] = stats.chi(df=1.01,loc=0.5,scale=1.5)
    gauge = Gauge(name,dists)
    gauge.lat = -3.3815
    gauge.lon = 127.113
    gauge.loc = 0.5
    gauge.scale = 1.5
    gauge.df = 1.01
    gauges.append(gauge)

    # Hulaliu
    name = 'Hulaliu'
    dists = dict()
    dists['height'] = stats.chi(df=1.01,loc=0.5,scale=2.0)
    gauge = Gauge(name,dists)
    gauge.lat = -3.543
    gauge.lon = 128.557
    gauge.df = 1.01
    gauge.loc = 0.5
    gauge.scale = 2.0
    gauges.append(gauge)

    # Saparua
    name = 'Saparua'
    dists = dict()
    dists['arrival'] = stats.norm(loc=45,scale=5)
    dists['height'] = stats.norm(loc=5,scale=1)
    dists['inundation'] = stats.norm(loc=125,scale=40)
    gauge = Gauge(name,dists)
    gauge.lat = -3.576
    gauge.lon = 128.657
    gauge.beta = 1.1067189507222546
    gauge.n = 0.06
    gauge.loc = 5
    gauge.scale = 1
    gauges.append(gauge)

    # Kulur
    name = 'Kulur'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=1)
    gauge = Gauge(name,dists)
    gauge.lat = -3.501
    gauge.lon = 128.562
    gauge.loc = 3
    gauge.scale = 1
    gauges.append(gauge)

    # Ameth
    name = 'Ameth'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=1)
    gauge = Gauge(name,dists)
    gauge.lat = -3.6455
    gauge.lon = 128.807
    gauge.loc = 3
    gauge.scale = 1
    gauges.append(gauge)

    # Amahai
    name = 'Amahai'
    dists = dict()
    dists['height'] = stats.norm(loc=3.5,scale=1)
    gauge = Gauge(name,dists)
    gauge.lat = -3.338
    gauge.lon = 128.921
    gauge.loc = 3.5
    gauge.scale = 1
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
