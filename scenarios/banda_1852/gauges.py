from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges():
    """Creates gauge objects for each observation points, and appends each to a list.
    
    Returns
    -------
    gagues : (list) of Gauge objects
    """
    gauges = list()

    # Pulu Ai
    name = 'Pulu Ai'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = -4.5175
    gauge.lon = 129.775
    gauges.append(gauge)

    # Ambon
    name = 'Ambon'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8,scale=0.4)
    gauge = Gauge(name,dists)
    gauge.lat = -3.691
    gauge.lon = 128.178
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
    gauges.append(gauge)

    # Buru
    name = 'Buru'
    dists = dict()
    dists['height'] = stats.chi(df=1.01,loc=0.5,scale=1.5)
    gauge = Gauge(name,dists)
    gauge.lat = -3.3815
    gauge.lon = 127.113
    gauges.append(gauge)

    # Hulaliu
    name = 'Hulaliu'
    dists = dict()
    dists['height'] = stats.chi(df=1.01,loc=0.5,scale=2.0)
    gauge = Gauge(name,dists)
    gauge.lat = -3.543
    gauge.lon = 128.557
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
    gauges.append(gauge)

    # Kulur
    name = 'Kulur'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=1)
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
    from sys import argv

    if len(argv) == 1:
        gauges_path = 'data/gauges.json'
    else:
        gauges_path = argv[1]

    dump_gauges(build_gauges(),gauges_path)
