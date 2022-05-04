from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges(delta):
    """Creates gauge object for each observation point's data and appends each to a list.

    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Pulu Ai
    name = 'Pulu Ai'
    dists = dict()
    dists['height'] = stats.norm(loc=3*delta,scale=0.8*delta)  #change loc /  scale tweek~10%
    gauge = Gauge(name,dists)
    gauge.lat = -4.5175
    gauge.lon = 129.775
    gauges.append(gauge)

    # Ambon
    name = 'Ambon'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8*delta,scale=0.4*delta)  #like above
    gauge = Gauge(name,dists)
    gauge.lat = -3.691
    gauge.lon = 128.178
    gauges.append(gauge)

    # Banda Neira
    name = 'Banda Neira'
    dists = dict()
    dists['arrival'] = stats.skewnorm(a=2*delta,loc=15*delta,scale=5*delta)# like above
    dists['height'] = stats.norm(loc=6.5*delta,scale=1.5*delta)
    #dists['inundation'] = stats.norm(loc=185,scale=65)
    gauge = Gauge(name,dists)
    gauge.lat = -4.5248
    gauge.lon = 129.8965
    gauge.beta = 4.253277987952933
    gauge.n = 0.06
    gauges.append(gauge)

    # Buru
    name = 'Buru'
    dists = dict()
    dists['height'] = stats.chi(df=1.01*delta,loc=0.5*delta,scale=1.5*delta) #like above
    gauge = Gauge(name,dists)
    gauge.lat = -3.3815
    gauge.lon = 127.113
    gauges.append(gauge)

    # Hulaliu
    name = 'Hulaliu'
    dists = dict()
    dists['height'] = stats.chi(df=1.01*delta,loc=0.5*delta,scale=2.0*delta)#like above
    gauge = Gauge(name,dists)
    gauge.lat = -3.543
    gauge.lon = 128.557
    gauges.append(gauge)

    # Saparua
    name = 'Saparua'
    dists = dict()
    dists['arrival'] = stats.norm(loc=45*delta,scale=5*delta)#likeabove
    dists['height'] = stats.norm(loc=5*delta,scale=1*delta)#like a bove
    #dists['inundation'] = stats.norm(loc=125,scale=40)
    gauge = Gauge(name,dists)
    gauge.lat = -3.576
    gauge.lon = 128.657
    gauge.beta = 1.1067189507222546
    gauge.n = 0.06
    gauges.append(gauge)

    # Kulur
    name = 'Kulur'
    dists = dict()
    dists['height'] = stats.norm(loc=3*delta,scale=1*delta) #likeabove
    gauge = Gauge(name,dists)
    gauge.lat = -3.501
    gauge.lon = 128.562
    gauges.append(gauge)

    # Ameth
    name = 'Ameth'
    dists = dict()
    dists['height'] = stats.norm(loc=3*delta,scale=1*delta)  #like  above
    gauge = Gauge(name,dists)
    gauge.lat = -3.6455
    gauge.lon = 128.807
    gauges.append(gauge)

    # Amahai
    name = 'Amahai'
    dists = dict()
    dists['height'] = stats.norm(loc=3.5*delta,scale=1*delta)
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
        gauges_path = 'data/gauges_step_'+str(delta)+'_.json'

    else:
        gauges_path = argv[1]

    dump_gauges(build_gauges(delta),gauges_path)
