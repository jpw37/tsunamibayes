from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.

    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Bulukumba
    name = 'Bulukumba'
    dists = dict()
    dists['height'] = stats.norm(loc=24,scale=5)
    dists['arrival'] = stats.norm(loc=15,scale=10)
    dists['PGA'] = stats.truncnorm(loc=1.0,scale=0.3,a=0.04,b=1.4) # center it around one.
    #dists['MMI'] = stats.truncnorm(loc=7,scale=1.5,a=4,b=11) # should probably be disabled.
    gauge = Gauge(name,dists)
    gauge.lat = -5.565079
    gauge.lon = 120.192826
    gauge.V_S30 = 260 # https://usgs.maps.arcgis.com/apps/webappviewer/index.html?id=8ac19bc334f747e486550f32837578e1
    gauges.append(gauge)

    # Sumenep
    name = 'Sumenep'
    dists = dict()
    dists['height'] = stats.truncnorm(loc=1.5,scale=1,a=-1,b=4)
    dists['arrival'] = stats.norm(loc=240,scale=45)
    gauge = Gauge(name,dists)
    gauge.lat = -7.049969
    gauge.lon = 113.908203
    gauges.append(gauge)

    # Nipa-Nipa
    name = 'Nipa-Nipa'
    dists = dict()
    dists['height'] = stats.truncnorm(loc=3,scale=2,a=-1,b=4)
    #dists['inundation'] = stats.norm(135,20)
    gauge = Gauge(name,dists)
    gauge.lat = -5.567525
    gauge.lon = 120.011503
    gauges.append(gauge)

    # Bima
    name = 'Bima'
    dists = dict()
    dists['height'] = stats.truncnorm(loc=10,scale=4,a=-2,b=4)
    gauge = Gauge(name,dists)
    gauge.lat =  -8.443485 # Note: this is within the town
    gauge.lon = 118.716085
    gauges.append(gauge)

    # Makassar
    # Actually, there is no recording of an event in Makassar.
    # Rather than treat this as evidence of lack of tsunami, we
    # omit it entirely.

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
