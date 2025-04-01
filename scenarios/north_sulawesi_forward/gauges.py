from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.
    
    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Bontang
    name = 'Bontang'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = 0.1068972
    gauge.lon = 117.5809318
    gauge.loc = 3
    gauge.scale = 0.8
    gauges.append(gauge)

    # Tandjoengbatoe
    name = 'Tandjoengbatoe'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8,scale=0.4)
    gauge = Gauge(name,dists)
    gauge.lat = 2.2686718
    gauge.lon = 118.1075566
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)


    # Tarakan
    name = 'Tarakan'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 3.2342551
    gauge.lon = 117.5669208
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tawau
    name = 'Tawau'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 4.2204775
    gauge.lon = 117.8909199
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Semporna
    name = 'Semporna'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 4.5194959
    gauge.lon = 118.6169157
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Bongao Island
    name = 'Bongao Island'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 5.0063209
    gauge.lon = 119.7659900
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Zamboanga
    name = 'Zamboanga'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 6.8588280
    gauge.lon = 122.0952185
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Pagadian
    name = 'Pagadian'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 7.7526962
    gauge.lon = 123.4887627
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # General Santos
    name = 'General Santos'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 6.0926422
    gauge.lon = 125.1775301
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Manado
    name = 'Manado'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 1.4847164
    gauge.lon = 124.8161636
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tahuna
    name = 'Tahuna'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 3.5933601
    gauge.lon = 125.4635514
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Boroko
    name = 'Boroko'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 0.9307962
    gauge.lon = 123.2932107
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Genlum
    name = 'Genlum'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 0.9444927
    gauge.lon = 123.0304609
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Boeol
    name = 'Boeol'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 1.1686772
    gauge.lon = 121.4511943
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tolotoli
    name = 'Tolotoli'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 0.9743408
    gauge.lon = 120.6464167
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Palu
    name = 'Palu'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = -0.8623012
    gauge.lon = 119.8639246
    gauge.loc = 1.8
    gauge.scale = 0.4
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
