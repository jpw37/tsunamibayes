from tsunamibayes import Gauge, dump_gauges
import scipy.stats as stats

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.
    
    Returns
    -------
    gauges : (list) of Gauge objects
    """
    gauges = list()

    # Balikpapan
    name = 'Balikpapan'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = -1.2968463 #[-1.2968463, -1.2902488, -1.2772992]
    gauge.lon = 116.8047846 #[116.8047846, 116.8590154, 116.9160517]
    gauge.loc = 3
    gauge.scale = 0.8
    gauges.append(gauge)

    # Bontang
    name = 'Bontang'
    dists = dict()
    dists['height'] = stats.norm(loc=3,scale=0.8)
    gauge = Gauge(name,dists)
    gauge.lat = 0.0936413 #[0.0936413, 0.0885611, 0.1068972]
    gauge.lon = 117.5044100 #[117.5044100, 117.5261494, 117.5809318]
    gauge.loc = 3
    gauge.scale = 0.8
    gauges.append(gauge)

    # Tandjoengbatoe
    name = 'Tandjoengbatoe'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8,scale=0.4)
    gauge = Gauge(name,dists)
    gauge.lat = 2.2671474 #[2.2671474, 2.2723291, 2.2803883]
    gauge.lon = 118.0996852 #[118.0996852, 118.1060936, 118.1070125]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tarakan
    name = 'Tarakan'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 3.3010757 #[3.3010757, 3.2844641, 3.2685585]
    gauge.lon = 117.5575058 #[117.5575058, 117.5739720, 117.5966022]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tawau
    name = 'Tawau'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 4.2606734 #[4.2606734, 4.2309717, 4.2315381]
    gauge.lon = 117.9005218 #[117.9005218, 117.8794758, 117.9244083]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Semporna
    name = 'Semporna'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 4.5058546 #[4.5058546, 4.4942023, 4.4858872]
    gauge.lon = 118.6074533 #[118.6074533, 118.6122207, 118.6162575]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Bongao Island
    name = 'Bongao Island'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 5.0082928 #[5.0082928, 5.0163516, 5.0260919]
    gauge.lon = 119.7686109 #[119.7686109, 119.7776486, 119.7823453]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Zamboanga
    name = 'Zamboanga'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 6.9015741 #[6.9015741, 6.8814426, 6.8746361]
    gauge.lon = 122.0322138 #[122.0322138, 122.0784756, 122.1042892]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Pagadian
    name = 'Pagadian'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 7.7815198 #[7.7815198, 7.7954067, 7.8057244]
    gauge.lon = 123.4436997 #[123.4436997, 123.4489195, 123.4682773]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # General Santos
    name = 'General Santos'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 6.0884893 #[6.0884893, 6.0960155, 6.0961621]
    gauge.lon = 125.1643202 #[125.1643202, 125.1736935, 125.1865727]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tahuna
    name = 'Tahuna'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 3.6045022 #[3.6045022, 3.6003375, 3.6035812]
    gauge.lon = 125.4702396 #[125.4702396, 125.4773798, 125.4877711]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Manado
    name = 'Manado'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 1.5140765 #[1.5140765, 1.4928330, 1.4682564]
    gauge.lon = 124.8366719 #[124.8366719, 124.8271245, 124.8163795]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Boroko
    name = 'Boroko'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 0.9261088 #[0.9261088, 0.9190513, 0.9177993]
    gauge.lon = 123.2759763 #[123.2759763, 123.2892930, 123.3082661]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Genlum
    name = 'Genluma'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 0.9347793 #[0.9347793, 0.9342088, 0.9360681]
    gauge.lon = 123.0321952 #[123.0321952, 123.0376047, 123.0399715]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Boeol
    name = 'Boeol'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 1.1797719 #[1.1797719, 1.1718516, 1.1660355]
    gauge.lon = 121.4355451 #[121.4355451, 121.4426615, 121.4496740]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Tolotoli
    name = 'Tolotoli'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = 1.0283203 #[1.0283203, 1.0365876, 1.0455639]
    gauge.lon = 120.7673113 #[120.7673113, 120.7894084, 120.7984559]
    gauge.loc = 1.8
    gauge.scale = 0.4
    gauges.append(gauge)

    # Palu
    name = 'Palu'
    dists = dict()
    dists['height'] = stats.norm(loc=1.8, scale=0.4)
    gauge = Gauge(name, dists)
    gauge.lat = -0.8477276 #[-0.8477276, -0.8707821, -0.8275325]
    gauge.lon = 119.8336537 #[119.8336537, 119.8548193, 119.8650675]
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
