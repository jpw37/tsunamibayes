import forward
from gauge import Gauge
import scipy.stats as stats

path = r"C:\Users\ashle\Documents\Whitehead Research\Research 2023\1852\etopo.tt3"
# coordinates = [
#     (129.775, -4.5175),
#     # (128.178, -3.691),
#     (129.8965, -4.5248),
#     (127.113, -3.3815),
#     (128.557, -3.543),
#     (128.657, -3.576),
#     (128.562, -3.501),
#     (128.807, -3.6455),
#     (128.921, -3.338)
# ]
# from scipy.stats import norm
#
# # Create some example frozen_rv objects (replace these with your actual distributions)
# height_distribution = norm(loc=5, scale=2)
# arrival_distribution = norm(loc=10, scale=3)
# inundation_distribution = norm(loc=3, scale=1)
#
# # Create the dictionary
# observations_dict = [
#     # {'height': height_distribution},
#     {'arrival': arrival_distribution},
#     {'inundation': inundation_distribution},
#     {'height': height_distribution},
#     {'arrival': arrival_distribution},
#     {'inundation': inundation_distribution},
#     {'height': height_distribution},
#     {'arrival': arrival_distribution},
#     {'inundation': inundation_distribution}
# ]
# names = ['name1', 'name2', 'name3', 'name4', 'name5', 'name6', 'name7', 'name8']
# zipped_list = zip(names, observations_dict, coordinates)
# gauge_obj_lst = [Gauge(name, obs, lat=coord[0], lon=coord[1]) for name, obs, coord in zipped_list]
earthquake_params = {
    'slip': 10.175,
    'length': 512087,
    'width': 145077,
    'rake': 90,
    'dip_offset': 10.993,
    'latitude': -5.522627782,
    'longitude': 131.7545311,
}

landslide_params = {'center_mass_depth': 2800,  # meters
    'thickness': 43.00891068,  # meters
    'width': 8,  # kilometers
    'length': 1,  # meters
    'initial_velocity': 92.50870467,  # meters per second
    'volume': 22167043776,
    'aspect_ratio': 0.39666532,
    'latitude': -6.2,
    'longitude': 130}


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
    dists['height'] = stats.norm(loc=3, scale=0.8)
    gauge = Gauge(name, dists)
    gauge.lat = -4.5175
    gauge.lon = 129.775
    gauges.append(gauge)

    # # Ambon
    # name = 'Ambon'
    # dists = dict()
    # dists['height'] = stats.norm(loc=1.8, scale=0.4)
    # gauge = Gauge(name, dists)
    # gauge.lat = -3.691
    # gauge.lon = 128.178
    # gauges.append(gauge)

    # Banda Neira
    name = 'Banda Neira'
    dists = dict()
    dists['arrival'] = stats.skewnorm(a=2, loc=15, scale=5)
    dists['height'] = stats.norm(loc=6.5, scale=1.5)
    dists['inundation'] = stats.norm(loc=185, scale=65)
    gauge = Gauge(name, dists)
    gauge.lat = -4.5248
    gauge.lon = 129.8965
    gauge.beta = 4.253277987952933
    gauge.n = 0.06
    gauges.append(gauge)

    # Buru
    name = 'Buru'
    dists = dict()
    dists['height'] = stats.chi(df=1.01, loc=0.5, scale=1.5)
    gauge = Gauge(name, dists)
    gauge.lat = -3.3815
    gauge.lon = 127.113
    gauges.append(gauge)

    # Hulaliu
    name = 'Hulaliu'
    dists = dict()
    dists['height'] = stats.chi(df=1.01, loc=0.5, scale=2.0)
    gauge = Gauge(name, dists)
    gauge.lat = -3.543
    gauge.lon = 128.557
    gauges.append(gauge)

    # Saparua
    name = 'Saparua'
    dists = dict()
    dists['arrival'] = stats.norm(loc=45, scale=5)
    dists['height'] = stats.norm(loc=5, scale=1)
    dists['inundation'] = stats.norm(loc=125, scale=40)
    gauge = Gauge(name, dists)
    gauge.lat = -3.576
    gauge.lon = 128.657
    gauge.beta = 1.1067189507222546
    gauge.n = 0.06
    gauges.append(gauge)

    # Kulur
    name = 'Kulur'
    dists = dict()
    dists['height'] = stats.norm(loc=3, scale=1)
    gauge = Gauge(name, dists)
    gauge.lat = -3.501
    gauge.lon = 128.562
    gauges.append(gauge)

    # Ameth
    name = 'Ameth'
    dists = dict()
    dists['height'] = stats.norm(loc=3, scale=1)
    gauge = Gauge(name, dists)
    gauge.lat = -3.6455
    gauge.lon = 128.807
    gauges.append(gauge)

    # Amahai
    name = 'Amahai'
    dists = dict()
    dists['height'] = stats.norm(loc=3.5, scale=1)
    gauge = Gauge(name, dists)
    gauge.lat = -3.338
    gauge.lon = 128.921
    gauges.append(gauge)
    # print(gauges)
    return gauges


gauge_obj_lst = build_gauges()
toy = forward.ToyForwardModel(gauge_obj_lst, None, path)

print(toy.run(earthquake_params))


# Now, observations_dict is a dictionary with keys representing types of observation
# and values representing the corresponding frozen_rv objects.

