import concurrent.futures
import dill
import pandas as pd
import forward
import random
import numpy as np
from gauge import Gauge
import scipy.stats as stats
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


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


path = r"C:\Users\ashle\Documents\Whitehead Research\Research 2023\1852\etopo.tt3"
gauge_obj_lst = build_gauges()

parameters_file_path = r'C:\Users\ashle\Documents\Whitehead Research\all_output_files\56292268_m8_output\model_params.csv'
sample_df = pd.read_csv(parameters_file_path)
sample_df = sample_df.drop(columns=['Unnamed: 0'])

result_file_path = r'C:\Users\ashle\Documents\Whitehead Research\all_output_files\56292268_m8_output\model_output.csv'
result_df = pd.read_csv(result_file_path)
result_df = result_df.drop(columns=['Unnamed: 0'])


# set a number of times to sample
number_of_samples = 3
# randomly choose n samples from the dataframe between start_index and end_index
start_index = 4000
end_index = 9580

random_idx = [random.randint(start_index, end_index) for i in range(number_of_samples)]

# make a list of all possible Lref values
# 0-5 every .1
# 5 to 50 every 1
# 50 to 500 every 10
# don't include zero
density = 50
small_lst = np.linspace(600, 5000, density, endpoint=True)   # np.linspace(0.1, 5, density, endpoint=True)
middle_lst = np.linspace(5000, 50000, density, endpoint=True)  # np.linspace(6, 50, density, endpoint=True)
big_lst = np.linspace(60000, 500000, density, endpoint=True)   # np.linspace(60, 500, density, endpoint=True)
# concatenate the lists
lref_lst = np.concatenate([small_lst, middle_lst, big_lst])

best_lref = 0
best_error = float('inf')


def compute_avg_error_for_lref(lref, random_idx, sample_df, gauge_obj_lst, path, result_df):
    def compute_error(row_index):
        landslide_params = sample_df.iloc[row_index].to_dict()
        result_row = result_df.iloc[row_index]

        toy = forward.ToyForwardModel(gauge_obj_lst, None, path)
        toy_results = toy.run(landslide_params, lref=lref)

        results_diff = toy_results - result_row
        squared_error = results_diff.pow(2).sum()
        return squared_error

    errors = [compute_error(row_index) for row_index in random_idx]
    return np.mean(errors)


with concurrent.futures.ProcessPoolExecutor() as executor:
    lref_error = list(executor.map(
        lambda lref: compute_avg_error_for_lref(lref, random_idx, sample_df, gauge_obj_lst, path, result_df),
        list(lref_lst),  # Convert dict_keys to a list
    ))

# make a plot with y = average difference and Lref on the x axis
plt.figure(figsize=(10, 5))
plt.plot(lref_lst, lref_error)
plt.xscale('log')
plt.xlabel('Lref')
plt.ylabel('Average Error')
plt.title('Optimizing Toy Height Model')
plt.show()
