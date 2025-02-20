import pandas as pd
import numpy as np

def convert_lat_lon(lat, lon):
    """Converts latitude and longitude from strings to decimals.
    Proper string form: '8-17-43.21S', '118-43-42.00E' etc.
    """
    latitude = sum(float(x) / 60 ** n for n, x in enumerate(lat[:-1].split('-')))  * (1 if 'N' == lat[-1] else -1)
    longitude = sum(float(x) / 60 ** n for n, x in enumerate(lon[:-1].split('-'))) * (1 if 'E' == lon[-1] else -1)
    return latitude, longitude

def load_data():
    #pass in data as arrays

    #small referes to 5.0-5.9 magnitude
    flores_small = pd.read_excel('./1820_fault_data/flores_trench5.0-5.9_filtered.xlsx')
    flores_lat_small = pd.DataFrame(flores_small, columns = ['lat(deg S)']).to_numpy().flatten()
    flores_long_small = pd.DataFrame(flores_small, columns = ['log (deg E)']).to_numpy().flatten()
    flores_mag_small = pd.DataFrame(flores_small, columns = ['Magnitude']).to_numpy().flatten()
    flores_depth_small = pd.DataFrame(flores_small, columns = ['Depth km']).to_numpy().flatten()
    flores_plate_boundary_small = pd.DataFrame(flores_small, columns = ['Distance from plate boundary km (estimated)']).to_numpy().flatten()
    flores_dist_ratio_small = pd.DataFrame(flores_small, columns = ['Calculated distance ratio (depth/TAN(Dip))']).to_numpy().flatten()
    flores_strike_small = pd.DataFrame(flores_small, columns = ['Stike']).to_numpy().flatten()
    flores_dip_small = pd.DataFrame(flores_small, columns = ['Dip']).to_numpy().flatten()
    flores_rake_small = pd.DataFrame(flores_small, columns = ['Rake']).to_numpy().flatten()

    #large refers to 6.0-7.2 magnitude
    flores_large = pd.read_excel('./1820_fault_data/flores-trench6-7.2_filtered.xlsx')
    flores_lat_large = pd.DataFrame(flores_large, columns = ['Lat (S & degree):']).to_numpy().flatten()
    flores_long_large = pd.DataFrame(flores_large, columns = ['Long (E & degree):']).to_numpy().flatten()
    flores_mag_large = pd.DataFrame(flores_large, columns = ['Magnitude']).to_numpy().flatten()
    flores_depth_large = pd.DataFrame(flores_large, columns = ['Depth km']).to_numpy().flatten()
    flores_plate_boundary_large = pd.DataFrame(flores_large, columns = ['Distance from plate boundary km (estimated)']).to_numpy().flatten()
    flores_dist_ratio_large = pd.DataFrame(flores_large, columns = ['Calculated distance ratio (depth/TAN(Dip))']).to_numpy().flatten()
    flores_strike_large = pd.DataFrame(flores_large, columns = ['Stike']).to_numpy().flatten()
    flores_dip_large = pd.DataFrame(flores_large, columns = ['Dip']).to_numpy().flatten()
    flores_rake_large = pd.DataFrame(flores_large, columns = ['Rake']).to_numpy().flatten()


    dataflores_large = [] #a list will be returned of lat and long
    for i in range(len(flores_lat_large)):
        dataflores_large.append(convert_lat_lon(flores_lat_large[i], flores_long_large[i]))

    dataflores_small = [] #a list will be returned of lat and long
    for i in range(len(flores_lat_small)):
        dataflores_small.append(convert_lat_lon(flores_lat_small[i], flores_long_small[i]))

    a_dataflores_large = np.array(dataflores_large) #convert list to array
    a_dataflores_lat_large = a_dataflores_large[:,0]
    a_dataflores_long_large = a_dataflores_large[:,1]

    a_dataflores_small = np.array(dataflores_small) #convert list to array
    a_dataflores_lat_small = a_dataflores_small[:,0]
    a_dataflores_long_small = a_dataflores_small[:,1]
    
    lats = np.append(a_dataflores_lat_large, a_dataflores_lat_small)
    lons = np.append(a_dataflores_long_large, a_dataflores_long_small)
    depths = np.append(flores_depth_large, flores_depth_small)
    dips = np.append(flores_dip_large, flores_dip_small)
    strikes = np.append(flores_strike_large, flores_strike_small)
    rakes = np.append(flores_rake_large, flores_rake_small)
    
    to_keep = depths < 35 # Only keep the data points where the depth is less than 35 km. We don't trust any other points.
    
    return [arr[to_keep] for arr in [lats, lons, depths, dips, strikes, rakes]]
    
