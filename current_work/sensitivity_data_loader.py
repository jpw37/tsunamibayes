import pandas as pd
import numpy as np

def convert_lat_lon(lat, lon):
    """Converts latitude and longitude from strings to floats.
    Proper string form: '8-17-43.21S', '118-43-42.00E' etc.
    """
    latitude = sum(float(x) / 60 ** n for n, x in enumerate(lat[:-1].split('-')))  * (1 if 'N' == lat[-1] else -1)
    longitude = sum(float(x) / 60 ** n for n, x in enumerate(lon[:-1].split('-'))) * (1 if 'E' == lon[-1] else -1)
    return latitude, longitude

def load_data():
    #pass in data as arrays

    #small referes to 5.0-5.9 magnitude
    small = pd.read_excel('Sumatra version 3.xlsx')
    lat_small = pd.DataFrame(small, columns = ['Lat N']).to_numpy().flatten()
    long_small = pd.DataFrame(small, columns = ['Long E']).to_numpy().flatten()
    mag_small = pd.DataFrame(small, columns = ['Magnitude']).to_numpy().flatten()
    depth_small = pd.DataFrame(small, columns = ['Depth km']).to_numpy().flatten()
    plate_boundary_small = pd.DataFrame(small, columns = ['Distance from plate boundary km (estimated)']).to_numpy().flatten()
    dist_ratio_small = pd.DataFrame(small, columns = ['Calculated distance ratio (depth/TAN(Dip))']).to_numpy().flatten()
    strike_small = pd.DataFrame(small, columns = ['Stike']).to_numpy().flatten()
    dip_small = pd.DataFrame(small, columns = ['Dip']).to_numpy().flatten()
    rake_small = pd.DataFrame(small, columns = ['Rake']).to_numpy().flatten()

    #large refers to 6.0-7.2 magnitude
    large = pd.read_excel('Sumatra version 3.xlsx')
    lat_large = pd.DataFrame(large, columns = ['Lat N']).to_numpy().flatten()
    long_large = pd.DataFrame(large, columns = ['Long E']).to_numpy().flatten()
    mag_large = pd.DataFrame(large, columns = ['Magnitude']).to_numpy().flatten()
    depth_large = pd.DataFrame(large, columns = ['Depth km']).to_numpy().flatten()
    plate_boundary_large = pd.DataFrame(large, columns = ['Distance from plate boundary km (estimated)']).to_numpy().flatten()
    dist_ratio_large = pd.DataFrame(large, columns = ['Calculated distance ratio (depth/TAN(Dip))']).to_numpy().flatten()
    strike_large = pd.DataFrame(large, columns = ['Stike']).to_numpy().flatten()
    dip_large = pd.DataFrame(large, columns = ['Dip']).to_numpy().flatten()
    rake_large = pd.DataFrame(large, columns = ['Rake']).to_numpy().flatten()


    datalarge = [] #a list will be returned of lat and long
    for i in range(len(lat_large)):
        if isinstance(lat_large[i], float) and isinstance(long_large[i], float):
            datalarge.append((lat_large[i], long_large[i]))            
        else:
            datalarge.append(convert_lat_lon(lat_large[i], long_large[i]))


    datasmall = [] #a list will be returned of lat and long
    for i in range(len(lat_small)):
        if isinstance(lat_small[i], float) and isinstance(long_small[i], float):
            datasmall.append((lat_small[i], long_small[i]))            
        else:
            datasmall.append(convert_lat_lon(lat_small[i], long_small[i]))

    a_datalarge = np.array(datalarge) #convert list to array
    a_data_lat_large = a_datalarge[:,0]
    a_data_long_large = a_datalarge[:,1]

    a_datasmall = np.array(datasmall) #convert list to array
    a_datalat_small = a_datasmall[:,0]
    a_datalong_small = a_datasmall[:,1]
    
    lats = np.append(a_data_lat_large, a_datalat_small)
    lons = np.append(a_data_long_large, a_datalong_small)
    depths = np.append(depth_large, depth_small)
    dips = np.append(dip_large, dip_small)
    strikes = np.append(strike_large, strike_small)
    rakes = np.append(rake_large, rake_small)
    
    to_keep = depths < 35 # Only keep the data points where the depth is less than 35 km. We don't trust any other points.
    
    return [arr[to_keep] for arr in [lats, lons, depths, dips, strikes, rakes]]
    
