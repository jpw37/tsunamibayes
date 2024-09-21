from fault import *

df = pd.read_csv('flores_sample.csv')

latFlores = df.iloc[:,8]    #We slice here the important information: lat/lon/strike coordinates. We don't really use the other columns according to my knowledge
lonFlores = df.iloc[:,7]
strikeFlores = df.iloc[:,9]

#From what I understand, these slices are not numpy Arrays, they still contain the column title/index.
latFlores = latFlores.values   #We slice here the important information: lat/lon/strike coordinates. We don't really use the other columns according to my knowledge
lonFlores = lonFlores.values
strikeFlores = strikeFlores.values

minlatF = latFlores.min()
maxlatF = latFlores.max()
minlonF = lonFlores.min()
maxlonF = lonFlores.max()

boundsFlores = {'lon_min' : minlonF,'lon_max' : maxlonF, 'lat_min' : minlatF, 'lat_max':maxlatF}

depthcurve = lambda depth : 25*1000
dipcurve = lambda dip : 25

model_bounds=None

floresFault = ReferenceCurveFault(latFlores,lonFlores,strikeFlores,depthcurve,dipcurve,boundsFlores,model_bounds)
distance, index = floresFault.distance(128.657, -3.576, True)
print(distance)
print(index)

width = 101765.8602 / 111111    #width in meters approximated to degrees
strike = 115.0212526
fault_point_lon, fault_point_lat = floresFault.lonpts[index], floresFault.latpts[index]
print(fault_point_lon, fault_point_lat)
shift_lat = width / 2 * np.cos(strike + 90)
shift_lon = width / 2 * np.sin(strike + 90)
print(shift_lon, shift_lat)
shifted_lat = shift_lat + fault_point_lat
shifted_lon = shift_lon + fault_point_lon
print(shifted_lon, shifted_lat)
distance = haversine(fault_point_lat, fault_point_lon, shifted_lat, shifted_lon)
print(101765.8602 - distance*2)