import pandas as pd


df = pd.DataFrame(
              {"Strike" : [4 ,5, 6],
               "Length" : [7, 8, 9],
               "Width" : [10, 11, 12],
               "Depth" : [10, 11, 12],
               "Slip" : [10, 11, 12],
               "Rake" : [10, 11, 12],
               "Dip" : [10, 11, 12],
               "Longitude" : [10, 11, 12],
               "Latitude" : [10, 11, 12],
               "Log Probability" : [10, 11, 12],
               "Accepts" : [10, 11, 12]})

df.to_csv('births1880.csv',index=False,header=True)

df = pd.read_csv('births1880.csv')

e_params = ['Strike', 'Length', 'Width', "Depth", "Slip", "Rake", "Longitude", "Latitude"]
print(df.get(e_params).tail(1))

print(df)


df.loc[len(df)] = [1,2,3,4,5,6,7,8,9,10]

df.to_csv('births1880.csv',index=False,header=True)

df = pd.read_csv('births1880.csv')

print(df)

print(df['Strike'].tail(1) + df['Strike'].tail(1))


