import pandas as pd

csv = pd.read_csv("samples.csv")
print("Length of csv", len(csv))

index = csv.index
print(index)

print(csv.loc[len(csv)-1])
