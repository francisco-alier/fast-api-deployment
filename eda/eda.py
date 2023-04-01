import pandas as pd

df = pd.read_csv("../data/census.csv")

#printing some basic dtatistics
print(df.head())
print(df.isnull().sum())