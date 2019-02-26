import pandas as pd

df = pd.read_csv('spam.csv', encoding = 'UTF-8')
print(type(df))
print(df.head())