import pandas as pd

df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
print(type(df))
print(df.head())
print()
print()
print()