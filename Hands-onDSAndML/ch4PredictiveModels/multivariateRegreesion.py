import pandas as pd
import xlrd
import statsmodels.api as sm

# read the file
df = pd.read_excel("http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls")
# convert text type to integer, because algorithm only support integer
df['Model_ord'] = pd.Categorical(df.Model).codes
# choose the variates
X = df[['Mileage', 'Model_ord', 'Doors']]
Y = df[['Price']]

X1 = sm.add_constant(X)
est = sm.OLS(Y, X1).fit()

print(est.summary())