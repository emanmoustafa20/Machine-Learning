# this model explains the SalePrice(ys) of a house depending on the LotFrontage(xs) measurement.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


url = 'https://raw.githubusercontent.com/rodmsmendes/reinforcementlearning4fun/master/data/house_prices.csv'
df = pd.read_csv(url)
#Here i select only the attributes with float64 data type and their statistics
df_float = df.select_dtypes(include=['float64']).copy()
#df_float.info()

#use fillna to fill the non values
df_float['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=False)

xs = df_float[['LotFrontage']]
ys = df['SalePrice']
"""
Here i create an instance of the LinearRegression class from scikit-learn  and call the fit function which accepts 
the predictor and predicted values
"""
lr1 = LinearRegression()
lr1.fit(xs, ys)

#print the model coefficients
print(lr1.coef_)
print(lr1.intercept_)
print(mean_squared_error(ys, lr1.predict(xs)))

#plot the results 
plt.scatter(xs,ys)
plt.plot(xs, lr1.predict(xs), linewidth=5.0, color='orange')
plt.show()