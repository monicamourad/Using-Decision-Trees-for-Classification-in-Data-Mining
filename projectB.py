import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import stats

DataSet = pd.read_csv("training_data.csv")
print(DataSet)

# clean DataSet
# remove irrelevant attributes, we will use original attributes
del DataSet['date']
del DataSet['Temperature pprox.']
del DataSet['Humidity approx.']
del DataSet['Light approx.']
del DataSet['CO2 aprrox.']
del DataSet['HumidityRatio x 1000000 approx.']
del DataSet['Occupancy']
print(DataSet)

# mean
print('\nmean:')
print(DataSet.mean())

# median
print('\nmedian:')
print(DataSet.median())

# mode
print('\nmode:')
print(DataSet.mode())

# skewness
print('\nskewness:')
print(DataSet.skew())

DataSet.hist(bins=100)
plt.xlim([0, 115000])
plt.show()

#   box plot
DataSet.boxplot(column=['Temperature'], grid=False)
plt.show()
DataSet.boxplot(column=['Humidity'], grid=False)
plt.show()
DataSet.boxplot(column=['Light'], grid=False)
plt.show()
DataSet.boxplot(column=['CO2'], grid=False)
plt.show()
DataSet.boxplot(column=['HumidityRatio '], grid=False)
plt.show()

#   Range
print('\nrange:')
print(DataSet.max() - DataSet.min())

#   IQR
print('\ninter quartile range:')
Q1 = DataSet.quantile(0.25)
Q3 = DataSet.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# variance
print('\nvariance:')
print(DataSet.var())

# standard deviation
print('\nstandard deviation:')
print(DataSet.std())

#   removing outliers
print('\nremoving outliers using z-score:')
z_scores = stats.zscore(DataSet)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_DataSet = DataSet[filtered_entries]
new_DataSet.to_csv('NewDataSet.csv')

