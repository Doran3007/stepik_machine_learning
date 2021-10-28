import pandas as pd
import numpy as np
df = pd.read_csv('StudentsPerformance.csv', header=0)
# print(df.head())
# print(df.describe())
#
# print('1')
# print(df.head(7))
#
# print('2')
# print(df.iloc[0:7])
#
# print('3')
# print(df.loc[:6])
#
# print('4')
# print(df.tail(7))
#
# print('5')
# print(df.loc[:7])
#
# print('6')
# print(df.iloc[:7])

titanic = pd.read_csv('titanic.csv', header=0)
print(titanic)
print(titanic.head())
print(titanic.describe())
print(titanic.dtypes)
print(titanic.columns)
print(titanic.index)
print(titanic.size)