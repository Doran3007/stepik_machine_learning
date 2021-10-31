import pandas as pd
import numpy as np
df = pd.read_csv('pandas_lesson/StudentsPerformance.csv', header=0)
titanic = pd.read_csv('pandas_lesson/titanic.csv', header=0)
dota_hero = pd.read_csv('pandas_lesson/dota_hero_stats.csv', header=0)
concentrations = pd.read_csv('pandas_lesson/algae.csv', header=0)


# ========Task_1
# print(df.head())
# print(df.describe())

# ========Task_2
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

# =======Task_3
# print(titanic)
# print(titanic.head())
# print(titanic.describe())
# print(titanic.dtypes)
# print(titanic.columns)
# print(titanic.index)
# print(titanic.size)

# ===========Task_4

# query_1 = df[df['lunch'] == 'free/reduced']
# arg_1 = query_1['lunch'].count()
# arg_2 = df['lunch'].count()
# finish_1 = arg_1/arg_2
# print(query_1)
# print(arg_1)
# print(arg_2)
# print(finish_1)

# =============Task_5
# query_1 = df[df['lunch'] == 'free/reduced']

# query_2 = df[df['lunch'] == 'standard']

# arg_1 = query_1['math score'].mean()
# arg_2 = query_1['math score'].var()

# arg_3 = query_2['math score'].mean()
# arg_4 = query_2['math score'].var()

# arg_5 = query_1['reading score'].mean()
# arg_6 = query_1['reading score'].var()

# arg_7 = query_2['reading score'].mean()
# arg_8 = query_2['reading score'].var()

# arg_9 = query_1['writing score'].mean()
# arg_10 = query_1['writing score'].var()

# arg_11 = query_2['writing score'].mean()
# arg_12 = query_2['writing score'].var()

# print('free/reduced')
# print(f'mean of math score: {arg_1}')
# print(f'var of math score: {arg_2}')
# print(f'mean of reading score: {arg_5}')
# print(f'var of reading score: {arg_6}')
# print(f'mean of writing score: {arg_9}')
# print(f'var of writing score: {arg_10}')

# print('standard')
# print(f'mean of math score: {arg_3}')
# print(f'var of math score: {arg_4}')
# print(f'mean of reading score: {arg_7}')
# print(f'var of reading score: {arg_8}')
# print(f'mean of writing score: {arg_11}')
# print(f'var of writing score: {arg_12}')

# =============task_6

# legs = dota_hero.groupby(['attack_type', 'primary_attr']).aggregate({'name': 'nunique'})
# print(legs)

# accountancy = pd.read_csv('accountancy.csv', header=0)

# acc = accountancy.groupby(['Executor', 'Type']).aggregate({'Salary': 'mean'})
# print(acc)

# ==========Task_7
concentrations = pd.read_csv('algae.csv', header=0)

# mean_concentrations = np.around(concentrations.query("genus == 'Fucus'").alanin.describe().loc[['min', 'mean', 'max']].values, decimals=2)
# mean_concentrations = concentrations.groupby('group')
# print(mean_concentrations)
vesiculosus = ['brown', 'green', 'red']
_var = concentrations.groupby('group').agg({'citrate': 'var'})
_var2 = concentrations.groupby('group').agg({'species': 'nunique'})
_var3 = concentrations.groupby('group').agg({'sucrose': 'std'})

print(_var)
print(_var2)
print(_var3)