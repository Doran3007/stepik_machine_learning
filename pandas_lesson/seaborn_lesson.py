import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

student_perf = pd.read_csv('StudentsPerformance.csv', header=0)
income = pd.read_csv('income.csv', header=0)
dataset = pd.read_csv('dataset_209770_6.txt', header=0, sep=" ")
genome = pd.read_csv('genome_matrix.csv', header=0, index_col=0)
dota_hero = pd.read_csv('dota_hero_stats.csv', header=0)
flower = pd.read_csv('iris.csv', header=0, index_col=0)
# =========task_1

# result = student_perf['math score'].hist()
# print(result)
# plt.show()

# =========task_2
# student_perf.plot.scatter(x = 'math score', y = 'reading score')
# plt.show()

# =========task_3
# hist_2 = sns.lmplot(x = 'math score', y = 'reading score', data = student_perf, hue='gender', fit_reg=False)
# hist_2.set_xlabels('math_score')
# hist_2.set_ylabels('reading_score')

# plt.show()

# =========task_4
# income.plot(kind='line')
# plt.show()

# income.plot()
# plt.show()

# income.income.plot()
# plt.show()

# sns.lineplot(x=income.index, y=income.income)
# plt.show()

# plt.plot(income.index, income.income)
# plt.show()

# income['income'].plot()
# plt.show()

# sns.lineplot(data=income)
# plt.show()

# =============task_5
# print(dataset)

# dataset.plot.scatter(x = 'x', y = 'y')
# plt.show()

# =============task_6

# g = sns.heatmap(data = genome)
# g.xaxis.set_ticks_position('top')
# g.xaxis.set_tick_params(rotation=90)
# plt.show()

# =============task_7

# dota_hero['cnt'] = dota_hero.roles.str.count(',')+1
# group_hero = dota_hero.groupby('cnt').aggregate({'localized_name': 'nunique'})
# print(group_hero)
# sns.lineplot(x='cnt', y = 'localized_name', data = group_hero)
# plt.show()

# =============task_8
# sns.distplot(flower['sepal length'], color='green',label="sepal length")
# sns.distplot(flower['sepal width'], color='blue',label="sepal width")
# sns.distplot(flower['petal length'], color='red',label="petal length")
# sns.distplot(flower['petal width'], color='yellow',label="petal width")
# plt.legend(labels=['sepal length', 'sepal width', 'petal length', 'petal width'])
# plt.show()

# =============task_9
# sns.violinplot(y='petal length', data=flower)
# plt.show()

# =============task_10

sns.pairplot(flower, hue="species")
plt.show()
