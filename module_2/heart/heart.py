import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score

hearts = pd.read_csv('module_2/heart/train_data_tree.csv')

X = hearts.drop(['num'], axis=1)
y = hearts['num']

clf = tree.DecisionTreeClassifier(criterion='entropy')

clf.fit(X,y)

tree.plot_tree(clf, feature_names=X.columns)
plt.show()