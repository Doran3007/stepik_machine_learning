import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score


# songs = pd.read_csv('module_2/song/songs.csv')

# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)

# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)
# predictions = clf.predict(X_test)
# clf.precision_score(y_test, predictions, average)