import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score

iris = pd.read_csv('module_2/iris/train_iris.csv')
iris_test = pd.read_csv('module_2/iris/test_iris.csv')

iris = iris.drop(['Unnamed: 0'], axis=1)
iris_test = iris_test.drop(['Unnamed: 0'], axis=1)

# drop columns with many nan values and unnecessary information for create 2 parameter, X=features, y=predicted value
X_train = iris.drop(['species'], axis=1)
y_train = iris.species
X_test = iris_test.drop(['species'], axis=1)
y_test= iris_test.species
rs = np.random.seed(0)
scores_data = pd.DataFrame()
max_depth_values = range(1, 100)
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=rs)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    temp_score = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score]})
    scores_data = scores_data.append(temp_score)
# we change the structure of the resulting data frame with model estimates, that there was one column with an estimate\
#  and columns with a type of estimate
data_scores_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score'], 
    var_name='set_types', value_name='score')
sns.lineplot(x='max_depth', y='score', hue='set_types', data=data_scores_long)
plt.show()