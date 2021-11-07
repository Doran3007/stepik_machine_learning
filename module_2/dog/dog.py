import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score

dog = pd.read_csv('module_2/dog/dogs_n_cats.csv')
dog_pred = pd.read_json('module_2/dog/dataset_209691_15.txt')
# drop columns with many nan values and unnecessary information for create 2 parameter, X=features, y=predicted value
X = dog.drop(['Вид'], axis=1)
y = dog.Вид
# X_pred = dog_pred.drop(['species'], axis=1)
# y_pred= dog_pred.species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


scores_data = pd.DataFrame()
max_depth_values = range(1, 100)
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_score = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'mean_cross_val_score': [mean_cross_val_score]})
    scores_data = scores_data.append(temp_score)
# we change the structure of the resulting data frame with model estimates, that there was one column with an estimate\
#  and columns with a type of estimate
data_scores_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'mean_cross_val_score'], 
    var_name='set_types', value_name='score')
# print(data_scores_long.head())

# sns.lineplot(x='max_depth', y='score', hue='set_types', data=data_scores_long)
# plt.show()
best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
best_clf.fit(X_train, y_train)
print(best_clf.score(X_test, y_test))
result = clf.predict(dog_pred)
print(pd.Series(result)[result == 'собачка'].count())