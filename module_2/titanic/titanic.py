import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv('module_2/titanic/train.csv')
# print(titanic.head())

# for check data nan values
# print(titanic.isnull().sum())

# drop columns with many nan values and unnecessary information for create 2 parameter, X=features, y=predicted value
X = titanic.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic.Survived
# Changing strings value to categorical values(1 or 0)
X = pd.get_dummies(X)
# Change nan values in columns to mediana
X = X.fillna({'Age': X.Age.median()})

# make simple model(it will be overfitting)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)
# displays a decision tree graph on the screen
# tree.plot_tree(clf, feature_names=X.columns)
# plt.show()

# the model came out overfitting, to avoid overfitting, we will use the maximal depth parameter, and divide the initial data into data for training and data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))

# to identify the most effective tree depth, do the following
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
sns.lineplot(x='max_depth', y='score', hue='set_types', data=data_scores_long)
# plt.show()
# Found the most effective tree depth in cross validation and trained the model with it, and check it on test data
best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
best_clf.fit(X_train, y_train)
# print(best_clf.score(X_test, y_test))
# ==========================================Random_forest
forest_clf = RandomForestClassifier()
parameters = {'n_estimators': [10, 20, 30], 'max_depth':[4,5,6], 'min_samples_leaf': [10,20],'min_samples_split':[10,30,50]}
grid_search_cv = GridSearchCV(forest_clf, parameters,cv=5)
grid_search_cv.fit(X_train, y_train)
print(grid_search_cv.fit(X_train, y_train))
print(grid_search_cv.best_params_)
best_clf = grid_search_cv.best_estimator_
print('best_clf', best_clf.score(X_test, y_test))
# find most valuable features
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'feature':list(X_train),'feature_importances': feature_importances}).sort_values(by='feature_importances', ascending=False)
print(feature_importances_df)