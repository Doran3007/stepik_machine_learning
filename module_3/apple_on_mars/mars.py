import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#  datasets
mushrooms = pd.read_csv('module_3/apple_on_mars/training_mush.csv')
testing_mush = pd.read_csv('module_3/apple_on_mars/testing_mush.csv')
y_test = pd.read_csv('module_3/apple_on_mars/testing_y_mush.csv')
mushrooms = pd.get_dummies(mushrooms)

#  split X, y, X_test
X = mushrooms.drop('class', axis=1)
y = mushrooms['class']

X_test = testing_mush

#  start learning
clf = RandomForestClassifier(random_state=0)

parametrs = {
    'n_estimators': range(10, 50, 10),
    'max_depth': range(1,12, 2),
    'min_samples_leaf': range(1, 7),
    'min_samples_split': range(2,9,2)
}
grid_search_cv  = GridSearchCV(clf, parametrs, cv=3, n_jobs=-1)
grid_search_cv.fit(X, y)
best_clf = grid_search_cv.best_estimator_
# print(grid_search_cv.best_params_)
# print(grid_search_cv.best_estimator_)

#  show the most significant features
features = best_clf.feature_importances_
features = pd.DataFrame({'feature':list(X),'feature_importances': features}).sort_values(by='feature_importances', ascending=False)
# print(features)
# sns.barplot(x="feature_importances", y="feature", data=features,
#             label="importance", color="b")
# plt.show()

#  show predicted value
predict = best_clf.predict(X_test)
poison_mush = predict.sum()
# print(poison_mush)

#  show confusion matrix
sns.heatmap(confusion_matrix(y_test, predict), annot=True, cmap="Blues")
plt.show()


