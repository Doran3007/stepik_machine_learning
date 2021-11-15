import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

space = pd.read_csv('module_3/attack_on_earth/space_can_be_a_dangerous_place.csv')


X = space.drop('dangerous', axis=1)
y = space['dangerous']

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
features = best_clf.feature_importances_
features = pd.DataFrame({'feature':list(X),'feature_importances': features}).sort_values(by='feature_importances', ascending=False)
print(features)
sns.barplot(x="feature_importances", y="feature", data=features,
            label="importance", color="b")
print(features)
plt.show()