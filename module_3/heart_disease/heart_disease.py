import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

hearts = pd.read_csv('module_3/heart_disease/heart.csv')
print(hearts.head())

# drop columns with many nan values and unnecessary information for create 2 parameter, X=features, y=predicted value
X = hearts.drop(['target'], axis=1)
y = hearts.target


# ==========================================Random_forest
np.random.seed(0)

rf = RandomForestClassifier(10, max_depth=5)
rf.fit(X, y)
# find most valuable features
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({'feature':list(X),'feature_importances': feature_importances}).sort_values(by='feature_importances', ascending=False)
print(feature_importances_df)
sns.set_color_codes("muted")
sns.barplot(x="feature_importances", y="feature", data=feature_importances_df,
            label="importance", color="b")
plt.show()
