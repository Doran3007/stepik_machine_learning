import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

event_data = pd.read_csv('ML_contest/event_data_train.csv', header=0)
submissions_data = pd.read_csv('ML_contest/submissions_data_train.csv', header=0)

event_data['date'] = pd.to_datetime(event_data['timestamp'], unit='s')
event_data['day'] = event_data['date'].dt.date

submissions_data['date'] = pd.to_datetime(submissions_data['timestamp'], unit='s')
submissions_data['day'] = submissions_data['date'].dt.date
# ----------check changed and data in plot
# print(event_data.head())
# user_activity = event_data.groupby('day').user_id.nunique().plot()
# plt.show()

# -------------info about users step
user_event = event_data.pivot_table(
    index = 'user_id',
    columns= 'action',
    values='step_id',
    aggfunc= 'count',
    fill_value=0
).reset_index()

# -------------info about users score
user_score = submissions_data.pivot_table(
    index = 'user_id',
    columns= 'submission_status',
    values='step_id',
    aggfunc= 'count',
    fill_value=0
).reset_index()

# ------------find info about diff about fist login and last login
gap_data = event_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']).groupby('user_id')['timestamp'] \
    .apply(list).apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data, axis=0))
# print(gap_data)

# --------------find lectors id
# event_data['month'] = event_data['date'].dt.month
# startdate = pd.to_datetime("2015-01-01").date()
# enddate = pd.to_datetime("2015-07-01").date()
# find_lector_data = event_data[(event_data['day'] > startdate) & (event_data['day'] < enddate)]
# find_lector_group = find_lector_data.groupby('user_id').agg({'month':'count'}).sort_values(by = ['month'], ascending=False)
# print(find_lector_group)

# ----------args period of returning user
period_ru = gap_data.quantile(0.90) / (24 * 60 * 60)

now = event_data['timestamp'].max()
drop_out_threshold = 30 * 24 * 60 * 60

users_data = event_data.groupby('user_id', as_index=False).agg({'timestamp':'max'}).rename(columns={'timestamp':'last_timestamp'})
users_data['is_gone_user'] = (now - users_data['last_timestamp']) > drop_out_threshold

# ----------dataset with count online day per user
user_days = event_data.groupby('user_id').day.nunique().to_frame().reset_index()

# ---------- do with 3 dataframes 1 large
users_data = users_data.merge(user_score, on='user_id', how='outer')
users_data = users_data.fillna(0)

users_data = users_data.merge(user_event, on='user_id', how='outer')
users_data = users_data.merge(user_days, on='user_id', how='outer')

# test about count unique user_id in our finish data and start data
# print(users_data.user_id.nunique())
# print(event_data.user_id.nunique())

users_data['passed_course'] = users_data['passed'] > 170

# ----------- find amount user, who finish course
# finish_course = users_data.groupby('passed_course', as_index=False).user_id.count()
# finish_course.index = finish_course['passed_course']
# false = finish_course.loc[False, 'user_id']
# true = finish_course.loc[True, 'user_id']
# finish_course_percent = (true/false) *100
# print(finish_course)
# print(finish_course_percent)
# print(users_data)

#  Create columns with first users actions
user_min_time = event_data.groupby('user_id', as_index=False).agg({'timestamp': 'min'}).rename({'timestamp': 'min_timestamp'}, axis=1)
users_data = users_data.merge(user_min_time, how='outer')

# Create columns with specify key, user_id+timestamp
event_data['user_time'] = event_data.user_id.map(str) + '_' + event_data.timestamp.map(str)
# Create args, that displays number of days without action
learning_time_treshold = 3 * 24 * 60 * 60
user_learning_time_treshold = user_min_time.user_id.map(str) + '_' + (user_min_time.min_timestamp + learning_time_treshold).map(str)
user_min_time['user_learning_time_treshold'] = user_learning_time_treshold
event_data = event_data.merge(user_min_time[['user_id', 'user_learning_time_treshold']], how='outer')
# Filter data with left user
events_data_train = event_data[event_data.user_time <= event_data.user_learning_time_treshold]
# print(events_data_train.groupby('user_id').day.nunique().max())

# Repeat same action to second data - submission_data
submissions_data['user_time'] = submissions_data.user_id.map(str) + '_' + submissions_data.timestamp.map(str)
submissions_data = submissions_data.merge(user_min_time[['user_id', 'user_learning_time_treshold']], how='outer')
submissions_data_train = submissions_data[submissions_data.user_time <= submissions_data.user_learning_time_treshold]
# submissions_data_train.groupby('user_id').day.nunique().max()

# Create main dataset
# number of days for user
X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index().rename(columns=({'day': 'days'}))

# number of step for user
steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index().rename(columns={'step_id': 'steps_tried'})
X = X.merge(steps_tried, on='user_id', how='outer')
# number of correct/wrong action
X = X.merge(submissions_data_train.pivot_table(index='user_id', columns='submission_status', values='step_id', aggfunc='count', 
                        fill_value=0).reset_index().head())
#  add some metrics
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
# add data from event_data
X = X.merge(events_data_train.pivot_table(index='user_id', columns='action', values='step_id', aggfunc='count', 
                        fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')
X = X.fillna(0)
#  extend dataset X, information from user_data, about finish/left course
X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']], how='outer')
# Filter only people who left course
X = X[~((X.is_gone_user == False) & (X.passed_course == False))]

#  Create args y, that we will prognosis
y = X.passed_course.map(int)

# Clear dataset X
X = X.drop(['passed_course', 'is_gone_user'], axis=1)
X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1)

# X.to_csv('ML_contest/X.csv')
# y.to_csv('ML_contest/y.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# linear logistick regresion
log_reg_clf = LogisticRegressionCV(cv=5)
log_reg_clf.fit(X_train, y_train)
y_pred = log_reg_clf.predict(X_test)
print(f'accuracy_score_Reg:{metrics.accuracy_score(y_test, y_pred)}')
print(f'f1_score_Reg:{metrics.f1_score(y_test, y_pred)}')
print(f'precision_score__Reg:{metrics.precision_score(y_test, y_pred)}')
print(f'recall_score_Reg:{metrics.recall_score(y_test, y_pred)}')
print(f'score_Reg:{log_reg_clf.score(X_test,y_pred)}')
print(f'coef_Reg:{log_reg_clf.coef_}')
print(f'intercept_Reg:{log_reg_clf.intercept_}')
print(f'cross_val_score_score:{cross_val_score(log_reg_clf,X_train,y_train,cv=4).mean()}')

# tree
dt = DecisionTreeClassifier(criterion='entropy')
parameters = {'max_depth': range(3,6), 'max_leaf_nodes':range(6,15), 'min_samples_leaf': range(1,4),'min_samples_split':range(2,5)}
grid_search_cv_tree_clf = GridSearchCV(dt, parameters, cv=4)
grid_search_cv_tree_clf.fit(X_train,y_train)
model = grid_search_cv_tree_clf.best_estimator_
y_pred_tree = model.predict(X_test)

print(f'accuracy_score_Tree:{metrics.accuracy_score(y_test, y_pred_tree)}')
print(f'f1_score_Tree:{metrics.f1_score(y_test, y_pred_tree)}')
print(f'precision_score_Tree:{metrics.precision_score(y_test, y_pred_tree)}')
print(f'recall_score_Tree:{metrics.recall_score(y_test, y_pred_tree)}')
print(f'DecisionTree:{grid_search_cv_tree_clf.best_params_}, cross_val_score_score: {cross_val_score(model,X_train,y_train,cv=4).mean()}')

#  KNeighbors
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(15,25), 'leaf_size':range(1,7)}
grid_search_cv_clf = GridSearchCV(knn,parameters,cv=4,n_jobs=-1)
grid_search_cv_clf.fit(X_train,y_train)
y_pred_kn = grid_search_cv_clf.predict(X_test)
model = grid_search_cv_clf.best_estimator_
print(f'accuracy_score_KN:{metrics.accuracy_score(y_test, y_pred_kn)}')
print(f'f1_score_KN:{metrics.f1_score(y_test, y_pred_kn)}')
print(f'precision_score_KN:{metrics.precision_score(y_test, y_pred_kn)}')
print(f'recall_score_KN:{metrics.recall_score(y_test, y_pred_kn)}')
print(f'KNeighbors:{grid_search_cv_clf.best_params_}, cross_val_score_score: {cross_val_score(model,X_train,y_train,cv=4).mean()}')