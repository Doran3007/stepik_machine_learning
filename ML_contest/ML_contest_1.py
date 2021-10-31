import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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