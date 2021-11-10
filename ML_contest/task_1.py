import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

event_data = pd.read_csv('ML_contest/event_data_train.csv', header=0)
submissions_data = pd.read_csv('ML_contest/submissions_data_train.csv', header=0)

# sub_data = submissions_data.groupby(['step_id', 'submission_status']).agg({'submission_status': 'count'}).rename({'submission_status':'num'}, axis=1).sort_values(by=['num'],ascending=False)
# print(sub_data) 31978 - step_id


# Давайте найдем такой стэп, используя данные о сабмитах. Для каждого пользователя найдите такой шаг, 
# который он не смог решить, и после этого не пытался решать другие шаги. Затем найдите id шага, 
# который стал финальной точкой практического обучения на курсе для максимального числа пользователей. 

# Create columns with last time actions
sub_data = submissions_data.groupby('user_id').agg({'timestamp':'max'}).rename({'timestamp': 'last_act'}, axis=1)
submissions_data = submissions_data.merge(sub_data, on='user_id', how='left')

# Filter data, to find wrong answer with last time actions
wrong = ['wrong']
submissions_data = submissions_data[(submissions_data['submission_status'].isin(wrong)) & (submissions_data['timestamp'] == submissions_data['last_act'])].sort_values(by=['unique_user_per_step'], ascending=False)
print(submissions_data)
# group data by step_id, to find count attempt with wrong status
count_user_for_step = submissions_data.groupby('step_id').agg({'user_id':'nunique'}).rename({'user_id': 'user_num'}, axis=1).sort_values(by=['user_num'], ascending=False)
print(count_user_for_step)
