import pandas as pd
import numpy as np
from time import time
# Comparison of duration of execution of functions
dataframe = pd.read_csv('module_3/bonus_python/iris.csv')

before_apply = time()
dataframe.apply('mean')
after_apply = time()
print('apply:', after_apply - before_apply)

before_mean = time()
dataframe.mean(axis=0)
after_mean = time()
print('mean:', after_mean - before_mean)

before_describe = time()
dataframe.describe().loc['mean']
after_describe = time()
print('describe:', after_describe - before_describe)

before_np = time()
dataframe.apply(np.mean)
after_np = time()
print('np:', after_np- before_np)