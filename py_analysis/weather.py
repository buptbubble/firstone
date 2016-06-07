#-*- coding: utf-8-*-
import os
import numpy as np
import pandas as pd
import time
import pickle

weathers = pd.DataFrame()
weather_data = dict()
for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    if file[0] is not '.' and not file.endswith('.py'):
        table = pd.read_table(file, header=None, names=[
                              'datetime', 'weather', 'temp', 'pm2.5'], index_col=None)
        last_time_slice, cur_time_slice = 0, 0
        for index, row in table.iterrows():
            dt = row['datetime']
            tm = time.strptime(dt, '%Y-%m-%d %H:%M:%S')
            cur_time_slice = (tm.tm_hour * 60 + tm.tm_min) / 10 + 1
            ts = '{}-{}-{}-'.format(tm.tm_year, tm.tm_mon,
                                    tm.tm_mday) + str(cur_time_slice)
            if cur_time_slice == last_time_slice:
                weather_data[ts] = (np.add(weather_data[ts],
                    (row['weather'], row['temp'], row['pm2.5'])) / 2).astype(np.float32)
            else:
                weather_data[ts] = np.array([row['weather'], row['temp'], row['pm2.5']],
                                            dtype=np.float32)
            last_time_slice = cur_time_slice

df = pd.DataFrame.from_dict(weather_data).T
df.columns = ['weather', 'temp', 'pm2.5']

path_to_wr = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.path.pardir, os.path.pardir,
                         os.path.pardir, os.path.pardir)

with open(os.path.join(path_to_wr, 'weather.pkl'), 'wb') as f:
    pickle.dump(df, f)
