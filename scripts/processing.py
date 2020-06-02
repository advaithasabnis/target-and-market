# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from targetandmarket.config import data_folder

# Uncomment following options for better viewability in IPython console
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 600)

#%% Constants
# Analyze users who joined during the six months before analysis period
# From 1 Nov 2019 to 31 April 2020
PERIOD_START = pd.Timestamp('2017-09-01').timestamp()
PERIOD_END = pd.Timestamp('2020-05-01').timestamp()

# Last modified timestamp (in seconds) - end of analysis period
LAST_MODIFIED = 1590969600


#%% Import Data

user_analytics_raw = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)
user_error = pd.read_csv(data_folder/'user_error_status.csv', index_col=0)
user_pro_status = pd.read_csv(data_folder/'user_pro_status.csv')
timeseries = pd.read_csv(data_folder/'timeseries.csv', index_col=0)

# Note that session numbers and engagement times have been scaled
# by a hidden factor to obfuscate true values

#%% Preprocessing
# Session numbers are absent for some sessions.
# Any user with at least one such session is removed.
user_analytics = user_analytics_raw.loc[
    ~user_analytics_raw['user_id'].isin(user_error['user_id'])
    ].copy()

# Users who started using the app between 01 Nov 2019 (6 months prior) and 01 May 2020 are selected
user_analytics = user_analytics.loc[
    (user_analytics['first_open']>=PERIOD_START) &
    (user_analytics['first_open']<PERIOD_END)
    ].copy()
print('Number of users between analysis duration:', len(user_analytics))

# Remove users with less than 5 sessions
user_analytics = user_analytics.loc[user_analytics.sessions>=5].copy()
user_analytics = user_analytics.reset_index(drop=True)
print('Number of users with more than 5 sessions:', len(user_analytics))

# Converting from seconds to days ago
user_analytics.loc[:, 'last_session'] = ((LAST_MODIFIED
                                         - user_analytics['last_session'])
                                         / (60*60*24))
user_analytics.loc[:, 'first_open'] = ((LAST_MODIFIED
                                        - user_analytics['first_open'])
                                       / (60*60*24))
user_analytics = user_analytics.astype('float64')

# Merge pro status with user analytics data
user_analytics = pd.merge(user_analytics,
                          user_pro_status,
                          how='inner',
                          left_on='user_id',
                          right_on='obfuscatedId')
print('Number of users with isPro (final count):', len(user_analytics))

# Pivot timeseries data
selected_columns = ['user_id', 'avg_session', 'last_session', 'first_open',
                    'isPro']
timeseries = timeseries.pivot(index='user_id',
                              columns='date',
                              values='day_eng_time')
user_data = pd.merge(timeseries,
                     user_analytics[selected_columns],
                     how='inner',
                     left_index=True,
                     right_on='user_id')
user_data = user_data.fillna(0)
user_data = user_data.drop(['user_id'], axis=1)

# Avg session time and last_session are heavily positively skewed
# Log transform them
user_data.loc[:, 'avg_session'] = np.log(user_data['avg_session'])
user_data.loc[:, 'last_session'] = np.log(user_data['last_session'])

#%% Save processed data
user_data.to_csv(data_folder/'user_data.csv')

