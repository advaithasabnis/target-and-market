# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from targetandmarket.config import data_folder

# Uncomment following options for better viewability in IPython console
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 600)

#%% Constants
# Analyze users who joined before May 01 2020 since we are looking at
# activity in May 2020
PERIOD_START = pd.Timestamp('2017-11-01').timestamp()
PERIOD_END = pd.Timestamp('2020-05-01').timestamp()

# Last modified timestamp (in seconds) - end of analysis period
LAST_MODIFIED = 1590969600


#%% Import Data
# Note that session numbers and engagement times have been scaled
# by a hidden factor to obfuscate true values
user_analytics_raw = pd.read_csv(data_folder/'user_analytics_raw.csv', index_col=0)
user_error = pd.read_csv(data_folder/'user_error_status.csv', index_col=0)
user_pro_status = pd.read_csv(data_folder/'user_pro_status.csv')
timeseries = pd.read_csv(data_folder/'timeseries.csv', index_col=0)
user_holdings = pd.read_csv(data_folder/'user_holdings.csv')
user_geo = pd.read_csv(data_folder/'user_geo.csv', index_col=0)
user_purchases = pd.read_csv(data_folder/'user_purchases_may.csv', index_col=0)
user_purchases_june = pd.read_csv(data_folder/'user_purchases_early_june.csv', index_col=0)

#%% Preprocessing
# Session numbers are absent for some sessions.
# Any user with at least one such session is removed.
user_analytics = user_analytics_raw.loc[
    ~user_analytics_raw['user_id'].isin(user_error['user_id'])
    ].copy()

# Users who started using the app since app launch and 01 May 2020 are selected
user_analytics = user_analytics.loc[
    (user_analytics['first_open']>=PERIOD_START) &
    (user_analytics['first_open']<PERIOD_END)
    ].copy()
print('Number of users between analysis duration:', len(user_analytics))

# Converting from seconds to days ago
user_analytics.loc[:, 'last_session'] = ((LAST_MODIFIED
                                         - user_analytics['last_session'])
                                         / (60*60*24))
user_analytics.loc[:, 'first_open'] = ((LAST_MODIFIED
                                        - user_analytics['first_open'])
                                       / (60*60*24))
user_analytics = user_analytics.astype('float64')

#%% Merge geolocation
user_geo = user_geo.dropna()
user_analytics = pd.merge(user_analytics, user_geo, how='inner', on='user_id')
print('Number of users with geolocation:', len(user_analytics))

#%% Merge pro status and holdings with user analytics data
# Holdings is the value of the portfolio managed in the app
# The pro status is as of June 03, 2020 and includes those in trial (not paid)
user_analytics = pd.merge(user_analytics,
                          user_holdings[['obfuscatedId', 'holdings', 'isPro']],
                          how='inner',
                          left_on='user_id',
                          right_on='obfuscatedId')
user_analytics = user_analytics.drop(['obfuscatedId'], axis=1)

# Remove users with negative holdings or unrealistically large holdings
user_analytics = user_analytics.loc[(user_analytics.holdings<10000000)
                                    & (user_analytics.holdings>10)]

# Since pro status includes those in trial, set label value for them as 0
user_purchases = user_purchases.dropna()
user_purchases = user_purchases.drop(['event_name'], axis=1)
mapping = {'io.getdelta.android.delta_pro_yearly_trial': 0,
           'io.getdelta.ios.DELTA_PRO_EARLY_BACKER_MONTHLY': 1,
           'io.getdelta.ios.DELTA_PRO_YEARLY_TRIAL': 0,
           'io.getdelta.ios.DELTA_PRO_EARLY_BACKER_MONTHLY_EQUALIZED': 1,
           'io.getdelta.ios.DELTA_PRO_EARLY_BACKER_YEARLY': 1,
           'io.getdelta.android.delta_pro_early_backer_yearly_equalized': 1
           }

user_purchases.loc[:, 'product_id'] = user_purchases['product_id'].map(mapping)
user_purchases = user_purchases.sort_values(by=['user_id', 'product_id'])
user_purchases = user_purchases.drop_duplicates(subset=['user_id'], keep='last')
user_analytics = pd.merge(user_analytics, user_purchases, how='left', on='user_id')
# Users that have pro status = 1 but are on trial, set pro status as 0
user_analytics.loc[((user_analytics.isPro==1) & (user_analytics.product_id==0)), 'isPro'] = 0
user_analytics = user_analytics.drop(['product_id'], axis=1)

# Pro status is as of June 03. Therefore, remove users who bought pro on and after June 01
# The model is based on activity data of May 2020 and the label data must be as of May 31, 2020
user_purchases_june = user_purchases_june.drop(['event_name'], axis=1)
user_purchases_june = user_purchases_june.dropna()
user_purchases_june.loc[:, 'product_id'] = user_purchases_june['product_id'].map(mapping)
user_purchases_june = user_purchases_june.sort_values(by=['user_id', 'product_id'])
user_purchases_june = user_purchases_june.drop_duplicates(subset=['user_id'], keep='last')
user_analytics = pd.merge(user_analytics, user_purchases_june, how='left', on='user_id')

# Saving those who bought pro for later
june_purchases = user_analytics.loc[((user_analytics.isPro==1) & (user_analytics.product_id==1)),
                                    'user_id'].copy()
june_purchases.to_csv(data_folder/'june_purchases.csv')

# Set pro status for those who purchased or tried after June 01 as 0
user_analytics.loc[((user_analytics.isPro==1) & (~pd.isna(user_analytics.product_id))), 'isPro'] = 0
user_analytics = user_analytics.drop(['product_id'], axis=1)

print('Final number of users:', len(user_analytics))

#%% Save processed data
user_analytics.to_csv(data_folder/'user_analytics.csv')
