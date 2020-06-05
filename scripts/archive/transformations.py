# -*- coding: utf-8 -*-

#%% Transformations did not have any effect on the models
# Some features heavily positively skewed so log transform them
# user_analytics.loc[:, 'sessions'] = np.log10(user_analytics['sessions'])
# user_analytics.loc[:, 'total_time'] = np.log10(user_analytics['total_time'])
# user_analytics.loc[:, 'avg_session'] = np.log10(user_analytics['avg_session'])
# user_analytics.loc[:, 'last_session'] = np.log10(user_analytics['last_session'])
# user_analytics.loc[:, 'holdings'] = np.log10(user_analytics['holdings'])

#%% Timeseries data not used - did not seem to improve AUC of NN
# =============================================================================
# # Pivot timeseries data
# timeseries = timeseries.pivot(index='user_id',
#                               columns='date',
#                               values='day_eng_time')
# selected_columns = ['user_id', 'sessions', 'total_time', 'avg_session', 'last_session', 'first_open',
#                     'holdings', 'isPro']
# user_data = pd.merge(timeseries,
#                      user_analytics[selected_columns],
#                      how='inner',
#                      left_index=True,
#                      right_on='user_id')
# user_data = user_data.fillna(0)
# user_data = user_data.reset_index(drop=True)
# 
# # Save processed data
# user_data.to_csv(data_folder/'user_data.csv')
# =============================================================================