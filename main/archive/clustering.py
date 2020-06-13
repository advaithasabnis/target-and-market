# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from targetandmarket.config import appData_folder, data_folder
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%%
df = pd.read_csv(appData_folder/'user_predictions.csv', index_col=0)
june_purchases = pd.read_csv(data_folder/'june_purchases.csv', index_col=0)

#%%
NUMBER = 10000
df = df.sort_values(by=['prediction', 'holdings', 'avg_session'], ascending=False)

dff = df.loc[df.isPro==0].copy()
dff = dff.iloc[:NUMBER, :].copy()

#%% Features
num_features = ['avg_session', 'active_days', 'holdings']

sc = StandardScaler()
X = sc.fit_transform(dff[num_features])

#%% Timing different algorithms
import timeit
from sklearn.cluster import DBSCAN, Birch
from hdbscan import HDBSCAN

methods = [KMeans(n_clusters=4), MiniBatchKMeans(n_clusters=4), DBSCAN(eps=0.5), HDBSCAN(min_cluster_size=100), Birch(n_clusters=4)]
methods = [MiniBatchKMeans(n_clusters=4, random_state=1, batch_size=1000), MiniBatchKMeans(n_clusters=4, random_state=1, batch_size=100), MiniBatchKMeans(n_clusters=4, random_state=1, batch_size=5000)]

for algo in methods:
    print(algo, timeit.timeit('algo.fit(X)', globals=globals(), number=10))

#%% Clustering
# n_clusters=4 chosen by comparing silhouette scores
km = MiniBatchKMeans(n_clusters=4, random_state=21, batch_size=1000).fit(X)
labels = km.labels_
print(np.unique(labels, return_counts=True))

df_cluster = dff[num_features].copy()
df_cluster.loc[:, 'cluster'] = labels
cluster_info = df_cluster.groupby(by='cluster').mean().copy()
print(cluster_info)

#%% Scaling for visualization
mm = MinMaxScaler()
clusters = mm.fit_transform(cluster_info)
clusters = pd.DataFrame(clusters, columns=num_features)

feature_range = pd.DataFrame({'avg_session': np.arange(0,1.25,0.25),
                              'active_days': np.arange(0,1.25,0.25),
                              'holdings': np.arange(0,1.25,0.25),
                              })
feature_range = pd.DataFrame(mm.inverse_transform(feature_range), columns=['avg_session', 'active_days', 'holdings'])
feature_range = feature_range.astype(int)

#%% Summary plot of clusters
fig = go.Figure()
for index, row in clusters.iterrows():
    fig.add_trace(go.Scatter(
        x=['Avg. Session Time', 'Avg. Active Days', 'Avg. Holdings'],
        y=row[num_features],
        mode='markers+lines',
        marker=dict(size=16),
        line=dict(width=3),
        hoverinfo='skip',
        name=f'Cluster {index+1}',
        yaxis=f'y{index+1}'
        ))
fig.update_layout(
    template='plotly_dark',
    xaxis=dict(
        domain=[0.2,1],
        showgrid=False,
        ),
    yaxis=dict(
        title='Avg. Session Time (sec)',
        title_standoff=0,
        range=[-0.05,1.05],
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=feature_range.avg_session.values,
        ),
    yaxis2=dict(
        title='Avg. Active Days',
        title_standoff=0,
        anchor='free',
        overlaying='y',
        side='left',
        position=0.05,
        range=[-0.05,1.05],
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=feature_range.active_days.values,
        ),
    yaxis3=dict(
        title='Avg. Holdings ($)',
        title_standoff=10,
        anchor='x',
        overlaying='y',
        side='right',
        range=[-0.05,1.05],
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=feature_range.holdings.values,
        ),
    yaxis4=dict(
        overlaying='y',
        range=[-0.05,1.05],
        visible=False
        ),
    margin=dict(l=0, t=0, b=0, r=0),
    )
fig.show()

#%% Validation
validation = pd.merge(df, june_purchases, how='inner', on='user_id')
