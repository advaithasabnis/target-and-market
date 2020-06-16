import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score


def kmeans_cluster(df, user_slider):
    dff = df.iloc[:user_slider, :].copy()
    num_features = ['avg_session', 'active_days', 'holdings']

    sc = StandardScaler()
    X = sc.fit_transform(dff[num_features])

    #%% Clustering
    # n_clusters=4 chosen by comparing silhouette scores
    km = MiniBatchKMeans(n_clusters=4, batch_size=1000, random_state=21).fit(X)
    labels = km.labels_

    df_cluster = dff[num_features].copy()
    df_cluster.loc[:, 'cluster'] = labels
    cluster_info = df_cluster.groupby(by='cluster').agg({'avg_session': 'mean', 'active_days': 'mean', 'holdings': ['mean', 'count']})
    cluster_info.columns = num_features + ['size']
    cluster_info.loc[cluster_info.avg_session.idxmax(), 'number'] = 1
    cluster_info.loc[cluster_info.active_days.idxmax(), 'number'] = 2
    cluster_info.loc[cluster_info.holdings.idxmax(), 'number'] = 3
    cluster_info.loc[:, 'number'] = cluster_info.number.fillna(4)
    cluster_info = cluster_info.sort_values(by='number')
    cluster_info = cluster_info.set_index('number')

    #%% Scaling for visualization
    mm = MinMaxScaler()
    clusters = mm.fit_transform(cluster_info[num_features])
    clusters = pd.DataFrame(clusters, columns=num_features)

    feature_range = pd.DataFrame({'avg_session': np.arange(0,1.25,0.25),
                                  'active_days': np.arange(0,1.25,0.25),
                                  'holdings': np.arange(0,1.25,0.25),
                                  })
    feature_range = pd.DataFrame(mm.inverse_transform(feature_range), columns=['avg_session', 'active_days', 'holdings'])
    feature_range = feature_range.astype(int)

    return cluster_info, clusters, feature_range