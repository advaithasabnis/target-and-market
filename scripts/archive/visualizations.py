# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:03:42 2020

@author: advai
"""

# Visualization of model performance for Demo Slides

import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from targetandmarket.config import data_folder
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

pio.templates["verdana"] = go.layout.Template(
    layout_font=dict(family="verdana, arial", color="#ffffff"),
    layout_hoverlabel=dict(font_family="verdana, arial")
    )
pio.templates.default = "plotly_dark+verdana"

#%% Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import catboost as cb

#%% Import and prepare data
user_data = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)
user_data = user_data.drop(['country'], axis=1)

#%% Preprocessing
num_features = ['avg_session', 'last_session', 'first_open', 'active_days', 'holdings']
nom_features = ['continent']

X = user_data.drop(['isPro'], axis=1).copy()
y = user_data['isPro'].copy()

preprocessor = ColumnTransformer(transformers=[
    ('numerical', StandardScaler(), num_features),
    ('nominal', OneHotEncoder(categories='auto', drop='first'), nom_features)
    ])

X_full = preprocessor.fit_transform(X)
y_full = y.copy()
features = ['avg_session', 'last_session', 'first_open', 'active_days', 'holdings', 'Asia', 'Europe', 'Oceania', 'Rest of Americas', 'US & Canada']
X_full = pd.DataFrame(data=X_full, columns=features)

#%%
xgb_classifier = xgb.XGBClassifier(n_estimators=90, max_depth=4, learning_rate=0.075,
                                   colsample_bytree=0.7, subsample=0.8, reg_lambda=16, gamma=1,
                                   min_child_weight=1.5, objective='binary:logistic',
                                   scale_pos_weight=20)
xgb_classifier.fit(X_full, y_full)
y_score = xgb_classifier.predict_proba(X_full)[:, 1:]
precision, recall, th = precision_recall_curve(y_full, y_score)

#%%
rf = RandomForestClassifier(n_estimators=110, max_depth=5, min_samples_leaf=12, class_weight='balanced')
rf.fit(X_full, y_full)
y_score_rf = rf.predict_proba(X_full)[:, 1:]
precision_rf, recall_rf, th_rf = precision_recall_curve(y_full, y_score_rf)

#%%
kn = KNeighborsClassifier(n_neighbors=100)
kn.fit(X_full, y_full)
y_score_kn = kn.predict_proba(X_full)[:, 1:]
precision_kn, recall_kn, th_kn = precision_recall_curve(y_full, y_score_kn)

#%%
lr = LogisticRegression(C=0.4, solver='saga', class_weight='balanced', max_iter=100)
lr.fit(X_full, y_full)
y_score_lr = lr.predict_proba(X_full)[:, 1:]
precision_lr, recall_lr, th_lr = precision_recall_curve(y_full, y_score_lr)

#%% Catboost. Categorical feature - continent, can be directly fed
num_features = ['avg_session', 'last_session', 'first_open', 'active_days', 'holdings']
nom_features = ['continent']

X = user_data[num_features+nom_features].copy()
y = user_data['isPro'].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), num_features)
        ],
    remainder='passthrough')

X_full = preprocessor.fit_transform(X)
y_full = y.copy()
features = ['avg_session', 'last_session', 'first_open', 'active_days', 'holdings', 'continent']
X_full = pd.DataFrame(data=X_full, columns=features)

catb = cb.CatBoostClassifier(depth=4, learning_rate=0.065, l2_leaf_reg=1, iterations=50)

catb.fit(X_full, y_full)
y_score_catb = catb.predict_proba(X_full)[:, 1:]
precision_catb, recall_catb, th_catb = precision_recall_curve(y_full, y_score_catb)


#%% Precision Recall Curves for multiple models
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recall,
    y=precision,
    name='XGBoost'
    ))
fig.add_trace(go.Scatter(
    x=recall_rf,
    y=precision_rf,
    name='Random Forest'
    ))
fig.add_trace(go.Scatter(
    x=recall_kn,
    y=precision_kn,
    name='KNN'
    ))
fig.add_trace(go.Scatter(
    x=recall_lr,
    y=precision_lr,
    name='Logistic Reg.'
    ))
fig.add_trace(go.Scatter(
    x=recall_catb,
    y=precision_catb,
    name='Catboost'
    ))
fig.show()

