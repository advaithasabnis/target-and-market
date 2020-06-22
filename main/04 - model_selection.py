# -*- coding: utf-8 -*-
"""
Created on Wed May 27 2020

@author: Advait Hasabnis

# Tuned various models and compared performance
# Area under precision recall curve used as metric
# XGBoost chosen as final model
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, auc, make_scorer
from targetandmarket.config import data_folder
import xgboost as xgb

#%% Custom scoring function
# Area Under Precision Recall Curve (AUPRC)


def auprc(y_true, y_score):
    precision, recall, th = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


#%% Import and prepare data
user_data = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)

#%% Features
num_features = ['avg_session', 'first_open', 'active_days', 'holdings', 'numberOfTransactions']

X = user_data[num_features].copy()
y = user_data['isPro'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=y)

#%% Preprocessing
preprocessor = StandardScaler()
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

performance = pd.Series(dtype='float', name='auprc')

#%% Random Forrest
# Parameter tuning for classifier
params = {'n_estimators': [50, 75, 100],
          'max_depth': [4],
          'min_samples_leaf': [50],
          'class_weight': ['balanced']
          }
grid = GridSearchCV(RandomForestClassifier(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_score_rf = grid.predict_proba(X_test)[:, 1:]
performance['Random Forest'] = auprc(y_test, y_score_rf)

#%% Logistic Regression
# Parameter tuning for classifier
params = {'C': [0.1, 0.2, 0.4, 0.8],
          'solver': ['saga', 'sag'],
          'class_weight': ['balanced'],
          'max_iter': [400]
          }
grid = GridSearchCV(LogisticRegression(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_score_lr = grid.predict_proba(X_test)[:, 1:]
performance['Logistic Reg.'] = auprc(y_test, y_score_lr)

#%% KNN
params = {'n_neighbors': [125, 100, 175]
          }
grid = GridSearchCV(KNeighborsClassifier(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_score_kn = grid.predict_proba(X_test)[:, 1:]
performance['KNN'] = auprc(y_test, y_score_kn)


#%% SVC
params = {'C': [0.1, 0.2, 0.3, 0.4],
          'kernel': ['linear', 'rbf'],
          'probability': [True],
          'class_weight': ['balanced']
          }
grid = GridSearchCV(SVC(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_score_svc = grid.predict_proba(X_test)[:, 1:]
performance['SVC'] = auprc(y_test, y_score_svc)

#%% XGBoost
params = {'n_estimators': [80, 90, 100],
          'max_depth': [3, 4],
          'subsample': [0.7, 0.8, 0.9],
          'learning_rate': [0.065, 0.07, 0.075],
          'colsample_bytree': [0.6, 0.7],
          'reg_lambda': [16, 32],
          'gamma': [1],
          'min_child_weight': [1.5],
          'objective': ['binary:logistic'],
          }
grid = GridSearchCV(xgb.XGBClassifier(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_score_xgb = grid.predict_proba(X_test)[:, 1:]
performance['XGBoost'] = auprc(y_test, y_score_xgb)

#%% Random Predictor (random values between 0 and 1)
performance['Random']=y_test.mean()
performance = performance.sort_values()

#%% Plotly settings
import plotly.io as pio
import plotly.graph_objs as go

pio.templates["verdana"] = go.layout.Template(
    layout=dict(paper_bgcolor='#212121', plot_bgcolor='#212121'),
    layout_font=dict(family="verdana, arial", color="#ffffff"),
    layout_hoverlabel=dict(font_family="verdana, arial")
    )
pio.templates.default = "plotly+verdana"
pio.renderers.default = 'browser'

#%% Bar plot of model comparison for demo slides
fig = go.Figure()
fig.add_trace(go.Bar(
    x=performance.index,
    y=performance.values,
    marker=dict(color='#6002EE'),
    ))
config = {
  'toImageButtonOptions': {
    'format': 'svg',
    'filename': 'custom_image',
    'height': 500,
    'width': 700,
    'scale': 1
  }
}
fig.show(config=config)