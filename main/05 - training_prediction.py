# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, make_scorer
from sklearn.model_selection import GridSearchCV
from targetandmarket.config import data_folder, appData_folder

#%% Custom scoring function
# Area Under Precision Recall Curve (AUPRC)


def auprc(y_true, y_score):
    precision, recall, th = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


#%% Import and prepare data
user_data = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)

#%% Features
num_features = ['avg_session', 'first_open', 'active_days', 'holdings', 'numberOfTransactions']
nom_features = ['continent']

X = user_data.drop(['isPro'], axis=1).copy()
y = user_data['isPro'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

#%% Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('numerical', StandardScaler(), num_features),
    ('nominal', OneHotEncoder(categories='auto', drop='first'), nom_features)
    ])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


#%% Tuning Xgboost
# Note: parameters were narrowed down from a wider range
# and the below params only show the last round of tuning
params = {'n_estimators': [80, 90, 100],
          'max_depth': [3, 4],
          'subsample': [0.7, 0.8, 0.9],
          'learning_rate': [0.065, 0.07, 0.075],
          'colsample_bytree': [0.6, 0.7],
          'reg_lambda': [16, 32],
          'gamma': [1],
          'min_child_weight': [1.5],
          'objective': ['binary:logistic'],
          'scale_pos_weight': [20]
          }
grid = GridSearchCV(xgb.XGBClassifier(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

#%% Check against test set
y_score_test = grid.predict_proba(X_test)[:, 1:]
print(f'Test ROC AUC: {auprc(y_test, y_score_test):0.3f}')

#%% Training model on full dataset
xgb_classifier = xgb.XGBClassifier(n_estimators=90, max_depth=4, learning_rate=0.07,
                                   colsample_bytree=0.7, subsample=0.9, reg_lambda=32, gamma=1,
                                   min_child_weight=1.5, objective='binary:logistic',
                                   scale_pos_weight=20)

X_full = preprocessor.fit_transform(X)
y_full = y.copy()
features = num_features + ['Asia', 'Europe', 'Oceania', 'Rest of Americas', 'US & Canada']
X_full = pd.DataFrame(data=X_full, columns=features)

xgb_classifier.fit(X_full, y_full)
y_score = xgb_classifier.predict_proba(X_full)[:, 1:]
print(f'ROC AUC: {roc_auc_score(y_full, y_score):0.3f}')
print(f'AUPRC: {auprc(y_full, y_score):0.3f}')

#%% Save predictions for Dash app
user_predictions = user_data.copy()
user_predictions.loc[:, 'prediction'] = y_score
user_predictions.to_csv(appData_folder/'user_predictions.csv')

