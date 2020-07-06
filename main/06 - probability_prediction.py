# -*- coding: utf-8 -*-
"""
Created on Wed May 27 2020

@author: Advait Hasabnis

Train and predict probability, for each user, that they are like a paid user using XGBoost.
Output probability predictions for front-end dashboard.
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from targetandmarket.config import data_folder, appData_folder

#%% Custom scoring function
# Area Under Precision Recall Curve (AUPRC)


def auprc(y_true, y_score):
    precision, recall, th = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


#%% Import and prepare data
user_data = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)

X = user_data.drop(['user_id', 'sessions', 'total_time', 'isPro'], axis=1).copy()
y = user_data['isPro'].copy()

#%% Training model on full dataset
xgb_classifier = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.075,
                                   colsample_bytree=0.6, subsample=0.8, reg_lambda=4, gamma=1,
                                   min_child_weight=1.5, objective='binary:logistic')

xgb_classifier.fit(X, y)
y_score = xgb_classifier.predict_proba(X)[:, 1:]
print(f'ROC AUC: {roc_auc_score(y, y_score):0.3f}')
print(f'AUPRC: {auprc(y, y_score):0.3f}')

#%% Save predictions for Dash app
user_predictions = user_data.copy()
user_predictions.loc[:, 'prediction'] = y_score
user_predictions.to_csv(appData_folder/'user_predictions.csv')
