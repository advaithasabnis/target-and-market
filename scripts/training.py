# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from targetandmarket.config import data_folder, appData_folder

#%% Import and prepare data
user_data = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)
# june_purchases = pd.read_csv(data_folder/'june_purchases.csv', index_col=0)

#%% Features
num_features = ['avg_session', 'last_session', 'first_open', 'holdings']
nom_features = ['continent']

X = user_data.drop(['isPro'], axis=1).copy()
y = user_data['isPro'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
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
params = {'n_estimators': [150, 175, 200],
                'max_depth': [2, 3],
                'learning_rate': [0.09, 0.095, 0.1],
                'colsample_bytree': [0.65],
                'reg_lambda': [16, 32],
                'gamma': [1],
                'min_child_weight': [1.5],
                'objective': ['binary:logistic'],
                'scale_pos_weight': [10, 20, 30]
                }
grid = GridSearchCV(xgb.XGBClassifier(), params,cv=5, verbose=1, n_jobs=-1, scoring='roc_auc')
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

#%% Check against test set
y_pred_test = grid.predict_proba(X_test)[:, 1:]
print(f'Test ROC AUC: {roc_auc_score(y_test, y_pred_test):0.3f}')

#%% Training model on full dataset
xgb_classifier = xgb.XGBClassifier(n_estimators=175, max_depth=2, learning_rate=0.095,
                                   colsample_bytree=0.65, reg_lambda=32, gamma=1,
                                   min_child_weight=1.5, objective='binary:logistic',
                                   scale_pos_weight=20)

X_full = preprocessor.fit_transform(X)
y_full = y.values

xgb_classifier.fit(X_full, y_full)
y_pred = xgb_classifier.predict_proba(X_full)[:, 1:]
print(f'ROC AUC: {roc_auc_score(y_full, y_pred):0.3f}')

#%% Save predictions for Dash app
user_predictions = user_data[['user_id', 'isPro']].copy()
user_predictions.loc[:, 'prediction'] = y_pred
user_predictions.to_csv(appData_folder/'user_predictions.csv')




