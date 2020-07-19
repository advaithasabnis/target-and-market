# -*- coding: utf-8 -*-
"""
Created on Wed May 27 2020

@author: Advait Hasabnis

# Feature Selection and Model Selection
# Area under precision recall curve used as metric
# XGBoost chosen as final model
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from targetandmarket.config import data_folder

#%% Definitions


class FeatureSelector(TransformerMixin, BaseEstimator):
    '''Select features using any sklearn method'''
    def __init__(self, transformer = None):
        self.transformer = transformer

    def fit(self, X, y=None, **kwargs):
        if self.transformer:
            self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.transformer:
            X_r = self.transformer.transform(X)
            return X_r
        else:
            return X


class ClfEstimator(BaseEstimator):
    '''Choose model for classification'''
    def __init__(self, estimator = LogisticRegression()):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        check_is_fitted(self.estimator)
        return self.estimator.predict(X)
    
    def predict_proba(self, X, y=None):
        check_is_fitted(self.estimator)
        return self.estimator.predict_proba(X)


def auprc(y_true, y_score):
    '''Custom scoring function: area under precision recall curve'''
    precision, recall, th = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def cv_score_statistics(estimator, X, y, cv=5, n_jobs=-1):
    '''Prints cross validation statistics'''
    scores = cross_val_score(estimator, X, y, scoring=make_scorer(auprc,needs_proba=True), cv=cv, n_jobs=n_jobs)
    print('All scores:\n', np.sort(scores), '\n')
    print('Mean:', np.round(np.mean(scores), 3))
    print('Median:', np.round(np.median(scores),3))
    print('SD:', np.round(np.std(scores), 3))
    

#%% Import data
user_data = pd.read_csv(data_folder/'user_analytics.csv', index_col=0)

#%% Train Test Split
X = user_data.drop(['user_id', 'sessions', 'total_time', 'isPro'], axis=1).copy()
y = user_data['isPro'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

#%% Pipeline
pipeline = Pipeline(steps=[
    ('sc', StandardScaler()),
    ('fs', FeatureSelector()),
    ('classifier', ClfEstimator())
])

#%% Baseline Model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_train)
cv_score_statistics(pipeline, X_train, y_train)
# Baseline AUPRC: 0.14

#%% Preliminary feature and model Selection
params = [
    {
        'fs__transformer': [SelectFromModel(LogisticRegression(), threshold=-np.inf)],
        'fs__transformer__max_features': np.arange(3,10),
        'classifier__estimator': [xgb.XGBClassifier(objective='binary:logisitc'),LogisticRegression(), RandomForestClassifier(), SVC()],
    },
    {
        'fs__transformer': [RFE(LogisticRegression()), RFE(xgb.XGBClassifier())],
        'fs__transformer__n_features_to_select': np.arange(3,10),
        'classifier__estimator': [xgb.XGBClassifier(objective='binary:logisitc'), LogisticRegression(), RandomForestClassifier(), SVC()],
    }]
grid = GridSearchCV(pipeline,
                    params,
                    cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc, needs_proba=True)
                   )
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)
# Best model: XGBoost, AUPRC = 0.20

#%% Tuning XGBoost model - Randomized Search over broad space
params = [
    {
        'fs__transformer': [RFE(xgb.XGBClassifier())],
        'fs__transformer__n_features_to_select': np.arange(3,8),
        'classifier__estimator': [xgb.XGBClassifier()],
        'classifier__estimator__n_estimators': np.arange(50,500,50),
        'classifier__estimator__max_depth': np.arange(2,10),
        'classifier__estimator__learning_rate': [0.1, 0.5, 0.8, 1],
        'classifier__estimator__subsample': [0.5, 0.8, 1],
        'classifier__estimator__colsample_by_tree': [0.5, 0.75, 1],
        'classifier__estimator__reg_lambda': [2, 4, 8, 16, 32],
        'classifier__estimator__gamma': [0, 1, 2, 4],
        'classifier__estimator__min_child_weight': [1, 1.25, 1.5, 1.75, 2],
        'classifier__estimator__objective': ['binary:logistic'],
        
    }]
grid = RandomizedSearchCV(pipeline,
                          params,
                          n_iter=100,
                          cv=5,
                          verbose=1,
                          n_jobs=-1,
                          scoring=make_scorer(auprc, needs_proba=True)
                         )
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

#%% Tuning XGBoost model - narrowed grid search
params = [
    {
        'classifier__estimator': [xgb.XGBClassifier()],
        'classifier__estimator__n_estimators': [250, 300],
        'classifier__estimator__max_depth': [3, 4],
        'classifier__estimator__learning_rate': [0.05, 0.075, 0.1],
        'classifier__estimator__subsample': [0.7, 0.8, 0.9],
        'classifier__estimator__colsample_by_tree': [0.6, 0.7],
        'classifier__estimator__reg_lambda': [3, 4, 5],
        'classifier__estimator__gamma': [1],
        'classifier__estimator__min_child_weight': [1.5],
        'classifier__estimator__objective': ['binary:logistic'],
        
    }]
grid = GridSearchCV(pipeline,
                    params,
                    cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc, needs_proba=True)
                   )
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)
# Best model: AUPRC = 0.235

#%% See performance on test set
y_pred = grid.predict_proba(X_test)[:, 1]
auprc(y_test, y_pred)
# AUPRC (Test Set): 0.230