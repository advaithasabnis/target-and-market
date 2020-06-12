# -*- coding: utf-8 -*-

# Tuning of some models that I tried

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb
from sklearn.model_selection import GridSearchCV

#%% Random Forrest
rf_classifier = RandomForestClassifier(n_estimators=110, max_depth=5, min_samples_leaf=12, class_weight='balanced')

# Parameter tuning for classifier
params = {'n_estimators': [110],
          'max_depth': [4, 5, 6],
          'min_samples_leaf': [10, 11, 12, 13],
          'class_weight': ['balanced']
          }
grid = GridSearchCV(RandomForestClassifier(), params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_pred = grid.predict_proba(X_test)[:, 1:]
print('Test AUC:', roc_auc_score(y_test, y_pred))

#%% Logistic Regression
lr_classifier = LogisticRegression(C=0.4, solver='saga', class_weight='balanced', max_iter=100)

# Parameter tuning for classifier
params = {'C': [0.1, 0.2, 0.3, 0.4],
          'solver': ['saga', 'sag'],
          'class_weight': ['balanced'],
          'max_iter': [400]
          }
grid = GridSearchCV(LogisticRegression(), params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_pred = grid.predict_proba(X_test)[:, 1:]
print('Test AUC:', roc_auc_score(y_test, y_pred))

#%%
import catboost as cb

catb_classifier = cb.CatBoostClassifier(depth=4, learning_rate=0.065, l2_leaf_reg=1, iterations=50)

params = {'depth': [3, 4, 5],
          'learning_rate' : [0.065, 0.07],
         'l2_leaf_reg': [1, 2, 4],
         'iterations': [50]}
grid = GridSearchCV(cb.CatBoostClassifier(),
                    params,cv=5,
                    verbose=1,
                    n_jobs=-1,
                    scoring=make_scorer(auprc,needs_proba=True))
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best score:', grid.best_score_)

y_pred = grid.predict_proba(X_test)[:, 1:]
print('Test AUC:', roc_auc_score(y_test, y_pred))