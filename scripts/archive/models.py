# -*- coding: utf-8 -*-

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

