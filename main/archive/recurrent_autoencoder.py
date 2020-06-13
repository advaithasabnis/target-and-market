# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from targetandmarket.config import data_folder, appData_folder

from tensorflow.keras import backend
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

def auprc(y_true, y_score):
    precision, recall, th = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch/20)

#%% Import preprocessed data
user_data = pd.read_csv(data_folder/'user_data.csv', index_col=0)
userIds = user_data[['user_id']].copy()
user_data = user_data.drop(['user_id'], axis=1)

# Scale split data into train and test
X = user_data.iloc[:, :-1].copy()
y = user_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

num_features = X.columns[:-1]
nom_features = ['continent']

#%% Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('numerical', StandardScaler(), num_features),
    ('nominal', OneHotEncoder(categories='auto', drop='first'), nom_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


#%%
# define model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(31,1)))
model.add(RepeatVector(31))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

#%%
X_train_A = X_train[:, :31]
X_test_A = X_test[:, :31]

X_train_B = X_train[:, 31:]
X_test_B = X_test[:, 31:]

X_train_A = X_train_A.reshape(X_train_A.shape[0], X_train_A.shape[1], 1)
X_test_A = X_test_A.reshape(X_test_A.shape[0], X_test_A.shape[1], 1)

#%%
lr_scheduler = LearningRateScheduler(exponential_decay_fn)
model.fit(X_train_A,
          X_train_A,
          validation_split=0.25,
          epochs=40,
          verbose=1,
          batch_size=4096,
          callbacks=[lr_scheduler])


#%%
train_preds = model.predict(X_train_A)
test_preds = model.predict(X_test_A)
from sklearn.metrics import mean_squared_error, r2_score
X_train_A = X_train_A.reshape(X_train_A.shape[0], 31)
train_preds = train_preds.reshape(train_preds.shape[0], 31)
X_test_A = X_test_A.reshape(X_test_A.shape[0], 31)
test_preds = test_preds.reshape(test_preds.shape[0], 31)
print(f'Train: {r2_score(X_train_A, train_preds)}')
print(f'Test: {r2_score(X_test_A, test_preds)}')
backend.clear_session()



#%%
recurrent_ae = Model(inputs=model.inputs, outputs=model.layers[0].output)
X_train_A_enc = recurrent_ae.predict(X_train_A)
X_test_A_enc = recurrent_ae.predict(X_test_A)

X_train_final = np.concatenate((X_train_A_enc, X_train_B), axis=1)
X_test_final = np.concatenate((X_test_A_enc, X_test_B), axis=1)

#%%
xgb_classifier = xgb.XGBClassifier(n_estimators=90, max_depth=4, learning_rate=0.075,
                                   colsample_bytree=0.7, subsample=0.8, reg_lambda=16, gamma=1,
                                   min_child_weight=1.5, objective='binary:logistic',
                                   scale_pos_weight=20)

xgb_classifier.fit(X_train_final, y_train)
y_score = xgb_classifier.predict_proba(X_train_final)[:, 1:]
print(f'ROC AUC: {roc_auc_score(y_train, y_score):0.3f}')
print(f'AUPRC: {auprc(y_train, y_score):0.3f}')

#%% SHAP
import shap
import matplotlib.pyplot as plt
shap.initjs()
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_train_final)
plt.figure()
shap.summary_plot(shap_values, X_train_final, plot_type='bar')
plt.show()
