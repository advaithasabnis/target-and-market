# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from targetandmarket.config import data_folder
from targetandmarket.customfuncs import create_model
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.columns[:-1]
nom_features = ['continent']

#%% Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('numerical', StandardScaler(), num_features),
    ('nominal', OneHotEncoder(categories='auto', drop='first'), nom_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


#%% Training and validation - Combined timeseries and aggregate
train_preds = np.zeros((len(X_train)))
test_preds = np.zeros((len(X_test)))

i = 1
skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, valid_index in skfolds.split(X_train, y_train):
    X_train_folds, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_folds, y_valid_fold = y_train[train_index], y_train[valid_index]
    
    X_train_folds_A = X_train_folds[:, :31]
    X_train_folds_B = X_train_folds[:, 31:]
    
    X_valid_fold_A = X_valid_fold[:, :31]
    X_valid_fold_B = X_valid_fold[:, 31:]
    
    X_test_A = X_test[:, :31]
    X_test_B = X_test[:, 31:]
    
    X_train_folds_A = X_train_folds_A.reshape(X_train_folds_A.shape[0],
                                              X_train_folds_A.shape[1],
                                              1)
    X_valid_fold_A = X_valid_fold_A.reshape(X_valid_fold_A.shape[0],
                                            X_valid_fold_A.shape[1],
                                            1)
    X_test_A = X_test_A.reshape(X_test_A.shape[0], X_test_A.shape[1],1)

    model = create_model(X_train_folds_B)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=AUC(curve='PR'))
    
    early_stopping = EarlyStopping(monitor='val_auc',
                                   min_delta=0.001,
                                   patience=5,
                                   mode='max',
                                   baseline=None,
                                   restore_best_weights=True,
                                   verbose=1)
    lr_scheduler = LearningRateScheduler(exponential_decay_fn)
    # reduce_lr = ReduceLROnPlateau(monitor='val_auc',
    #                               factor=0.5,
    #                               patience=3,
    #                               min_lr=1e-6,
    #                               mode='max',
    #                               verbose=1)
    model.fit([X_train_folds_A, X_train_folds_B],
              y_train_folds,
              validation_data=([X_valid_fold_A, X_valid_fold_B], y_valid_fold),
              verbose=1,
              batch_size=4096,
              callbacks=[early_stopping, lr_scheduler],
              epochs=100,
              class_weight={0: 1, 1: 20}
              )
    
    valid_fold_preds = model.predict([X_valid_fold_A, X_valid_fold_B])
    test_fold_preds = model.predict([X_test_A, X_test_B])
    train_preds[valid_index] = valid_fold_preds.ravel()
    test_preds += test_fold_preds.ravel()
    precision, recall, thresholds = precision_recall_curve(y_valid_fold, valid_fold_preds)
    print("Fold Number ",i," - AUPRC = ", auc(recall, precision))
    backend.clear_session()
    i += 1

precision, recall, thresholds = precision_recall_curve(y_train, train_preds)
print("Overall Training AUPRC=",auc(recall, precision))
print("Overall Training AUC ROC", roc_auc_score(y_train, train_preds))

test_preds /= skfolds.get_n_splits()
precision, recall, thresholds = precision_recall_curve(y_test, test_preds)
print("Overall Test AUPRC=",auc(recall, precision))
print("Overall Test AUC ROC", roc_auc_score(y_test, test_preds))


