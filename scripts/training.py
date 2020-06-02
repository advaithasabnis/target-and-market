# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from targetandmarket.config import data_folder
from targetandmarket.customfuncs import create_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay

#%%
user_data = pd.read_csv(data_folder/'user_data.csv', index_col=0)

X = user_data.iloc[:, :34].values
y = user_data['isPro'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%% Training and validation
train_preds = np.zeros((len(X_train)))
test_preds = np.zeros((len(X_test)))

i = 1
skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, valid_index in skfolds.split(X_train, y_train):
    X_train_folds, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_folds, y_valid_fold = y_train[train_index], y_train[valid_index]
    
    X_train_folds_A = X_train_folds[:, :31]
    X_train_folds_B = X_train_folds[:, -3:]
    
    X_valid_fold_A = X_valid_fold[:, :31]
    X_valid_fold_B = X_valid_fold[:, -3:]
    
    X_test_A = X_test[:, :31]
    X_test_B = X_test[:, -3:]
    
    X_train_folds_A = X_train_folds_A.reshape(X_train_folds_A.shape[0],
                                              X_train_folds_A.shape[1],
                                              1)
    X_valid_fold_A = X_valid_fold_A.reshape(X_valid_fold_A.shape[0],
                                            X_valid_fold_A.shape[1],
                                            1)
    X_test_A = X_test_A.reshape(X_test_A.shape[0], X_test_A.shape[1],1)

    model = create_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=AUC(curve='PR'))
    
    early_stopping = EarlyStopping(monitor='val_auc',
                                   min_delta=0.001,
                                   patience=5,
                                   mode='max',
                                   baseline=None,
                                   restore_best_weights=True,
                                   verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc',
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-6,
                                  mode='max',
                                  verbose=1)
    model.fit([X_train_folds_A, X_train_folds_B],
              y_train_folds,
              validation_data=([X_valid_fold_A, X_valid_fold_B], y_valid_fold),
              verbose=1,
              batch_size=256,
              callbacks=[early_stopping, reduce_lr],
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

test_preds /= skfolds.get_n_splits()
precision, recall, thresholds = precision_recall_curve(y_test, test_preds)
print("Overall Training AUPRC=",auc(recall, precision))

#%% Plot PR Curve
PrecisionRecallDisplay(precision, recall).plot()
