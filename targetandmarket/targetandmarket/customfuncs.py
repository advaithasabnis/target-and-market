# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate, BatchNormalization
from tensorflow.keras.models import Model


def create_model():
    # two sets of input, one for timeseries, other for aggregate features
    inputA = Input(shape=(31,1))
    inputB = Input(shape=(3,))
    
    # first branch operates on the timeseries
    x = LSTM(128)(inputA)
    x = Dropout(0.2)(x)
    
    # combine two branches
    combined = concatenate([x, inputB])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.2)(z)
    z = BatchNormalization()(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(32, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[inputA, inputB], outputs=[z])

    return model


def simple_model():
    inputB = Input(shape=(3,))
    z = Dense(256, activation='relu')(inputB)
    z = Dropout(0.2)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(32, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[inputB], outputs=[z])
    
    return model