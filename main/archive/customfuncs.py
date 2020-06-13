# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate, BatchNormalization, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


def create_model(data):
    # two sets of input, one for timeseries, other for aggregate features
    inputA = Input(shape=(31,1))
    inputB = Input(shape=(data.shape[1],))
    
    # first branch operates on the timeseries
    x = LSTM(300)(inputA)
    x = Dense(300, activation='relu')(x)
    
    # combine two branches
    y = Dense(300, activation='relu')(inputB)
    combined = concatenate([x, y])
    z = Dense(300, activation='relu')(combined)
    z = Dropout(0.1)(z)
    z = Dense(300, activation='relu')(z)
    z = Dropout(0.1)(z)
    z = Dense(300, activation='relu')(z)
    z = Dropout(0.1)(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[inputA, inputB], outputs=[z])

    return model


def simple_model(data):
    inputB = Input(shape=(data.shape[1],))
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


def recurrent_aedc():
    inputA = Input(shape=(31,1))
    
    x = LSTM(100, activation='relu')(inputA)
    x = RepeatVector(31)(x)
    x = LSTM(100, activation='relu', return_sequences=True)
    x = TimeDistributed(Dense(1, activation='sigmoid'))
    
    model = Model(inputs=[inputA], output=[x])
    return model