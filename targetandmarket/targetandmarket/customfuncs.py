# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.models import Model


def create_model():
    # two sets of input, one for timeseries, other for aggregate features
    inputA = Input(shape=(31,1))
    inputB = Input(shape=(3,))
    
    # first branch operates on the timeseries
    x = LSTM(256)(inputA)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Model(inputs=inputA, outputs=x)
    
    # second branch operators on aggregate features
    y = Dense(256, activation='relu')(inputB)
    y = Dropout(0.2)(y)
    y = Dense(256, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(256, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(64, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Model(inputs=inputB, outputs=y)
    
    # combine two branches
    combined = Concatenate()([x.output, y.output])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.2)(z)
    z = Dense(16, activation='relu')(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[x.input, y.input], outputs=z)

    return model