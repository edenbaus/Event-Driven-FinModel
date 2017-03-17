#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential


def make_timeseries_regressor(Vec_d, filter_length, nb_outputs=1, nb_filter=4):

    model = Sequential((
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(Vec_d, 1)),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='sigmoid'),     
    ))
    #model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # To perform (binary) classification instead:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def evaluate_timeseries(df,y, Vec_d):

    filter_length = 5
    nb_filter = 4
    df = np.atleast_2d(df)     # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = df.shape
    model = make_timeseries_regressor(Vec_d=Vec_d, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)

    X = np.atleast_3d(np.asarray(df))
    test_size = int(0.01 * nb_samples)       
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    model.fit(X_train, y_train, nb_epoch=25, batch_size=2, validation_data=(X_test, y_test))

    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    for actual, predicted in zip(y_test, pred.squeeze()):
        print(actual.squeeze(), predicted, sep='\t')
    print('next', model.predict(q).squeeze(), sep='\t')


def main():


    #####download data and import

if __name__ == '__main__':
