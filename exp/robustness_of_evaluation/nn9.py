#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# Standard import
from random import randint
import sys

# Third-party import
import keras
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.layers import Dense, Input, Activation
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run(data, input_n):
    batch_size = 64
    epochs = 500
    test_acc = []

    for _ in range(20):
        random_seed = randint(0, 10000)
        tr_x, tmp_x, tr_y, tmp_y = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=random_seed)
        va_x, te_x, va_y, te_y = train_test_split(tmp_x, tmp_y, test_size=0.5, random_state=random_seed)
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(tr_x)
        va_x = scaler.transform(va_x)
        te_x = scaler.transform(te_x)

        X_input = Input((input_n, ))
        X = Dense(9)(X_input)
        X = Activation('tanh')(X)
        X = Dense(1)(X)
        X_outputs = Activation('sigmoid')(X)

        model = Model(inputs=X_input, outputs=X_outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
        model.fit(tr_x, tr_y, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(va_x, va_y), callbacks=[es])

        train_loss, train_score = model.evaluate(tr_x, tr_y, verbose=0)
        test_loss, test_score = model.evaluate(te_x, te_y, verbose=0)

        test_acc.append(test_score)
        K.clear_session()

    return np.mean(test_acc), np.std(test_acc)


if "__main__" == __name__:
    if sys.argv[1] == 'german':
        input_n = 24
    elif sys.argv[1] == 'australian':
        input_n = 14

    data = np.load(f'../../data/{sys.argv[1]}.npy')
    times = 20

    print('1 round:')
    mean, std = run(data, input_n)
    print(f'mean: {mean}, std: {std}')

    print('\n20 round:')
    test_acc_20 = []
    for _ in range(20):
        test_acc_20.append(run(data, input_n)[0])
    print(f'mean: {np.mean(test_acc_20)}, std: {np.std(test_acc_20)}')
