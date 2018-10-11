from __future__ import print_function
import sys
import random
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional, loguniform, quniform
from hyperas import optim
from keras.layers import Dense, Input, Dropout, Reshape, LSTM, Activation, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Subtract, Lambda, Add, Concatenate, GaussianNoise, ThresholdedReLU, PReLU, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers, initializers
from keras import backend as K
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
sys.path.append('../../data')
from split import split

if __name__ == '__main__':
    data = split('../../data/%s.npy' %sys.argv[1])
    seed = np.load('./seed.npy')

    if sys.argv[1] == 'german':
        input_n = 24
    elif sys.argv[1] == 'australian':
        input_n = 15
    times = 20
    batch_size = 32
    epochs = 500

    train_scores_20_mean = []
    val_scores_20_mean = []
    test_scores_20_mean = []
    train_losses_20_mean = []
    val_losses_20_mean = []
    test_losses_20_mean = []

    for i in range(21):
        train_scores = []
        val_scores = []
        test_scores = []
        train_losses = []
        val_losses = []
        test_losses = []
        for j in range(times):
            random_seed = int(seed[i, j])
            train, tmp, _, _ = train_test_split(data, data, test_size=0.2, random_state=random_seed)
            val, test, _, _ = train_test_split(tmp, tmp, test_size=0.5, random_state=random_seed)

            x_train, y_train = train[:, :-1], train[:, -1]
            x_val, y_val = val[:, :-1], val[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            X_input = Input((input_n, ))
            X = Dense(9)(X_input)
            X = Activation('tanh')(X)
            X = Dense(1)(X)
            X_outputs = Activation('sigmoid')(X)

            model = Model(inputs = X_input, outputs = X_outputs)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            es = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
            model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(x_val, y_val), callbacks=[es])

            train_loss, train_score = model.evaluate(x_train, y_train, verbose=0)
            val_loss, val_score = model.evaluate(x_val, y_val, verbose=0)
            test_loss, test_score = model.evaluate(x_test, y_test, verbose=0)
            print('time = %d' %i)
            print('tr_loss: %f, tr_score: %f' %(train_loss, train_score))
            print('va_loss: %f, va_score: %f' %(val_loss, val_score))
            print('te_loss: %f, te_score: %f' %(test_loss, test_score))
            print('')
            train_scores.append(train_score)
            val_scores.append(val_score)
            test_scores.append(test_score)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            K.clear_session()

        train_scores_20_mean.append(np.mean(train_scores))
        val_scores_20_mean.append(np.mean(val_scores))
        test_scores_20_mean.append(np.mean(test_scores))
        train_losses_20_mean.append(np.mean(train_losses))
        val_losses_20_mean.append(np.mean(val_losses))
        test_losses_20_mean.append(np.mean(test_losses))
        print('======================================================')
        print(i, ' Seed: ', seed[i])
        print('tr_loss: %f, tr_score: %f' %(train_losses_20_mean[i], train_scores_20_mean[i]))
        print('va_loss: %f, va_score: %f' %(val_losses_20_mean[i], val_scores_20_mean[i]))
        print('te_loss: %f, te_score: %f' %(test_losses_20_mean[i], test_scores_20_mean[i]))

    sorted_index = np.argsort(test_scores_20_mean)
    train_scores_20_mean = np.array(train_scores_20_mean)
    val_scores_20_mean = np.array(val_scores_20_mean)
    test_scores_20_mean = np.array(test_scores_20_mean)
    train_losses_20_mean = np.array(train_losses_20_mean)
    val_losses_20_mean = np.array(val_losses_20_mean)
    test_losses_20_mean = np.array(test_losses_20_mean)
    seed = seed[sorted_index]
    train_scores_20_mean = train_scores_20_mean[sorted_index]
    val_scores_20_mean = val_scores_20_mean[sorted_index]
    test_scores_20_mean = test_scores_20_mean[sorted_index]
    train_losses_20_mean = train_losses_20_mean[sorted_index]
    val_losses_20_mean = val_losses_20_mean[sorted_index]
    test_losses_20_mean = test_losses_20_mean[sorted_index]
    print('\n')
    print('mid seed: ', seed[10])
    print('mid test scores', test_scores_20_mean[10])
