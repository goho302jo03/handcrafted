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

    if sys.argv[1] == 'german':
        input_n = 24
    elif sys.argv[1] == 'australian':
        input_n = 15
    times = 20
    batch_size = 64
    epochs = 500
    inits = ['Zeros', 'Ones', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'Orthogonal', 'lecun_uniform', 'lecun_normal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal']
    acts = ['tanh', 'softsign', 'sigmoid', 'hard_sigmoid', 'relu', 'softplus', 'LeakyReLU', 'PReLU', 'elu', 'selu']

    best_scores = []

    for init in inits:
        for act in acts:
            print('============================')
            print(init + ' + ' + act)
            print('============================')
            seeds = []
            train_scores = []
            val_scores = []
            test_scores = []
            train_losses = []
            val_losses = []
            test_losses = []
            for i in range(times):
                random_seed = 1234*i + 5678
                train, val, test = split('../../data/%s.npy' %sys.argv[1], random_seed)

                x_train, y_train = train[:, :-1], train[:, -1]
                x_val, y_val = val[:, :-1], val[:, -1]
                x_test, y_test = test[:, :-1], test[:, -1]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                x_test = scaler.transform(x_test)

                X_input = Input((input_n, ))
                X = Dense(9, kernel_initializer=init)(X_input)

                if act == 'LeakyReLU':
                    X = LeakyReLU()(X)
                elif act == 'PReLU':
                    X = PReLU()(X)
                else:
                    X = Activation(act)(X)

                X = Dense(1, kernel_initializer=init)(X)
                X_outputs = Activation('sigmoid')(X)

                model = Model(inputs = X_input, outputs = X_outputs)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(x_val, y_val))

                train_loss, train_score = model.evaluate(x_train, y_train, verbose=0)
                val_loss, val_score = model.evaluate(x_val, y_val, verbose=0)
                test_loss, test_score = model.evaluate(x_test, y_test, verbose=0)
                print('time = %d' %i)
                print('seed = %d' %random_seed)
                print('tr_loss: %f, tr_score: %f' %(train_loss, train_score))
                print('va_loss: %f, va_score: %f' %(val_loss, val_score))
                print('te_loss: %f, te_score: %f' %(test_loss, test_score))
                print('')
                seeds.append(random_seed)
                train_scores.append(train_score)
                val_scores.append(val_score)
                test_scores.append(test_score)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                K.clear_session()

            sorted_index = np.argsort(train_losses)
            seeds = np.array(seeds)
            train_scores = np.array(train_scores)
            val_scores = np.array(val_scores)
            test_scores = np.array(test_scores)
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)
            test_losses = np.array(test_losses)
            seeds = seeds[sorted_index]
            train_scores = train_scores[sorted_index]
            val_scores = val_scores[sorted_index]
            test_scores = test_scores[sorted_index]
            train_losses = train_losses[sorted_index]
            val_losses = val_losses[sorted_index]
            test_losses = test_losses[sorted_index]
            best_scores.append(np.mean(train_losses))
            print('tr_loss: %f, tr_score: %f' %(np.mean(train_losses), np.mean(train_scores)))
            print('va_loss: %f, va_score: %f' %(np.mean(val_losses), np.mean(val_scores)))
            print('te_loss: %f, te_score: %f' %(np.mean(test_losses), np.mean(test_scores)))
            print('The medium of train loss is:')
            print('seed = %d' %seeds[10])
            print('tr_loss: %f, tr_score: %f' %(train_losses[10], train_scores[10]))
            print('va_loss: %f, va_score: %f' %(val_losses[10], val_scores[10]))
            print('te_loss: %f, te_score: %f' %(test_losses[10], test_scores[10]))
            print('\n')

    for i, loss in enumerate(best_scores):
        print(loss)
        if (i+1)%10 == 0:
            print('')
