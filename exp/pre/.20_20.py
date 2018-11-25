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
sys.path.append('../data')
from split import split

def data():

    x_train, y_train = np.load('./tmp/x_train.npy'), np.load('./tmp/y_train.npy')
    x_val, y_val = np.load('./tmp/x_val.npy'), np.load('./tmp/y_val.npy')
    return x_train, y_train, x_val, y_val

def create_model(x_train, y_train, x_val, y_val):

    batch_size = 32
    epochs = 500
    init = 'lecun_normal'
    act = 'selu'

    neurons = 9
    layers = 1
    reg = None

    X_input = Input((24, ))
    X = X_input

    for _ in range(layers):
        X = Dense(neurons, kernel_initializer=init, kernel_regularizer=reg)(X)
        X = Activation(act)(X)

    X = Dense(1, kernel_initializer=init, kernel_regularizer=reg)(X)
    X_outputs = Activation('sigmoid')(X)

    model = Model(
        inputs = X_input,
        outputs = X_outputs,
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    model.fit(x_train,
        y_train,
        batch_size=batch_size,
        verbose=0,
        epochs=epochs,
        validation_data=(x_val, y_val),
    )
    loss_t, score_t = model.evaluate(
        x_train,
        y_train,
        verbose=0,
    )
    loss_v, score_v = model.evaluate(x_val, y_val, verbose=0)

    print(p)
    return {'loss': loss_v, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    datas = split('../data/%s.npy' %sys.argv[1])

    times = 20
    train_scores = []
    val_scores = []
    test_scores = []
    train_losses = []
    val_losses = []
    test_losses = []
    train_100_scores = []
    val_100_scores = []
    test_100_scores = []
    train_100_losses = []
    val_100_losses = []
    test_100_losses = []

    for j in range(20):
        for i in range(times):
            print('------------------------------------------------')
            print('Times = %d' %(i+1))
            print('------------------------------------------------')
            random_seed = random.randint(0, 500)
            train, tmp, _, _ = train_test_split(datas, datas, test_size=0.2, random_state=random_seed)
            val, test, _, _ = train_test_split(tmp, tmp, test_size=0.5, random_state=random_seed)

            x_train, y_train = train[:, :-1], train[:, -1]
            x_val, y_val = val[:, :-1], val[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

            np.save('./tmp/x_train.npy', x_train)
            np.save('./tmp/y_train.npy', y_train)
            np.save('./tmp/x_val.npy', x_val)
            np.save('./tmp/y_val.npy', y_val)

            best_run, model = optim.minimize(model=create_model,
                                             data=data,
                                             algo=tpe.suggest,
                                             max_evals=40,
                                             trials=Trials(),
                                             rseed=random_seed,
                                             verbose=False)
            print('')
            print('Best Param:')
            print(best_run)
            train_loss, train_score = model.evaluate(x_train, y_train, verbose=0)
            val_loss, val_score = model.evaluate(x_val, y_val, verbose=0)
            test_loss, test_score = model.evaluate(x_test, y_test, verbose=0)
            print('tr_loss: %f, tr_score: %f' %(train_loss, train_score))
            print('va_loss: %f, va_score: %f' %(val_loss, val_score))
            print('te_loss: %f, te_score: %f' %(test_loss, test_score))
            train_scores.append(train_score)
            val_scores.append(val_score)
            test_scores.append(test_score)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            K.clear_session()

        print('\n\n========== every 20 Summary ==========')
        print('tr_loss: %f, tr_score: %f' %(np.mean(train_losses), np.mean(train_scores)))
        print('va_loss: %f, va_score: %f' %(np.mean(val_losses), np.mean(val_scores)))
        print('te_loss: %f, te_score: %f' %(np.mean(test_losses), np.mean(test_scores)))
        train_100_scores.append(np.mean(train_scores))
        val_100_scores.append(np.mean(val_scores))
        test_100_scores.append(np.mean(test_scores))
        train_100_losses.append(np.mean(train_losses))
        val_100_losses.append(np.mean(val_losses))
        test_100_losses.append(np.mean(test_losses))

    print('\n\n============== Summary ==============')
    print('tr_loss: %f, tr_score: %f' %(np.mean(train_100_losses), np.mean(train_100_scores)))
    print('va_loss: %f, va_score: %f' %(np.mean(val_100_losses), np.mean(val_100_scores)))
    print('te_loss: %f, te_score: %f' %(np.mean(test_100_losses), np.mean(test_100_scores)))

