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

def data():

    x_train, y_train = np.load('./tmp/x_train.npy'), np.load('./tmp/y_train.npy')
    x_val, y_val = np.load('./tmp/x_val.npy'), np.load('./tmp/y_val.npy')
    return x_train, y_train, x_val, y_val

def create_model(x_train, y_train, x_val, y_val):

    batch_size = 64
    epochs = 500
    init = 'Orthogonal'
    act = 'tanh'

    neurons = int({{quniform(9, 180, 9)}})
    layers = {{choice([1, 2, 4, 8])}}
    norm = {{choice(['no', 'l1', 'l2'])}}
    dropout = {{choice([0, 1])}}
    earlystop = {{choice([0, 1])}}
    k = None
    p = None
    patience = None

    if norm == 'no':
        reg = None
    elif norm == 'l1':
        k = {{loguniform(-9.2, -2.3)}}
        reg = regularizers.l1(k)
    elif norm == 'l2':
        k = {{loguniform(-9.2, -2.3)}}
        reg = regularizers.l2(k)

    X_input = Input((24, ))
    X = Reshape((-1, ))(X_input)

    for _ in range(layers):
        X = Dense(neurons, kernel_initializer=init, kernel_regularizer=reg)(X)
        X = Activation(act)(X)

        if dropout == 1:
            p = {{uniform(0, 1)}}
            X = Dropout(p)(X)

    X = Dense(1, kernel_initializer=init, kernel_regularizer=reg)(X)
    X_outputs = Activation('sigmoid')(X)

    model = Model(inputs = X_input, outputs = X_outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if earlystop == 0:
        model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(x_val, y_val))
    elif earlystop == 1:
        patience = int({{quniform(1, 500, 1)}})
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')
        model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(x_val, y_val), callbacks=[es])

    loss_t, score_t = model.evaluate(x_train, y_train, verbose=0)
    loss_v, score_v = model.evaluate(x_val, y_val, verbose=0)

    print(str(neurons) + '\t' + str(layers) + '\t' + str(norm) + '\t' + str(dropout) + '\t' + str(earlystop) + '\t\t' + '%-24s%-24s%s'%(str(k), str(p), str(patience)))
    return {'loss': loss_v, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    train_scores = []
    val_scores = []
    test_scores = []
    param = []

    random_seed = random.randint(0, 100)
    train, val, test = split('../../data/%s.npy' %sys.argv[1], random_seed)

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

    print('neurons\tlayers\tnorm\tdropout\tearlystop\tk\t\t\tp\t\t\tpatience')
    best_run, model = optim.minimize(model=create_model,
                                     data=data,
                                     algo=tpe.suggest,
                                     max_evals=300,
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

