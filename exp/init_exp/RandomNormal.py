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
    x_test, y_test = np.load('./tmp/x_test.npy'), np.load('./tmp/y_test.npy')
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_model(x_train, y_train, x_val, y_val, x_test, y_test):

    if sys.argv[1] == 'german':
        input_n = 24
    elif sys.argv[1] == 'australian':
        input_n = 15

    batch_size = 32
    epochs = 500
    inits = ['Zeros', 'Ones', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'Orthogonal', 'lecun_uniform', 'lecun_normal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal']
    acts = ['tanh', 'softsign', 'sigmoid', 'hard_sigmoid', 'relu', 'softplus', 'LeakyReLU', 'PReLU', 'elu', 'selu']
    init = inits[2]
    act = acts[int({{quniform(0, 9, 1)}})]

    neurons = int({{quniform(9, 180, 9)}})
    layers = {{choice([1, 2, 4, 8])}}
    norm = {{choice(['no', 'l1', 'l2'])}}
    dropout = {{choice([0, 1])}}
    earlystop = {{choice([0, 1])}}
    k1 = None
    k2 = None
    p = None

    if norm == 'no':
        reg = None
    elif norm == 'l1':
        k1 = {{loguniform(-9.2, -2.3)}}
        reg = regularizers.l1(k1)
    elif norm == 'l2':
        k2 = {{loguniform(-9.2, -2.3)}}
        reg = regularizers.l2(k2)

    X_input = Input((input_n, ))
    X = X_input

    for _ in range(layers):
        X = Dense(
            neurons,
            kernel_initializer=init,
            kernel_regularizer=reg,
        )(X)

        if act == 'LeakyReLU':
            X = LeakyReLU()(X)
        elif act == 'PReLU':
            X = PReLU()(X)
        else:
            X = Activation(act)(X)

        if dropout == 1:
            p = {{uniform(0, 1)}}
            X = Dropout(p)(X)

    X = Dense(1, kernel_initializer=init, kernel_regularizer=reg)(X)
    X_outputs = Activation('sigmoid')(X)

    model = Model(inputs = X_input, outputs = X_outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    patience = int({{quniform(1, 500, 1)}})
    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=0,
        mode='auto',
    )
    if earlystop == 1:
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            verbose=0,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[es],
        )
    else:
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            verbose=0,
            epochs=epochs,
            validation_data=(x_val, y_val),
        )

    loss_t, score_t = model.evaluate(x_train, y_train, verbose=0)
    loss_v, score_v = model.evaluate(x_val, y_val, verbose=0)
    loss_te, score_te = model.evaluate(x_test, y_test, verbose=0)

    print(init + '\t' + act + '\t' + str(neurons) + '\t' + str(layers) + '\t' + str(norm) + '\t' + str(dropout) + '\t' + str(earlystop) + '%-24s%-24s%-24s%s'%(str(k1), str(k2), str(p), str(patience)) + '  ' + str(score_v) + '  ' + str(loss_v) + '  ' + str(score_te) + '  ' + str(loss_te))
    return {'loss': loss_v, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    seed = np.load('../pre/seed.npy')

    times = 20
    train_scores = []
    val_scores = []
    test_scores = []
    train_losses = []
    val_losses = []
    test_losses = []

    for i in range(times):
        datas = split('../../data/%s.npy' %sys.argv[1])
        random_seed = int(seed[15][i])
        # random_seed = 123456*i + 987654
        print('------------------------------------------------')
        print('Seed = %d' %(random_seed))
        print('------------------------------------------------')
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
        np.save('./tmp/x_test.npy', x_test)
        np.save('./tmp/y_test.npy', y_test)

        best_run, model = optim.minimize(model=create_model,
                                         data=data,
                                         algo=tpe.suggest,
                                         max_evals=100,
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
        print('=========================================================')
        train_scores.append(np.mean(train_score))
        val_scores.append(np.mean(val_score))
        test_scores.append(np.mean(test_score))
        train_losses.append(np.mean(train_loss))
        val_losses.append(np.mean(val_loss))
        test_losses.append(np.mean(test_loss))
        K.clear_session()

    print('\n\n========== every 20 Summary ==========')
    print('tr_loss: %f, tr_score: %f' %(np.mean(train_losses), np.mean(train_scores)))
    print('va_loss: %f, va_score: %f' %(np.mean(val_losses), np.mean(val_scores)))
    print('te_loss: %f, te_score: %f' %(np.mean(test_losses), np.mean(test_scores)))
