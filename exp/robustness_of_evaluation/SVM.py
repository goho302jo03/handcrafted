#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# Standard import
from random import randint
import sys

# Third-party import
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run(data):
    test_acc = []
    for _ in range(20):
        random_seed = randint(0, 10000)
        tr_x, tmp_x, tr_y, tmp_y = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=random_seed)
        va_x, te_x, va_y, te_y = train_test_split(tmp_x, tmp_y, test_size=0.5, random_state=random_seed)
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(tr_x)
        va_x = scaler.transform(va_x)
        te_x = scaler.transform(te_x)
        clf = SVC(gamma='auto')
        clf.fit(tr_x, tr_y)
        test_acc.append(clf.score(te_x, te_y))

    return np.mean(test_acc), np.std(test_acc)


if "__main__" == __name__:
    data = np.load(f'../../data/{sys.argv[1]}.npy')

    print('1 round:')
    mean, std = run(data)
    print(f'mean: {mean}, std: {std}')

    print('\n20 round:')
    test_acc_20 = []
    for _ in range(20):
        test_acc_20.append(run(data)[0])
    print(f'mean: {np.mean(test_acc_20)}, std: {np.std(test_acc_20)}')

