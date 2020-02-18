#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# Third-party import
import numpy as np


def main(file):
    with open('%s.dat' %file, 'r') as f:
        data = f.readlines()

    data = np.array([v.split() for v in data], dtype='float32')
    if file == 'german':
        data[:, -1] -= 1
    np.save('./%s.npy' %file, data)


if "__main__" == __name__:
    main('australian')
    main('german')
