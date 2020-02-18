#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# Standard import
import random
import sys

# Third-party import
import numpy as np


if "__main__" == __name__:
    seed = np.zeros((21, 20))
    for i in range(21):
        for j in range(20):
            seed[i, j] = int(random.randint(0, 10000))
    np.save('seed_%s.npy' %str(sys.argv[1]), seed)
