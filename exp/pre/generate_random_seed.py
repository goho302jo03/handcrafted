import numpy as np
import random
import sys

if '__main__' == __name__:
    seed = np.zeros((21, 20))
    for i in range(21):
        for j in range(20):
            seed[i, j] = int(random.randint(0, 10000))
    np.save('seed_%s.npy' %str(sys.argv[1]), seed)
