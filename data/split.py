import numpy as np
from sklearn.model_selection import train_test_split

def old_split(file, seed):
    data = np.load(file)
    train, tmp, _, _ = train_test_split(data, data, test_size=0.2, random_state=seed)
    val, test, _, _ = train_test_split(tmp, tmp, test_size=0.5, random_state=seed)

    return train, val, test

def split(file):
    data = np.load(file)

    return data
