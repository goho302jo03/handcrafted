import numpy as np

def main(file):
    with open('%s.dat' %file, 'r') as f:
        data = f.readlines()

    data = np.array([v.split() for v in data], dtype='float32')
    if file == 'german':
        data[:, -1] -= 1
    np.save('./%s.npy' %file, data)

if __name__ == "__main__":
    main('australian')
    main('german')
