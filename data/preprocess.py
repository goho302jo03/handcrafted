import numpy as np

def main(file):
    with open('%s.dat' %file, 'r') as f:
        data = f.readlines()

    data = [v.split() for v in data]
    np.save('./%s.npy' %file, np.array(data, dtype='float32'))

if __name__ == "__main__":
    main('australian')
    main('german')
