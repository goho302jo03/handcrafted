
if '__main__' == __name__:
    count = dict()

    for i in range(100):
        with open('./log/mlp9_choose_seed_%d.log' %(i+1)) as f:
            data = int(f.readlines()[-3].split('.')[0].split(' ')[-1].replace('[', ''))
        if data in count:
            count[data] += 1
        else:
            count[data] = 1
    print(count)
