import numpy as np
from ast import literal_eval

files = ['best']
lines = [107, 217, 327, 437, 547, 657, 767, 877, 987, 1097,
         1207, 1317, 1427, 1537, 1647, 1757, 1867, 1977, 2087, 2197]

for file in files:
    count = {
        'layers': {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        },
        'int': {
            0.0: 0,
            1.0: 0,
            2.0: 0,
            3.0: 0,
            4.0: 0,
            5.0: 0,
            6.0: 0,
            7.0: 0,
            8.0: 0,
            9.0: 0,
            10.0: 0,
            11.0: 0
        },
        'int_1': {
            0.0: 0,
            1.0: 0,
            2.0: 0,
            3.0: 0,
            4.0: 0,
            5.0: 0,
            6.0: 0,
            7.0: 0,
            8.0: 0,
            9.0: 0,
        },
        'norm': {
            0: 0,
            1: 0,
            2: 0
        },
        'dropout': {
            0: 0,
            1: 0
        },
        # 'dropout_1': {
        #     0: 0,
        #     1: 0
        # }
    }
    with open('log/%s.log' %file, 'r') as f:
        data = f.readlines()
    for line in lines:
        line_dict = literal_eval(data[line-1])
        for ele in ['layers', 'int', 'norm', 'dropout', 'int_1']:
            count[ele][line_dict[ele]] += 1
    print(file)
    print(count)
    print('---------------------------------------------------------------')
