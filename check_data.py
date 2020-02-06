import numpy as np
import sys
from os import listdir
from os import walk
from os.path import isfile, join
import pickle

d = sys.argv[1]
print(f'Searching {d}')
for (dirpath, dirnames, filenames) in walk(d):
    for f in filenames:
        if f.endswith('npy'):
            f = dirpath + '/' + f
            try:
                #     with open(f, 'rb') as f_:
                #         a = pickle.load(f_)

                a = np.load(f)
                print(f'Successfully loaded {f}, {a.shape}')
            except Exception as e:
                print(f, e)
