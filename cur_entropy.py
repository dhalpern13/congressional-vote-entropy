from sys import argv

import pandas as pd

from entropy_funcs import prediction_entropy

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

with open(argv[1]) as f:
    columns = f.read().splitlines()

print('train')
for i in range(1, len(columns) + 1):
    print(i, prediction_entropy(columns[:i], train))

print('test')
for i in range(1, len(columns) + 1):
    print(i, prediction_entropy(columns[:i], test))