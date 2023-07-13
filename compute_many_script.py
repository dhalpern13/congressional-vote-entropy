import pandas as pd
from entropy_funcs import greedy_prediction_entropy, overall_prediction_entropy, prediction_entropy, \
    new_greedy_k_entropy, new_greedy_k_entropy2

df = pd.read_csv('data/train.csv')

new_greedy_k_entropy2(50, df)