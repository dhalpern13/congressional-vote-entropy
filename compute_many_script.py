import pandas as pd
from entropy_funcs import greedy_prediction_entropy, overall_prediction_entropy, prediction_entropy

df = pd.read_csv('data/train.csv')

greedy_prediction_entropy(50, df)