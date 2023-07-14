import pandas as pd

from entropy_funcs_global import greedy_k_entropy

if __name__ == '__main__':
    df = pd.read_csv('data/cleaned_survey.csv')

    greedy_k_entropy(104, df)