from itertools import combinations
from math import comb

import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

from pandarallel import pandarallel

pandarallel.initialize()


def total_entropy(columns, data):
    restricted_votes = data[list(columns)]
    counts = restricted_votes.value_counts()
    return entropy(counts, base=2)


def prediction_entropy(columns, data):
    group = data.groupby(list(columns))

    def scaled_entropy(data_subset):
        value_counts_per_column = data_subset.apply(pd.Series.value_counts, axis=0).fillna(0)
        column_entropies = entropy(value_counts_per_column, axis=0, base=2)
        mean_entropy = column_entropies.mean()
        scaled_down = mean_entropy * len(data_subset) / len(data)
        return scaled_down

    return group.apply(scaled_entropy).sum()


def optimal_k_entropy(k, data):
    unique_columns = list(data.columns)
    return max((combo for combo in tqdm(combinations(unique_columns, r=k), total=comb(len(unique_columns), k))),
               key=lambda x: total_entropy(x, data))


def greedy_k_entropy(k, data):
    cur = []
    remaining_columns = set(data.columns)
    for _ in tqdm(range(k)):
        best_addition = max((bill for bill in remaining_columns), key=lambda bill: total_entropy(cur + [bill], data))
        cur.append(best_addition)
        print(best_addition)
        remaining_columns.remove(best_addition)
    return cur

def scaled_pred_entropy(data):
    value_counts_per_column = data.apply(pd.Series.value_counts, axis=0).fillna(0)
    column_entropies = entropy(value_counts_per_column, axis=0, base=2)
    mean_entropy = column_entropies.mean()
    scaled = mean_entropy * len(data)
    return scaled


def new_greedy_k_entropy(k, data):
    cur_group_dfs = [data]
    cur_columns = []
    remaining_columns = set(data.columns)
    for i in range(k):
        def col_to_entropy(col):
            return sum(group_df.groupby(col).apply(scaled_pred_entropy).sum() for group_df in cur_group_dfs)

        cols_to_entropy_dict = {col: col_to_entropy(col) for col in tqdm(remaining_columns, desc=f'{i + 1}:')}
        best_col = min(cols_to_entropy_dict, key=cols_to_entropy_dict.get)
        print(f'{best_col}: {cols_to_entropy_dict[best_col] / len(data)}')
        cur_columns.append(best_col)
        remaining_columns.remove(best_col)
        cur_group_dfs = [sub_group_df for group_df in cur_group_dfs for _, sub_group_df in group_df.groupby(best_col)]
    return cur_columns


def optimal_prediction_entropy(k, data):
    unique_columns = list(data.columns)
    return min((combo for combo in tqdm(combinations(unique_columns, r=k), total=comb(len(unique_columns), k))),
               key=lambda x: prediction_entropy(x, data))


def greedy_prediction_entropy(k, data):
    cur = []
    remaining_columns = set(data.columns)
    for i in range(k):
        best_addition = min((bill for bill in tqdm(remaining_columns, desc=f'{i + 1}.')),
                            key=lambda bill: prediction_entropy(cur + [bill], data))
        cur.append(best_addition)
        print(best_addition)
        remaining_columns.remove(best_addition)
    return cur


def overall_prediction_entropy(data):
    return entropy(data.apply(pd.Series.value_counts, axis=0).fillna(0), axis=0, base=2).mean()
