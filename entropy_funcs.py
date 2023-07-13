from itertools import combinations, product
from math import comb
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

np.seterr(invalid='ignore')


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


def col_entropies(file_name):
    df = pd.read_csv(file_name)
    return {
        col: df.groupby(col).apply(scaled_pred_entropy).sum() for col in df.columns
    }


def new_greedy_k_entropy2(k, data):
    cur_group_dfs = [data]
    cur_columns = []
    remaining_columns = set(data.columns)
    for iteration in range(k):
        for j, df in enumerate(cur_group_dfs):
            df.to_csv(f'temp/{j}.csv', index=False)

        with Pool() as p:
            df_col_entropies = list(tqdm(p.imap(col_entropies, [f'temp/{i}.csv' for i in range(len(cur_group_dfs))]),
                                         total=len(cur_group_dfs)))

        cols_to_entropy = {
            col: sum(d[col] for d in df_col_entropies) for col in data.columns
        }
        best_col = min(cols_to_entropy, key=cols_to_entropy.get)
        print(f'{best_col}: {cols_to_entropy[best_col] / len(data)}')
        cur_columns.append(best_col)
        remaining_columns.remove(best_col)
        cur_group_dfs = [sub_group_df for group_df in cur_group_dfs for _, sub_group_df in group_df.groupby(best_col) if
                         len(sub_group_df) > 1]
    return cur_columns


def new_greedy_k_entropy3(k, data):
    as_np = data.to_numpy()
    num_rows, num_cols = as_np.shape
    indicators = np.stack([(as_np == i) for i in range(3)])
    row_to_indices = dict()
    for i in range(3):
        row_id, col_id = np.nonzero(indicators[i].T)
        keys, locs = np.unique(row_id, return_index=True)
        as_dict = dict(zip(keys, np.split(col_id, locs[1:])))
        with_sets = {
            col: set(as_dict[col]) if col in as_dict else {} for col in range(num_cols)
        }
        row_to_indices[i] = with_sets

    cur_group_indices = [set(range(num_rows))]
    cur_columns = []
    remaining_columns = set(range(num_cols))
    for iteration in range(k):
        col_to_entropy = {}
        for col in tqdm(remaining_columns):
            total = 0
            for indices in cur_group_indices:
                index_subsets = [list(indices.intersection(row_to_indices[i][col])) for i in range(3)]

                for index_subset in index_subsets:
                    subset_counts = indicators[:, index_subset, :].sum(axis=1)
                    with np.errstate(divide='ignore'):
                        ent = entropy(subset_counts, axis=0, base=2)
                    mean_ent = np.nan_to_num(ent).mean()
                    scaled = mean_ent * len(index_subset)
                    total += scaled
            col_to_entropy[col] = total

        best_col = min(col_to_entropy, key=col_to_entropy.get)
        print(f'{data.columns[best_col]}: {col_to_entropy[best_col] / len(data)}')
        cur_columns.append(best_col)
        remaining_columns.remove(best_col)
        cur_group_indices = [indices.intersection(row_to_indices[i][best_col]) for i, indices in
                             product(range(3), cur_group_indices)]
        cur_group_indices = [s for s in cur_group_indices if len(s) > 1]
    return cur_columns


def new_greedy_k_entropy4(k, data):
    as_np = data.to_numpy()
    num_rows, num_cols = as_np.shape
    indicators = np.stack([(as_np == i) for i in range(3)])
    cur_group_indices = np.ones((1, num_rows))
    cur_columns = []
    remaining_columns = set(range(num_cols))

    for iteration in range(k):
        col_to_entropy = {}
        for col in remaining_columns:
            group_counts = cur_group_indices[:, None, :] * indicators[:, :, col]
            flattened = group_counts.reshape(-1, group_counts.shape[-1])
            split_group_counts = flattened.sum(axis=1)
            response_counts = flattened @ indicators
            ent = entropy(response_counts, axis=0, base=2)
            nan_filled = np.nan_to_num(ent)
            mean_column = nan_filled.mean(axis=1)
            scaled = mean_column * split_group_counts
            col_to_entropy[col] = scaled.sum()

        best_col = min(col_to_entropy, key=col_to_entropy.get)
        print(f'{data.columns[best_col]}: {col_to_entropy[best_col] / len(data)}')
        cur_columns.append(best_col)
        remaining_columns.remove(best_col)

        best_group_counts = cur_group_indices[:, None, :] * indicators[:, :, best_col]
        best_flattened = best_group_counts.reshape(-1, best_group_counts.shape[-1])

        cur_group_indices = best_flattened[best_flattened.sum(axis=1) > 1]
    return cur_columns


def col_entropy(args):
    col, cur_group_indices, indicators = args
    group_counts = cur_group_indices[:, None, :] * indicators[:, :, col]
    flattened = group_counts.reshape(-1, group_counts.shape[-1])
    split_group_counts = flattened.sum(axis=1)
    response_counts = flattened @ indicators
    with np.errstate(divide='ignore'):
        ent = entropy(response_counts, axis=0, base=2)
    nan_filled = np.nan_to_num(ent)
    mean_column = nan_filled.mean(axis=1)
    scaled = mean_column * split_group_counts
    return scaled.sum()


def new_greedy_k_entropy5(k, data):
    as_np = data.to_numpy()
    num_rows, num_cols = as_np.shape
    indicators = np.stack([(as_np == i) for i in range(3)])
    cur_group_indices = np.ones((1, num_rows))
    cur_columns = []
    remaining_columns = list(range(num_cols))

    for iteration in range(k):
        entropies = process_map(col_entropy, [(c, cur_group_indices, indicators) for c in remaining_columns])
        col_to_entropy = dict(zip(remaining_columns, entropies))

        best_col = min(col_to_entropy, key=col_to_entropy.get)
        print(f'{data.columns[best_col]}: {col_to_entropy[best_col] / len(data)}')
        cur_columns.append(best_col)
        remaining_columns.remove(best_col)

        best_group_counts = cur_group_indices[:, None, :] * indicators[:, :, best_col]
        best_flattened = best_group_counts.reshape(-1, best_group_counts.shape[-1])

        cur_group_indices = best_flattened[best_flattened.sum(axis=1) > 1]
    return cur_columns


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
