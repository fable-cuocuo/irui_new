from collections import defaultdict

import numpy as np


def _normalize_nonnegative(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr[arr < 0] = 0.0
    m = arr.max()
    if m <= 0:
        return np.ones_like(arr, dtype=np.float64)
    return arr / m


def load_user_sequences(seq_path):
    user_seqs = defaultdict(list)
    max_item_id = 0
    with open(seq_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                user = int(float(parts[0]))
                item = int(float(parts[1]))
            except ValueError:
                continue
            if item <= 0:
                continue
            user_seqs[user].append(item)
            if item > max_item_id:
                max_item_id = item
    return user_seqs, max_item_id


def build_item_importance_scores(seq_path, method="popularity"):
    user_seqs, max_item_id = load_user_sequences(seq_path)

    if method == "uniform":
        scores = np.ones(max_item_id + 1, dtype=np.float64)
        scores[0] = 0.0
        return scores.tolist()

    if method == "popularity":
        pop = np.zeros(max_item_id + 1, dtype=np.float64)
        for seq in user_seqs.values():
            for item in seq:
                pop[item] += 1.0
        pop[0] = 0.0
        pop = _normalize_nonnegative(pop)
        pop[0] = 0.0
        return pop.tolist()

    raise ValueError(f"Unsupported item importance method: {method}")
