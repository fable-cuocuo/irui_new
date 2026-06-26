import copy
import math
from collections import defaultdict

import numpy as np

from multi_dataset_benchmark import prepare_cfg, train_eval, steam_cfg, yelp_cfg, ele_cfg


def _load_user_sequences(seq_path):
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


def _build_transition_graph(user_seqs):
    out_sum = defaultdict(float)
    in_edges = defaultdict(list)
    undirected = defaultdict(dict)

    for seq in user_seqs.values():
        if len(seq) <= 1:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if a <= 0 or b <= 0:
                continue
            out_sum[a] += 1.0
            in_edges[b].append((a, 1.0))

            undirected[a][b] = undirected[a].get(b, 0.0) + 1.0
            undirected[b][a] = undirected[b].get(a, 0.0) + 1.0

    return out_sum, in_edges, undirected


def _normalize_nonnegative(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr[arr < 0] = 0.0
    m = arr.max()
    if m <= 0:
        return np.ones_like(arr, dtype=np.float64)
    return arr / m


def _popularity_scores(user_seqs, max_item_id):
    pop = np.zeros(max_item_id + 1, dtype=np.float64)
    for seq in user_seqs.values():
        for item in seq:
            pop[item] += 1.0
    pop[0] = 0.0
    pop = _normalize_nonnegative(pop)
    pop[0] = 0.0
    return pop.tolist()


def _pagerank_scores(max_item_id, out_sum, in_edges, damping=0.85, iters=30):
    item_ids = [i for i in range(1, max_item_id + 1)]
    if not item_ids:
        return [0.0]

    n = len(item_ids)
    pr = np.full(max_item_id + 1, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n

    dangling_nodes = [i for i in item_ids if out_sum.get(i, 0.0) <= 0.0]

    for _ in range(iters):
        new_pr = np.zeros_like(pr)
        dangling_mass = pr[dangling_nodes].sum() if dangling_nodes else 0.0
        dangling_share = damping * dangling_mass / n

        for dst in item_ids:
            acc = 0.0
            for src, w in in_edges.get(dst, []):
                denom = out_sum.get(src, 0.0)
                if denom > 0:
                    acc += pr[src] * (w / denom)
            new_pr[dst] = teleport + damping * acc + dangling_share

        s = new_pr[item_ids].sum()
        if s > 0:
            new_pr[item_ids] /= s
        pr = new_pr

    pr[0] = 0.0
    pr = _normalize_nonnegative(pr)
    pr[0] = 0.0
    return pr.tolist()


def _eigenvector_scores(max_item_id, undirected, iters=20):
    item_ids = [i for i in range(1, max_item_id + 1)]
    if not item_ids:
        return [0.0]

    x = np.full(max_item_id + 1, 1.0, dtype=np.float64)
    x[0] = 0.0
    for _ in range(iters):
        x_new = np.zeros_like(x)
        for i in item_ids:
            neighbors = undirected.get(i, {})
            if not neighbors:
                continue
            s = 0.0
            for j, w in neighbors.items():
                s += w * x[j]
            x_new[i] = s
        norm = math.sqrt(float(np.sum(x_new[item_ids] ** 2)))
        if norm > 0:
            x_new[item_ids] /= norm
        x = x_new

    x[0] = 0.0
    x = _normalize_nonnegative(x)
    x[0] = 0.0
    return x.tolist()


def build_item_importance(seq_path, method):
    user_seqs, max_item_id = _load_user_sequences(seq_path)
    out_sum, in_edges, undirected = _build_transition_graph(user_seqs)

    if method == "uniform":
        scores = np.ones(max_item_id + 1, dtype=np.float64)
        scores[0] = 0.0
        return scores.tolist()
    if method == "popularity":
        return _popularity_scores(user_seqs, max_item_id)
    if method == "pagerank":
        return _pagerank_scores(max_item_id, out_sum, in_edges)
    if method == "eigenvector":
        return _eigenvector_scores(max_item_id, undirected)
    raise ValueError(f"Unknown method: {method}")


def main():
    dataset_cfgs = {
        "yelp": prepare_cfg(yelp_cfg),
        "electronics": prepare_cfg(ele_cfg),
        "ml1m": prepare_cfg(
            {
                **steam_cfg,
                "dataset": "ml1m",
                "input_len": 10,
                "max_seq_len": 10,
            }
        ),
    }

    methods = ["uniform", "popularity", "pagerank", "eigenvector"]
    seq_paths = {
        "yelp": "./datasets/yelp/seq/seq.dat",
        "electronics": "./datasets/electronics/seq/seq.dat",
        "ml1m": "./datasets/ml1m/seq/seq.dat",
    }

    scores_cache = {}
    for ds_name in dataset_cfgs.keys():
        scores_cache[ds_name] = {}
        for method in methods:
            print(f"\n>>> Building item_importance for {ds_name} with {method}")
            scores_cache[ds_name][method] = build_item_importance(seq_paths[ds_name], method)

    results = {}
    for method in methods:
        results[method] = {}
        for ds_name, cfg in dataset_cfgs.items():
            run_cfg = copy.deepcopy(cfg)
            run_cfg["item_importance_scores"] = scores_cache[ds_name][method]
            print(f"\n===== Running Fuzzy on {ds_name} with {method} importance =====")
            results[method][ds_name] = train_eval(run_cfg, "Fuzzy")

    print("\n=== METHOD_TABLE ===")
    print("| Method | Dataset | NDCG@10 | Hit@10 | AP | NDCG@5 | Hit@5 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for method in methods:
        for ds_name in dataset_cfgs.keys():
            met = results[method][ds_name]
            print(
                f"| {method} | {ds_name} | {met['ndcg_10']:.6f} | {met['hit_10']:.6f} | "
                f"{met['ap']:.6f} | {met['ndcg_5']:.6f} | {met['hit_5']:.6f} |"
            )

    print("\n=== METHOD_SUMMARY ===")
    print("| Method | Avg NDCG@10 | Avg Hit@10 | Avg AP |")
    print("|---|---:|---:|---:|")
    best_method = None
    best_avg_ndcg10 = -1.0
    for method in methods:
        avg_ndcg10 = float(np.mean([results[method][d]["ndcg_10"] for d in dataset_cfgs.keys()]))
        avg_hit10 = float(np.mean([results[method][d]["hit_10"] for d in dataset_cfgs.keys()]))
        avg_ap = float(np.mean([results[method][d]["ap"] for d in dataset_cfgs.keys()]))
        print(f"| {method} | {avg_ndcg10:.6f} | {avg_hit10:.6f} | {avg_ap:.6f} |")
        if avg_ndcg10 > best_avg_ndcg10:
            best_avg_ndcg10 = avg_ndcg10
            best_method = method

    print(f"\nBEST_METHOD_BY_AVG_NDCG10={best_method}")


if __name__ == "__main__":
    main()
