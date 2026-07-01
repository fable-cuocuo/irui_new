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


def _build_transition_graph(user_seqs, max_item_id):
    graph = np.zeros((max_item_id + 1, max_item_id + 1), dtype=np.float64)
    for seq in user_seqs.values():
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            src = seq[i]
            dst = seq[i + 1]
            if src <= 0 or dst <= 0:
                continue
            graph[src, dst] += 1.0
    return graph


def _popularity_scores(user_seqs, max_item_id):
    pop = np.zeros(max_item_id + 1, dtype=np.float64)
    for seq in user_seqs.values():
        for item in seq:
            if item > 0:
                pop[item] += 1.0
    pop[0] = 0.0
    pop = _normalize_nonnegative(pop)
    pop[0] = 0.0
    return pop


def _pagerank_scores(transition_graph, max_iter=100, damping=0.85, tol=1e-8):
    n = transition_graph.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float64)

    trans = transition_graph.copy()
    row_sum = trans.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    trans = trans / row_sum

    pr = np.ones(n, dtype=np.float64) / n
    teleport = np.ones(n, dtype=np.float64) / n
    for _ in range(max_iter):
        nxt = (1.0 - damping) * teleport + damping * trans.T.dot(pr)
        if np.linalg.norm(nxt - pr, ord=1) < tol:
            pr = nxt
            break
        pr = nxt
    pr[0] = 0.0
    pr = _normalize_nonnegative(pr)
    pr[0] = 0.0
    return pr


def _eigenvector_scores(transition_graph, max_iter=200, tol=1e-8):
    n = transition_graph.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float64)

    # Symmetrize for stable undirected-style centrality.
    mat = transition_graph + transition_graph.T
    vec = np.ones(n, dtype=np.float64) / np.sqrt(max(n, 1))
    for _ in range(max_iter):
        nxt = mat.dot(vec)
        norm = np.linalg.norm(nxt, ord=2)
        if norm <= 0:
            break
        nxt = nxt / norm
        if np.linalg.norm(nxt - vec, ord=2) < tol:
            vec = nxt
            break
        vec = nxt
    vec = np.abs(vec)
    vec[0] = 0.0
    vec = _normalize_nonnegative(vec)
    vec[0] = 0.0
    return vec


def _blend_scores(user_seqs, transition_graph, max_item_id, spec):
    # spec format: popularity:0.5,pagerank:0.3,eigenvector:0.2
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Blend spec is empty.")

    scores = np.zeros(max_item_id + 1, dtype=np.float64)
    total_w = 0.0
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid blend part: {part}")
        method, weight = part.split(":", 1)
        method = method.strip().lower()
        w = float(weight.strip())
        if w <= 0:
            continue
        if method == "popularity":
            cur = _popularity_scores(user_seqs, max_item_id)
        elif method == "pagerank":
            cur = _pagerank_scores(transition_graph)
        elif method == "eigenvector":
            cur = _eigenvector_scores(transition_graph)
        else:
            raise ValueError(f"Unsupported blend method: {method}")
        scores += w * cur
        total_w += w

    if total_w <= 0:
        raise ValueError("Blend weights must contain positive values.")
    scores = scores / total_w
    scores[0] = 0.0
    scores = _normalize_nonnegative(scores)
    scores[0] = 0.0
    return scores


def build_item_importance_scores(seq_path, method="popularity"):
    user_seqs, max_item_id = load_user_sequences(seq_path)
    method = str(method).strip().lower()

    if max_item_id <= 0:
        return [0.0]

    if method == "uniform":
        scores = np.ones(max_item_id + 1, dtype=np.float64)
        scores[0] = 0.0
        return scores.tolist()

    transition_graph = _build_transition_graph(user_seqs, max_item_id)

    if method == "popularity":
        return _popularity_scores(user_seqs, max_item_id).tolist()
    if method == "pagerank":
        return _pagerank_scores(transition_graph).tolist()
    if method == "eigenvector":
        return _eigenvector_scores(transition_graph).tolist()
    if method.startswith("blend:"):
        blend_spec = method[len("blend:") :]
        return _blend_scores(user_seqs, transition_graph, max_item_id, blend_spec).tolist()

    raise ValueError(f"Unsupported item importance method: {method}")
