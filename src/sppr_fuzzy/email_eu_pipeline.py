#!/usr/bin/env python
# coding: utf-8

# In[24]:


# # Cell 1 
# %pip install tqdm


# In[25]:


# Cell 2 — imports, global config, paths
from __future__ import annotations

import os
import math
import gzip
import json
import time
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from bisect import bisect_right

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy import sparse

# Reproducibility
np.random.seed(0)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

# SNAP file
SNAP_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
RAW_GZ = DATA_DIR / "email-Eu-core-temporal.txt.gz"
RAW_TXT = DATA_DIR / "email-Eu-core-temporal.txt"

# Paper constants
ALPHA = 0.85
SIGMA_THWPR = 0.60  # your \sigmaThWPR macro

# Strong-edge tolerance
STRONG_EPS = 1e-12

# Default sigma list used in the paper tables (as in your manuscript)
SIGMA_TABLE = [0.2626, 0.4415, 0.4989, 0.5550, 0.5825, 0.7215, 0.9900]


# In[26]:


# Cell 3 — download SNAP data (runs only if missing)
import urllib.request

if not RAW_GZ.exists():
    print(f"Downloading: {SNAP_URL}")
    urllib.request.urlretrieve(SNAP_URL, RAW_GZ)
    print(f"Saved to: {RAW_GZ}")
else:
    print(f"Found: {RAW_GZ}")


# In[27]:


# Cell 4 — unzip to a .txt for faster reloading (runs only if missing)
if not RAW_TXT.exists():
    print(f"Decompressing {RAW_GZ} -> {RAW_TXT}")
    with gzip.open(RAW_GZ, "rt") as f_in, open(RAW_TXT, "w") as f_out:
        for line in f_in:
            if line.strip() and not line.startswith("#"):
                f_out.write(line)
    print("Done.")
else:
    print(f"Found: {RAW_TXT}")


# In[28]:


# Cell 5 — load events into a DataFrame
# Format: src dst ts (ts in seconds; starts at 0 in this dataset)
df = pd.read_csv(RAW_TXT, sep=r"\s+", names=["src", "dst", "t"], dtype=np.int64)
df = df.sort_values("t").reset_index(drop=True)
df.head(), df.shape


# In[29]:


# Cell 6 — utilities: splitting, stable ranking, overlap, etc.

def time_quantiles(df_events: pd.DataFrame, q_train: float, q_eval: float) -> tuple[int, int]:
    """
    Returns (tau, tau') where tau is q_train-quantile and tau' is q_eval-quantile of timestamps.
    Uses numpy quantile (linear interpolation).
    """
    ts = df_events["t"].to_numpy()
    tau = int(np.quantile(ts, q_train))
    tau2 = int(np.quantile(ts, q_eval))
    return tau, tau2

def split_events(df_events: pd.DataFrame, q_train: float, q_eval: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: t <= tau
    Eval : tau < t <= tau'
    """
    tau, tau2 = time_quantiles(df_events, q_train=q_train, q_eval=q_eval)
    train = df_events[df_events["t"] <= tau].copy()
    eval_ = df_events[(df_events["t"] > tau) & (df_events["t"] <= tau2)].copy()
    return train, eval_

def stable_topk_set(values: np.ndarray, node_ids: np.ndarray, k: int) -> set[int]:
    """
    Deterministic top-k by sorting on (-value, node_id).
    """
    k = min(k, len(node_ids))
    order = np.lexsort((node_ids, -values))
    top = node_ids[order[:k]]
    return set(map(int, top))

def overlap_at_k(score: np.ndarray, target: np.ndarray, node_ids: np.ndarray, k: int) -> float:
    pred = stable_topk_set(score, node_ids, k)
    truth = stable_topk_set(target, node_ids, k)
    return len(pred & truth) / float(k)

def spearman_rho(score: np.ndarray, target: np.ndarray) -> float:
    rho, _ = spearmanr(score, target)
    return float(rho)


# In[30]:


# Cell 7 — build dyadic counts and fuzzy memberships (lambda), then high-confidence cutoff + mu

@dataclass
class FuzzyDigraph:
    node_ids: np.ndarray          # original node ids
    id_to_idx: dict[int,int]      # mapping node id -> [0..n-1]
    idx_to_id: np.ndarray         # inverse mapping
    n: int

    # high-confidence directed edges
    src: np.ndarray               # int32 indices
    dst: np.ndarray               # int32 indices
    w: np.ndarray                 # float64 memberships in [0,1]
    m: int

    # vertex memberships mu in [0,1]
    mu: np.ndarray                # float64 length n

    # in-strength counts from training window (baseline)
    instrength: np.ndarray        # float64 length n

    # helpful metadata
    lambda_min: float
    cmax: int
    q_train: float
    q_eval: float
    tau: int
    tau2: int


def build_fuzzy_graph_from_train(
    df_train: pd.DataFrame,
    q_train: float,
    q_eval: float,
    tau: int,
    tau2: int,
    highconf_quantile: float = 0.70,
) -> tuple[FuzzyDigraph, pd.DataFrame]:
    """
    From training events, build dyadic counts c_uv, compute lambda via log scaling,
    then threshold by the 70th percentile of lambda values (high-confidence subgraph).
    Also compute mu(v) = max incident lambda in the high-confidence graph.
    Returns (FuzzyDigraph, edges_df_full_train) where edges_df_full_train includes lambda for all dyads.
    """
    # Dyadic counts from training window
    grp = df_train.groupby(["src", "dst"], sort=False).size().reset_index(name="c")
    cmax = int(grp["c"].max())
    grp["lambda"] = np.log1p(grp["c"].to_numpy(dtype=np.float64)) / np.log1p(float(cmax))

    # Node set: all nodes that appear in training (as src or dst)
    nodes = np.unique(np.concatenate([df_train["src"].unique(), df_train["dst"].unique()]))
    nodes = np.sort(nodes)
    n = len(nodes)
    id_to_idx = {int(v): i for i, v in enumerate(nodes)}
    idx_to_id = nodes.copy()

    # Training-window in-strength baseline (counts, not lambdas)
    instr = np.zeros(n, dtype=np.float64)
    # aggregate incoming counts
    for _, row in grp.iterrows():
        v = int(row["dst"])
        instr[id_to_idx[v]] += float(row["c"])

    # High-confidence cutoff on lambda values over observed dyads
    lambdas = grp["lambda"].to_numpy(dtype=np.float64)
    lambda_min = float(np.quantile(lambdas, highconf_quantile))

    high = grp[grp["lambda"] >= lambda_min].copy()

    # Build high-confidence edge arrays
    src = high["src"].map(id_to_idx).to_numpy(dtype=np.int32)
    dst = high["dst"].map(id_to_idx).to_numpy(dtype=np.int32)
    w = high["lambda"].to_numpy(dtype=np.float64)

    # Vertex memberships mu(v) from high-confidence incident edges
    mu = np.zeros(n, dtype=np.float64)
    # outgoing
    for u_i, wt in zip(src, w):
        if wt > mu[u_i]:
            mu[u_i] = wt
    # incoming
    for v_i, wt in zip(dst, w):
        if wt > mu[v_i]:
            mu[v_i] = wt

    F = FuzzyDigraph(
        node_ids=nodes,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        n=n,
        src=src,
        dst=dst,
        w=w,
        m=len(w),
        mu=mu,
        instrength=instr,
        lambda_min=lambda_min,
        cmax=cmax,
        q_train=q_train,
        q_eval=q_eval,
        tau=tau,
        tau2=tau2,
    )
    return F, grp


# In[31]:


# Cell 8 — build adjacency lists (needed for widest-path and SCC thresholding)

@dataclass
class AdjLists:
    out_nbrs: list[np.ndarray]     # out_nbrs[u] = array of neighbors
    out_wts: list[np.ndarray]      # out_wts[u]  = array of weights aligned with out_nbrs[u]
    out_edge_idx: list[np.ndarray] # indices into edge arrays (src,dst,w) for each u

    in_nbrs: list[np.ndarray]
    in_wts: list[np.ndarray]

def make_adjlists(F: FuzzyDigraph) -> AdjLists:
    n = F.n
    # group edges by src
    buckets = [[] for _ in range(n)]
    for ei, u in enumerate(F.src):
        buckets[int(u)].append(ei)

    out_nbrs, out_wts, out_edge_idx = [], [], []
    for u in range(n):
        idxs = np.array(buckets[u], dtype=np.int32)
        out_edge_idx.append(idxs)
        if len(idxs) == 0:
            out_nbrs.append(np.array([], dtype=np.int32))
            out_wts.append(np.array([], dtype=np.float64))
        else:
            out_nbrs.append(F.dst[idxs])
            out_wts.append(F.w[idxs])

    # build reverse adjacency
    in_buckets = [[] for _ in range(n)]
    for ei, v in enumerate(F.dst):
        in_buckets[int(v)].append(ei)

    in_nbrs, in_wts = [], []
    for v in range(n):
        idxs = np.array(in_buckets[v], dtype=np.int32)
        if len(idxs) == 0:
            in_nbrs.append(np.array([], dtype=np.int32))
            in_wts.append(np.array([], dtype=np.float64))
        else:
            in_nbrs.append(F.src[idxs])
            in_wts.append(F.w[idxs])

    return AdjLists(out_nbrs=out_nbrs, out_wts=out_wts, out_edge_idx=out_edge_idx,
                    in_nbrs=in_nbrs, in_wts=in_wts)


# In[32]:


# Cell 9 — PageRank on a row-stochastic kernel represented as a sparse CSR (with dangling fix)

def pagerank_rowstochastic_csr(
    P: sparse.csr_matrix,
    dangling_rows: np.ndarray,
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 500,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """
    Computes p = alpha * P^T p + (1-alpha) u, with dangling rows replaced by u.

    P: CSR row-stochastic on non-dangling rows; dangling rows are all zeros.
    dangling_rows: boolean array length n indicating dangling rows.
    u: teleportation distribution (column). If None, uniform.
    """
    n = P.shape[0]
    if u is None:
        u = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        u = u.astype(np.float64, copy=False)
        u = u / u.sum()

    PT = P.transpose().tocsr()
    p = np.full(n, 1.0 / n, dtype=np.float64)

    dangling_rows = dangling_rows.astype(bool, copy=False)

    for _ in range(max_iter):
        dangling_mass = p[dangling_rows].sum()
        p_next = alpha * (PT @ p + dangling_mass * u) + (1.0 - alpha) * u
        # normalize defensively
        p_next = np.asarray(p_next).reshape(-1)
        p_next /= p_next.sum()

        if np.abs(p_next - p).sum() <= tol:
            p = p_next
            break
        p = p_next

    return p


# In[33]:


# Cell 10 — build CSR row-stochastic matrix from edges (node-level baselines)

def build_rowstochastic_csr_from_edges(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Builds a CSR matrix P where P[u,v] = weight(u,v)/sum_out(u), with zero rows for dangling.
    Returns (P, dangling_rows).
    """
    src = src.astype(np.int32, copy=False)
    dst = dst.astype(np.int32, copy=False)
    weight = weight.astype(np.float64, copy=False)

    out_sum = np.zeros(n, dtype=np.float64)
    np.add.at(out_sum, src, weight)

    dangling = (out_sum == 0)

    # Normalize weights where out_sum>0
    norm_w = weight / out_sum[src]

    P = sparse.csr_matrix((norm_w, (src, dst)), shape=(n, n), dtype=np.float64)
    return P, dangling


# In[34]:


# Cell 11 — widest-path (max–min) routine + strong-edge extraction (streaming per source)
import heapq

def widest_path_from_source(adj: AdjLists, s: int) -> np.ndarray:
    """
    Computes Con_F(s, v) for all v under max–min (widest path) semantics on the *high-confidence* support.
    Returns best[v] in [0,1], with best[s]=1.
    """
    n = len(adj.out_nbrs)
    best = np.zeros(n, dtype=np.float64)
    best[s] = 1.0

    heap = [(-1.0, s)]  # max-heap via negative
    while heap:
        neg_b, u = heapq.heappop(heap)
        b = -neg_b
        if b < best[u] - 1e-15:
            continue
        nbrs = adj.out_nbrs[u]
        wts = adj.out_wts[u]
        for v, w in zip(nbrs, wts):
            cand = b if b < w else w  # min(b, w)
            if cand > best[int(v)] + 1e-15:
                best[int(v)] = cand
                heapq.heappush(heap, (-cand, int(v)))
    return best

def compute_strong_edges(
    F: FuzzyDigraph,
    adj: AdjLists,
    eps: float = 1e-12,
    cache_key: str | None = None,
) -> np.ndarray:
    """
    Strong edge: support edge (u,v) with w(u,v) == Con_F(u,v) (within eps).
    Returns an array of edge indices into F.src/F.dst/F.w that are strong.
    Caches to disk if cache_key is provided.
    """
    if cache_key is not None:
        cache_path = CACHE_DIR / f"strong_edges_{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    strong_idx = []

    for u in tqdm(range(F.n), desc="Strong edges: widest-path per source"):
        best = widest_path_from_source(adj, u)
        eidxs = adj.out_edge_idx[u]
        if len(eidxs) == 0:
            continue
        dsts = F.dst[eidxs]
        ws = F.w[eidxs]
        # compare each outgoing edge weight to Con(u, v)
        con_vals = best[dsts]
        mask = np.abs(ws - con_vals) <= eps
        if np.any(mask):
            strong_idx.extend(eidxs[mask].tolist())

    strong_idx = np.array(strong_idx, dtype=np.int32)

    if cache_key is not None:
        np.save(cache_path, strong_idx)

    return strong_idx


# In[35]:


# Cell 12 — SCC partition of the sigma-cut digraph (custom Kosaraju; uses threshold sigma on weights)

def scc_kosaraju_sigma(adj: AdjLists, sigma: float) -> tuple[np.ndarray, int]:
    """
    Computes SCCs of the sigma-cut digraph: edges with weight >= sigma.
    Returns (comp_id, num_comps), where comp_id[v] in [0..m-1].
    """
    n = len(adj.out_nbrs)
    visited = np.zeros(n, dtype=bool)
    order = []

    # iterative DFS for finishing order
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            u, i = stack[-1]
            nbrs = adj.out_nbrs[u]
            wts = adj.out_wts[u]
            # advance i until we find an admissible edge or exhaust
            while i < len(nbrs) and wts[i] < sigma:
                i += 1
            if i >= len(nbrs):
                stack.pop()
                order.append(u)
                continue
            v = int(nbrs[i])
            stack[-1] = (u, i + 1)
            if not visited[v]:
                visited[v] = True
                stack.append((v, 0))

    # second pass on reversed graph
    comp = np.full(n, -1, dtype=np.int32)
    comp_count = 0
    for start in reversed(order):
        if comp[start] != -1:
            continue
        # BFS/DFS on reverse edges with same sigma threshold
        stack = [start]
        comp[start] = comp_count
        while stack:
            u = stack.pop()
            nbrs = adj.in_nbrs[u]
            wts = adj.in_wts[u]
            for v, w in zip(nbrs, wts):
                if w < sigma:
                    continue
                v = int(v)
                if comp[v] == -1:
                    comp[v] = comp_count
                    stack.append(v)
        comp_count += 1

    return comp, comp_count


# In[36]:


# Cell 13 — SPPR kernel build at a given sigma + lift to vertices

def sppr_scores_for_sigma(
    F: FuzzyDigraph,
    adj: AdjLists,
    strong_edge_idx: np.ndarray,
    sigma: float,
    alpha: float = 0.85,
) -> tuple[np.ndarray, int]:
    """
    Returns (vertex_scores, m) where m = number of SCC-classes at threshold sigma.

    Class construction: SCCs of sigma-cut digraph.
    Backbone coefficients: max over sigma-admissible strong edges between classes.
    Diagonal set to 0.
    """
    comp, m = scc_kosaraju_sigma(adj, sigma)

    # class memberships bar_mu = max mu(v) within class
    bar_mu = np.zeros(m, dtype=np.float64)
    for v in range(F.n):
        c = int(comp[v])
        if F.mu[v] > bar_mu[c]:
            bar_mu[c] = F.mu[v]

    # aggregate Lambda^{ssp}_{ij} from strong edges with w>=sigma
    Lambda = {}  # (ci,cj) -> max weight
    for ei in strong_edge_idx:
        w = float(F.w[ei])
        if w < sigma:
            continue
        u = int(F.src[ei]); v = int(F.dst[ei])
        ci = int(comp[u]); cj = int(comp[v])
        if ci == cj:
            continue
        key = (ci, cj)
        prev = Lambda.get(key, 0.0)
        if w > prev:
            Lambda[key] = w

    if len(Lambda) == 0:
        # no inter-class edges; PageRank becomes uniform on classes and lift via mu
        p_class = np.full(m, 1.0/m, dtype=np.float64)
        scores = F.mu * p_class[comp]
        return scores, m

    # build sparse row-stochastic P on classes
    rows = np.fromiter((k[0] for k in Lambda.keys()), dtype=np.int32, count=len(Lambda))
    cols = np.fromiter((k[1] for k in Lambda.keys()), dtype=np.int32, count=len(Lambda))
    vals_raw = np.fromiter((v for v in Lambda.values()), dtype=np.float64, count=len(Lambda))

    # w_ij = min(bar_mu_i, bar_mu_j, Lambda_ij) — kept for consistency with manuscript
    vals = np.minimum(vals_raw, np.minimum(bar_mu[rows], bar_mu[cols]))

    out_sum = np.zeros(m, dtype=np.float64)
    np.add.at(out_sum, rows, vals)
    dangling = (out_sum == 0)

    vals_norm = vals / out_sum[rows]
    P = sparse.csr_matrix((vals_norm, (rows, cols)), shape=(m, m), dtype=np.float64)

    u = np.full(m, 1.0/m, dtype=np.float64)
    p_class = pagerank_rowstochastic_csr(P, dangling_rows=dangling, alpha=alpha, u=u)

    # lift to vertices: fPR(v)=mu(v)*p_[v]
    scores = F.mu * p_class[comp]
    return scores, m


# In[37]:


# Cell 14 — Baseline scores: PR, WPR, thWPR, plus InStrength

def baseline_scores(
    F: FuzzyDigraph,
    alpha: float = 0.85,
    sigma_thwpr: float = 0.60,
) -> dict[str, np.ndarray]:
    n = F.n
    u = np.full(n, 1.0/n, dtype=np.float64)

    # PR: unweighted on high-confidence support
    w_un = np.ones(F.m, dtype=np.float64)
    P_pr, dang_pr = build_rowstochastic_csr_from_edges(n, F.src, F.dst, w_un)
    pr = pagerank_rowstochastic_csr(P_pr, dang_pr, alpha=alpha, u=u)

    # WPR: weighted by lambda on high-confidence
    P_wpr, dang_wpr = build_rowstochastic_csr_from_edges(n, F.src, F.dst, F.w)
    wpr = pagerank_rowstochastic_csr(P_wpr, dang_wpr, alpha=alpha, u=u)

    # thWPR: threshold at sigma_thwpr on top of high-confidence cutoff
    mask = (F.w >= sigma_thwpr)
    src_t = F.src[mask]; dst_t = F.dst[mask]; w_t = F.w[mask]
    if len(w_t) == 0:
        thwpr = np.full(n, 1.0/n, dtype=np.float64)
    else:
        P_th, dang_th = build_rowstochastic_csr_from_edges(n, src_t, dst_t, w_t)
        thwpr = pagerank_rowstochastic_csr(P_th, dang_th, alpha=alpha, u=u)

    # InStrength: training-window incoming counts (no propagation)
    instr = F.instrength.copy()

    return {
        "PR": pr,
        "WPR": wpr,
        f"thWPR_sigma={sigma_thwpr:.2f}": thwpr,
        "InStrength": instr,
    }


# In[38]:


# Cell 15 — Targets (paper + new ones): future volume, distinct senders, new incoming contacts, reply likelihood

def target_future_incoming_volume(
    df_eval: pd.DataFrame,
    F: FuzzyDigraph,
) -> np.ndarray:
    """
    Future incoming volume in eval window: sum_u (#events u->v in eval)
    """
    n = F.n
    tgt = np.zeros(n, dtype=np.float64)
    # restrict to node set
    mask = df_eval["dst"].isin(F.id_to_idx)
    tmp = df_eval[mask]
    counts = tmp.groupby("dst").size()
    for v_id, c in counts.items():
        tgt[F.id_to_idx[int(v_id)]] = float(c)
    return tgt

def target_future_distinct_senders(
    df_eval: pd.DataFrame,
    F: FuzzyDigraph,
) -> np.ndarray:
    """
    Diversity target: number of distinct senders in eval window for each recipient v.
    """
    n = F.n
    tgt = np.zeros(n, dtype=np.float64)
    tmp = df_eval[df_eval["dst"].isin(F.id_to_idx) & df_eval["src"].isin(F.id_to_idx)]
    # unique src per dst
    distinct = tmp.groupby("dst")["src"].nunique()
    for v_id, c in distinct.items():
        tgt[F.id_to_idx[int(v_id)]] = float(c)
    return tgt

def target_future_new_incoming_contacts(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    F: FuzzyDigraph,
) -> np.ndarray:
    """
    Novelty target: number of distinct eval senders u->v such that u->v did NOT occur in training.
    Uses raw training events (not high-confidence threshold), as "contact history".
    """
    n = F.n
    tgt = np.zeros(n, dtype=np.float64)

    train_pairs = set(
        zip(df_train["src"].to_numpy(dtype=np.int64), df_train["dst"].to_numpy(dtype=np.int64))
    )

    tmp = df_eval[df_eval["dst"].isin(F.id_to_idx) & df_eval["src"].isin(F.id_to_idx)]
    # distinct incoming senders in eval
    pairs_eval = tmp.groupby(["src", "dst"]).size().reset_index()[["src","dst"]]

    new_counts = defaultdict(int)
    for u, v in pairs_eval.to_numpy(dtype=np.int64):
        if (int(u), int(v)) not in train_pairs:
            new_counts[int(v)] += 1

    for v_id, c in new_counts.items():
        tgt[F.id_to_idx[int(v_id)]] = float(c)

    return tgt

def target_future_reply_likelihood(
    df_all: pd.DataFrame,
    df_eval: pd.DataFrame,
    F: FuzzyDigraph,
    reply_horizon_days: float = 7.0,
    use_replies_after_eval_end: bool = True,
) -> np.ndarray:
    """
    Approximate reply likelihood from directionality and time.

    For each pair (u->v) in eval, define t_in = first time u emailed v in eval window.
    v is considered to have "replied to u" if there exists an event v->u at time t_out
    with t_in < t_out <= t_in + H, where H = reply_horizon_days.

    reply_likelihood(v) = (#distinct incoming senders u in eval that v replied to) / (#distinct incoming senders u in eval).
    If denominator is 0, target is 0.

    If use_replies_after_eval_end=True, we search replies in df_all (full timeline),
    otherwise we restrict reply search to df_eval only.
    """
    H = int(reply_horizon_days * 24 * 3600)

    # incoming pairs in eval: earliest incoming time per (u,v)
    tmp = df_eval[df_eval["dst"].isin(F.id_to_idx) & df_eval["src"].isin(F.id_to_idx)]
    first_in = tmp.groupby(["src","dst"])["t"].min().reset_index()  # columns: src=u, dst=v, t=t_in

    # build outgoing time lists for candidate replies: (src=v, dst=u) -> sorted times
    if use_replies_after_eval_end:
        df_reply = df_all[df_all["src"].isin(F.id_to_idx) & df_all["dst"].isin(F.id_to_idx)]
    else:
        df_reply = tmp

    out_times = defaultdict(list)
    for s, d, t in df_reply[["src","dst","t"]].to_numpy(dtype=np.int64):
        out_times[(int(s), int(d))].append(int(t))
    for k in out_times:
        out_times[k].sort()

    denom = np.zeros(F.n, dtype=np.float64)
    numer = np.zeros(F.n, dtype=np.float64)

    # for each incoming dyad u->v with earliest time t_in, check if v->u occurs soon after
    for u_id, v_id, t_in in first_in.to_numpy(dtype=np.int64):
        u_id = int(u_id); v_id = int(v_id); t_in = int(t_in)
        v_idx = F.id_to_idx[v_id]
        denom[v_idx] += 1.0

        times = out_times.get((v_id, u_id), None)
        if not times:
            continue
        j = bisect_right(times, t_in)
        if j < len(times) and times[j] <= t_in + H:
            numer[v_idx] += 1.0

    # likelihood in [0,1]
    tgt = np.zeros(F.n, dtype=np.float64)
    mask = denom > 0
    tgt[mask] = numer[mask] / denom[mask]
    return tgt


# In[39]:


# Cell 16 — evaluation wrapper for a set of methods against a target
def evaluate_methods(
    method_scores: dict[str, np.ndarray],
    target: np.ndarray,
    node_ids: np.ndarray,
    ks: tuple[int, ...] = (50, 100),
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    mask: optional boolean mask of nodes to include in evaluation (stress tests).
    """
    if mask is None:
        mask = np.ones_like(target, dtype=bool)

    out = []
    for name, score in method_scores.items():
        s = score[mask]
        t = target[mask]
        ids = node_ids[mask]
        row = {"method": name, "rho": spearman_rho(s, t)}
        for k in ks:
            row[f"overlap@{k}"] = overlap_at_k(s, t, ids, k)
        out.append(row)

    return pd.DataFrame(out).set_index("method")


# In[40]:


# Cell 17 — full pipeline for one split: builds graph, computes baselines + SPPR (at chosen sigmas), evaluates (paper target)

def run_split_pipeline(
    df_all: pd.DataFrame,
    q_train: float,
    q_eval: float,
    sigma_table: list[float] = SIGMA_TABLE,
    alpha: float = ALPHA,
    sigma_thwpr: float = SIGMA_THWPR,
    highconf_quantile: float = 0.70,
    eps: float = STRONG_EPS,
) -> dict:
    """
    Returns dict with:
      F, adj, strong_edges_idx,
      baselines_scores,
      sppr_scores_table (dict sigma->scores),
      eval_targets (dict name->target),
      results_paper_target_table (DataFrame like Table)
    """
    tau, tau2 = time_quantiles(df_all, q_train=q_train, q_eval=q_eval)
    df_train, df_eval = split_events(df_all, q_train=q_train, q_eval=q_eval)

    F, edges_full = build_fuzzy_graph_from_train(
        df_train=df_train,
        q_train=q_train,
        q_eval=q_eval,
        tau=tau,
        tau2=tau2,
        highconf_quantile=highconf_quantile
    )

    adj = make_adjlists(F)

    # cache key: depends on split and cutoff
    cache_key = f"q{int(q_train*100)}_{int(q_eval*100)}_lammin{F.lambda_min:.6f}_n{F.n}_m{F.m}"
    strong_idx = compute_strong_edges(F, adj, eps=eps, cache_key=cache_key)

    base = baseline_scores(F, alpha=alpha, sigma_thwpr=sigma_thwpr)

    # SPPR for the table sigmas
    sppr = {}
    m_by_sigma = {}
    for s in sigma_table:
        scores, m = sppr_scores_for_sigma(F, adj, strong_idx, sigma=float(s), alpha=alpha)
        sppr[f"SPPR({s:.4f},{m})"] = scores
        m_by_sigma[float(s)] = int(m)

    # Paper target: future incoming volume (counts)
    tgt_vol = target_future_incoming_volume(df_eval, F)

    # Evaluate baselines
    methods_for_table = dict(base)
    # Evaluate SPPR rows for table
    methods_for_table.update(sppr)

    res = evaluate_methods(methods_for_table, tgt_vol, node_ids=F.idx_to_id)

    # Return everything + raw split frames for computing additional targets
    return dict(
        q_train=q_train, q_eval=q_eval, tau=tau, tau2=tau2,
        df_train=df_train, df_eval=df_eval,
        F=F, adj=adj, strong_idx=strong_idx,
        edges_full_train=edges_full,
        baselines=base,
        sppr_table=sppr,
        m_by_sigma=m_by_sigma,
        target_volume=tgt_vol,
        results_table=res
    )


# In[41]:


# Cell 18 — run the two splits used in the paper
run_80_90 = run_split_pipeline(df, q_train=0.80, q_eval=0.90)
run_70_80 = run_split_pipeline(df, q_train=0.70, q_eval=0.80)

# sanity prints matching the manuscript
for run in (run_80_90, run_70_80):
    F = run["F"]
    print(f"\nSplit q_train={run['q_train']:.2f}, q_eval={run['q_eval']:.2f}")
    print(f"tau={run['tau']}, tau'={run['tau2']}")
    print(f"Train events: {len(run['df_train']):,}, Eval events: {len(run['df_eval']):,}")
    print(f"Training nodes n={F.n}, training dyads (all)={len(run['edges_full_train']):,}")
    print(f"High-conf lambda_min={F.lambda_min:.6f}, high-conf edges={F.m:,}")
    print(f"Strong edges (cached)={len(run['strong_idx']):,}")


# In[42]:


# Cell 23 — sigma sweep utilities for figures (m(sigma), rho(sigma), overlap@100(sigma))

def sigma_grid_quantiles(weights: np.ndarray, q_low: float, q_high: float, num: int) -> np.ndarray:
    qs = np.linspace(q_low, q_high, num)
    return np.unique(np.quantile(weights, qs))

def sppr_sweep(
    run: dict,
    sigmas: np.ndarray,
    target: np.ndarray,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    F = run["F"]
    adj = run["adj"]
    strong_idx = run["strong_idx"]
    node_ids = F.idx_to_id

    rows = []
    for s in tqdm(sigmas, desc="SPPR sweep"):
        scores, m = sppr_scores_for_sigma(F, adj, strong_idx, sigma=float(s), alpha=alpha)
        rho = spearman_rho(scores, target)
        ov100 = overlap_at_k(scores, target, node_ids, 100)
        rows.append({"sigma": float(s), "m": int(m), "rho": rho, "overlap@100": ov100})
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


# In[43]:


# Cell 27 — helpers for consistent ordering + LaTeX export

def ordered_results(res_df: pd.DataFrame, run: dict) -> pd.DataFrame:
    """
    Orders rows as: baselines first, then SPPR rows in the exact order they were created for that run.
    Safe even if some rows are missing.
    """
    order = list(run["baselines"].keys()) + list(run["sppr_table"].keys())
    order = [k for k in order if k in res_df.index]
    return res_df.loc[order]

def format_results_for_paper(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Paper-style formatting:
      - rho: 4 decimals
      - overlaps: 2 decimals
    Returns a copy with rounded floats (still numeric).
    """
    out = df_res.copy()
    if "rho" in out.columns:
        out["rho"] = out["rho"].astype(float).round(4)
    for c in out.columns:
        if c.startswith("overlap@"):
            out[c] = out[c].astype(float).round(2)
    return out

def results_to_latex(df_res: pd.DataFrame) -> str:
    """
    LaTeX export with column-specific formatting (rho=4dp, overlaps=2dp).
    """
    df_fmt = format_results_for_paper(df_res)

    def _fmt(x, col):
        if pd.isna(x):
            return ""
        if col == "rho":
            return f"{float(x):.4f}"
        if str(col).startswith("overlap@"):
            return f"{float(x):.2f}"
        return f"{float(x):.4f}"

    # Build formatters dict
    fmts = {col: (lambda x, col=col: _fmt(x, col)) for col in df_fmt.columns}
    return df_fmt.to_latex(formatters=fmts)


# In[44]:


# Cell 31 — Reply likelihood target (0.80/0.90 split)

F = run_80_90["F"]
df_eval = run_80_90["df_eval"]

tgt_reply_7d_80_90 = target_future_reply_likelihood(
    df_all=df,              # full timeline to find replies after incoming
    df_eval=df_eval,        # incoming events in the evaluation window
    F=F,
    reply_horizon_days=7.0,
    use_replies_after_eval_end=True
)

methods = {}
methods.update(run_80_90["baselines"])
methods.update(run_80_90["sppr_table"])

res_reply_80_90 = evaluate_methods(methods, tgt_reply_7d_80_90, node_ids=F.idx_to_id, ks=(50, 100))
ordered_results(res_reply_80_90, run_80_90)


# In[45]:


# Cell 32 — LaTeX for reply likelihood (0.80/0.90 split)

tab_reply_80_90 = ordered_results(res_reply_80_90, run_80_90)
print(results_to_latex(tab_reply_80_90))


# In[46]:


# Cell — Reply likelihood target + table (0.70/0.80 split)

F = run_70_80["F"]
df_eval = run_70_80["df_eval"]

tgt_reply_7d_70_80 = target_future_reply_likelihood(
    df_all=df,              # full timeline to find replies after incoming
    df_eval=df_eval,        # incoming events in the evaluation window
    F=F,
    reply_horizon_days=7.0,
    use_replies_after_eval_end=True
)

methods = {}
methods.update(run_70_80["baselines"])     # includes PR, WPR, thWPR_sigma=..., InStrength
methods.update(run_70_80["sppr_table"])    # SPPR rows for SIGMA_TABLE

res_reply_70_80 = evaluate_methods(methods, tgt_reply_7d_70_80, node_ids=F.idx_to_id, ks=(50, 100))

tab_reply_70_80 = ordered_results(res_reply_70_80, run_70_80)
display(tab_reply_70_80)

print(results_to_latex(tab_reply_70_80))


# In[ ]:




