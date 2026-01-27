#!/usr/bin/env python
# coding: utf-8

# In[36]:


from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional (if available); code works without SciPy too.
try:
    from scipy.stats import spearmanr, kendalltau
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

RNG = np.random.default_rng(0)

DATA_DIR = "data_email_eu"
os.makedirs(DATA_DIR, exist_ok=True)

# === Dataset (choose ONE) ===
# Full core temporal dataset:
DATA_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
LOCAL_GZ = os.path.join(DATA_DIR, "email-Eu-core-temporal.txt.gz")

# (Optional) Smaller department subnetworks for faster experiments:
# DATA_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept1.txt.gz"
# LOCAL_GZ = os.path.join(DATA_DIR, "email-Eu-core-temporal-Dept1.txt.gz")

# === Temporal split ===
TRAIN_FRAC = 0.80
TEST_FRAC  = 0.90  # test window is (tau, tau_prime]

# === PageRank ===
ALPHA = 0.85
MAX_ITER = 200
TOL = 1e-12

# === Fuzzy weight construction from email counts ===
# method="count" or "decay"
AGG_METHOD = "count"
HALF_LIFE_DAYS = 60.0   # only used if AGG_METHOD="decay"
SCALE_METHOD = "log"    # "log" or "minmax" or "cauchy"

# === Edge filtering (base vs high-confidence) ===
LAMBDA_MIN = 0.0        # noise floor on fuzzy edges (after scaling), keep base edges with w>=LAMBDA_MIN
HC_QUANTILE = 0.70      # high-confidence cutoff via quantile of base weights (controls SPPR speed)

# === Sigma grid strategy ===
SIGMA_GRID_MODE = "quantiles"  # "quantiles" or "linspace"
SIGMA_POINTS = 9


# In[37]:


import urllib.request

if not os.path.exists(LOCAL_GZ):
    print("Downloading:", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, LOCAL_GZ)
    print("Saved to:", LOCAL_GZ)
else:
    print("Already downloaded:", LOCAL_GZ)

# Data format: SRC DST UNIXTS (whitespace separated), comment lines start with '#'
events = pd.read_csv(
    LOCAL_GZ,
    sep=r"\s+",
    comment="#",
    header=None,
    names=["src", "dst", "time"],
    compression="gzip",
    dtype={"src": np.int32, "dst": np.int32, "time": np.int64},
)

print("Events loaded:", events.shape)
print(events.head())

tmin, tmax = int(events["time"].min()), int(events["time"].max())
print("Time span:", tmin, "to", tmax, "(seconds); ~", (tmax - tmin)/86400, "days")

all_nodes = np.union1d(events["src"].unique(), events["dst"].unique()).astype(np.int32)
all_nodes.sort()
print("Unique nodes:", all_nodes.size, "min id:", int(all_nodes.min()), "max id:", int(all_nodes.max()))


# In[38]:


def pick_time_cutoffs(df_events: pd.DataFrame, train_frac: float, test_frac: float) -> Tuple[int, int]:
    t = df_events["time"].to_numpy()
    tau = int(np.quantile(t, train_frac))
    tau_prime = int(np.quantile(t, test_frac))
    if tau_prime <= tau:
        tau_prime = tau + 1
    return tau, tau_prime

tau, tau_prime = pick_time_cutoffs(events, TRAIN_FRAC, TEST_FRAC)
print("tau (train end)      =", tau)
print("tau_prime (test end) =", tau_prime)

train_events = events[events["time"] <= tau].copy()
test_events  = events[(events["time"] > tau) & (events["time"] <= tau_prime)].copy()

print("Train events:", train_events.shape[0])
print("Test  events:", test_events.shape[0])


# In[39]:


def scale_to_unit_interval(raw: np.ndarray, method: str = "log") -> np.ndarray:
    raw = raw.astype(np.float64)
    if raw.size == 0:
        return raw
    mx = raw.max()
    if mx <= 0:
        return np.zeros_like(raw)

    if method == "log":
        return np.log1p(raw) / np.log1p(mx)
    if method == "minmax":
        return raw / mx
    if method == "cauchy":
        # saturating: x/(x+tau), choose tau as median positive
        pos = raw[raw > 0]
        tau = np.median(pos) if pos.size else 1.0
        return raw / (raw + tau)

    raise ValueError(f"Unknown scale method: {method}")

def aggregate_email_edges(
    df_events: pd.DataFrame,
    end_time: Optional[int] = None,
    method: str = "count",
    half_life_days: float = 60.0,
    scale: str = "log"
) -> pd.DataFrame:
    if method == "count":
        grp = df_events.groupby(["src", "dst"]).size().reset_index(name="raw")
        raw = grp["raw"].to_numpy(dtype=np.float64)

    elif method == "decay":
        if end_time is None:
            raise ValueError("end_time required for method='decay'")
        half_life_sec = float(half_life_days) * 86400.0
        dt = (end_time - df_events["time"].to_numpy(dtype=np.float64))
        contrib = np.exp(-dt / half_life_sec)
        tmp = df_events[["src", "dst"]].copy()
        tmp["raw"] = contrib
        grp = tmp.groupby(["src", "dst"])["raw"].sum().reset_index(name="raw")
        raw = grp["raw"].to_numpy(dtype=np.float64)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    w = scale_to_unit_interval(raw, method=scale)
    out = grp.copy()
    out["w"] = w
    return out[["src", "dst", "w", "raw"]]

edges_train = aggregate_email_edges(
    train_events,
    end_time=tau,
    method=AGG_METHOD,
    half_life_days=HALF_LIFE_DAYS,
    scale=SCALE_METHOD,
)

print("Aggregated directed edges (train):", edges_train.shape)
print(edges_train.head())
print("Weight stats:", edges_train["w"].describe())


# In[43]:


def filter_edges(df_edges: pd.DataFrame, lambda_min: float) -> pd.DataFrame:
    return df_edges[df_edges["w"] >= lambda_min].copy()


edges_base = filter_edges(edges_train, LAMBDA_MIN)

# Keep node set from base (so G_base and G_hc share it)
nodes = np.union1d(edges_base["src"].unique(), edges_base["dst"].unique()).astype(np.int32)
nodes.sort()
print("Nodes in base graph:", nodes.size)

# High-confidence cutoff by quantile of base weights
w_base = edges_base["w"].to_numpy()
weight_min = float(np.quantile(w_base, HC_QUANTILE)) if w_base.size else 1.0
edges_hc = edges_base[edges_base["w"] >= weight_min].copy()

print("Base edges:", edges_base.shape[0], "HC edges:", edges_hc.shape[0], "weight_min:", weight_min)
print("HC weight stats:", edges_hc["w"].describe())


# In[44]:


@dataclass
class IndexedDiGraph:
    nodes: np.ndarray                     # original node ids
    node_to_idx: Dict[int, int]           # original id -> 0..n-1
    out_adj: List[List[Tuple[int, float]]]  # adjacency in indices
    out_wsum: np.ndarray                  # sum of outgoing weights
    out_deg: np.ndarray                   # out-degree (count)
    edge_set: set                         # set of (u_idx,v_idx)

def build_indexed_digraph(df_edges: pd.DataFrame, nodes: np.ndarray) -> IndexedDiGraph:
    node_to_idx = {int(v): i for i, v in enumerate(nodes)}
    n = len(nodes)
    out_adj = [[] for _ in range(n)]
    out_wsum = np.zeros(n, dtype=np.float64)
    out_deg = np.zeros(n, dtype=np.float64)
    edge_set = set()

    for row in df_edges.itertuples(index=False):
        u0, v0, w = int(row.src), int(row.dst), float(row.w)
        if u0 not in node_to_idx or v0 not in node_to_idx:
            continue
        u = node_to_idx[u0]
        v = node_to_idx[v0]
        out_adj[u].append((v, w))
        out_wsum[u] += w
        out_deg[u] += 1.0
        edge_set.add((u, v))

    return IndexedDiGraph(
        nodes=nodes, node_to_idx=node_to_idx,
        out_adj=out_adj, out_wsum=out_wsum, out_deg=out_deg,
        edge_set=edge_set
    )

def make_support_graph(G: IndexedDiGraph) -> Tuple[List[List[Tuple[int, float]]], np.ndarray]:
    n = len(G.out_adj)
    out_adj = [[] for _ in range(n)]
    out_deg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbrs = G.out_adj[i]
        if nbrs:
            out_adj[i] = [(j, 1.0) for (j, _) in nbrs]
            out_deg[i] = float(len(nbrs))
    return out_adj, out_deg

def pagerank_power(out_adj: List[List[Tuple[int, float]]],
                   out_rowsum: np.ndarray,
                   alpha: float = 0.85,
                   tol: float = 1e-12,
                   max_iter: int = 200,
                   u: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(out_adj)
    if u is None:
        u = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        u = np.asarray(u, dtype=np.float64)
        u = u / u.sum()

    p = u.copy()
    dangling = (out_rowsum <= 0)

    for _ in range(max_iter):
        new = (1.0 - alpha) * u.copy()

        # dangling mass redistributed according to u
        dm = p[dangling].sum()
        if dm > 0:
            new += alpha * dm * u

        # push along edges
        for i, nbrs in enumerate(out_adj):
            rs = out_rowsum[i]
            if rs > 0 and nbrs:
                coeff = alpha * p[i] / rs
                for j, w in nbrs:
                    new[j] += coeff * w

        if np.linalg.norm(new - p, ord=1) <= tol:
            return new
        p = new

    return p

G_base = build_indexed_digraph(edges_base, nodes)
G_hc   = build_indexed_digraph(edges_hc, nodes)

print("Graphs built. n =", nodes.size, "|E_base| =", len(edges_base), "|E_hc| =", len(edges_hc))


# In[45]:


# PR on base support
PR_adj, PR_rowsum = make_support_graph(G_base)
p_pr = pagerank_power(PR_adj, PR_rowsum, alpha=ALPHA, tol=TOL, max_iter=MAX_ITER)

# WPR on base weights
p_wpr = pagerank_power(G_base.out_adj, G_base.out_wsum, alpha=ALPHA, tol=TOL, max_iter=MAX_ITER)

# thWPR on high-confidence weights
p_thwpr = pagerank_power(G_hc.out_adj, G_hc.out_wsum, alpha=ALPHA, tol=TOL, max_iter=MAX_ITER)

print("PR/WPR/thWPR computed.")
print("Sanity sums:", p_pr.sum(), p_wpr.sum(), p_thwpr.sum())


# In[47]:


def compute_mu_fixB(df_edges: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    node_to_idx = {int(v): i for i, v in enumerate(nodes)}
    n = len(nodes)
    max_out = np.zeros(n, dtype=np.float64)
    max_in  = np.zeros(n, dtype=np.float64)

    for row in df_edges.itertuples(index=False):
        u0, v0, w = int(row.src), int(row.dst), float(row.w)
        if u0 not in node_to_idx or v0 not in node_to_idx:
            continue
        u = node_to_idx[u0]
        v = node_to_idx[v0]
        if w > max_out[u]:
            max_out[u] = w
        if w > max_in[v]:
            max_in[v] = w

    mu = np.maximum(max_in, max_out)
    return mu


mu = compute_mu_fixB(edges_hc, nodes)
print("mu stats:", pd.Series(mu).describe())


# In[49]:


from typing import List, Tuple, Optional
import numpy as np
import heapq
import time

def widest_path_caps(
    out_adj: List[List[Tuple[int, float]]],
    src: int,
    targets: Optional[List[int]] = None,
    eps: float = 1e-15
) -> np.ndarray:
    """
    caps[v] = Con_F(src, v) under max–min path semantics.
    Early stop if targets provided: stop once all targets are settled.
    """
    n = len(out_adj)
    caps = np.zeros(n, dtype=np.float64)
    caps[src] = 1.0

    visited = np.zeros(n, dtype=bool)
    heap = [(-1.0, src)]  # max-heap via negative

    remaining = None
    if targets is not None:
        remaining = set(int(t) for t in targets)
        remaining.discard(src)

    while heap:
        negc, x = heapq.heappop(heap)
        c = -negc
        if visited[x]:
            continue
        visited[x] = True

        if remaining is not None and x in remaining:
            remaining.remove(x)
            if not remaining:
                break

        for y, w in out_adj[x]:
            cand = min(c, float(w))
            if cand > caps[y] + eps:
                caps[y] = cand
                heapq.heappush(heap, (-cand, y))

    return caps


def compute_strong_edges(
    out_adj: List[List[Tuple[int, float]]],
    tol: float = 1e-12,
    early_stop: bool = True
) -> np.ndarray:
    """
    Returns array rows [u, v, w] where (u,v) is strong iff w == Con_F(u,v).
    Only needs Con_F(u,v) for v in out-neighbors of u; early-stop after those are settled.
    """
    n = len(out_adj)
    strong = []

    t0 = time.time()
    for u in range(n):
        nbrs = [v for v, _ in out_adj[u]]
        if not nbrs:
            continue

        caps = widest_path_caps(out_adj, u, targets=nbrs if early_stop else None)

        for v, w in out_adj[u]:
            if abs(float(w) - caps[v]) <= tol:
                strong.append((u, v, float(w)))

        if (u + 1) % 100 == 0:
            print(f"  processed {u+1}/{n} sources...")

    strong = np.array(strong, dtype=np.float64)
    print("Strong edges computed:", strong.shape, "time:", round(time.time() - t0, 2), "sec")
    return strong


# WARNING: this can be the slowest step; increase HC_QUANTILE if needed.
strong_edges = compute_strong_edges(G_hc.out_adj, tol=1e-12, early_stop=True)


# In[50]:


def threshold_adj(out_adj: List[List[Tuple[int, float]]], sigma: float) -> List[List[Tuple[int, float]]]:
    return [[(v, w) for (v, w) in nbrs if w >= sigma] for nbrs in out_adj]

def scc_kosaraju(out_adj: List[List[Tuple[int, float]]]) -> Tuple[np.ndarray, List[List[int]]]:
    n = len(out_adj)
    rev = [[] for _ in range(n)]
    for u, nbrs in enumerate(out_adj):
        for v, _ in nbrs:
            rev[v].append(u)

    visited = np.zeros(n, dtype=bool)
    order = []

    def dfs1(u: int):
        visited[u] = True
        for v, _ in out_adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for u in range(n):
        if not visited[u]:
            dfs1(u)

    comp_id = np.full(n, -1, dtype=np.int32)
    comps: List[List[int]] = []

    def dfs2(u: int, cid: int):
        comp_id[u] = cid
        comps[cid].append(u)
        for v in rev[u]:
            if comp_id[v] == -1:
                dfs2(v, cid)

    for u in reversed(order):
        if comp_id[u] == -1:
            comps.append([])
            dfs2(u, len(comps) - 1)

    return comp_id, comps

def sppr_vertex_scores(G_sp: IndexedDiGraph,
                       mu: np.ndarray,
                       strong_edges: np.ndarray,
                       sigma: float,
                       alpha: float = 0.85,
                       tol: float = 1e-12,
                       max_iter: int = 200) -> Tuple[np.ndarray, Dict]:
    # 1) SCC classes for thresholded graph at sigma
    out_thr = threshold_adj(G_sp.out_adj, sigma)
    comp_id, comps = scc_kosaraju(out_thr)
    m = len(comps)

    # 2) bar_mu per class
    bar_mu = np.zeros(m, dtype=np.float64)
    for i, vs in enumerate(comps):
        bar_mu[i] = mu[vs].max() if vs else 0.0

    # 3) Lambda^{ssp}_{ij}: max strong-edge weight from class i to class j
    lambda_max: List[Dict[int, float]] = [dict() for _ in range(m)]
    for u, v, w in strong_edges:
        u = int(u); v = int(v); w = float(w)
        i = int(comp_id[u]); j = int(comp_id[v])
        if i == j:
            continue
        prev = lambda_max[i].get(j, 0.0)
        if w > prev:
            lambda_max[i][j] = w

    # 4) Build class adjacency with w_ij = min(bar_mu[i], bar_mu[j], lambda_max[i][j])
    class_adj: List[List[Tuple[int, float]]] = [[] for _ in range(m)]
    class_rowsum = np.zeros(m, dtype=np.float64)
    for i in range(m):
        for j, lam in lambda_max[i].items():
            wij = min(bar_mu[i], bar_mu[j], lam)
            if wij > 0:
                class_adj[i].append((j, wij))
                class_rowsum[i] += wij

    # 5) PageRank on classes
    p_class = pagerank_power(class_adj, class_rowsum, alpha=alpha, tol=tol, max_iter=max_iter)

    # 6) Lift to vertex fuzzy scores: fPR(v)=mu(v)*p_class[class(v)]
    p_vertex = mu * p_class[comp_id.astype(np.int32)]

    meta = {
        "sigma": float(sigma),
        "m_classes": int(m),
        "class_sizes": np.array([len(c) for c in comps], dtype=np.int32),
        "bar_mu": bar_mu,
        "p_class": p_class,
        "comp_id": comp_id,
    }
    return p_vertex, meta


# In[51]:


hc_weights = edges_hc["w"].to_numpy(dtype=np.float64)
hc_weights = hc_weights[hc_weights > 0]

if hc_weights.size == 0:
    raise RuntimeError("No HC weights > 0; decrease HC_QUANTILE or check aggregation.")

if SIGMA_GRID_MODE == "quantiles":
    qs = np.linspace(0.10, 0.95, SIGMA_POINTS)
    sigmas = np.quantile(hc_weights, qs)
    sigmas = np.unique(np.clip(sigmas, 1e-6, 1.0))
elif SIGMA_GRID_MODE == "linspace":
    sigmas = np.linspace(hc_weights.min(), hc_weights.max(), SIGMA_POINTS)
    sigmas = np.unique(np.clip(sigmas, 1e-6, 1.0))
else:
    raise ValueError("Unknown SIGMA_GRID_MODE")

sigmas = np.sort(sigmas)
print("Sigmas:", sigmas)

sppr_runs = []
for s in sigmas:
    t0 = time.time()
    scores, meta = sppr_vertex_scores(G_hc, mu, strong_edges, sigma=float(s), alpha=ALPHA, tol=TOL, max_iter=MAX_ITER)
    meta["runtime_sec"] = time.time() - t0
    sppr_runs.append((scores, meta))
    print(f"sigma={s:.4f}  m={meta['m_classes']}  time={meta['runtime_sec']:.2f}s")


# In[53]:


def future_incoming_volume(df_test: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    node_to_idx = {int(v): i for i, v in enumerate(nodes)}
    y = np.zeros(len(nodes), dtype=np.float64)

    counts = df_test.groupby("dst").size()
    for dst, c in counts.items():
        dst = int(dst)
        if dst in node_to_idx:
            y[node_to_idx[dst]] = float(c)

    return y


y = future_incoming_volume(test_events, nodes)
print("Target y stats:", pd.Series(y).describe())


# In[54]:


def rankdata_desc(x: np.ndarray) -> np.ndarray:
    # rank 1 = largest; ties get average rank
    order = np.argsort(-x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    # average ties
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks

def spearman_manual(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata_desc(a)
    rb = rankdata_desc(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)

def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = int(min(k, len(a)))
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / float(k)

def eval_scores(name: str, scores: np.ndarray, y: np.ndarray, ks=(10, 25, 50, 100)) -> Dict:
    out = {"method": name}
    if SCIPY_OK:
        out["spearman"] = float(spearmanr(scores, y).correlation)
        out["kendall"]  = float(kendalltau(scores, y).correlation)
    else:
        out["spearman"] = spearman_manual(scores, y)
        out["kendall"]  = float("nan")

    for k in ks:
        out[f"overlap@{k}"] = overlap_at_k(scores, y, k)
    return out

rows = []
rows.append(eval_scores("PR", p_pr, y))
rows.append(eval_scores("WPR", p_wpr, y))
rows.append(eval_scores("thWPR", p_thwpr, y))

for scores, meta in sppr_runs:
    rows.append(eval_scores(f"SPPR sigma={meta['sigma']:.4f} (m={meta['m_classes']})", scores, y))

results = pd.DataFrame(rows)
results


# In[61]:


import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

sig_list = []
m_list = []
spearman_list = []

for scores, meta in sppr_runs:
    sig_list.append(float(meta["sigma"]))
    m_list.append(int(meta["m_classes"]))
    if SCIPY_OK:
        spearman_list.append(float(spearmanr(scores, y).correlation))
    else:
        spearman_list.append(spearman_manual(scores, y))

# Sort by sigma (defensive)
order = np.argsort(sig_list)
sig_arr = np.array(sig_list)[order]
m_arr = np.array(m_list)[order]
sp_arr = np.array(spearman_list)[order]

# --- Figure 1: m vs sigma ---
fig1 = plt.figure()
plt.plot(sig_arr, m_arr, marker="o")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"Quotient size $m$ (# SCC classes)")
plt.title(r"Multiscale SCC quotient size vs $\sigma$")
plt.tight_layout()

path1 = os.path.join(FIG_DIR, "mail_eu_m_vs_sigma.png")
fig1.savefig(path1, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig1)

# --- Figure 2: Spearman vs sigma ---
fig2 = plt.figure()
plt.plot(sig_arr, sp_arr, marker="o")
plt.xlabel(r"$\sigma$")
plt.ylabel("Spearman(score, future incoming volume)")
plt.title(r"Predictive correlation vs $\sigma$")
plt.tight_layout()

path2 = os.path.join(FIG_DIR, "mail_eu_spearman_vs_sigma.png")
fig2.savefig(path2, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig2)

print("Saved figures:")
print(" -", path1)
print(" -", path2)


# In[56]:


results_path = os.path.join(DATA_DIR, "ranking_results.csv")
results.to_csv(results_path, index=False)
print("Saved:", results_path)

# Also capture the m(sigma) curve
ms_df = pd.DataFrame({"sigma": sig_list, "m_classes": m_list, "spearman": spearman_list})
ms_path = os.path.join(DATA_DIR, "sigma_multiscale_curve.csv")
ms_df.to_csv(ms_path, index=False)
print("Saved:", ms_path)

# LaTeX table snippet (copy into paper)
print(results.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))


# In[64]:


# - Dense sampling in (0.50, 0.61)
# - Additional high-sigma values above 0.61 (upper quantiles + linear ramp)
# Produces a compact results table with Spearman + overlaps and exports LaTeX.

import numpy as np
import pandas as pd

# --- prerequisites ---
_required = ["edges_hc", "G_hc", "mu", "strong_edges", "y",
             "sppr_vertex_scores", "p_pr", "p_wpr", "p_thwpr",
             "ALPHA", "TOL", "MAX_ITER", "SCIPY_OK"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise NameError(f"Missing variables: {_missing}. Run earlier cells first.")

# ---- correlation helpers (use your existing spearman_manual if SciPy absent) ----
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if SCIPY_OK and spearmanr is not None:
        return float(spearmanr(a, b).correlation)
    return spearman_manual(a, b)

def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = int(min(k, len(a)))
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / float(k)

def eval_row(method: str, scores: np.ndarray, y: np.ndarray, ks=(10, 50, 100)) -> dict:
    d = {"method": method, "rho": spearman_corr(scores, y)}
    for k in ks:
        d[f"overlap@{k}"] = overlap_at_k(scores, y, k)
    return d

# ----------------------------
# 1) Build expanded sigma grid
# ----------------------------
hc_w = edges_hc["w"].to_numpy(dtype=np.float64)
hc_w = hc_w[hc_w > 0]
if hc_w.size == 0:
    raise RuntimeError("No HC weights > 0. Check edges_hc / preprocessing.")

# Existing sigmas you already reported (keep them)
sig_existing = np.array([0.2626, 0.2774, 0.3028, 0.3239, 0.3577, 0.3959, 0.4415, 0.4989, 0.6117], dtype=np.float64)

# Dense between 0.50 and 0.61 (excluding endpoints because they are already included)
sig_dense = np.linspace(0.50, 0.61, 13, dtype=np.float64)  # 0.50, 0.508..., ..., 0.61
# High-sigma candidates above 0.61 (quantiles + a short linear ramp)
q_hi = np.array([0.955, 0.965, 0.975, 0.985, 0.990, 0.992, 0.995, 0.997, 0.999], dtype=np.float64)
sig_hi_q = np.quantile(hc_w, q_hi)

sig_hi_lin = np.linspace(0.62, min(0.99, float(hc_w.max())), 8, dtype=np.float64)

# Combine and clean
sigmas = np.unique(np.concatenate([sig_existing, sig_dense, sig_hi_q, sig_hi_lin]))
sigmas = np.clip(sigmas, 1e-6, 1.0)
sigmas = np.unique(np.round(sigmas, 6))  # stabilize printing
sigmas.sort()

print(f"Expanded sigma grid: {len(sigmas)} values")
print(sigmas)

# -----------------------------------
# 2) Run SPPR for all expanded sigmas
# -----------------------------------
sppr_scores = []   # list of (sigma, m, scores)
for s in sigmas:
    scores, meta = sppr_vertex_scores(G_hc, mu, strong_edges, sigma=float(s), alpha=ALPHA, tol=TOL, max_iter=MAX_ITER)
    sppr_scores.append((float(s), int(meta["m_classes"]), scores))

# ---------------------------------------
# 3) Build the replacement results table
# ---------------------------------------
rows = []
rows.append(eval_row("PR", p_pr, y))
rows.append(eval_row("WPR", p_wpr, y))
rows.append(eval_row("thWPR", p_thwpr, y))

for s, m, scores in sppr_scores:
    rows.append(eval_row(f"SPPR({s:.4f},{m})", scores, y))

results_ext = pd.DataFrame(rows)

# Keep baselines at top, then SPPR sorted by sigma
mask_sp = results_ext["method"].str.startswith("SPPR")
results_ext = pd.concat(
    [results_ext[~mask_sp], results_ext[mask_sp]],
    ignore_index=True
)

display(results_ext)

# ---------------------------------------
# 4) Convenience: show "best" SPPR rows
# ---------------------------------------
sp_only = results_ext[mask_sp].copy()
if not sp_only.empty:
    best_rho = sp_only["rho"].astype(float).idxmax()
    best_o10 = sp_only["overlap@10"].astype(float).idxmax()
    best_o50 = sp_only["overlap@50"].astype(float).idxmax()
    best_o100 = sp_only["overlap@100"].astype(float).idxmax()

    print("\nBest SPPR by rho:")
    display(results_ext.loc[[best_rho]])
    print("\nBest SPPR by overlap@10:")
    display(results_ext.loc[[best_o10]])
    print("\nBest SPPR by overlap@50:")
    display(results_ext.loc[[best_o50]])
    print("\nBest SPPR by overlap@100:")
    display(results_ext.loc[[best_o100]])

# ---------------------------------------
# 5) Export LaTeX table for the paper
# ---------------------------------------
latex = results_ext.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
print("\nLaTeX table:\n")
print(latex)


# In[65]:


# Plot SPPR overlap@100 vs sigma and save to: figures/mail_eu_overlap100_vs_sigma.png
#
# This cell is robust: it will work if you have either
#   - sppr_runs_dense / sppr_runs (list of (scores, meta)), and y
# or
#   - a pandas DataFrame named results_ext/results_dense/results with columns method + overlap@100.
#
# It will recompute overlaps from the raw SPPR scores if needed.

import os
import re
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
OUT_PATH = "figures/mail_eu_overlap100_vs_sigma.png"

def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = int(min(k, len(a)))
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / float(k)

# ---------- Path A: prefer raw SPPR runs (most reliable) ----------
sppr_source = None
for name in ["sppr_runs_dense", "sppr_runs"]:
    if name in globals():
        sppr_source = globals()[name]
        break

if sppr_source is not None:
    if "y" not in globals():
        raise NameError("Found sppr runs but missing y. Run the evaluation window target cell first.")

    sig_list, ov100_list = [], []
    for scores, meta in sppr_source:
        sig_list.append(float(meta["sigma"]))
        ov100_list.append(overlap_at_k(np.asarray(scores), np.asarray(y), 100))

    order = np.argsort(sig_list)
    sig = np.asarray(sig_list)[order]
    ov100 = np.asarray(ov100_list)[order]

else:
    # ---------- Path B: fall back to a results DataFrame ----------
    import pandas as pd

    df = None
    for name in ["results_ext", "results_dense", "results"]:
        if name in globals():
            df = globals()[name]
            break

    if df is None:
        raise NameError(
            "Could not find sppr_runs_dense/sppr_runs OR results_ext/results_dense/results.\n"
            "Run the cell that computes SPPR runs and/or builds the results DataFrame first."
        )

    if "method" not in df.columns:
        raise KeyError("Results DataFrame has no 'method' column.")

    sp = df[df["method"].astype(str).str.startswith("SPPR")].copy()
    if sp.empty:
        raise ValueError("No SPPR rows found in the results DataFrame.")

    def parse_sigma(method_str: str) -> float:
        s = str(method_str)
        if "sigma=" in s:  # "SPPR sigma=0.2626 (m=...)"
            return float(s.split("sigma=")[1].split()[0])
        if s.startswith("SPPR("):  # "SPPR(0.2626,387)"
            inner = s[s.find("(") + 1 : s.find(")")]
            return float(inner.split(",")[0])
        m = re.search(r"([0-9]*\.[0-9]+)", s)
        return float(m.group(1)) if m else float("nan")

    sp["sigma"] = sp["method"].apply(parse_sigma)

    # Find overlap@100 column
    col_ov100 = None
    for c in sp.columns:
        if str(c).strip().lower() == "overlap@100":
            col_ov100 = c
            break
    if col_ov100 is None:
        candidates = [c for c in sp.columns if "overlap" in str(c).lower() and "100" in str(c)]
        if not candidates:
            raise KeyError("Could not find overlap@100 in the results DataFrame.")
        col_ov100 = candidates[0]

    sp = sp.dropna(subset=["sigma"]).sort_values("sigma")
    sig = sp["sigma"].to_numpy(dtype=float)
    ov100 = sp[col_ov100].to_numpy(dtype=float)

# ---------- Plot ----------
plt.figure()
plt.plot(sig, ov100, marker="o")
plt.xlabel(r"$\sigma$")
plt.ylabel("overlap@100")
plt.title(r"SPPR overlap@100 vs $\sigma$")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PATH)


# In[ ]:




