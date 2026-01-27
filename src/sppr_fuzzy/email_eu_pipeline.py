from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable
from pathlib import Path
import os
import urllib.request
import numpy as np
import pandas as pd

from .pagerank import pagerank_power
from .widest_path import compute_strong_edges
from .scc import scc_kosaraju

# Optional: use SciPy correlations when available
try:
    from scipy.stats import spearmanr  # type: ignore
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


SNAP_CORE_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
SNAP_DEPT_URLS = {
    "dept1": "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept1.txt.gz",
    "dept2": "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept2.txt.gz",
    "dept3": "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept3.txt.gz",
    "dept4": "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept4.txt.gz",
}


@dataclass
class IndexedDiGraph:
    nodes: np.ndarray
    node_to_idx: Dict[int, int]
    out_adj: List[List[Tuple[int, float]]]
    out_wsum: np.ndarray
    out_deg: np.ndarray


def _download_if_missing(url: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, local_path.as_posix())
    print(f"Saved: {local_path}")


def load_snap_email_events(gz_path: Path) -> pd.DataFrame:
    """Load SNAP email temporal events (SRC DST TIME)."""
    events = pd.read_csv(
        gz_path.as_posix(),
        sep=r"\s+",
        comment="#",
        header=None,
        names=["src", "dst", "time"],
        compression="gzip",
        dtype={"src": np.int32, "dst": np.int32, "time": np.int64},
    )
    return events


def pick_time_cutoffs(events: pd.DataFrame, train_frac: float, eval_frac: float) -> Tuple[int, int]:
    t = events["time"].to_numpy()
    tau = int(np.quantile(t, float(train_frac)))
    tau_prime = int(np.quantile(t, float(eval_frac)))
    if tau_prime <= tau:
        tau_prime = tau + 1
    return tau, tau_prime


def scale_to_unit_interval(raw: np.ndarray, method: str = "log") -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size == 0:
        return raw
    mx = float(raw.max())
    if mx <= 0:
        return np.zeros_like(raw)

    method = str(method).lower()
    if method == "log":
        return np.log1p(raw) / np.log1p(mx)
    if method == "minmax":
        return raw / mx
    if method == "cauchy":
        pos = raw[raw > 0]
        tau = float(np.median(pos)) if pos.size else 1.0
        return raw / (raw + tau)
    raise ValueError(f"Unknown scale method: {method}")


def aggregate_email_edges(
    events: pd.DataFrame,
    *,
    end_time: Optional[int] = None,
    method: str = "count",
    half_life_days: float = 60.0,
    scale: str = "log",
) -> pd.DataFrame:
    """Aggregate temporal events to a weighted directed edge list.

    Output columns: src, dst, w (scaled membership in [0,1]), raw (unscaled intensity).
    """
    method = str(method).lower()
    if method == "count":
        grp = events.groupby(["src", "dst"]).size().reset_index(name="raw")
        raw = grp["raw"].to_numpy(dtype=np.float64)

    elif method == "decay":
        if end_time is None:
            raise ValueError("end_time required for method='decay'")
        half_life_sec = float(half_life_days) * 86400.0
        dt = (float(end_time) - events["time"].to_numpy(dtype=np.float64))
        contrib = np.exp(-dt / half_life_sec)
        tmp = events[["src", "dst"]].copy()
        tmp["raw"] = contrib
        grp = tmp.groupby(["src", "dst"])["raw"].sum().reset_index(name="raw")
        raw = grp["raw"].to_numpy(dtype=np.float64)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    w = scale_to_unit_interval(raw, method=scale)
    out = grp.copy()
    out["w"] = w
    return out[["src", "dst", "w", "raw"]]


def build_indexed_digraph(edges: pd.DataFrame, nodes: np.ndarray) -> IndexedDiGraph:
    node_to_idx = {int(v): i for i, v in enumerate(nodes)}
    n = int(len(nodes))
    out_adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    out_wsum = np.zeros(n, dtype=np.float64)
    out_deg = np.zeros(n, dtype=np.float64)

    for row in edges.itertuples(index=False):
        u0, v0, w = int(row.src), int(row.dst), float(row.w)
        if u0 not in node_to_idx or v0 not in node_to_idx:
            continue
        u = node_to_idx[u0]
        v = node_to_idx[v0]
        out_adj[u].append((v, w))
        out_wsum[u] += w
        out_deg[u] += 1.0

    return IndexedDiGraph(nodes=nodes, node_to_idx=node_to_idx, out_adj=out_adj, out_wsum=out_wsum, out_deg=out_deg)


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


def compute_mu_from_edges(edges: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    """Vertex membership mu(v) := max{ max_out(v), max_in(v) } from the given edge list."""
    node_to_idx = {int(v): i for i, v in enumerate(nodes)}
    n = len(nodes)
    max_out = np.zeros(n, dtype=np.float64)
    max_in = np.zeros(n, dtype=np.float64)

    for row in edges.itertuples(index=False):
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


def threshold_adj(out_adj: List[List[Tuple[int, float]]], sigma: float) -> List[List[Tuple[int, float]]]:
    s = float(sigma)
    return [[(v, w) for (v, w) in nbrs if float(w) >= s] for nbrs in out_adj]


def sppr_vertex_scores(
    G: IndexedDiGraph,
    *,
    mu: np.ndarray,
    strong_edges: np.ndarray,
    sigma: float,
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """Compute SPPR vertex scores at threshold sigma on a weighted digraph.

    Returns
    -------
    scores: np.ndarray
        Vertex-level scores fPR(v)=mu(v)*p_class[class(v)].
    meta: dict
        Includes sigma, m_classes, class sizes, and class PageRank vector.
    """
    # 1) SCC classes for the sigma-cut graph (computed from original weights)
    out_thr = threshold_adj(G.out_adj, sigma)
    comp_id, comps = scc_kosaraju(out_thr)
    m = len(comps)

    # 2) bar_mu per class
    bar_mu = np.zeros(m, dtype=np.float64)
    for i, vs in enumerate(comps):
        bar_mu[i] = float(mu[vs].max()) if vs else 0.0

    # 3) Lambda^{ssp}_{ij}: max strong-edge weight from class i to class j
    # store as per-row dict for sparsity
    lambda_max: List[Dict[int, float]] = [dict() for _ in range(m)]
    for u, v, w in strong_edges:
        u_i = int(u); v_i = int(v); w_f = float(w)
        i = int(comp_id[u_i]); j = int(comp_id[v_i])
        if i == j:
            continue
        prev = float(lambda_max[i].get(j, 0.0))
        if w_f > prev:
            lambda_max[i][j] = w_f

    # 4) Build class adjacency with w_ij = min(bar_mu[i], bar_mu[j], lambda_max[i][j])
    class_adj: List[List[Tuple[int, float]]] = [[] for _ in range(m)]
    class_rowsum = np.zeros(m, dtype=np.float64)
    for i in range(m):
        for j, lam in lambda_max[i].items():
            wij = min(float(bar_mu[i]), float(bar_mu[j]), float(lam))
            if wij > 0:
                class_adj[i].append((int(j), float(wij)))
                class_rowsum[i] += float(wij)

    # 5) PageRank on classes
    p_class = pagerank_power(class_adj, class_rowsum, alpha=alpha, tol=tol, max_iter=max_iter)

    # 6) Lift to vertices: fPR(v) = mu(v) * p_class[class(v)]
    scores = np.asarray(mu, dtype=np.float64) * p_class[comp_id.astype(np.int32)]

    meta = {
        "sigma": float(sigma),
        "m_classes": int(m),
        "class_sizes": np.array([len(c) for c in comps], dtype=np.int32),
        "bar_mu": bar_mu,
        "p_class": p_class,
        "comp_id": comp_id,
    }
    return scores, meta


def future_incoming_volume(test_events: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    node_to_idx = {int(v): i for i, v in enumerate(nodes)}
    y = np.zeros(len(nodes), dtype=np.float64)
    counts = test_events.groupby("dst").size()
    for dst, c in counts.items():
        dst = int(dst)
        if dst in node_to_idx:
            y[node_to_idx[dst]] = float(c)
    return y


def _rankdata_desc(x: np.ndarray) -> np.ndarray:
    # rank 1 = largest; ties get average rank
    x = np.asarray(x)
    order = np.argsort(-x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if SCIPY_OK:
        return float(spearmanr(a, b).correlation)
    ra = _rankdata_desc(a)
    rb = _rankdata_desc(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)


def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = int(min(int(k), len(a)))
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return float(len(ta & tb)) / float(k)


def choose_sigma_grid(hc_weights: np.ndarray, mode: str, points: int) -> np.ndarray:
    w = np.asarray(hc_weights, dtype=np.float64)
    w = w[w > 0]
    if w.size == 0:
        raise ValueError("No positive HC weights.")
    mode = str(mode).lower()
    points = int(points)

    if mode == "quantiles":
        qs = np.linspace(0.10, 0.95, points)
        sigmas = np.quantile(w, qs)
    elif mode == "linspace":
        sigmas = np.linspace(float(w.min()), float(w.max()), points)
    else:
        raise ValueError("sigma grid mode must be 'quantiles' or 'linspace'.")

    sigmas = np.unique(np.clip(sigmas, 1e-6, 1.0))
    return np.sort(sigmas)


def run_email_eu_experiment(
    *,
    dataset: str = "core",
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    train_frac: float = 0.80,
    eval_frac: float = 0.90,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-12,
    agg_method: str = "count",
    half_life_days: float = 60.0,
    scale_method: str = "log",
    lambda_min: float = 0.0,
    hc_quantile: float = 0.70,
    sigma_grid: str = "quantiles",
    sigma_points: int = 9,
    strong_tol: float = 1e-12,
    early_stop: bool = True,
) -> pd.DataFrame:
    """End-to-end reproduction of the email-Eu experiment."""
    data_dir_p = Path(data_dir)
    outputs_dir_p = Path(outputs_dir)
    (outputs_dir_p / "figures").mkdir(parents=True, exist_ok=True)

    dataset_key = str(dataset).lower()
    if dataset_key == "core":
        url = SNAP_CORE_URL
        filename = "email-Eu-core-temporal.txt.gz"
    else:
        if dataset_key not in SNAP_DEPT_URLS:
            raise ValueError("dataset must be 'core' or one of dept1..dept4")
        url = SNAP_DEPT_URLS[dataset_key]
        filename = f"email-Eu-core-temporal-{dataset_key.upper()}.txt.gz"

    gz_path = data_dir_p / filename
    _download_if_missing(url, gz_path)
    events = load_snap_email_events(gz_path)

    tau, tau_prime = pick_time_cutoffs(events, float(train_frac), float(eval_frac))
    train_events = events[events["time"] <= tau].copy()
    test_events = events[(events["time"] > tau) & (events["time"] <= tau_prime)].copy()

    edges_train = aggregate_email_edges(
        train_events,
        end_time=tau,
        method=agg_method,
        half_life_days=half_life_days,
        scale=scale_method,
    )
    edges_base = edges_train[edges_train["w"] >= float(lambda_min)].copy()

    nodes = np.union1d(edges_base["src"].unique(), edges_base["dst"].unique()).astype(np.int32)
    nodes.sort()

    # high-confidence edges by quantile cutoff of base weights
    w_base = edges_base["w"].to_numpy(dtype=np.float64)
    wmin = float(np.quantile(w_base, float(hc_quantile))) if w_base.size else 1.0
    edges_hc = edges_base[edges_base["w"] >= wmin].copy()

    G_base = build_indexed_digraph(edges_base, nodes)
    G_hc = build_indexed_digraph(edges_hc, nodes)

    # Baselines
    PR_adj, PR_rowsum = make_support_graph(G_base)
    p_pr = pagerank_power(PR_adj, PR_rowsum, alpha=alpha, tol=tol, max_iter=max_iter)
    p_wpr = pagerank_power(G_base.out_adj, G_base.out_wsum, alpha=alpha, tol=tol, max_iter=max_iter)
    p_thwpr = pagerank_power(G_hc.out_adj, G_hc.out_wsum, alpha=alpha, tol=tol, max_iter=max_iter)

    # mu from HC edges
    mu = compute_mu_from_edges(edges_hc, nodes)

    # strong edges on HC graph
    strong_edges = compute_strong_edges(G_hc.out_adj, tol=strong_tol, early_stop=early_stop)

    # evaluation target: future incoming volume
    y = future_incoming_volume(test_events, nodes)

    # sigma grid
    hc_w = edges_hc["w"].to_numpy(dtype=np.float64)
    sigmas = choose_sigma_grid(hc_w, mode=sigma_grid, points=sigma_points)

    # run SPPR per sigma
    sppr_runs: List[Tuple[np.ndarray, Dict]] = []
    for s in sigmas:
        scores, meta = sppr_vertex_scores(
            G_hc,
            mu=mu,
            strong_edges=strong_edges,
            sigma=float(s),
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )
        sppr_runs.append((scores, meta))

    # collect results table
    rows: List[Dict] = []
    def add_row(name: str, scores: np.ndarray) -> None:
        rows.append({
            "method": name,
            "rho": spearman_corr(scores, y),
            "overlap@10": overlap_at_k(scores, y, 10),
            "overlap@50": overlap_at_k(scores, y, 50),
            "overlap@100": overlap_at_k(scores, y, 100),
        })

    add_row("PR", p_pr)
    add_row("WPR", p_wpr)
    add_row("thWPR", p_thwpr)
    for scores, meta in sppr_runs:
        add_row(f"SPPR({meta['sigma']:.4f},{meta['m_classes']})", scores)

    results = pd.DataFrame(rows)

    # export CSV
    results.to_csv(outputs_dir_p / "ranking_results.csv", index=False)

    # export sigma curve
    sig_list = [float(meta["sigma"]) for _, meta in sppr_runs]
    m_list = [int(meta["m_classes"]) for _, meta in sppr_runs]
    rho_list = [spearman_corr(scores, y) for scores, _ in sppr_runs]
    curve = pd.DataFrame({"sigma": sig_list, "m_classes": m_list, "spearman": rho_list})
    curve.to_csv(outputs_dir_p / "sigma_multiscale_curve.csv", index=False)

    return results
