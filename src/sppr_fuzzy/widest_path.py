from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import heapq
import time


def widest_path_caps(
    out_adj: List[List[Tuple[int, float]]],
    src: int,
    *,
    targets: Optional[List[int]] = None,
    eps: float = 1e-15,
) -> np.ndarray:
    """Widest-path (max–min) connectivity from `src`.

    Returns an array `caps` with
        caps[v] = Con_F(src, v) = max_{P: src->v} min_{e in P} w(e)

    If `targets` is provided, the run stops once all target nodes have been settled.
    """
    n = len(out_adj)
    caps = np.zeros(n, dtype=np.float64)
    caps[int(src)] = 1.0

    visited = np.zeros(n, dtype=bool)
    heap = [(-1.0, int(src))]  # max-heap via negative key

    remaining = None
    if targets is not None:
        remaining = set(int(t) for t in targets)
        remaining.discard(int(src))

    while heap:
        negc, x = heapq.heappop(heap)
        c = -float(negc)
        x = int(x)
        if visited[x]:
            continue
        visited[x] = True

        if remaining is not None and x in remaining:
            remaining.remove(x)
            if not remaining:
                break

        for y, w in out_adj[x]:
            y = int(y)
            cand = min(c, float(w))
            if cand > caps[y] + float(eps):
                caps[y] = cand
                heapq.heappush(heap, (-cand, y))

    return caps


def compute_strong_edges(
    out_adj: List[List[Tuple[int, float]]],
    *,
    tol: float = 1e-12,
    early_stop: bool = True,
    progress_every: int = 200,
) -> np.ndarray:
    """Compute strong edges (u,v) satisfying w(u,v) == Con_F(u,v).

    The routine computes widest-path connectivity from each source u.
    If `early_stop` is True, it only needs to settle the out-neighbors of u,
    which can reduce runtime on sparse graphs.

    Returns
    -------
    strong_edges: np.ndarray
        Array of rows [u, v, w] (dtype float64).
    """
    n = len(out_adj)
    strong = []
    t0 = time.time()

    for u in range(n):
        nbrs = [v for (v, _) in out_adj[u]]
        if not nbrs:
            continue

        caps = widest_path_caps(out_adj, u, targets=nbrs if early_stop else None)

        for v, w in out_adj[u]:
            if abs(float(w) - float(caps[int(v)])) <= float(tol):
                strong.append((float(u), float(v), float(w)))

        if progress_every and (u + 1) % int(progress_every) == 0:
            elapsed = time.time() - t0
            print(f"  strong-edge sources processed: {u+1}/{n} (elapsed {elapsed:.1f}s)")

    strong_edges = np.asarray(strong, dtype=np.float64)
    print(f"Strong edges: {strong_edges.shape[0]} (computed in {time.time() - t0:.2f}s)")
    return strong_edges
