from __future__ import annotations

from typing import List, Tuple
import numpy as np


def scc_kosaraju(out_adj: List[List[Tuple[int, float]]]) -> Tuple[np.ndarray, List[List[int]]]:
    """Strongly connected components via Kosaraju (iterative).

    Parameters
    ----------
    out_adj:
        adjacency list with weighted edges (weights ignored for SCC).

    Returns
    -------
    comp_id:
        np.ndarray of length n mapping node -> component index in [0, m-1].
    comps:
        list of components; comps[c] is list of nodes in component c.
    """
    n = len(out_adj)
    rev: List[List[int]] = [[] for _ in range(n)]
    for u, nbrs in enumerate(out_adj):
        for v, _ in nbrs:
            rev[int(v)].append(int(u))

    visited = np.zeros(n, dtype=bool)
    order: List[int] = []

    # first pass: compute finishing order
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            u, idx = stack[-1]
            nbrs = out_adj[u]
            if idx < len(nbrs):
                v = int(nbrs[idx][0])
                stack[-1] = (u, idx + 1)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                stack.pop()
                order.append(u)

    # second pass on reversed graph
    comp_id = np.full(n, -1, dtype=np.int32)
    comps: List[List[int]] = []

    for start in reversed(order):
        if comp_id[start] != -1:
            continue
        cid = len(comps)
        comps.append([])
        stack = [start]
        comp_id[start] = cid
        while stack:
            u = stack.pop()
            comps[cid].append(u)
            for v in rev[u]:
                if comp_id[v] == -1:
                    comp_id[v] = cid
                    stack.append(v)

    return comp_id, comps
