from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np


def pagerank_power(
    out_adj: List[List[Tuple[int, float]]],
    out_rowsum: np.ndarray,
    *,
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 200,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute PageRank with row-stochastic convention via power iteration.

    Parameters
    ----------
    out_adj:
        Adjacency list. Each entry out_adj[i] is a list of (j, weight_ij).
    out_rowsum:
        Row sums of weights for each node i (sum_j weight_ij).
    alpha:
        Damping factor in (0,1).
    tol:
        L1 convergence tolerance.
    max_iter:
        Maximum number of iterations.
    u:
        Teleportation distribution (length n). If None, uniform.

    Returns
    -------
    p: np.ndarray
        Stationary distribution (length n), sum(p)=1.
    """
    n = len(out_adj)
    if n == 0:
        return np.array([], dtype=np.float64)

    if u is None:
        u = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        u = np.asarray(u, dtype=np.float64)
        s = float(u.sum())
        if s <= 0:
            raise ValueError("Teleportation vector u must have positive sum.")
        u = u / s

    p = u.copy()
    out_rowsum = np.asarray(out_rowsum, dtype=np.float64)
    dangling = (out_rowsum <= 0)

    for _ in range(int(max_iter)):
        new = (1.0 - float(alpha)) * u.copy()

        # dangling mass redistributed according to u
        dm = float(p[dangling].sum())
        if dm > 0:
            new += float(alpha) * dm * u

        # push along edges
        for i, nbrs in enumerate(out_adj):
            rs = float(out_rowsum[i])
            if rs > 0 and nbrs:
                coeff = float(alpha) * float(p[i]) / rs
                for j, w in nbrs:
                    new[int(j)] += coeff * float(w)

        if float(np.linalg.norm(new - p, ord=1)) <= float(tol):
            return new
        p = new

    return p
