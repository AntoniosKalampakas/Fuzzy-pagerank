"""SPPR on directed fuzzy graphs.

This package provides a minimal implementation of:
- widest-path (max–min) connectivity,
- strong-edge detection,
- SCC-based quotient construction at threshold sigma,
- strong-path PageRank (SPPR) on quotient classes and lift back to vertices.
"""

from .pagerank import pagerank_power
from .widest_path import widest_path_caps, compute_strong_edges
from .scc import scc_kosaraju
from .email_eu_pipeline import run_email_eu_experiment

__all__ = [
    "pagerank_power",
    "widest_path_caps",
    "compute_strong_edges",
    "scc_kosaraju",
    "run_email_eu_experiment",
]
