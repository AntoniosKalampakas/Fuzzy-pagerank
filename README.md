# Strong-Path PageRank on Directed Fuzzy Graphs

Reference implementation and experimental code for the paper:

**Antonios Kalampakas**, *Strong-Path PageRank on Directed Fuzzy Graphs*.

This repository reproduces the experiments on the SNAP temporal email dataset
`email-Eu-core-temporal` and provides a baseline implementation of:

- classical PageRank (PR) on the unweighted support graph,
- weighted PageRank (WPR) on fuzzy edge memberships,
- thresholded weighted PageRank (thWPR) on a high-confidence subgraph,
- strong-path PageRank (SPPR) on directed fuzzy graphs via strongest–strong–path admissibility.

## Repository structure

- `scripts/run_email_eu.py` — end-to-end experiment runner (downloads data, runs baselines + SPPR, saves tables/figures).
- `src/sppr_fuzzy/` — core routines (PageRank power iteration, widest-path connectivity, SCC quotienting, SPPR lift).
- `paper/` — the LaTeX source of the manuscript 
- `data/` — raw downloaded data 

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e .
```

Optional (for faster/standard correlations):

```bash
pip install -e .[scipy]
```

### 2) Reproduce the email-Eu-core-temporal experiment

```bash
python scripts/run_email_eu.py --dataset core
```

Outputs are written to `outputs/`:

- `ranking_results.csv` — main table (Spearman + top-k overlaps)
- `sigma_multiscale_curve.csv` — (sigma, m(sigma), Spearman) sweep
- `figures/` — plots used in the paper

### Useful options

```bash
# run on a smaller department subnetwork for quicker iteration
python scripts/run_email_eu.py --dataset dept1

# change the high-confidence quantile cutoff (default 0.70)
python scripts/run_email_eu.py --dataset core --hc-quantile 0.80

# choose sigma grid strategy
python scripts/run_email_eu.py --sigma-grid quantiles --sigma-points 9
python scripts/run_email_eu.py --sigma-grid linspace  --sigma-points 9
```

## Data

The temporal email data are public and distributed by SNAP (Stanford Network Analysis Project).
The runner script downloads the selected dataset automatically into `data/` if it is missing.

- Dataset page: https://snap.stanford.edu/data/email-Eu-core-temporal.html

## Reproducibility notes

- Time split is time-respecting and uses timestamp quantiles: train <= 0.80-quantile, evaluate in (0.80, 0.90].
- Strong edges are detected with a floating-point tolerance `--strong-tol` (default `1e-12`).
- Dangling nodes/classes are handled by standard PageRank dangling correction and teleportation.

## License

Choose a license appropriate for your distribution goals (e.g., MIT, BSD-3-Clause, CC-BY-4.0 for the paper text).
This skeleton includes an MIT license by default.

## Citation

See `CITATION.cff`.
