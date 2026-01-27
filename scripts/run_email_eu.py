#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sppr_fuzzy.email_eu_pipeline import run_email_eu_experiment


def _parse_sppr_method(s: str) -> tuple[float, int] | None:
    """Parse method label SPPR(0.5825,859) -> (0.5825, 859)."""
    m = re.match(r"SPPR\(([^,]+),([^\)]+)\)", s.strip())
    if not m:
        return None
    try:
        sigma = float(m.group(1))
        mcls = int(float(m.group(2)))
        return sigma, mcls
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce SPPR experiment on email-Eu-core-temporal (SNAP).")

    ap.add_argument("--dataset", default="core", choices=["core", "dept1", "dept2", "dept3", "dept4"],
                    help="Which SNAP file to download/use.")
    ap.add_argument("--data-dir", default="data", help="Directory for raw downloaded data.")
    ap.add_argument("--outputs-dir", default="outputs", help="Directory to save CSV/figures.")

    ap.add_argument("--train-frac", type=float, default=0.80, help="Training window fraction (timestamp quantile).")    
    ap.add_argument("--eval-frac", type=float, default=0.90, help="Evaluation window end fraction (timestamp quantile).")

    ap.add_argument("--alpha", type=float, default=0.85, help="PageRank damping factor.")
    ap.add_argument("--max-iter", type=int, default=200, help="Max power iterations.")
    ap.add_argument("--tol", type=float, default=1e-12, help="Power iteration L1 tolerance.")

    ap.add_argument("--agg-method", default="count", choices=["count", "decay"],
                    help="Edge intensity aggregation: count or exponential decay.")
    ap.add_argument("--half-life-days", type=float, default=60.0, help="Half-life in days if --agg-method=decay.")
    ap.add_argument("--scale-method", default="log", choices=["log", "minmax", "cauchy"],
                    help="Scaling from raw counts/intensity to [0,1].")

    ap.add_argument("--lambda-min", type=float, default=0.0, help="Noise floor: keep edges with w>=lambda_min.")
    ap.add_argument("--hc-quantile", type=float, default=0.70, help="High-confidence cutoff quantile.")
    ap.add_argument("--sigma-grid", default="quantiles", choices=["quantiles", "linspace"],
                    help="How to choose sigma thresholds.")
    ap.add_argument("--sigma-points", type=int, default=9, help="# of sigma points.")
    ap.add_argument("--strong-tol", type=float, default=1e-12, help="Tolerance for strong-edge test.")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="Disable early stopping in widest-path runs (slower, more conservative).")

    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Run experiment (also exports CSVs)
    results = run_email_eu_experiment(
        dataset=args.dataset,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        agg_method=args.agg_method,
        half_life_days=args.half_life_days,
        scale_method=args.scale_method,
        lambda_min=args.lambda_min,
        hc_quantile=args.hc_quantile,
        sigma_grid=args.sigma_grid,
        sigma_points=args.sigma_points,
        strong_tol=args.strong_tol,
        early_stop=(not args.no_early_stop),
    )

    print("\nSaved:", outputs_dir / "ranking_results.csv")
    print(results.to_string(index=False))

    # Load sigma curve and plot
    curve_path = outputs_dir / "sigma_multiscale_curve.csv"
    curve = pd.read_csv(curve_path)
    curve = curve.sort_values("sigma")

    # Figure: m vs sigma
    plt.figure()
    plt.plot(curve["sigma"], curve["m_classes"], marker="o")
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"Quotient size $m(\sigma)$")
    plt.title(r"Quotient size vs $\sigma$")
    plt.tight_layout()
    fig1 = outputs_dir / "figures" / "mail_eu_m_vs_sigma.png"
    plt.savefig(fig1, dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: Spearman vs sigma
    plt.figure()
    plt.plot(curve["sigma"], curve["spearman"], marker="o")
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"Spearman $\rho$ (score vs future incoming)")
    plt.title(r"Predictive rank correlation vs $\sigma$")
    plt.tight_layout()
    fig2 = outputs_dir / "figures" / "mail_eu_spearman_vs_sigma.png"
    plt.savefig(fig2, dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: overlap@100 vs sigma (parse SPPR rows)
    sp = results[results["method"].astype(str).str.startswith("SPPR(")].copy()
    if not sp.empty:
        parsed = sp["method"].apply(_parse_sppr_method)
        sp["sigma"] = [p[0] if p else np.nan for p in parsed]
        sp = sp.dropna(subset=["sigma"]).sort_values("sigma")
        plt.figure()
        plt.plot(sp["sigma"], sp["overlap@100"], marker="o")
        plt.xlabel(r"$\sigma$")
        plt.ylabel("overlap@100")
        plt.title(r"SPPR elite-set agreement vs $\sigma$")
        plt.tight_layout()
        fig3 = outputs_dir / "figures" / "mail_eu_overlap100_vs_sigma.png"
        plt.savefig(fig3, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved figures:")
        print(" -", fig1)
        print(" -", fig2)
        print(" -", fig3)
    else:
        print("No SPPR rows found; skipping overlap@100 plot.")
        print("Saved figures:")
        print(" -", fig1)
        print(" -", fig2)


if __name__ == "__main__":
    main()
