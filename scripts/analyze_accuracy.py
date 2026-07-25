#!/usr/bin/env python3
"""Analyze Monte Carlo accuracy from a QuantCUDA convergence benchmark CSV.

Produces the convergence table (error vs Black-Scholes by path count) and,
when matplotlib is available, a log-log figure showing the expected 1/sqrt(N)
decay. Pure analysis/validation tooling - pricing is done by the C++ engine.

Usage:
    quantcuda benchmark --suite convergence --output benchmarks/raw/conv.csv
    python3 scripts/analyze_accuracy.py benchmarks/raw/conv.csv \
        --figure benchmarks/figures/convergence.png
"""
import argparse
import csv
import math
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="convergence benchmark CSV")
    ap.add_argument("--figure", help="optional PNG output path")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for line_number, r in enumerate(csv.DictReader(f), start=2):
            if r.get("suite") != "convergence":
                continue
            # New reports carry an explicit abs_error column. Retain a
            # fallback for pre-1.0 CSVs where that value occupied the generic
            # throughput field.
            error_field = r.get("abs_error", "")
            if not error_field:
                error_field = r.get("throughput_per_sec")
            try:
                row = {
                    "paths": int(r["paths"]),
                    "price": float(r["price"]),
                    "std_error": float(r["std_error"]),
                    "abs_error": float(error_field),
                }
            except (KeyError, TypeError, ValueError) as exc:
                print(f"invalid convergence row at CSV line {line_number}: {exc}",
                      file=sys.stderr)
                return 1
            if row["paths"] <= 0:
                print(f"invalid non-positive path count at CSV line {line_number}",
                      file=sys.stderr)
                return 1
            if (not math.isfinite(row["price"]) or
                    not math.isfinite(row["std_error"]) or
                    not math.isfinite(row["abs_error"]) or
                    row["std_error"] < 0 or row["abs_error"] < 0):
                print(f"invalid non-finite or negative metric at CSV line {line_number}",
                      file=sys.stderr)
                return 1
            rows.append(row)
    if not rows:
        print("no convergence rows found in CSV", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["paths"])
    duplicate_paths = sorted({
        r["paths"] for i, r in enumerate(rows[1:], start=1)
        if r["paths"] == rows[i - 1]["paths"]
    })
    if duplicate_paths:
        values = ", ".join(str(value) for value in duplicate_paths)
        print(f"duplicate convergence path counts: {values}", file=sys.stderr)
        return 1

    zero_se_count = sum(r["std_error"] == 0.0 for r in rows)
    if zero_se_count:
        print(f"warning: {zero_se_count} row(s) have zero standard error; "
              "excluding them from slope fitting", file=sys.stderr)

    print(f"{'paths':>12} {'price':>12} {'std_error':>12} {'abs_error':>12} "
          f"{'error/SE':>10}")
    for r in rows:
        z = (f"{r['abs_error'] / r['std_error']:>10.2f}"
             if r["std_error"] > 0 else f"{'n/a':>10}")
        print(f"{r['paths']:>12,} {r['price']:>12.6f} {r['std_error']:>12.6f} "
              f"{r['abs_error']:>12.6f} {z}")

    # Absolute error from one Monte Carlo realization need not be monotone.
    # Standard error is the statistically meaningful 1/sqrt(N) quantity, so
    # fit its log-log slope across all positive observations.
    positive = [(math.log(r["paths"]), math.log(r["std_error"]))
                for r in rows if r["std_error"] > 0]
    if len(positive) >= 2:
        mean_x = sum(x for x, _ in positive) / len(positive)
        mean_y = sum(y for _, y in positive) / len(positive)
        denom = sum((x - mean_x) ** 2 for x, _ in positive)
        if denom > 0.0:
            slope = (sum((x - mean_x) * (y - mean_y)
                         for x, y in positive) / denom)
            print(f"\nstandard-error log-log slope: {slope:.3f} "
                  f"(theory: -0.500)")
        else:
            print("cannot fit slope: path counts have zero log variance",
                  file=sys.stderr)
    elif len(rows) >= 2:
        print("cannot fit slope: need at least two positive standard errors",
              file=sys.stderr)

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping figure", file=sys.stderr)
            return 0
        ns = [r["paths"] for r in rows]
        err = [max(r["abs_error"], 1e-12) for r in rows]
        plt.figure(figsize=(7, 4.5))
        plt.loglog(ns, err, "o-", label="abs error vs Black-Scholes")
        positive_se = [(r["paths"], r["std_error"])
                       for r in rows if r["std_error"] > 0]
        if positive_se:
            plt.loglog([point[0] for point in positive_se],
                       [point[1] for point in positive_se],
                       "s--", label="standard error")
        plt.xlabel("paths")
        plt.ylabel("error")
        plt.title("Monte Carlo convergence (QuantCUDA)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.figure, dpi=150)
        print(f"figure written: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
