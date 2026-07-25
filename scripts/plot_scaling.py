#!/usr/bin/env python3
"""Plot throughput vs instrument count from a QuantCUDA scaling benchmark CSV.

Usage:
    quantcuda benchmark --suite scaling --output benchmarks/raw/scaling.csv
    python3 scripts/plot_scaling.py benchmarks/raw/scaling.csv \
        --figure benchmarks/figures/scaling.png
"""
import argparse
import csv
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="scaling benchmark CSV")
    ap.add_argument("--figure", help="optional PNG output path")
    args = ap.parse_args()

    by_backend = {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["suite"] != "scaling" or "skipped" in r["note"]:
                continue
            by_backend.setdefault(r["backend"], []).append(
                (int(r["instruments"]), float(r["throughput_per_sec"])))
    if not by_backend:
        print("no scaling rows found", file=sys.stderr)
        return 1

    for backend, pts in by_backend.items():
        pts.sort()
        print(backend)
        for instr, tp in pts:
            print(f"  {instr:>6} instruments: {tp / 1e6:8.2f} M paths/sec")

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping figure", file=sys.stderr)
            return 0
        plt.figure(figsize=(7, 4.5))
        for backend, pts in by_backend.items():
            pts.sort()
            plt.plot([p[0] for p in pts], [p[1] / 1e6 for p in pts],
                     "o-", label=backend)
        plt.xscale("log")
        plt.xlabel("instruments")
        plt.ylabel("throughput (M paths/sec)")
        plt.title("QuantCUDA scaling suite")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.figure, dpi=150)
        print(f"figure written: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
