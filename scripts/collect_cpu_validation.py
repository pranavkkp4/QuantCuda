#!/usr/bin/env python3
"""Collect a reproducible CPU Monte Carlo/Black-Scholes convergence dataset.

The C++ benchmark owns pricing and CSV metadata; this wrapper only locates the
cross-platform executable, selects an output, invokes the convergence suite,
and runs the repository's accuracy analyzer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "raw" / "convergence_scalar.csv"


def executable_candidates() -> list[Path]:
    names = ("quantcuda.exe", "quantcuda")
    directories = (
        REPO_ROOT / "build",
        REPO_ROOT / "build" / "Release",
        REPO_ROOT / "build-cpu",
        REPO_ROOT / "build-cpu" / "Release",
        REPO_ROOT / "build-gpu",
        REPO_ROOT / "build-gpu" / "Release",
    )
    return [directory / name for directory in directories for name in names]


def resolve_executable(requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"QuantCUDA executable not found: {path}")
        return path
    for candidate in executable_candidates():
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in executable_candidates())
    raise FileNotFoundError(
        "QuantCUDA executable not found; build the project or pass --executable. "
        f"Searched:\n  {searched}")


def output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        help="quantcuda executable (auto-detected in common build directories)",
    )
    parser.add_argument(
        "--backend",
        choices=("scalar", "openmp"),
        default="scalar",
        help="CPU backend used for all convergence rows (default: scalar)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)),
        help="CSV destination, relative paths are resolved from the repository root",
    )
    parser.add_argument("--figure", help="optional convergence PNG destination")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="collect CSV without printing the accuracy table",
    )
    args = parser.parse_args()
    if not 0 <= args.seed <= (1 << 64) - 1:
        parser.error("--seed must be in [0, 18446744073709551615]")

    try:
        executable = resolve_executable(args.executable)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    csv_path = output_path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    collect = [
        str(executable),
        "benchmark",
        "--suite",
        "convergence",
        "--backends",
        args.backend,
        "--seed",
        str(args.seed),
        "--output",
        str(csv_path),
    ]
    print("Collecting:", " ".join(collect))
    try:
        subprocess.run(collect, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"collection failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    if not args.skip_analysis:
        analyze = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_accuracy.py"),
            str(csv_path),
        ]
        if args.figure:
            figure_path = output_path(args.figure)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            analyze.extend(("--figure", str(figure_path)))
        try:
            subprocess.run(analyze, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"analysis failed with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode or 1
    print(f"Validation CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
