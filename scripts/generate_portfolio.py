#!/usr/bin/env python3
"""Generate portfolio JSON configs for QuantCUDA (e.g. 100 European + 50 Asian
+ 20 barrier options across correlated assets, per the project specification).

Usage:
    python3 scripts/generate_portfolio.py --assets 4 --european 100 \
        --asian 50 --barrier 20 --output configs/generated_portfolio.json
"""
import argparse
import json
import random


MAX_PATHS = 100_000_000
MAX_SCENARIOS = 10_000_000
MAX_PRICING_WORK = 5_000_000_000
MAX_SCENARIO_VALUES = 50_000_000
MAX_INSTRUMENTS = 25_000  # keeps generated, indented JSON below the 8 MiB loader cap
MAX_SEED = (1 << 64) - 1


def bounded_int(name: str, minimum: int, maximum: int):
    """Return an argparse converter with a clear, field-specific range error."""
    def parse(value: str) -> int:
        try:
            parsed = int(value, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{minimum}, {maximum}]")
        return parsed
    return parse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=bounded_int("assets", 1, 64), default=4)
    ap.add_argument("--european", type=bounded_int("european", 0, MAX_INSTRUMENTS),
                    default=100)
    ap.add_argument("--asian", type=bounded_int("asian", 0, MAX_INSTRUMENTS), default=50)
    ap.add_argument("--barrier", type=bounded_int("barrier", 0, MAX_INSTRUMENTS), default=20)
    ap.add_argument("--paths", type=bounded_int("paths", 1, MAX_PATHS),
                    default=200_000)
    ap.add_argument("--scenarios", type=bounded_int("scenarios", 1, MAX_SCENARIOS),
                    default=100_000)
    ap.add_argument("--seed", type=bounded_int("seed", 0, MAX_SEED), default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.european + args.asian + args.barrier == 0:
        ap.error("at least one instrument count must be positive")
    instrument_count = args.european + args.asian + args.barrier
    if instrument_count > MAX_INSTRUMENTS:
        ap.error(f"total instruments must not exceed {MAX_INSTRUMENTS}")
    conservative_path_units = args.european + 252 * (args.asian + args.barrier)
    if args.paths * conservative_path_units > MAX_PRICING_WORK:
        ap.error("paths x conservative path steps exceeds 5000000000")
    if args.scenarios * (args.assets + instrument_count) > MAX_SCENARIO_VALUES:
        ap.error("scenarios x (assets + instruments) exceeds 50000000")
    rng = random.Random(args.seed)

    symbols = [f"ASSET_{i + 1:02d}" for i in range(args.assets)]
    spots = {s: round(rng.uniform(50.0, 400.0), 2) for s in symbols}
    assets = [
        {
            "symbol": s,
            "spot": spots[s],
            "volatility": round(rng.uniform(0.12, 0.35), 4),
            "dividend_yield": round(rng.uniform(0.0, 0.025), 4),
        }
        for s in symbols
    ]

    # Random symmetric diagonally-dominant correlation matrix (guaranteed PD).
    corr = [[0.0] * args.assets for _ in range(args.assets)]
    for i in range(args.assets):
        off = [round(rng.uniform(0.0, 0.35 / max(args.assets - 1, 1)), 4)
               for _ in range(args.assets)]
        for j in range(args.assets):
            corr[i][j] = 1.0 if i == j else off[j]
    for i in range(args.assets):
        for j in range(i + 1, args.assets):
            corr[j][i] = corr[i][j] = round((corr[i][j] + corr[j][i]) / 2, 4)

    portfolio = []

    def strike_for(s: str) -> float:
        return round(spots[s] * rng.uniform(0.85, 1.15), 2)

    for n in range(args.european):
        s = rng.choice(symbols)
        portfolio.append({
            "id": f"EU_{'CALL' if n % 2 == 0 else 'PUT'}_{n:03d}",
            "type": "european_call" if n % 2 == 0 else "european_put",
            "underlying": s,
            "strike": strike_for(s),
            "maturity": round(rng.uniform(0.25, 2.0), 2),
            "quantity": rng.choice([10, 20, 50, 100]) * rng.choice([1, 1, 1, -1]),
        })
    for n in range(args.asian):
        s = rng.choice(symbols)
        portfolio.append({
            "id": f"ASIAN_{'CALL' if n % 2 == 0 else 'PUT'}_{n:03d}",
            "type": "asian_call" if n % 2 == 0 else "asian_put",
            "underlying": s,
            "strike": strike_for(s),
            "maturity": round(rng.uniform(0.5, 1.5), 2),
            "quantity": rng.choice([10, 25, 50]),
            "time_steps": rng.choice([12, 52, 252]),
        })
    for n in range(args.barrier):
        s = rng.choice(symbols)
        up = n % 2 == 0
        call = n % 4 in (0, 1)
        barrier = round(spots[s] * (rng.uniform(1.1, 1.6) if up
                                    else rng.uniform(0.5, 0.9)), 2)
        portfolio.append({
            "id": f"BARRIER_{'UP' if up else 'DOWN'}_{n:03d}",
            "type": f"barrier_{'up' if up else 'down'}_out_{'call' if call else 'put'}",
            "underlying": s,
            "strike": strike_for(s),
            "maturity": round(rng.uniform(0.5, 1.5), 2),
            "quantity": rng.choice([10, 20]),
            "barrier": barrier,
        })

    doc = {
        "simulation": {
            "paths": args.paths,
            "time_steps": 252,
            "seed": args.seed,
            "precision": "double",
            "backend": "openmp",
        },
        "market": {
            "risk_free_rate": 0.0425,
            "assets": assets,
            "correlation": corr,
        },
        "portfolio": portfolio,
        "risk": {
            "scenarios": args.scenarios,
            "horizon_years": round(1.0 / 252.0, 6),
            "confidence_levels": [0.95, 0.99],
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {args.output}: {len(portfolio)} instruments, "
          f"{args.assets} assets")


if __name__ == "__main__":
    main()
