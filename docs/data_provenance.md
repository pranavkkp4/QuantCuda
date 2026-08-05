# Data provenance and reproducibility

## Why no external market dataset is required

QuantCUDA's correctness targets are mathematical models, not a fitted trading
strategy. European Monte Carlo prices are checked against the Black-Scholes
closed form under the same explicit inputs `(S, K, r, q, sigma, T)`. Asian and
barrier checks compare independently implemented CPU/GPU estimators under
risk-neutral GBM, while VaR/CVaR tests use controlled synthetic distributions
with known properties. Downloading historical prices would add vendor choices,
corporate-action adjustments, missing observations, and calibration error
without improving any of those implementation checks.

No network download is therefore part of the validated build. The repository
contains no opaque third-party financial dataset and does not require API keys.

## Version-controlled model inputs

- `configs/*.json` contains the small, reviewable sample markets and portfolios.
- `scripts/generate_portfolio.py` creates larger synthetic books from a local
  seeded PRNG. The same Python version, arguments, and seed reproduce the same
  JSON byte-for-byte.
- The C++ engines use counter-based Philox streams keyed by seed and path or
  scenario ID, so CPU results do not depend on thread scheduling.
- `benchmarks/raw/*.csv` contains captured benchmark outputs. Each row records
  its workload and build/runtime environment; these files are measurements,
  not model inputs.

Generate a deterministic synthetic portfolio:

```bash
python scripts/generate_portfolio.py --assets 4 --european 100 \
  --asian 50 --barrier 20 --seed 42 \
  --output configs/generated_portfolio.json
```

## Reproducing the analytical validation dataset

After building `quantcuda`, collect the four-point scalar convergence sweep
(10,000 through 10,000,000 paths) and analyze its error/standard-error slope:

```bash
python scripts/collect_cpu_validation.py
```

The default output is `benchmarks/raw/convergence_scalar.csv`. Both the output
and executable can be selected explicitly, which is useful for out-of-tree and
multi-config builds:

```bash
python scripts/collect_cpu_validation.py \
  --executable build/Release/quantcuda.exe \
  --output benchmarks/raw/my_cpu_convergence.csv \
  --figure benchmarks/figures/my_cpu_convergence.png
```

The collector uses argument-vector subprocess calls rather than a shell and is
portable across Windows, Linux, and macOS. Relative output paths are resolved
from the repository root. The generated CSV contains the Black-Scholes
absolute error, Monte Carlo standard error, seed-dependent estimates, and the
compiler/OS/backend environment emitted by the C++ benchmark runner.

## When external data would become appropriate

Historical datasets become necessary for volatility/correlation calibration,
backtesting, stress-scenario construction, or P&L attribution. Those are
separate concerns from the pricing-kernel and analytical-validation scope here.
Any future downloaded dataset should be accompanied by its source URL, access
date, license, checksum, adjustment policy, and a reproducible acquisition
script; raw licensed data should not be committed unless redistribution is
explicitly permitted.
