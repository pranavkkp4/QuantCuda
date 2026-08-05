# QuantCUDA Risk Engine

A **C++20 / CUDA derivatives pricing and portfolio risk engine** built as a
series of measurable optimization experiments. It prices European, Asian and
barrier options with Monte Carlo simulation on scalar CPU, OpenMP multicore
CPU, and CUDA GPU backends, computes Greeks three different ways, and measures
portfolio VaR/CVaR from correlated multi-asset scenarios.

Every GPU optimization generation **V0–V7 is preserved as a benchmarkable
version** — the project treats performance as a sequence of hypotheses tested
against measurements, not as folklore. CPU baselines are first-class citizens,
not afterthoughts.

> **Runs anywhere.** The CUDA backend is optional: on a machine with an NVIDIA
> GPU and CUDA toolkit you get the full GPU engine; on any other machine the
> same binary builds CPU-only with the scalar and OpenMP backends fully
> functional (pricing, Greeks, risk, benchmarks, tests — everything except
> `cuda` backend rows).

---

## Table of contents

1. [Features](#features)
2. [Measured release results](#measured-release-results)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Building](#building)
6. [Usage](#usage)
7. [Portfolio JSON schema](#portfolio-json-schema)
8. [GPU implementation generations V0–V7](#gpu-implementation-generations-v0v7)
9. [Benchmark suites](#benchmark-suites)
10. [Testing](#testing)
11. [Repository layout](#repository-layout)
12. [Numerical and benchmarking methodology](#numerical-and-benchmarking-methodology)
13. [Documentation](#documentation)
14. [Roadmap](#roadmap)
15. [License](#license)

---

## Features

**Pricing (Monte Carlo, geometric Brownian motion)**

| Instrument | CPU scalar | CPU OpenMP | CUDA | Analytic check |
|---|---|---|---|---|
| European call/put | ✅ | ✅ | ✅ (V0–V7) | Black-Scholes |
| Asian call/put (arithmetic average) | ✅ | ✅ | ✅ | — |
| Barrier (up/down-and-out call/put, discrete monitoring) | ✅ | ✅ | ✅ (3 divergence strategies) | — |

- Deterministic, reproducible results: counter-based Philox4x32-10 streams
  keyed by `(seed, path_id)`. Same seed ⇒ same price, independent of thread
  scheduling (CPU and OpenMP agree to ~1e-10; see `tests/test_monte_carlo.cpp`).
- Confidence reporting: every Monte Carlo price ships with its standard error.

**Greeks**

- Analytic Black-Scholes delta/gamma/vega (double precision).
- Central finite differences with **common random numbers** (one draw prices
  all bumps — lower estimator variance than independent re-simulation;
  measured in `tests/test_greeks.cpp` and `docs/financial_models.md`).
- **Pathwise delta** under GBM, compared against analytic and FD estimators.
- Fused single-launch GPU kernel producing price + delta + gamma + vega.

**Portfolio risk**

- Correlated multi-asset scenario engine: Cholesky factorization (host, once),
  one-step horizon shocks, per-scenario repricing (Black-Scholes full
  revaluation for Europeans; delta-gamma approximation for path-dependent
  instruments — clearly documented as an approximation).
- VaR and CVaR at configurable confidence levels (default 95%/99%), scenario
  P&L statistics, worst loss / best gain.
- GPU path: scenario P&L kernel + Thrust device sort + tail reduction.

**Benchmarking**

- Timed suites use warm-up runs, repeated measurements, median/min/p95,
  identical seeds and parameters across backends, and an environment block in
  every CSV.
- The dedicated GPU studies separate CUDA-event intervals from end-to-end wall
  time where both measurements are meaningful.
- CSV output for the Python analysis scripts.

## Measured release results

The release measurements below were captured on Windows with MSVC 19.44,
CUDA 13.3.73, an Intel Core i5-12400F (6 cores / 12 threads), and an NVIDIA
GeForce RTX 3050 6 GB (sm_86, 18 SMs; driver 595.79).

| Study | Measured result |
|---|---|
| European, 10⁷ paths | scalar 647.54 ms; OpenMP 134.05 ms; CUDA 68.90 ms end-to-end (1.95× vs OpenMP) |
| 8 Asian options, 10⁵ paths × 252 steps | OpenMP 1269.83 ms; CUDA 193.38 ms (6.57×) |
| V0 → V2 European kernel | 58.900 → 30.159 ms (1.95×); end-to-end 84.412 → 52.451 ms |
| V7 precision | double 42.905 ms; mixed 23.908 ms; float 4.814 ms GPU interval; prices 7.167997 / 7.163358 / 7.163358 |
| V1 layout, 64 × 10⁵ paths | AoS 36.309 ms; SoA 28.359 ms end-to-end (1.28×) |
| V4 near barrier, 10⁶ × 252 steps | mask 235.642 ms; compacted 178.628 ms GPU interval (1.32×); all strategies agree on price |
| Portfolio risk, 7 instruments / 50,000 scenarios | OpenMP 29.696 ms; CUDA 26.009 ms (1.14×) |

The CUDA release passed all 56 tests with no failures or skips. Compute
Sanitizer reported zero errors for V6 async pricing, V1 layout, V4 barrier
divergence/compaction, and GPU portfolio risk. At one million CUDA paths, the
European call and put were 0.45 and 0.36 standard errors from Black-Scholes.
See `docs/performance_report.md` and `docs/correctness_report.md` for complete
tables, methodology, and timing caveats.

## Requirements

| Component | Minimum | Notes |
|---|---|---|
| C++ compiler | GCC 11 / Clang 14 / MSVC 2022 | C++20 |
| CMake | 3.24+ | |
| CUDA toolkit | 12.x | optional — enables the GPU engine |
| NVIDIA GPU | compute capability ≥ 7.5 | RTX 20xx+ (RTX 3050 = sm_86) |
| Python | 3.9+ | analysis scripts only (matplotlib optional) |

## Quick start

The commands below use a single-config generator on Linux or macOS and are run
from the repository root.

```bash
# 1. Configure & build (auto-detects CUDA)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Run the test suite (56 unit + statistical tests)
ctest --test-dir build --output-on-failure

# 3. Price a portfolio
./build/quantcuda price --portfolio configs/european_call.json \
    --backend openmp --paths 1000000 --greeks

# 4. Validate Monte Carlo against Black-Scholes
./build/quantcuda validate --portfolio configs/european_call.json --paths 2000000

# 5. Portfolio risk (VaR/CVaR)
./build/quantcuda risk --portfolio configs/multi_asset_portfolio.json \
    --scenarios 100000

# 6. Benchmark CPU backends
./build/quantcuda benchmark --suite european-throughput \
    --backends scalar,openmp --paths 10000000 --output results.csv
```

On Windows with the Visual Studio 2022 generator, the equivalent PowerShell
commands are:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
.\build\Release\quantcuda.exe price --portfolio configs\european_call.json --backend openmp --paths 1000000 --greeks
```

Visual Studio is a multi-config generator, so its release executables live in
`build\Release\`. With Ninja on Windows, use `build\quantcuda.exe` instead.

On a GPU machine the same commands accept `--backend cuda`, e.g.:

```bash
./build/quantcuda price --portfolio configs/european_call.json \
    --backend cuda --gpu-version v3 --precision mixed --paths 10000000
./build/quantcuda benchmark --suite version-study   # V0-V7 + reduction flavors
```

## Building

### CPU-only (single-config generator)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DQUANTCUDA_ENABLE_CUDA=OFF
cmake --build build --parallel
```

### With CUDA (GPU machine)

CMake auto-detects the toolkit. If it is not on `PATH`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="86"   # RTX 3050; or "native"
cmake --build build --parallel
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `75;86;89;90` (Turing → Hopper).
Verify the build with `./build/quantcuda info`.

For a Visual Studio CUDA build on Windows, use the PowerShell commands from
Quick start and verify with `.\build\Release\quantcuda.exe info`. The CUDA
Toolkit's Visual Studio integration must be installed.

### Build options

| Option | Default | Meaning |
|---|---|---|
| `QUANTCUDA_ENABLE_CUDA` | ON | Build the GPU engine when a toolkit is found |
| `QUANTCUDA_BUILD_TESTS` | ON | Build `quantcuda_tests` |
| `CMAKE_BUILD_TYPE` | Release | Use Debug for development |

## Usage

```text
quantcuda <command> [options]

  price       price every instrument in a portfolio JSON
  risk        portfolio VaR/CVaR from correlated scenarios
  validate    Monte Carlo vs Black-Scholes comparison table
  greeks      analytic vs FD-CRN vs pathwise Greeks
  benchmark   run a benchmark suite
  versions    list GPU implementation generations V0-V7
  info        print build/environment/device information
```

The usage examples below assume `quantcuda` is on `PATH`. Otherwise invoke
`./build/quantcuda` on a single-config POSIX build,
`.\build\quantcuda.exe` with Ninja on Windows, or
`.\build\Release\quantcuda.exe` with Visual Studio.

### `price`

```bash
quantcuda price --portfolio configs/barrier_portfolio.json \
    --backend cuda --paths 2000000 \
    --precision mixed --seed 42 --steps 252 --greeks
```

Prints per-instrument price, standard error, Black-Scholes reference and
deviation for Europeans, optional Greeks, portfolio market value, timing and
throughput.

### `risk`

```bash
quantcuda risk --portfolio configs/multi_asset_portfolio.json \
    --scenarios 1000000 --var-levels 0.95,0.99 --horizon-days 1 --backend cuda
```

### `benchmark`

```bash
quantcuda benchmark --suite european-throughput --backends scalar,openmp,cuda \
    --paths 10000000 --repeats 5 --output benchmarks/raw/european.csv
```

### Common options

| Option | Values | Default | Notes |
|---|---|---|---|
| `--portfolio` | path | — | JSON input (see `configs/`) |
| `--backend` | scalar, openmp, cuda | from config | compute backend |
| `--paths` | 1 … 100,000,000 | from config | Monte Carlo paths; aggregate-work cap also applies |
| `--precision` | float, double, mixed | from config | float/mixed are CUDA-only; CPUs use double |
| `--seed` | uint64 | 42 | RNG seed |
| `--steps` | 1 … 100000 | 252 | path steps (Asian/barrier) |
| `--gpu-version` | v0…v7 | v7 | GPU generation (CUDA builds) |
| `--block-size` | 32…1024 | 256 | CUDA block size |
| `--scenarios` | 1 … 10,000,000 | 100000 | risk scenarios; scenarios × assets is capped |
| `--var-levels` | e.g. `0.95,0.99` | 0.95,0.99 | confidence levels |
| `--horizon-days` | 1…252 | 1 | risk horizon |
| `--repeats` | 1…101 | 5 | benchmark repetitions |
| `--warmup` | 0…101 | 1 | unmeasured benchmark warm-up runs |

## Portfolio JSON schema

```jsonc
{
  "simulation": {"paths": 1000000, "time_steps": 252, "seed": 42,
                 "precision": "double", "backend": "openmp"},
  "market": {
    "risk_free_rate": 0.0425,
    "assets": [{"symbol": "A", "spot": 100.0, "volatility": 0.20,
                "dividend_yield": 0.01}],
    "correlation": [[1.0]]        // d×d, symmetric, PD; optional
  },
  "portfolio": [
    {"id": "C1", "type": "european_call", "underlying": "A",
     "strike": 105.0, "maturity": 1.0, "quantity": 100},
    {"id": "B1", "type": "barrier_up_out_call", "underlying": "A",
     "strike": 110.0, "maturity": 1.0, "quantity": 20, "barrier": 140.0}
  ],
  "risk": {"scenarios": 100000, "horizon_years": 0.003968,
           "confidence_levels": [0.95, 0.99]}
}
```

Option types: `european_call`, `european_put`, `asian_call`, `asian_put`,
`barrier_up_out_call`, `barrier_up_out_put`, `barrier_down_out_call`,
`barrier_down_out_put`. All fields are strictly validated (ranges, matrix
symmetry, positive-definiteness via Cholesky, instrument limits — see
`src/core/json_config.cpp` and `tests/test_json_config.cpp`).

Generate large synthetic books:

```bash
python3 scripts/generate_portfolio.py --assets 4 --european 100 \
    --asian 50 --barrier 20 --output configs/generated_portfolio.json
```

These model-validation workloads do not require a downloaded market dataset:
the GBM inputs are explicit and European results are checked against the
Black-Scholes closed form. See `docs/data_provenance.md` for the rationale and
fully reproducible data-generation commands.

## GPU implementation generations V0–V7

| Version | One-line description | Select |
|---|---|---|
| **V0** naive | AoS inputs, separate RNG/sim/payoff kernels, global arrays, per-thread `atomicAdd`, synchronous copies | `--gpu-version v0` |
| **V1** SoA | structure-of-arrays batch layout (coalesced instrument reads); AoS-vs-SoA study | `--gpu-version v1` + `benchmark --suite layout-study` |
| **V2** fused | single kernel = draw + terminal price + payoff + shared-memory block reduction | `--gpu-version v2` |
| **V3** reduction | warp-shuffle reductions (4 flavors kept: atomic, shared tree, shuffle, shuffle+smem) | `--gpu-version v3` |
| **V4** specialized | per-instrument-type kernels; barrier early-exit vs boolean-mask vs compacted | `--gpu-version v4` |
| **V5** memory | `__constant__` option parameters, shared-memory staging for Cholesky shocks | `--gpu-version v5` |
| **V6** async | pinned host memory + 4-stream chunked asynchronous execution | `--gpu-version v6` |
| **V7** precision | float / double / mixed (float simulation, double accumulation) study | `--gpu-version v7` |

`--gpu-version` selects the European implementation generation. Asian and
barrier options dispatch to their specialized path kernels independently of
that flag; use `path-dependent`, `divergence`, and `gpu-divergence` to compare
those workloads. In the V6 benchmark, the field named `kernel_ms` covers the
asynchronous GPU interval through D2H completion, not compute alone; use the
end-to-end column for whole-operation comparisons.

The core European-kernel comparison is one command; V1 and V4 have dedicated
workloads because layout and barrier divergence are not meaningful on the
single-European workload:

```bash
quantcuda benchmark --suite version-study --paths 10000000
quantcuda benchmark --suite layout-study --instruments 64 --paths 100000
quantcuda benchmark --suite gpu-divergence --paths 1000000 --steps 252
```

## Benchmark suites

| Suite | Workload | Backends |
|---|---|---|
| `european-throughput` | 1 European call, 10⁷ paths (configurable) | all |
| `path-dependent` | 64 Asian options × paths × 12/52/252 steps | all |
| `divergence` | barrier near/medium/far × early-exit/mask (+compacted on GPU) | all |
| `portfolio-risk` | multi-asset scenario engine, VaR/CVaR | all |
| `scaling` | 1→1024 instruments at fixed paths/instrument | all |
| `convergence` | abs error vs Black-Scholes at 10⁴…10⁷ paths | CPU |
| `version-study` | V0–V7 plus four reduction flavors | GPU |
| `gpu-divergence` | kernel-time divergence study with knockout fractions | GPU |
| `layout-study` | AoS vs SoA batch layout | GPU |

Timed suites use warm-up + repeats, median/min/p95, identical seeds, CSV +
stdout reports, and an environment block. `convergence` is an accuracy sweep,
so its rows report price, standard error, and absolute error rather than timing
quantiles. Workload sizes are overridable (`--paths`, `--steps`,
`--instruments`, `--scenarios`).

Recreate the canonical CPU convergence CSV and run the accuracy analysis with
one cross-platform command (defaults to the scalar backend and seed 42):

```bash
python3 scripts/collect_cpu_validation.py
```

## Testing

```bash
ctest --test-dir build --output-on-failure
# or directly:
./build/quantcuda_tests
```

**56 tests** across 10 suites (**56 passed, 0 failed, 0 skipped** in the CUDA
release): Black-Scholes reference
values, put-call parity, payoff mechanics, Monte Carlo convergence and
thread-count invariance, Greeks (analytic vs FD-CRN vs pathwise), Cholesky
reconstruction and non-PD rejection, empirical correlation, VaR/CVaR statistics,
JSON validation, edge cases (T=0, σ=0, S=0, breached barriers, deep ITM/OTM,
tiny path counts), and CPU/GPU agreement (compiled into every build; executes
fully when a GPU is present).

See `docs/test_plan.md` for the QA strategy and quality gates.

## Repository layout

```text
quantcuda-risk-engine/
├── CMakeLists.txt
├── README.md                  ← you are here
├── LICENSE
├── configs/                   ← sample portfolios (European, Asian, barrier, multi-asset)
├── include/
│   ├── quantcuda/             ← public headers (CPU engines are header-only)
│   └── third_party/nlohmann/  ← JSON parser (pinned v3.11.3)
├── src/
│   ├── core/                  ← types + JSON config
│   ├── cuda/                  ← kernels: rng / european / asian / barrier /
│   │                            greeks / correlation / portfolio_risk /
│   │                            reduction + cuda_pricer dispatch + stub
│   ├── benchmark/             ← benchmark runner
│   ├── cli/                   ← command-line interface
│   └── main.cpp
├── tests/                     ← 56 unit + statistical tests (custom micro-framework)
├── scripts/                   ← Python analysis/validation (generate, plot, accuracy)
├── benchmarks/{raw,processed,figures}/
├── profiling/                 ← Nsight workflow + capture commands
└── docs/                      ← architecture, models, correctness, performance,
                                 test plan, security review
```

## Numerical and benchmarking methodology

- **Model:** risk-neutral GBM; Europeans priced from terminal draws, path
  options from discrete fixings (arithmetic average; discrete barrier
  monitoring). Double-precision CPU reference; GPU precision study in V7.
- **RNG:** Philox4x32-10 counter-based streams; CPU implements the same
  algorithm family as cuRAND's device generator. Per-path substreams ⇒
  reproducibility independent of scheduling.
- **Statistics:** standard error with every price; CPU and optimized CUDA paths
  use Welford/Chan moments to avoid cancellation at large payoff scales;
  agreement tests compare within 4 combined standard errors; convergence
  verified ~1/√N.
- **Benchmarks:** warm-up + median/min/p95; identical seeds across backends;
  CSVs label the primary timing scope and carry separate GPU-interval and
  end-to-end columns; environment recorded per report.
- **Honesty:** every GPU speedup in the shipped report was measured on the RTX
  3050 named above; inconclusive changes and timing-boundary caveats remain in
  the tables instead of being presented as wins.

## Documentation

| Document | Contents |
|---|---|
| `docs/architecture.md` | component design, data flow, version strategy |
| `docs/financial_models.md` | GBM, payoffs, Black-Scholes, Greeks estimators, Cholesky scenarios, VaR/CVaR definitions |
| `docs/correctness_report.md` | test results + validation tables (real outputs) |
| `docs/performance_report.md` | measured CPU/GPU results for all nine suites, optimization conclusions, timing caveats |
| `docs/test_plan.md` | QA strategy, coverage, quality gates |
| `docs/data_provenance.md` | no-download rationale, synthetic inputs, reproducible validation CSV collection |
| `docs/security_review.md` | OWASP Top-10 review of the codebase |
| `profiling/README.md` | Nsight Compute/Systems capture workflows |

## Roadmap

1. **Antithetic variates** — variance reduction for Europeans and Asians
   (one-line change per path: price z and −z; expected ~2× effective paths).
2. **American options** — Longstaff-Schwartz least-squares MC on GPU.
3. **Multi-GPU** — split path space across devices; merged block partial sums.
4. **Exotic expansion** — lookback, cliquet, basket options on the correlated
   engine; continuous barrier monitoring via Brownian-bridge correction.
5. **Nested-simulation risk** — full MC revaluation of path-dependent books
   inside the scenario engine (GPU-only workload).

## License

MIT — see [LICENSE](LICENSE).
