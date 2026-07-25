# Correctness report

All values below are real outputs from the local release CUDA build. The GPU
test branches executed on the target RTX 3050; none were skipped.

## 1. Verified environment

```text
Operating system:          Windows
Compiler:                  MSVC 19.44, release (NDEBUG)
CPU:                       12th Gen Intel Core i5-12400F (6 cores / 12 threads)
OpenMP:                    12 max threads
GPU:                       NVIDIA GeForce RTX 3050, 6 GB, sm_86, 18 SMs
NVIDIA driver:             595.79
CUDA toolkit / nvcc:       13.3.73
```

The configured fat binary contains code for sm_75, sm_86, sm_89, and sm_90;
all four targets were verified with `cuobjdump`.

## 2. Release verification summary

| Check | Result |
|---|---|
| Custom unit/statistical harness | **56 passed, 0 failed, 0 skipped** |
| CTest integration | **1/1 passed** (`quantcuda_unit_and_statistical`) |
| CUDA runtime test branches | executed on RTX 3050; no skips |
| CUDA architecture images | sm_75 / sm_86 / sm_89 / sm_90 present |

Compute Sanitizer memcheck was also run against the high-risk GPU paths:

| Sanitized workload | Result |
|---|---|
| V6 streams + pinned asynchronous pricing | 0 errors |
| V1 AoS/SoA layout study | 0 errors |
| V4 barrier early-exit, mask, and compaction | 0 errors |
| GPU portfolio scenario/VaR engine | 0 errors |

## 3. Unit and statistical test coverage

| Suite | Tests | What they prove |
|---|---:|---|
| black_scholes | 8 | Hull call/put references, parity, hand-verified Greeks, T=0, σ=0, monotonicity, invalid inputs, deep ITM/OTM |
| rng | 1 | Philox seed determinism and normal mean/variance sanity |
| payoffs | 6 | European/Asian/barrier payoff mechanics, discounting, knock-out bounds, early-exit/mask equality |
| monte_carlo | 6 | Black-Scholes convergence, SE ∝ 1/√N, scalar/OpenMP agreement and thread invariance, Asian ordering, determinism |
| greeks | 5 | analytic vs FD-CRN vs pathwise delta, gamma/vega tolerances, and CRN variance reduction |
| correlation | 5 | Cholesky reconstruction, overlap-safe in-place transforms, non-PD rejection, near-singular accuracy, empirical correlation |
| risk | 6 | empirical VaR/CVaR, CVaR ≥ VaR, normal references, empty input, end-to-end risk invariants |
| json_config | 7 | valid input and full uint64 seed, malformed/missing/invalid fields, unknown instruments, non-PD correlation, workload caps |
| edge_cases | 9 | T=0, σ=0, S=0, breached barriers, deep ITM/OTM, zero/tiny/odd path counts, empty portfolio, tiny T |
| cpu_gpu_agreement | 3 | CPU reference draws; GPU European/Asian/barrier agreement; scale-stable V0–V7 moments; GPU correlation/risk invariants |
| **Total** | **56** | **all passed** |

## 4. CUDA Monte Carlo vs Black-Scholes

Command: `quantcuda validate --portfolio configs/european_call.json --backend
cuda --paths 1000000`, seed 42.

| Instrument | Black-Scholes | CUDA MC | MC standard error | Deviation / SE | Result |
|---|---:|---:|---:|---:|---|
| EU_CALL_001 | 7.164333 | 7.158748 | 0.012484 | 0.45 | OK |
| EU_PUT_001 | 3.601600 | 3.599290 | 0.006493 | 0.36 | OK |

Both estimates are comfortably inside the four-standard-error release gate.

## 5. Scalar convergence vs Black-Scholes

`configs/european_call.json` call leg, scalar backend, seed 42. Reference:
Black-Scholes = 7.164333252.

| Paths | MC price | Standard error | Absolute error | Error / SE |
|---:|---:|---:|---:|---:|
| 10,000 | 7.160951504 | 0.124096174 | 0.003381748 | 0.027 |
| 100,000 | 7.184893644 | 0.039322236 | 0.020560392 | 0.523 |
| 1,000,000 | 7.151236279 | 0.012460688 | 0.013096973 | 1.051 |
| 10,000,000 | 7.165623834 | 0.003948596 | 0.001290582 | 0.327 |

All four estimates are within 1.051 standard errors of the analytic value. A
least-squares fit of `log(SE)` against `log(paths)` gives slope **-0.499**,
matching the theoretical -0.5. Absolute Monte Carlo error is noisy and is not
used to estimate that slope. Source: `benchmarks/raw/convergence_scalar.csv`;
analysis/figure: `scripts/analyze_accuracy.py`.

## 6. CUDA Greeks agreement

500,000 paths, seed 42. Vega is per 1.00 change in volatility.

| Instrument | Greek | Analytic | CUDA FD-CRN | CUDA pathwise |
|---|---|---:|---:|---:|
| call | delta | 0.502351 | 0.503223 | 0.503335 |
| call | gamma | 0.019745 | 0.019957 | — |
| call | vega | 39.490479 | 39.467641 | — |
| put | delta | -0.298078 | -0.297807 | -0.297679 |
| put | gamma | 0.019934 | 0.019578 | — |
| put | vega | 29.900592 | 29.855625 | — |

The finite-difference estimates meet the test tolerances (Δ ±0.01,
Γ ±0.005, vega ±1.0), and both pathwise deltas meet their
four-standard-error checks.

## 7. Correlation engine

For the three-asset test matrix and 200,000 scenarios, maximum empirical
correlation error is **< 0.01**. Cholesky reconstruction error is **< 1e-12**;
non-positive-definite and singular matrices are rejected. Exact in-place and
partially overlapping factor applications are snapshot-safe. The CUDA test
also generated 20,000 two-asset shocks and recovered the requested 0.45
correlation within 0.03.

## 8. Portfolio risk consistency

The measured `portfolio-risk` workload uses the seven-instrument sample book,
50,000 one-day scenarios, and the same release seed/configuration.

| Backend | Base V₀ | VaR₉₅ | Median suite time |
|---|---:|---:|---:|
| OpenMP | 1538.093901 | 213.660639 | 29.696 ms |
| CUDA | 1538.093901 | 215.063410 | 26.009 ms |

The base valuation agrees exactly. CPU and GPU scenario generators are
independent implementations, so sampled VaR is compared statistically rather
than bitwise. The tests assert VaR₉₅ ≤ VaR₉₉, CVaR ≥ VaR at both levels,
finite P&L moments, and a positive finite best gain on the GPU path.

## 9. Edge-case matrix

| Case | Expected | Measured result |
|---|---|---|
| T = 0 | intrinsic value, zero SE, requested path count retained | exact |
| σ = 0 | discounted deterministic payoff | within 1e-10 |
| S = 0 | call 0 / put = K·e⁻ʳᵀ | exact |
| barrier pre-breached | 0 | exact |
| deep ITM | Black-Scholes within 4 SE | passed |
| deep OTM | < 1e-8 | passed |
| 8 paths | finite price and SE | passed |
| 0 paths / negative volatility | reject | passed |
| empty portfolio | V₀ = 0, risk estimates present | passed |
| paths not divisible by block | grid-stride execution succeeds | 123,457 paths passed |
| payoff near 10¹⁶ with ~10⁹ spread | finite, nonzero analytic-scale SE | V0–V7 passed within 3% |

## 10. Reproducibility and scope

- Scalar runs are bit-identical for a fixed seed. Scalar and OpenMP agree to
  1e-10 and OpenMP is invariant to thread count at that tolerance.
- CPU and CUDA both use counter-based Philox streams but different
  implementations, so cross-backend checks use combined sampling error.
- Optimized GPU reductions merge deterministic Welford/Chan block moments on
  the host. V0/V1 intentionally retain unordered centered device atomics and
  may vary in their final roundoff bits.
- Asian and barrier pricing select their specialized CUDA kernels independently
  of `--gpu-version`; the V0–V7 version table is the European experiment.
- Canonical benchmark CSVs record the workload fields and full environment;
  release validation commands in this report use seed 42.
