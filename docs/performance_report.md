# Performance report

All values in this report are real measurements from the release CUDA build on
the local target machine. Every table is populated from captured benchmark output.

## 1. Measurement environment and protocol

```text
Operating system:          Windows
Compiler:                  MSVC 19.44, release (NDEBUG)
CPU:                       12th Gen Intel Core i5-12400F (6 cores / 12 threads)
OpenMP:                    12 max threads
GPU:                       NVIDIA GeForce RTX 3050, 6 GB, sm_86, 18 SMs
NVIDIA driver:             595.79
CUDA toolkit / nvcc:       13.3.73
CUDA fatbins verified:     sm_75, sm_86, sm_89, sm_90
```

Timed suites use a warm-up followed by repeated runs with seed 42. Tables show
min/median/p95 where the suite records timing samples. Standard backend rows
measure end-to-end wall time. The dedicated GPU studies use CUDA events for a
GPU interval and report end-to-end wall time separately where available. Every
CSV row embeds the build/environment block and labels `timing_scope`; the
`gpu_interval_ms` and `end_to_end_ms` columns remove any timing-boundary
ambiguity. For V6, the GPU interval extends through asynchronous D2H completion
as noted in §5.

All nine benchmark suites were executed:

| Suite | Measured workload | Repeats |
|---|---|---:|
| `european-throughput` | 1 European call × 10⁷ paths | 5 |
| `path-dependent` | 8 Asian calls × 10⁵ paths × 252 steps | 3 |
| `divergence` | near/medium/far barrier × 10⁵ paths × 252 steps | 3 |
| `portfolio-risk` | 7 instruments × 50,000 scenarios | 5 |
| `scaling` | 1–1024 instruments × 50,000 paths | 3 |
| `convergence` | 10⁴, 10⁵, 10⁶, 10⁷ paths | accuracy sweep |
| `version-study` | V0–V7 + four reductions × 10⁷ values | 5 |
| `gpu-divergence` | 3 barriers × 3 strategies × 10⁶ paths × 252 steps | 5 |
| `layout-study` | AoS/SoA × 64 instruments × 10⁵ paths | 5 |

## 2. European throughput and CPU scaling

### 2.1 European throughput

European call: S=100, K=105, r=4.25%, q=1%, σ=20%, T=1, seed 42,
10,000,000 paths. These are end-to-end suite times.

| Backend | Min ms | Median ms | P95 ms | Throughput | Price |
|---|---:|---:|---:|---:|---:|
| scalar | 643.146 | 647.544 | 652.144 | 15.44 M paths/s | 7.165624 |
| OpenMP | 123.400 | 134.055 | 141.141 | 74.60 M paths/s | 7.165624 |
| CUDA V7 double | 68.060 | 68.900 | 87.008 | 145.14 M paths/s | 7.167997 |

CUDA is **1.95× faster than OpenMP** and **9.40× faster than scalar** on
this workload. The scalar/OpenMP and CUDA prices are each within one standard
error of the Black-Scholes reference. Source:
`benchmarks/raw/european_throughput_10M.csv`.

### 2.2 Instrument scaling

Each entry uses 50,000 paths per instrument. Throughput stays nearly flat as
the book grows, which is the expected linear-work signature.

| Instruments | Scalar median ms | Scalar M paths/s | OpenMP median ms | OpenMP M paths/s |
|---:|---:|---:|---:|---:|
| 1 | 3.196 | 15.65 | 0.751 | 66.58 |
| 8 | 25.951 | 15.41 | 5.789 | 69.10 |
| 64 | 200.772 | 15.94 | 43.585 | 73.42 |
| 256 | 801.771 | 15.96 | 193.161 | 66.27 |
| 1024 | 3212.304 | 15.94 | 743.564 | 68.86 |

Source: `benchmarks/raw/scaling_50k.csv`.

## 3. Path-dependent pricing, divergence, and portfolio risk

### 3.1 Asian path workload

Eight arithmetic-average Asian calls, 100,000 paths per option, 252 fixings.

| Backend | Min ms | Median ms | P95 ms | Throughput | Last-option price ± SE |
|---|---:|---:|---:|---:|---:|
| OpenMP | 1266.158 | 1269.829 | 1324.271 | 0.630 M paths/s | 6.961642 ± 0.027124 |
| CUDA | 193.191 | 193.385 | 197.768 | 4.137 M paths/s | 6.934856 ± 0.027134 |

CUDA is **6.57× faster than OpenMP**. The two estimates agree within their
combined Monte Carlo uncertainty.

### 3.2 CPU barrier divergence

Scalar up-and-out call, 100,000 paths and 252 fixings. The ratio is boolean-mask
median divided by early-exit median.

| Barrier | Strategy | Median ms | Price | Mask / early ratio |
|---|---|---:|---:|---:|
| near 110 | early-exit | 905.038 | 0.024288 | **1.71×** |
| near 110 | boolean mask | 1551.034 | 0.024288 | 1.00× |
| medium 130 | early-exit | 1449.512 | 2.022629 | **1.09×** |
| medium 130 | boolean mask | 1576.714 | 2.022629 | 1.00× |
| far 180 | early-exit | 1558.947 | 6.861154 | ≈1.00× |
| far 180 | boolean mask | 1564.823 | 6.861154 | 1.00× |

The CPU hypothesis is confirmed: early termination matters near the barrier
and disappears when almost every path survives. Prices are identical between
strategies because they consume the same per-path streams. OpenMP and CUDA
end-to-end rows are retained in `benchmarks/raw/divergence_100k_252.csv`; the
dedicated GPU kernel experiment is in §6.3.

### 3.3 Portfolio risk

`configs/multi_asset_portfolio.json`, 7 instruments, 50,000 one-day scenarios.

| Backend | Min ms | Median ms | P95 ms | Throughput | Base V₀ | VaR₉₅ |
|---|---:|---:|---:|---:|---:|---:|
| OpenMP | 28.361 | 29.696 | 38.530 | 1.684 M scenarios/s | 1538.093901 | 213.660639 |
| CUDA | 25.532 | 26.009 | 27.775 | 1.922 M scenarios/s | 1538.093901 | 215.063410 |

CUDA is **1.14× faster than OpenMP** here. The modest gain is an honest result:
this small seven-instrument workload includes fixed host preparation and
transfer costs. CPU and GPU use independent scenario streams, so their sampled
VaR values need not be bit-identical.

## 4. Monte Carlo convergence

For the scalar European call, the fitted log-log slope of standard error versus
path count is **-0.499** (theory: -0.5). All four estimates are within 1.051
standard errors of Black-Scholes. The detailed price/error table is in
`docs/correctness_report.md`; source data is
`benchmarks/raw/convergence_scalar.csv`.

## 5. European GPU generation study (V0–V7)

Command workload: 10,000,000 paths, five measured repeats after warm-up.

| Version / change | GPU interval min / median / p95 ms | E2E median ms | Price | Result |
|---|---:|---:|---:|---|
| V0 naive: split + centered atomic + sync | 57.908 / 58.900 / 59.450 | 84.412 | 7.167997 | baseline |
| V1 SoA parameter path | 26.031 / 29.601 / 30.968 | 51.593 | 7.167997 | 1.99× GPU-interval gain vs V0 |
| V2 fused + shared-tree moments | 28.505 / 30.159 / 30.657 | 52.451 | 7.167997 | 1.95× GPU interval vs V0 |
| V3 warp/Chan reduction | 41.854 / 42.646 / 44.681 | 66.287 | 7.167997 | 1.38× GPU interval, 1.27× E2E vs V0 |
| V4 European-specialized | 42.429 / 43.501 / 44.408 | 65.922 | 7.167997 | statistically near V3 |
| V5 constant parameters | 41.622 / 42.941 / 43.973 | 65.410 | 7.167997 | no measurable win here |
| V6 four streams + pinned memory | 41.554 / 42.895 / 43.501 | 66.758 | 7.167997 | E2E within 3% of V5 |
| V7 float | 4.431 / 4.814 / 5.024 | 27.814 | 7.163358 | 8.91× GPU interval vs double |
| V7 mixed | 22.955 / 23.908 / 25.721 | 46.402 | 7.163358 | 1.79× GPU interval vs double |
| V7 double | 41.694 / 42.905 / 44.498 | 65.898 | 7.167997 | precision reference |

The V6 field historically named `kernel_ms` is broader than a compute-only
kernel: it covers the asynchronous CUDA-event interval through D2H completion.
Its E2E column is the valid whole-operation comparison. The float and mixed
prices differed from the double estimate by 0.004639, or 1.17 of the reported
double-estimator standard error. A single seed cannot establish numerical
bias, so no precision mode is declared universally superior from this row.

V2–V7 compute mergeable Welford/Chan moments rather than cancellation-prone
raw second moments. On this GPU the shared-tree Chan merge in V2 is faster than
shuffling the three-field moment tuple in V3. That measured trade-off is kept
visible: V3 remains faster than V0, but it is not a win over V1 or V2 for this
numerically hardened statistic.

This table is specifically the European generation study. Asian and barrier
pricing always route to their specialized path kernels independently of
`--gpu-version`; compare those paths with the `path-dependent`, `divergence`,
and `gpu-divergence` suites.

## 6. Focused GPU studies

### 6.1 Reduction flavors

Ten million generated values; error is absolute difference from the host
reference sum.

| Flavor | Min / median / p95 ms | Speedup vs atomic | Abs error |
|---|---:|---:|---:|
| atomic per thread | 18.385 / 19.018 / 19.203 | 1.00× | 6.28e-12 |
| shared-memory tree | 1.491 / 1.495 / 1.503 | 12.72× | 1.59e-12 |
| warp shuffle + atomic | 1.034 / 1.049 / 1.399 | **18.14×** | 2.08e-11 |
| warp shuffle + shared memory | 1.166 / 1.181 / 1.736 | 16.11× | 0 |

The warp-shuffle + atomic flavor was fastest on this RTX 3050. All four sums
agreed with the host reference to better than 5e-11.

### 6.2 V1 layout

64 instruments × 100,000 paths, five repeats.

| Layout | GPU interval ms | E2E ms | Throughput |
|---|---:|---:|---:|
| AoS | 18.577 | 36.309 | 176.26 M paths/s |
| SoA | 13.232 | 28.359 | 225.68 M paths/s |

SoA is **1.40× faster in the GPU interval** and **1.28× end-to-end**.

### 6.3 V4 GPU barrier divergence

One million paths, 252 steps, five repeats. These are CUDA-event kernel times.

| Barrier | Knockout | Strategy | Min / median / p95 ms | Price |
|---|---:|---|---:|---:|
| near 110 | 62.7% | early-exit | 234.036 / 234.948 / 235.152 | 0.023360 |
| near 110 | 62.7% | boolean mask | 234.540 / 235.642 / 236.726 | 0.023360 |
| near 110 | 62.7% | compacted | 177.648 / 178.628 / 179.451 | 0.023360 |
| medium 130 | 19.4% | early-exit | 234.074 / 235.568 / 235.970 | 2.015462 |
| medium 130 | 19.4% | boolean mask | 233.973 / 234.582 / 235.363 | 2.015462 |
| medium 130 | 19.4% | compacted | 227.043 / 228.206 / 229.263 | 2.015462 |
| far 180 | 0.4% | early-exit | 234.507 / 234.590 / 235.473 | 6.910883 |
| far 180 | 0.4% | boolean mask | 233.633 / 235.555 / 235.683 | 6.910883 |
| far 180 | 0.4% | compacted | 234.411 / 235.444 / 235.875 | 6.910883 |

On the GPU, early-exit and boolean-mask kernels are effectively tied because
warp divergence prevents a single knocked-out lane from retiring the whole
warp. Compaction is **1.32× faster than masking near the barrier**, only
1.03× faster at the medium barrier, and neutral at the far barrier. This is
the measured GPU counterpart to the different CPU branch result in §3.2.

## 7. Release verification and conclusions

- Compute Sanitizer memcheck reported **0 errors** for V6 async pricing, V1
  layout, all V4 divergence/compaction paths, and GPU portfolio risk.
- `cuobjdump` verified code objects for **sm_75, sm_86, sm_89, and sm_90**.
- V0 → V2 is a clear improvement; V3's stable warp/Chan tuple merge is slower
  than V2's shared tree on this device, while V5 and V6 are neutral. These
  measured non-wins are retained rather than recast as optimizations.
- GPU speedups are always stated against a named scalar, OpenMP, layout, or
  previous-version baseline.
- Nsight capture commands and the per-kernel metric checklist are in
  `profiling/README.md`.
