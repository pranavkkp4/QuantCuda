# QuantCUDA architecture

## Design goals

1. **Every optimization generation stays alive.** GPU kernels V0–V7 are not
   replaced by their successors; they coexist behind one dispatch layer so the
   benchmark runner can time them against each other on identical workloads.
2. **CPU baselines are first-class.** The scalar engine is the correctness
   reference; OpenMP is the honest multicore baseline. GPU speedups are always
   reported against a stated baseline, never against a strawman.
3. **No virtual dispatch on the device.** Each instrument type has a
   specialized kernel. Dispatch happens once, on the host.
4. **Reproducibility by construction.** Counter-based Philox streams keyed by
   `(seed, path_id)` decouple randomness from scheduling.

## Component map

```text
                 ┌─────────────────────────────────────────────┐
                 │ main.cpp → cli.cpp (price/risk/validate/    │
                 │ greeks/benchmark/versions/info)             │
                 └───────────────┬─────────────────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────────┐
        │                        │                             │
┌───────▼────────┐     ┌─────────▼─────────┐        ┌──────────▼───────────┐
│ json_config.cpp │     │ CPU engines        │        │ cuda_pricer (public  │
│ + validation    │     │ (header-only)      │        │ GPU interface)       │
│  types.hpp      │     │                    │        │  or cuda_stub.cpp    │
└────────────────┘     │  black_scholes.hpp │        └──────────┬───────────┘
                       │  monte_carlo_cpu   │                   │
                       │  greeks_cpu.hpp    │        ┌──────────▼───────────┐
                       │  risk_cpu.hpp      │        │ .cu kernel modules:  │
                       │  cholesky.hpp      │        │ rng / european /     │
                       │  rng_cpu.hpp       │        │ asian / barrier /    │
                       └─────────┬─────────┘        │ greeks / correlation │
                                 │                  │ portfolio_risk /     │
                       ┌─────────▼─────────┐        │ reduction            │
                       │ benchmark_runner  │◄───────┤ kernels_common.cuh   │
                       │ (suites, timing,  │        └──────────────────────┘
                       │ CSV/env reports)  │
                       └───────────────────┘
```

## Data flow — pricing

1. `load_engine_input_file` parses and validates the portfolio JSON
   (ranges, matrix shape/symmetry/PD via Cholesky, workload caps).
2. The CLI merges config with CLI overrides into a `SimulationConfig`.
3. `price_one` dispatches on backend:
   - scalar → `CpuMonteCarloPricer` (accumulate per-path payoffs, return
     discounted mean + standard error),
   - openmp → `OpenMpMonteCarloPricer` (per-thread Welford moments combined in
     fixed order — deterministic for a fixed thread count),
   - cuda → `gpu::price_option` → version-dispatched kernels.
4. The report prints price, standard error, Black-Scholes reference and
   deviation for Europeans, optional Greeks, portfolio value and throughput.

## Data flow — portfolio risk

```text
portfolio + market ──► prepare()           (once, host: base prices + Greeks)
                   ──► Cholesky L          (once, host)
                   ──► correlated shocks   (scenarios × d; CPU or GPU kernel)
                   ──► scenario repricing  (European: BS full revaluation;
                                            path-dependent: delta-gamma)
                   ──► losses → sort → VaR/CVaR tails   (CPU std::sort or
                                                          GPU thrust::sort)
```

Base valuation happens **once** on the host even in the GPU path: it is a
one-time cost of a few hundred thousand paths and removes all instrument-type
complexity from the scenario kernel, which then needs only Black-Scholes
(closed form) and a delta-gamma formula.

## GPU version strategy

| Version | File | What changes vs previous |
|---|---|---|
| V0 naive | `european.cu` | separate simulate/payoff/centered-atomic kernels, global `ST`/payoff arrays, synchronous copies |
| V1 SoA | `european.cu` | batch kernels templated on AoS vs SoA parameter layout; centered atomics retained |
| V2 fused | `european.cu` | one fused kernel + shared-memory tree merge of Welford moments |
| V3 reduction | `european.cu`, `reduction.cu` | warp-shuffle Chan merge; standalone 4-flavor scalar-sum study |
| V4 specialized | `asian.cu`, `barrier.cu` | per-type kernels; barrier early-exit / boolean-mask / stream-compacted variants |
| V5 memory | `european.cu`, `correlation.cu` | `__constant__` params; constant-vs-global Cholesky staging |
| V6 async | `european.cu` | pinned host buffers + 4-stream chunked launches |
| V7 precision | everywhere via templates | `float` / `double` / mixed (`float` sim, `double` accumulate) |

Dispatch: `SimulationConfig::gpu_version` → `european_gpu` /
`european_async_gpu` switch in `european.cu`; `cuda_pricer.cu` routes public
calls. Version rows are timed with CUDA events (kernel) and wall clock (e2e).

## Determinism and reproducibility

- CPU: `make_path_rng(seed, path)` → Philox keyed per path. OpenMP reduces
  per-thread partials in fixed order ⇒ bitwise-stable results per thread
  count; scalar/OpenMP agree ≤ 1e-10 (tested).
- GPU: one `curandStatePhilox4_32_10_t` per path, `sequence == path_id`.
  Optimized kernels update Welford moments per thread, merge them with Chan's
  recurrence in each block, then merge block moments on the host in a fixed
  order. V0/V1 intentionally retain unordered centered device atomics as part
  of their baseline identities, so repeated runs may differ in final roundoff
  bits.
- The CPU/GPU agreement test (`tests/test_cpu_gpu_agreement.cpp`) compares
  prices within 4 combined standard errors (independent RNG implementations),
  and the pre-generated-draws strategy pins the CPU half of the comparison.

## Error handling

- Config/CLI errors: `std::runtime_error` with a specific message; exit code 2.
- CUDA errors: `QC_CUDA_CHECK` macro wraps every API call and throws with
  file:line. Device memory is RAII (`DeviceBuffer`).
- No-GPU-at-runtime is a first-class case: `gpu::available()` guards every
  GPU path with actionable messages; benchmark GPU rows are recorded as
  `skipped: no GPU available` rather than failing.

## Why header-only CPU engines

The CPU engine layer (`monte_carlo_cpu.hpp`, `greeks_cpu.hpp`, `risk_cpu.hpp`,
`cholesky.hpp`) is templated/header-only so tests, the CLI, the benchmark
runner and the CUDA dispatch layer share one implementation with zero
duplication and full inlining. The build links one static `quantcuda_core`
library (types + JSON) plus the GPU library (kernels or stub).
