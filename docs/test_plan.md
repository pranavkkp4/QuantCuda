# Test plan — QuantCUDA Risk Engine

## 1. Strategy

Behavior-driven tests through **public interfaces only** (engines, JSON
loader, statistics functions): tests describe *what* the system computes, not
*how*, so internal refactors (e.g. a new reduction flavor) never break them.
Every test follows **AAA** (Arrange–Act–Assert) and carries a traceable ID
(`TC-<CATEGORY>-<NUMBER>` in file headers).

Two test kinds:

- **Unit tests** — exact values, identities, and error handling.
- **Statistical tests** — Monte Carlo estimates validated against standard
  errors (z-score < 4 ⇒ false-failure probability ~ 10⁻⁵ per check) and
  distributional properties (correlation, tail statistics).

## 2. Coverage matrix

| Category | File | Tests | Priority | Type |
|---|---|---|---|---|
| TC-BS | test_black_scholes.cpp | 8 | P0 | unit |
| TC-RNG | (in test_black_scholes) | 1 | P1 | statistical |
| TC-PAY | test_payoffs.cpp | 6 | P0 | unit/statistical |
| TC-MC | test_monte_carlo.cpp | 6 | P0 | statistical |
| TC-GRK | test_greeks.cpp | 5 | P0 | statistical |
| TC-COR | test_correlation.cpp | 5 | P0 | unit/statistical |
| TC-RSK | test_risk.cpp | 6 | P0 | unit/statistical |
| TC-JSN | test_json_config.cpp | 7 | P1 | unit (negative) |
| TC-EDG | test_edge_cases.cpp | 9 | P1 | unit |
| TC-GPU | test_cpu_gpu_agreement.cpp | 3 | P1 | statistical (GPU-gated) |
| **Total** | | **56** | | |

## 3. Quality gates (release criteria)

| Gate | Target | Status |
|---|---|---|
| Test execution | 100% of planned tests | ✅ 56/56 |
| Pass rate | ≥ 80% (target 100%) | ✅ 100% |
| P0 bugs open | 0 | ✅ |
| P1 bugs open | ≤ 5 | ✅ 0 |
| Compiler warnings (-Wall -Wextra) | 0 | ✅ |
| Black-Scholes agreement | every European within 4 SE | ✅ |

## 4. Bug severity policy (P0–P4)

| Severity | Definition | Example in this project |
|---|---|---|
| P0 blocker | wrong financial result, crash, data loss | MC price outside 4 SE of Black-Scholes |
| P1 critical | major feature broken with workaround | OpenMP results differ from scalar |
| P2 high | minor feature issue, edge case | JSON rejects a valid config |
| P3 medium | cosmetic | misaligned report column |
| P4 low | documentation typo | — |

### Bug log (development history)

| ID | Severity | Summary | Root cause | Resolution |
|---|---|---|---|---|
| BUG-001 | P2 (test) | Greeks reference test failed (γ=0.0638, ν=15.2093 expected) | **test expectation wrong**, engine correct: hand-verified d1=0.7693 → Γ=0.0500, ν=8.8134 | fixed test constants; added derivation comment |
| BUG-002 | P2 | `Real` used untemplated in barrier compaction kernels | draft kernels not templated | made phase kernels double-precision explicitly |
| BUG-003 | P3 | confusing no-GPU error message | `cudaGetDeviceCount` driver error surfaced raw | `ensure_device` maps to actionable message |

## 5. Statistical-test discipline

- Deterministic seeds everywhere: statistical tests are **repeatable**, not
  flaky by construction.
- Agreement tests use combined standard errors (`4·√(se₁² + se₂²)`) when both
  sides are estimates.
- CPU/GPU agreement uses the pre-generated-draws strategy for the CPU half
  and 4-SE tolerance for the GPU comparison, per the specification.

## 6. Environment coverage

| Environment | What runs | Verified |
|---|---|---|
| CPU-only build | all 56 tests, all CLI commands, all CPU benchmark suites | ✅ |
| CUDA build, no GPU | tests pass, GPU tests SKIP, GPU CLI paths error cleanly | ✅ |
| CUDA build + GPU | full suite incl. TC-GPU execution | run on target hardware |

## 7. Regression procedure

1. `cmake --build build && ctest --test-dir build --output-on-failure`
2. `./build/quantcuda validate --portfolio configs/european_call.json --paths 2000000`
3. `./build/quantcuda benchmark --suite convergence --backends scalar`
4. Review deviations > 4 SE before accepting any numerical change.
