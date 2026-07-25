# Financial models

## 1. Underlying dynamics

All engines simulate **geometric Brownian motion** under the risk-neutral
measure. Over a step of length Δt,

```
S(t+Δt) = S(t) · exp( (r − q − σ²/2)·Δt + σ·√Δt·Z ),   Z ~ N(0,1)
```

- `r` — continuously compounded risk-free rate
- `q` — continuous dividend yield
- `σ` — annualized volatility

The log-Euler step is exact for GBM, so European options need only the
terminal draw; path-dependent options are stepped over discrete fixings.

## 2. Payoffs

| Instrument | Payoff at maturity T |
|---|---|
| European call | max(S_T − K, 0) |
| European put | max(K − S_T, 0) |
| Asian call/put | max(Â − K, 0) / max(K − Â, 0), Â = arithmetic mean of S over the fixings |
| Barrier (up/down-and-out) | European payoff if S never touches/crosses the barrier at any fixing, else 0 |

Barrier monitoring is **discrete** (at the simulated fixings). A discretely
monitored knock-out is worth more than its continuously monitored twin
(fewer chances to breach); this is stated here and in the test suite rather
than silently approximated.

Prices are discounted means: `V = e^{−rT} · E[payoff]`, reported with the
Monte Carlo standard error `σ_payoff / √N`. CPU engines use Welford moments;
optimized CUDA kernels use per-thread Welford updates and Chan merges across
threads and blocks. This avoids the catastrophic cancellation of
`E[X²] − E[X]²` when a payoff has a large location but small real dispersion.

## 3. Black-Scholes reference

`BlackScholesPricer` implements the standard closed forms in double
precision:

```
C = S·e^{−qT}·N(d1) − K·e^{−rT}·N(d2)
P = K·e^{−rT}·N(−d2) − S·e^{−qT}·N(−d1)
d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),   d2 = d1 − σ√T
```

Edge cases are explicit, not emergent:

- `T = 0` → intrinsic value;
- `σ = 0` → deterministic limit `e^{−rT}·payoff(S·e^{(r−q)T})`;
- `S = 0` → call is 0, put is `K·e^{−rT}`;
- negative inputs throw `std::invalid_argument`.

Analytic Greeks: `Δ = ±e^{−qT}N(±d1)`, `Γ = e^{−qT}n(d1)/(Sσ√T)`,
`ν = S·e^{−qT}n(d1)√T` (per unit σ). Reference values are verified against
Hull's worked example in `tests/test_black_scholes.cpp` and put-call parity
holds to 1e-10.

## 4. Monte Carlo Greeks

### Finite differences with common random numbers (CRN)

Central differences need three prices. Running **three independent
simulations** makes the difference of two noisy estimates: the variance of
`V(S+h) − V(S−h)` is the *sum* of both variances.

The engine instead draws **one** set of normals per path and prices all bumps
from the same draw. The bumps are then perfectly correlated samples, the
difference cancels the shared randomness, and the estimator variance drops
materially (demonstrated by `greeks.crn_reduces_fd_variance`). The same trick
supplies vega via `σ ± h_σ`.

```
Δ ≈ [V(S+h) − V(S−h)] / 2h          (central, CRN)
Γ ≈ [V(S+h) − 2V(S) + V(S−h)] / h²  (central, CRN)
ν ≈ [V(σ+h_σ) − V(σ−h_σ)] / 2h_σ    (central, CRN)
```

### Pathwise delta (European, GBM)

Interchanging derivative and expectation:

```
∂V/∂S₀ = e^{−rT} · E[ 1{S_T > K} · ∂S_T/∂S₀ ] = e^{−rT} · E[ 1{S_T > K} · S_T/S₀ ]   (call)
```

(negative sign for puts). The estimator is unbiased in the limit and is
validated against Black-Scholes delta within 4 standard errors
(`greeks.pathwise_delta_matches_analytic`).

The GPU exposes both: one fused CRN kernel (price + Δ + Γ + ν in a single
launch, five accumulators) and a separate pathwise-delta kernel.

## 5. Correlated multi-asset scenarios

For a `d`-asset book, scenario shocks are correlated standard normals:

```
R = L·Lᵀ        (Cholesky, computed once on the host)
z_corr = L·z,   z ~ N(0, I_d)
```

Each asset then takes one GBM step over the risk horizon `h` (default one
trading day). Correctness checks: Frobenius reconstruction `‖R − LLᵀ‖_F`
< 1e-12, empirical correlation of generated samples within 0.01 of target
(200k scenarios), non-positive-definite and singular matrices rejected, and
near-singular matrices (`ρ = 1 − 10⁻¹⁰`) decomposed accurately
(`tests/test_correlation.cpp`).

## 6. VaR and CVaR

Per scenario `i`: `P&L_i = V_i − V₀`, loss `L_i = −P&L_i`.

- **VaR_α** — the α-quantile of the loss distribution: the smallest `L` with
  `P(loss ≤ L) ≥ α` (empirical quantile after sorting).
- **CVaR_α** (expected shortfall) — the quantile integral
  `(1/(1-α)) ∫_α^1 VaR_u du`. For empirical samples, the implementation uses
  exactly the upper `(1-α)` probability mass, including a fractional boundary
  observation when `αN` is not an integer.

Statistical validation: on synthetic normal losses, VaR₉₉ and CVaR₉₉ recover
`μ + 2.326σ` and `μ + 2.665σ` within sampling tolerance
(`tests/test_risk.cpp`).

### Repricing policy inside scenarios

| Instrument | Scenario repricing |
|---|---|
| European | full Black-Scholes revaluation at the shocked spot (exact) |
| Asian / barrier | delta-gamma approximation around the base MC price: `Vᵢ ≈ V₀ᵢ + δᵢΔS + ½γᵢΔS²`, floored at 0 |

The delta-gamma branch is an **approximation by design** (documented here and
in the risk report): exact scenario revaluation of path-dependent books would
require nested simulation, which is roadmap item 5. All-European books are
exact under this engine.

The current Asian/barrier approximation holds the base maturity and monitoring
state fixed over the short risk horizon, so it captures spot delta/gamma but
not theta or newly observed fixings/barrier state. Use full nested repricing
when those effects are material.

## 7. Precision policy

- CPU: double precision throughout.
- GPU V7: three configurations are benchmarked — full `float`, full `double`
  (the accuracy reference), and **mixed** (`float` path simulation with
  `double` payoff accumulation). Select a production default only after the
  version study quantifies both bias and throughput on the target GPU.
