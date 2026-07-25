// TC-BS-001..008: Black-Scholes analytical engine (P0 - correctness reference).
#include "quantcuda/black_scholes.hpp"
#include "quantcuda/rng_cpu.hpp"
#include "test_util.hpp"

using quantcuda::BlackScholesPricer;
using quantcuda::OptionType;

// Hull, Options, Futures and Other Derivatives, standard worked example:
// S=42, K=40, r=10%, sigma=20%, T=0.5, q=0 -> call 4.7594, put 0.8086.
QC_TEST(black_scholes, hull_reference_call_put) {
    double c = BlackScholesPricer::call(42.0, 40.0, 0.10, 0.0, 0.20, 0.5);
    double p = BlackScholesPricer::put(42.0, 40.0, 0.10, 0.0, 0.20, 0.5);
    QC_CHECK_NEAR(c, 4.7594, 1e-3);
    QC_CHECK_NEAR(p, 0.8086, 1e-3);
}

QC_TEST(black_scholes, put_call_parity) {
    double S = 100.0, K = 95.0, r = 0.0425, q = 0.01, v = 0.25, T = 1.25;
    double c = BlackScholesPricer::call(S, K, r, q, v, T);
    double p = BlackScholesPricer::put(S, K, r, q, v, T);
    QC_CHECK_NEAR(c - p, BlackScholesPricer::parity_rhs(S, K, r, q, T), 1e-10);
}

QC_TEST(black_scholes, analytic_greeks_reference_values) {
    // Same Hull example, verified by hand:
    // d1 = 0.7693, N(d1) = 0.7791, N'(d1) = 0.2968
    // delta = N(d1) = 0.7791; gamma = N'(d1)/(S*sigma*sqrt(T)) = 0.0500;
    // vega  = S*N'(d1)*sqrt(T) = 8.8134 (per 1.00 change in sigma).
    auto g = BlackScholesPricer::greeks(OptionType::EuropeanCall,
                                        42.0, 40.0, 0.10, 0.0, 0.20, 0.5);
    QC_CHECK_NEAR(g.delta, 0.7791, 1e-3);
    QC_CHECK_NEAR(g.gamma, 0.0500, 1e-3);
    QC_CHECK_NEAR(g.vega, 8.8134, 1e-3);

    // A zero-strike call is S*exp(-qT), so its boundary delta at S=K=0
    // is the discounted carry factor rather than zero.
    auto zero_strike_call = BlackScholesPricer::greeks(
        OptionType::EuropeanCall, 0.0, 0.0, 0.05, 0.03, 0.20, 1.0);
    auto zero_strike_put = BlackScholesPricer::greeks(
        OptionType::EuropeanPut, 0.0, 0.0, 0.05, 0.03, 0.20, 1.0);
    QC_CHECK_NEAR(zero_strike_call.delta, std::exp(-0.03), 1e-12);
    QC_CHECK_NEAR(zero_strike_put.delta, 0.0, 1e-12);
}

QC_TEST(black_scholes, maturity_zero_is_intrinsic) {
    QC_CHECK_NEAR(BlackScholesPricer::call(110.0, 100.0, 0.05, 0.0, 0.20, 0.0), 10.0, 1e-12);
    QC_CHECK_NEAR(BlackScholesPricer::call(90.0, 100.0, 0.05, 0.0, 0.20, 0.0), 0.0, 1e-12);
    QC_CHECK_NEAR(BlackScholesPricer::put(90.0, 100.0, 0.05, 0.0, 0.20, 0.0), 10.0, 1e-12);
}

QC_TEST(black_scholes, zero_vol_is_deterministic_discount) {
    // sigma = 0 -> S grows deterministically at r-q; value = df * payoff(forward).
    double S = 100.0, K = 105.0, r = 0.05, q = 0.0, T = 1.0;
    double fwd = S * std::exp((r - q) * T);   // ~105.127 -> ITM call
    double expected = std::exp(-r * T) * (fwd - K);
    QC_CHECK_NEAR(BlackScholesPricer::call(S, K, r, q, 0.0, T), expected, 1e-12);
    // Deep OTM under zero vol stays worthless.
    QC_CHECK_NEAR(BlackScholesPricer::call(S, 120.0, r, q, 0.0, T), 0.0, 1e-12);
}

QC_TEST(black_scholes, monotonicity_in_spot_and_strike) {
    double base = BlackScholesPricer::call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
    QC_CHECK(BlackScholesPricer::call(110.0, 100.0, 0.05, 0.0, 0.20, 1.0) > base);
    QC_CHECK(BlackScholesPricer::call(100.0, 110.0, 0.05, 0.0, 0.20, 1.0) < base);
    QC_CHECK(BlackScholesPricer::put(100.0, 110.0, 0.05, 0.0, 0.20, 1.0) >
             BlackScholesPricer::put(100.0, 100.0, 0.05, 0.0, 0.20, 1.0));
}

QC_TEST(black_scholes, invalid_inputs_throw) {
    QC_CHECK_THROWS(BlackScholesPricer::call(-1.0, 100.0, 0.05, 0.0, 0.2, 1.0));
    QC_CHECK_THROWS(BlackScholesPricer::call(100.0, -5.0, 0.05, 0.0, 0.2, 1.0));
    QC_CHECK_THROWS(BlackScholesPricer::call(100.0, 100.0, 0.05, 0.0, -0.2, 1.0));
    QC_CHECK_THROWS(BlackScholesPricer::call(100.0, 100.0, 0.05, 0.0, 0.2, -1.0));
}

QC_TEST(black_scholes, deep_itm_otm_limits) {
    // Deep ITM call ~ discounted forward intrinsic; deep OTM ~ 0.
    double deep_itm = BlackScholesPricer::call(500.0, 100.0, 0.05, 0.0, 0.20, 1.0);
    QC_CHECK_NEAR(deep_itm, 500.0 - 100.0 * std::exp(-0.05), 1.0);
    QC_CHECK(BlackScholesPricer::call(1.0, 500.0, 0.05, 0.0, 0.20, 1.0) < 1e-8);
}

// Philox CPU generator: determinism and basic distribution sanity.
QC_TEST(rng, philox_deterministic_and_sane) {
    quantcuda::Philox4x32_10 g1(42, 0), g2(42, 0), g3(43, 0);
    double a = g1.next_normal();
    QC_CHECK(a == g2.next_normal());   // same seed -> identical stream
    QC_CHECK(a != g3.next_normal());   // different seed -> different stream

    quantcuda::Philox4x32_10 g(7, 0);
    double sum = 0.0, sq = 0.0;
    const int n = 200'000;
    for (int i = 0; i < n; ++i) {
        double z = g.next_normal();
        sum += z; sq += z * z;
    }
    double mean = sum / n, var = sq / n - mean * mean;
    QC_CHECK_NEAR(mean, 0.0, 0.02);    // ~4 sigma of sample mean
    QC_CHECK_NEAR(var, 1.0, 0.02);

    // Regression: the old fixed 4096-counter stride made these two streams
    // identical, correlating long Asian/barrier paths with later paths.
    auto late_path_zero = quantcuda::make_path_rng(42, 0, 4096);
    auto path_one_start = quantcuda::make_path_rng(42, 1, 0);
    bool any_different = false;
    for (int i = 0; i < 4; ++i)
        any_different = any_different ||
                        late_path_zero.next_u32() != path_one_start.next_u32();
    QC_CHECK(any_different);
}
