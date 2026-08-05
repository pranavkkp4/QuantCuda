// TC-MC-001..006: Monte Carlo engines - convergence, agreement, determinism (P0/P1).
#include <cmath>

#include "quantcuda/black_scholes.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
#include "test_util.hpp"

using namespace quantcuda;

namespace {
const double S0 = 100.0, K = 105.0, R = 0.0425, Q = 0.01, V = 0.20, T = 1.0;
}

QC_TEST(monte_carlo, european_converges_to_black_scholes) {
    double bs = BlackScholesPricer::call(S0, K, R, Q, V, T);
    // Increasing path counts: estimate should approach the analytic price and
    // the absolute error should stay within ~4 standard errors.
    for (std::size_t n : {10'000UL, 100'000UL, 1'000'000UL}) {
        McResult r = CpuMonteCarloPricer::price_european(
            OptionType::EuropeanCall, S0, K, R, Q, V, T, n, 42);
        double z = std::fabs(r.price - bs) / r.std_error;
        QC_CHECK(z < 4.0);
    }
}

QC_TEST(monte_carlo, error_shrinks_like_inverse_sqrt_n) {
    McResult r1 = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 10'000, 42);
    McResult r2 = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 1'000'000, 42);
    double ratio = r1.std_error / r2.std_error;  // expect ~10x (sqrt(100))
    QC_CHECK(ratio > 6.0 && ratio < 16.0);
}

QC_TEST(monte_carlo, scalar_openmp_agreement) {
    McResult a = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 200'000, 42);
    OptionSpec o; o.type = OptionType::EuropeanCall; o.strike = K; o.maturity = T;
    SimulationConfig cfg; cfg.paths = 200'000; cfg.seed = 42;
    McResult b = OpenMpMonteCarloPricer::price(o, S0, R, Q, V, cfg);
    // Per-path draws identical; only reduction order differs.
    QC_CHECK_NEAR(a.price, b.price, 1e-10);

    // The same schedule-independent stream contract also covers paths with
    // many draws and barrier early exits, not just one-draw Europeans.
    cfg.paths = 20'000;
    cfg.time_steps = 24;
    OptionSpec asian;
    asian.type = OptionType::AsianPut;
    asian.strike = K;
    asian.maturity = T;
    McResult asian_scalar = CpuMonteCarloPricer::price(asian, S0, R, Q, V, cfg);
    McResult asian_openmp = OpenMpMonteCarloPricer::price(asian, S0, R, Q, V, cfg);
    QC_CHECK_NEAR(asian_scalar.price, asian_openmp.price, 1e-10);

    OptionSpec barrier;
    barrier.type = OptionType::BarrierUpOutCall;
    barrier.strike = K;
    barrier.barrier = 125.0;
    barrier.maturity = T;
    McResult barrier_scalar = CpuMonteCarloPricer::price(barrier, S0, R, Q, V, cfg);
    McResult barrier_openmp = OpenMpMonteCarloPricer::price(
        barrier, S0, R, Q, V, cfg, BarrierCheckMode::EarlyExit);
    QC_CHECK_NEAR(barrier_scalar.price, barrier_openmp.price, 1e-10);
}

QC_TEST(monte_carlo, openmp_thread_count_invariance) {
    OptionSpec o; o.type = OptionType::EuropeanCall; o.strike = K; o.maturity = T;
    SimulationConfig cfg; cfg.paths = 100'000; cfg.seed = 42;
    cfg.openmp_threads = 1;
    McResult one = OpenMpMonteCarloPricer::price(o, S0, R, Q, V, cfg);
    cfg.openmp_threads = 0;  // all cores
    McResult all = OpenMpMonteCarloPricer::price(o, S0, R, Q, V, cfg);
    QC_CHECK_NEAR(one.price, all.price, 1e-10);
}

QC_TEST(monte_carlo, asian_below_vanilla_for_same_strike) {
    // Arithmetic-average call is worth less than the otherwise identical
    // European call (average has lower volatility than the terminal price).
    SimulationConfig cfg; cfg.paths = 100'000; cfg.seed = 42; cfg.time_steps = 252;
    OptionSpec eu; eu.type = OptionType::EuropeanCall; eu.strike = S0; eu.maturity = T;
    OptionSpec as; as.type = OptionType::AsianCall; as.strike = S0; as.maturity = T;
    double veu = CpuMonteCarloPricer::price(eu, S0, R, Q, V, cfg).price;
    double vas = CpuMonteCarloPricer::price(as, S0, R, Q, V, cfg).price;
    QC_CHECK(vas < veu);
    QC_CHECK(vas > 0.0);
}

QC_TEST(monte_carlo, determinism_same_seed) {
    McResult a = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 50'000, 99);
    McResult b = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 50'000, 99);
    QC_CHECK(a.price == b.price);
    QC_CHECK(a.std_error == b.std_error);
}
