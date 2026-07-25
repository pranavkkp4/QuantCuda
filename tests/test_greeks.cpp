// TC-GRK-001..005: Greeks engines - analytic vs FD vs pathwise (P0/P1).
#include "quantcuda/black_scholes.hpp"
#include "quantcuda/greeks_cpu.hpp"
#include "test_util.hpp"

using namespace quantcuda;

namespace {
const double S0 = 100.0, K = 105.0, R = 0.0425, Q = 0.01, V = 0.20, T = 1.0;
}

QC_TEST(greeks, fd_matches_analytic_with_crn) {
    Greeks ref = BlackScholesPricer::greeks(OptionType::EuropeanCall,
                                            S0, K, R, Q, V, T);
    McGreeksResult fd = CpuGreeksEngine::european_fd(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 500'000, 42);
    QC_CHECK_NEAR(fd.greeks.delta, ref.delta, 0.01);
    QC_CHECK_NEAR(fd.greeks.gamma, ref.gamma, 0.005);
    QC_CHECK_NEAR(fd.greeks.vega, ref.vega, 1.0);
}

QC_TEST(greeks, pathwise_delta_matches_analytic) {
    Greeks ref = BlackScholesPricer::greeks(OptionType::EuropeanCall,
                                            S0, K, R, Q, V, T);
    McResult pw = CpuGreeksEngine::european_delta_pathwise(
        OptionType::EuropeanCall, S0, K, R, Q, V, T, 500'000, 42);
    QC_CHECK_NEAR(pw.price, ref.delta, 4.0 * pw.std_error + 1e-3);
}

QC_TEST(greeks, pathwise_put_delta_matches_analytic) {
    Greeks ref = BlackScholesPricer::greeks(OptionType::EuropeanPut,
                                            S0, K, R, Q, V, T);
    McResult pw = CpuGreeksEngine::european_delta_pathwise(
        OptionType::EuropeanPut, S0, K, R, Q, V, T, 500'000, 42);
    QC_CHECK_NEAR(pw.price, ref.delta, 4.0 * pw.std_error + 1e-3);
}

QC_TEST(greeks, crn_reduces_fd_variance) {
    // Same draws (CRN) vs independent draws for the delta difference:
    // CRN per-path delta contributions must have lower variance.
    const std::size_t n = 20'000;
    double h = 1.0;
    double drift = (R - Q - 0.5 * V * V) * T, diff = V * std::sqrt(T);
    detail::Accumulator crn, indep;
    for (std::size_t i = 0; i < n; ++i) {
        auto rng1 = make_path_rng(42, i);
        double z1 = rng1.next_normal();
        double up = detail::european_path(OptionType::EuropeanCall, S0 + h, K, drift, diff, z1);
        double dn = detail::european_path(OptionType::EuropeanCall, S0 - h, K, drift, diff, z1);
        crn.add((up - dn) / (2 * h));
        auto rng2 = make_path_rng(4242, i);  // independent second stream
        double z2 = rng2.next_normal();
        double up2 = detail::european_path(OptionType::EuropeanCall, S0 + h, K, drift, diff, z2);
        double dn2 = detail::european_path(OptionType::EuropeanCall, S0 - h, K, drift, diff, z1);
        indep.add((up2 - dn2) / (2 * h));
    }
    McResult rc = detail::finalize(crn, n, 1.0);
    McResult ri = detail::finalize(indep, n, 1.0);
    QC_CHECK(rc.std_error < ri.std_error);
}

QC_TEST(greeks, analytic_edge_cases) {
    // T=0: delta is 1 for ITM call, 0 for OTM.
    Greeks itm = BlackScholesPricer::greeks(OptionType::EuropeanCall,
                                            110.0, 100.0, 0.05, 0.0, 0.2, 0.0);
    QC_CHECK_NEAR(itm.delta, 1.0, 1e-12);
    Greeks otm = BlackScholesPricer::greeks(OptionType::EuropeanCall,
                                            90.0, 100.0, 0.05, 0.0, 0.2, 0.0);
    QC_CHECK_NEAR(otm.delta, 0.0, 1e-12);
    // Low-volatility vega uses a one-sided bump rather than dividing a clamped
    // interval by the central-difference denominator.
    McGreeksResult low_vol = CpuGreeksEngine::european_fd(
        OptionType::EuropeanCall, 100.0, 100.0, 0.05, 0.0, 0.005, 1.0,
        50'000, 42);
    QC_CHECK(std::isfinite(low_vol.greeks.vega));
    QC_CHECK_THROWS(CpuGreeksEngine::european_delta_pathwise(
        OptionType::EuropeanCall, 0.0, 100.0, 0.05, 0.0, 0.2, 1.0,
        100, 42));
}
