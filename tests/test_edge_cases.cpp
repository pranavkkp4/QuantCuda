// TC-EDG-001..008: edge cases from the specification (P1).
#include "quantcuda/json_config.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
#include "quantcuda/risk_cpu.hpp"
#include "test_util.hpp"

using namespace quantcuda;

QC_TEST(edge_cases, maturity_zero_intrinsic) {
    McResult r = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 110.0, 100.0, 0.05, 0.0, 0.2, 0.0, 10'000, 42);
    QC_CHECK_NEAR(r.price, 10.0, 1e-12);
    QC_CHECK_NEAR(r.std_error, 0.0, 1e-15);
    QC_CHECK(r.paths == 10'000);
}

QC_TEST(edge_cases, sigma_zero_deterministic) {
    McResult r = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanPut, 100.0, 120.0, 0.05, 0.0, 0.0, 1.0, 10'000, 42);
    double fwd = 100.0 * std::exp(0.05);
    QC_CHECK_NEAR(r.price, std::exp(-0.05) * (120.0 - fwd), 1e-10);
}

QC_TEST(edge_cases, spot_zero) {
    // Call on worthless underlying is worthless.
    McResult c = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 0.0, 100.0, 0.05, 0.0, 0.2, 1.0, 1'000, 42);
    QC_CHECK_NEAR(c.price, 0.0, 1e-15);
    // Put is worth the discounted strike.
    McResult p = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanPut, 0.0, 100.0, 0.05, 0.0, 0.2, 1.0, 1'000, 42);
    QC_CHECK_NEAR(p.price, 100.0 * std::exp(-0.05), 1e-10);
    QC_CHECK_NEAR(BlackScholesPricer::price(OptionType::EuropeanCall, 0.0, 100.0,
                                            0.05, 0.0, 0.2, 1.0),
                  0.0, 1e-15);
    QC_CHECK_NEAR(BlackScholesPricer::price(OptionType::EuropeanPut, 0.0, 100.0,
                                            0.05, 0.0, 0.2, 1.0),
                  100.0 * std::exp(-0.05), 1e-12);
}

QC_TEST(edge_cases, barrier_already_breached) {
    McResult up = CpuMonteCarloPricer::price_barrier(
        OptionType::BarrierUpOutCall, 150.0, 100.0, 140.0, 0.05, 0.0, 0.2, 1.0,
        52, 1'000, 42);
    QC_CHECK_NEAR(up.price, 0.0, 1e-15);
    McResult dn = CpuMonteCarloPricer::price_barrier(
        OptionType::BarrierDownOutPut, 5.0, 100.0, 10.0, 0.05, 0.0, 0.2, 1.0,
        52, 1'000, 42);
    QC_CHECK_NEAR(dn.price, 0.0, 1e-15);
}

QC_TEST(edge_cases, deep_itm_otm) {
    // Deep ITM call: price ~ discounted intrinsic (forward - K).
    McResult itm = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 500.0, 100.0, 0.05, 0.0, 0.20, 1.0, 100'000, 42);
    double reference = BlackScholesPricer::call(500.0, 100.0, 0.05, 0.0, 0.20, 1.0);
    QC_CHECK_NEAR(itm.price, reference, 4.0 * itm.std_error);
    // Deep OTM: worth ~nothing.
    McResult otm = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 10.0, 500.0, 0.05, 0.0, 0.20, 1.0, 100'000, 42);
    QC_CHECK(otm.price < 1e-8);
}

QC_TEST(edge_cases, very_small_path_count) {
    McResult r = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 8, 42);
    QC_CHECK(r.paths == 8);
    QC_CHECK(std::isfinite(r.price) && std::isfinite(r.std_error));
    QC_CHECK(r.price >= 0.0);
    QC_CHECK_THROWS(CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0, 42));
    QC_CHECK_THROWS(CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 100.0, 100.0, 0.05, 0.0, -0.2, 1.0, 8, 42));
}

QC_TEST(edge_cases, empty_portfolio) {
    Portfolio pf;
    MarketData m;
    m.risk_free_rate = 0.05;
    m.assets.push_back({"A", 100.0, 0.2, 0.0});
    pf.validate(m);  // empty portfolio is structurally valid
    SimulationConfig sc; sc.paths = 1'000; sc.seed = 1;
    RiskConfig rc; rc.scenarios = 100;
    RiskResult rr = PortfolioRiskEngineCpu::run(pf, m, sc, rc);
    QC_CHECK_NEAR(rr.base_value, 0.0, 1e-15);
    QC_CHECK(rr.estimates.size() == 2);
}

QC_TEST(edge_cases, path_count_not_divisible_by_block) {
    // CPU mirror of the GPU grid-size edge case: odd path counts must work.
    OptionSpec o; o.type = OptionType::EuropeanCall; o.strike = 100.0; o.maturity = 1.0;
    SimulationConfig cfg; cfg.paths = 123'457; cfg.seed = 42;
    McResult a = OpenMpMonteCarloPricer::price(o, 100.0, 0.05, 0.0, 0.2, cfg);
    QC_CHECK(a.paths == 123'457);
    QC_CHECK(std::isfinite(a.price));
}

QC_TEST(edge_cases, at_maturity_value_equals_intrinsic) {
    // T tiny: MC price within a few cents of intrinsic as T -> 0.
    McResult r = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 105.0, 100.0, 0.05, 0.0, 0.2, 1e-6, 50'000, 42);
    QC_CHECK_NEAR(r.price, 5.0, 0.02);
}
