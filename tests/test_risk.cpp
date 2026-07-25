// TC-RSK-001..006: VaR / CVaR statistics and portfolio risk engine (P0/P1).
#include <cmath>

#include "quantcuda/risk_cpu.hpp"
#include "test_util.hpp"

using namespace quantcuda;

QC_TEST(risk, var_percentile_selection) {
    // 100 ordered losses 1..100: 95% VaR = 95, 99% VaR = 99.
    std::vector<double> losses(100);
    for (int i = 0; i < 100; ++i) losses[static_cast<std::size_t>(i)] = i + 1;
    auto est = var_cvar_from_losses(losses, {0.95, 0.99});
    QC_CHECK_NEAR(est[0].var, 95.0, 1e-12);
    QC_CHECK_NEAR(est[1].var, 99.0, 1e-12);
}

QC_TEST(risk, cvar_tail_averaging) {
    // Quantile-integral expected shortfall uses exactly the upper 5%/1% mass:
    // losses 1..100 therefore give means of 96..100 and 100, respectively.
    std::vector<double> losses(100);
    for (int i = 0; i < 100; ++i) losses[static_cast<std::size_t>(i)] = i + 1;
    auto est = var_cvar_from_losses(losses, {0.95, 0.99});
    QC_CHECK_NEAR(est[0].cvar, 98.0, 1e-12);
    QC_CHECK_NEAR(est[1].cvar, 100.0, 1e-12);

    // alpha*n = 2.5: half of loss 3 plus all of loss 4, divided by 1.5.
    auto fractional = var_cvar_from_losses({1.0, 2.0, 3.0, 4.0}, {0.625});
    QC_CHECK_NEAR(fractional[0].var, 3.0, 1e-12);
    QC_CHECK_NEAR(fractional[0].cvar, 11.0 / 3.0, 1e-12);
}

QC_TEST(risk, cvar_at_least_var) {
    std::vector<double> losses = {5, -3, 12, 0, 7, 7, 9, -20, 3, 11, 100, 2};
    auto est = var_cvar_from_losses(losses, {0.5, 0.9, 0.99});
    for (const auto& e : est) QC_CHECK(e.cvar >= e.var - 1e-12);
}

QC_TEST(risk, normal_analytic_check) {
    // Losses ~ N(mu, sigma): VaR_99 ~ mu + 2.326 sigma (sampling tolerance).
    double mu = 1.0, sigma = 2.0;
    const std::size_t n = 400'000;
    std::vector<double> losses(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto rng = make_path_rng(555, i);
        losses[i] = mu + sigma * rng.next_normal();
    }
    auto est = var_cvar_from_losses(std::move(losses), {0.99});
    QC_CHECK_NEAR(est[0].var, mu + 2.326 * sigma, 0.06);
    QC_CHECK_NEAR(est[0].cvar, mu + 2.665 * sigma, 0.08);  // normal ES closed form
}

QC_TEST(risk, empty_losses_safe) {
    auto est = var_cvar_from_losses({}, {0.95});
    QC_CHECK(est.empty());
    QC_CHECK_THROWS(var_cvar_from_losses({}, {std::nan("")}));
    QC_CHECK_THROWS(var_cvar_from_losses({1.0, std::nan("")}, {0.95}));
}

QC_TEST(risk, single_asset_portfolio_engine) {
    // Long 1 ATM European call, single asset: run end-to-end and check
    // internal consistency (VaR ordering, CVaR >= VaR, finite values).
    MarketData m;
    m.risk_free_rate = 0.05;
    m.assets.push_back({"AAA", 100.0, 0.20, 0.0});
    Portfolio pf;
    OptionSpec o;
    o.id = "C1"; o.type = OptionType::EuropeanCall; o.underlying = "AAA";
    o.strike = 100.0; o.maturity = 0.5; o.quantity = 10.0;
    pf.options.push_back(o);
    SimulationConfig sc; sc.paths = 20'000; sc.seed = 42;
    RiskConfig rc; rc.scenarios = 50'000; rc.horizon = 1.0 / 252.0;
    RiskResult rr = PortfolioRiskEngineCpu::run(pf, m, sc, rc);
    QC_CHECK(rr.base_value > 0.0);
    QC_CHECK(rr.estimates.size() == 2);
    QC_CHECK(rr.estimates[0].var <= rr.estimates[1].var + 1e-12);  // 95 <= 99
    QC_CHECK(rr.estimates[0].cvar >= rr.estimates[0].var - 1e-12);
    QC_CHECK(rr.estimates[1].cvar >= rr.estimates[1].var - 1e-12);
    QC_CHECK(std::isfinite(rr.pnl_mean) && std::isfinite(rr.pnl_stddev));
    RiskResult scalar = PortfolioRiskEngineCpu::run(pf, m, sc, rc, Backend::Scalar);
    QC_CHECK(scalar.backend == Backend::Scalar);
    QC_CHECK_NEAR(scalar.base_value, rr.base_value, 1e-12);
    QC_CHECK_NEAR(scalar.estimates[0].var, rr.estimates[0].var, 1e-12);
    RiskConfig bad = rc;
    bad.scenarios = 0;
    QC_CHECK_THROWS(PortfolioRiskEngineCpu::run(pf, m, sc, bad, Backend::Scalar));
    bad = rc;
    bad.horizon = 1.0;
    QC_CHECK_THROWS(PortfolioRiskEngineCpu::run(pf, m, sc, bad, Backend::Scalar));
}
