// TC-GPU-001..004: CPU/GPU agreement (P1).
//
// The first check proves CPU stream identity against explicitly generated
// draws. CUDA uses cuRAND rather than externally supplied CPU draws, so the
// CPU/GPU comparisons below use independent RNG streams and statistical
// tolerances based on their combined standard errors.
#include <limits>

#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/cholesky.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
#include "test_util.hpp"

using namespace quantcuda;

namespace {
// CPU reference evaluated from pre-generated normals: identical z values
// must produce the engine's identical discounted payoff sum.
double cpu_price_with_draws(OptionType t, double S0, double K, double r, double q,
                            double sigma, double T, const std::vector<double>& zs) {
    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double diff = sigma * std::sqrt(T);
    double sum = 0.0;
    for (double z : zs) sum += detail::european_path(t, S0, K, drift, diff, z);
    return std::exp(-r * T) * sum / static_cast<double>(zs.size());
}
}  // namespace

QC_TEST(cpu_gpu_agreement, cpu_reference_with_precomputed_draws) {
    std::vector<double> zs;
    zs.reserve(100'000);
    for (std::size_t i = 0; i < 100'000; ++i) {
        auto rng = make_path_rng(777, i);
        zs.push_back(rng.next_normal());
    }
    double with_draws = cpu_price_with_draws(OptionType::EuropeanCall,
                                             100.0, 105.0, 0.0425, 0.01, 0.2, 1.0, zs);
    // Same seed through the engine => identical sequence => identical price.
    McResult eng = CpuMonteCarloPricer::price_european(
        OptionType::EuropeanCall, 100.0, 105.0, 0.0425, 0.01, 0.2, 1.0,
        zs.size(), 777);
    QC_CHECK_NEAR(with_draws, eng.price, 1e-10);
}

QC_TEST(cpu_gpu_agreement, gpu_matches_cpu_when_cuda_built) {
    if (!gpu::available()) {
        QC_SKIP("CUDA device unavailable");
    }
    OptionSpec o; o.type = OptionType::EuropeanCall; o.strike = 105.0; o.maturity = 1.0;
    SimulationConfig cfg; cfg.paths = 200'000; cfg.seed = 42;
    cfg.precision = Precision::Double;
    McResult cpu = CpuMonteCarloPricer::price(o, 100.0, 0.0425, 0.01, 0.2, cfg);
    McResult gpu_r = gpu::price_european(o, 100.0, 0.0425, 0.01, 0.2, cfg);
    // Independent RNG implementations: agree within combined sampling error.
    double tol = 4.0 * std::hypot(cpu.std_error, gpu_r.std_error);
    QC_CHECK(std::fabs(cpu.price - gpu_r.price) < tol);
}

QC_TEST(cpu_gpu_agreement, gpu_path_types_match_cpu_when_built) {
    if (!gpu::available()) {
        QC_SKIP("CUDA device unavailable");
    }
    SimulationConfig cfg; cfg.paths = 100'000; cfg.seed = 42; cfg.time_steps = 52;
    cfg.precision = Precision::Double;
    OptionSpec as; as.type = OptionType::AsianCall; as.strike = 100.0; as.maturity = 1.0;
    OptionSpec bo; bo.type = OptionType::BarrierUpOutCall; bo.strike = 105.0;
    bo.maturity = 1.0; bo.barrier = 140.0;
    McResult a_cpu = CpuMonteCarloPricer::price(as, 100.0, 0.05, 0.0, 0.25, cfg);
    McResult a_gpu = gpu::price_asian(as, 100.0, 0.05, 0.0, 0.25, cfg);
    QC_CHECK(std::fabs(a_cpu.price - a_gpu.price) <
             4.0 * std::hypot(a_cpu.std_error, a_gpu.std_error));
    McResult b_cpu = CpuMonteCarloPricer::price(bo, 100.0, 0.05, 0.0, 0.25, cfg);
    McResult b_gpu = gpu::price_barrier(bo, 100.0, 0.05, 0.0, 0.25, cfg, 0);
    QC_CHECK(std::fabs(b_cpu.price - b_gpu.price) <
             4.0 * std::hypot(b_cpu.std_error, b_gpu.std_error));
    OptionSpec breached = bo;
    breached.barrier = 95.0;
    McResult knocked_out =
        gpu::price_barrier(breached, 100.0, 0.05, 0.0, 0.25, cfg, 0);
    QC_CHECK_NEAR(knocked_out.price, 0.0, 0.0);
    QC_CHECK_NEAR(knocked_out.std_error, 0.0, 0.0);

    // Below a one-vol-point bump, vega is a one-sided forward difference.
    // This catches both the lower bump location and the denominator.
    OptionSpec low_vol;
    low_vol.type = OptionType::EuropeanCall;
    low_vol.strike = 100.0;
    low_vol.maturity = 1.0;
    McGreeksResult cpu_low_vol = CpuGreeksEngine::european_fd(
        low_vol.type, 100.0, low_vol.strike, 0.04, 0.0, 0.005,
        low_vol.maturity, cfg.paths, cfg.seed);
    McGreeksResult gpu_low_vol =
        gpu::price_greeks_european(low_vol, 100.0, 0.04, 0.0, 0.005, cfg);
    QC_CHECK_NEAR(gpu_low_vol.greeks.vega, cpu_low_vol.greeks.vega, 2.0);

    // Raw second moments are numerically unresolved for very large constant
    // payoffs; the GPU finalizer must not report roundoff as real variance.
    OptionSpec deterministic = low_vol;
    deterministic.strike = 0.0;
    McResult stable =
        gpu::price_european(deterministic, 1.0e16, 0.0, 0.0, 0.0, cfg);
    QC_CHECK_NEAR(stable.price, 1.0e16, 16.0);
    QC_CHECK_NEAR(stable.std_error, 0.0, 0.0);

    // Exercise the correlated-scenario and portfolio-risk kernels too; these
    // previously had no GPU regression coverage.
    MarketData market;
    market.risk_free_rate = 0.04;
    market.assets.push_back({"A", 100.0, 0.20, 0.0});
    market.assets.push_back({"B", 80.0, 0.25, 0.01});
    market.correlation = {{1.0, 0.45}, {0.45, 1.0}};
    const std::size_t scenarios = 20'000;
    auto shocks = gpu::correlated_shocks(market, scenarios, 991, false);
    QC_CHECK(shocks.size() == scenarios * 2);
    auto empirical = CorrelationEngine::empirical_correlation(shocks, scenarios, 2);
    QC_CHECK_NEAR(empirical[0][1], 0.45, 0.03);

    Portfolio pf;
    OptionSpec call;
    call.id = "GPU_RISK_CALL"; call.type = OptionType::EuropeanCall;
    call.underlying = "A"; call.strike = 100.0; call.maturity = 0.5;
    OptionSpec put = call;
    put.id = "GPU_RISK_PUT"; put.type = OptionType::EuropeanPut;
    pf.options = {call, put};
    SimulationConfig risk_cfg = cfg;
    risk_cfg.paths = 20'000;
    risk_cfg.backend = Backend::Cuda;
    RiskConfig rcfg;
    rcfg.scenarios = scenarios;
    rcfg.horizon = 1.0 / 252.0;
    RiskResult risk = gpu::portfolio_risk(pf, market, risk_cfg, rcfg);
    QC_CHECK(risk.scenarios == scenarios);
    QC_CHECK(risk.estimates.size() == 2);
    QC_CHECK(risk.estimates[0].var <= risk.estimates[1].var + 1e-12);
    QC_CHECK(risk.estimates[0].cvar >= risk.estimates[0].var - 1e-12);
    QC_CHECK(risk.estimates[1].cvar >= risk.estimates[1].var - 1e-12);
    QC_CHECK(std::isfinite(risk.best_gain) && risk.best_gain > 0.0);

    // Degenerate FD/pathwise inputs must fail explicitly, never leak NaNs.
    OptionSpec expired = call;
    expired.maturity = 0.0;
    QC_CHECK_THROWS(gpu::price_greeks_european(call, 0.0, 0.04, 0.0, 0.2, cfg));
    QC_CHECK_THROWS(gpu::delta_pathwise_european(expired, 100.0, 0.04, 0.0, 0.2,
                                                  cfg));

    // Public CUDA APIs enforce the same domains as the CPU engines, plus
    // launch-shape and allocation-overflow guards.
    SimulationConfig no_paths = cfg;
    no_paths.paths = 0;
    QC_CHECK_THROWS(gpu::price_european(call, 100.0, 0.04, 0.0, 0.2, no_paths));
    SimulationConfig no_steps = cfg;
    no_steps.time_steps = 0;
    QC_CHECK_THROWS(gpu::price_asian(as, 100.0, 0.04, 0.0, 0.2, no_steps));
    SimulationConfig bad_block = cfg;
    bad_block.block_size = 48;
    QC_CHECK_THROWS(gpu::price_european(call, 100.0, 0.04, 0.0, 0.2, bad_block));
    QC_CHECK_THROWS(gpu::price_european(as, 100.0, 0.04, 0.0, 0.2, cfg));
    QC_CHECK_THROWS(gpu::price_barrier(bo, 100.0, 0.04, 0.0, 0.2, cfg, 3));
    QC_CHECK_THROWS(gpu::price_european(
        call, std::numeric_limits<double>::quiet_NaN(), 0.04, 0.0, 0.2, cfg));
    QC_CHECK_THROWS(gpu::correlated_shocks(market, 0, 991, false));
    QC_CHECK_THROWS(gpu::correlated_shocks(
        market, std::numeric_limits<std::size_t>::max(), 991, false));
    QC_CHECK_THROWS(gpu::reduction_study(100, 42, 0));
    QC_CHECK_THROWS(gpu::reduction_study(100'000'001, 42, 256));

    RiskConfig bad_risk = rcfg;
    bad_risk.scenarios = 0;
    QC_CHECK_THROWS(gpu::portfolio_risk(pf, market, risk_cfg, bad_risk));
    bad_risk = rcfg;
    bad_risk.horizon = 0.0;
    QC_CHECK_THROWS(gpu::portfolio_risk(pf, market, risk_cfg, bad_risk));
    bad_risk = rcfg;
    bad_risk.confidence_levels = {1.0};
    QC_CHECK_THROWS(gpu::portfolio_risk(pf, market, risk_cfg, bad_risk));
    bad_risk = rcfg;
    bad_risk.horizon = 1.0;
    QC_CHECK_THROWS(gpu::portfolio_risk(pf, market, risk_cfg, bad_risk));
}
