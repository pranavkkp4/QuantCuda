// QuantCUDA Risk Engine - portfolio scenario risk (VaR / CVaR) on CPU.
//
// Workflow per market scenario over the risk horizon h:
//   1. draw correlated shocks z_corr = L z for the d underlyings,
//   2. propagate each asset one GBM step of length h,
//   3. reprice every instrument (European: Black-Scholes full revaluation;
//      path-dependent: delta-gamma approximation around the base price -
//      documented as an approximation, exact for the European part),
//   4. sum to scenario portfolio value, P&L_i = V_i - V0, loss L_i = -P&L_i.
// VaR_alpha is the alpha-quantile of the loss distribution; CVaR_alpha is the
// quantile-integral expected shortfall over exactly the upper (1-alpha) mass.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "black_scholes.hpp"
#include "cholesky.hpp"
#include "greeks_cpu.hpp"
#include "monte_carlo_cpu.hpp"
#include "rng_cpu.hpp"
#include "types.hpp"

namespace quantcuda {

// VaR/CVaR from an unsorted loss vector (losses positive). Levels in (0,1).
std::vector<VarEstimate> var_cvar_from_losses(std::vector<double> losses,
                                              const std::vector<double>& levels);

// Per-instrument risk snapshot used for scenario repricing.
struct InstrumentRisk {
    const OptionSpec* opt = nullptr;
    int asset_index = 0;
    double base_price = 0.0;   // per unit
    double delta = 0.0;        // per unit
    double gamma = 0.0;        // per unit
    bool analytic_repricing = false;  // European -> exact BS under scenario
};

class PortfolioRiskEngineCpu {
public:
    static constexpr std::size_t max_paths = 100'000'000;
    static constexpr std::size_t max_scenarios = 10'000'000;
    static constexpr std::size_t max_pricing_work = 5'000'000'000ull;
    static constexpr std::size_t max_scenario_work = 50'000'000;

    // Build per-instrument base prices and Greeks once (reused by all scenarios).
    static std::vector<InstrumentRisk> prepare(const Portfolio& pf,
                                               const MarketData& market,
                                               const SimulationConfig& price_cfg) {
        std::vector<InstrumentRisk> out;
        out.reserve(pf.options.size());
        for (const auto& o : pf.options) {
            int ai = market.asset_index(o.underlying);
            if (ai < 0) throw std::invalid_argument("risk: option has unknown underlying");
            const AssetData& a = market.assets[static_cast<std::size_t>(ai)];
            InstrumentRisk ir;
            ir.opt = &o;
            ir.asset_index = ai;
            if (is_european(o.type)) {
                ir.base_price = BlackScholesPricer::price(o.type, a.spot, o.strike,
                                                          market.risk_free_rate,
                                                          a.dividend_yield, a.volatility,
                                                          o.maturity);
                Greeks g = BlackScholesPricer::greeks(o.type, a.spot, o.strike,
                                                      market.risk_free_rate,
                                                      a.dividend_yield, a.volatility,
                                                      o.maturity);
                ir.delta = g.delta;
                ir.gamma = g.gamma;
                ir.analytic_repricing = true;
            } else {
                McResult r = price_cfg.backend == Backend::OpenMP
                                 ? OpenMpMonteCarloPricer::price(
                                       o, a.spot, market.risk_free_rate,
                                       a.dividend_yield, a.volatility, price_cfg)
                                 : CpuMonteCarloPricer::price(
                                       o, a.spot, market.risk_free_rate,
                                       a.dividend_yield, a.volatility, price_cfg);
                ir.base_price = r.price;
                Greeks g = CpuGreeksEngine::analytic_or_fd(o, a.spot, market.risk_free_rate,
                                                           a.dividend_yield, a.volatility,
                                                           price_cfg);
                ir.delta = g.delta;
                ir.gamma = g.gamma;
                ir.analytic_repricing = false;
            }
            out.push_back(ir);
        }
        return out;
    }

    static double base_value(const Portfolio& pf, const std::vector<InstrumentRisk>& instr) {
        double v = 0.0;
        for (std::size_t i = 0; i < instr.size(); ++i)
            v += pf.options[i].quantity * instr[i].base_price;
        return v;
    }

    // Full scenario engine. Returns losses (positive = loss), one per scenario.
    static std::vector<double> scenario_losses(
        const MarketData& market,
        const std::vector<InstrumentRisk>& instr, double v0,
        const RiskConfig& rcfg, std::uint64_t seed,
        Backend backend = Backend::OpenMP) {
        const std::size_t d = market.assets.size();
        if (d == 0) throw std::invalid_argument("risk: at least one asset is required");
        if (d > 256) throw std::invalid_argument("risk: at most 256 assets supported");
        if (rcfg.scenarios == 0)
            throw std::invalid_argument("risk: scenarios must be >= 1");
        if (rcfg.scenarios > max_scenarios)
            throw std::invalid_argument("risk: scenarios exceeds limit 10000000");
        if (d > std::numeric_limits<std::size_t>::max() - instr.size() ||
            rcfg.scenarios > max_scenario_work / (d + instr.size()))
            throw std::invalid_argument(
                "risk: scenarios x (assets + instruments) exceeds limit 50000000");
        if (!std::isfinite(rcfg.horizon) || rcfg.horizon <= 0.0)
            throw std::invalid_argument("risk: horizon must be finite and > 0");
        std::vector<double> L;
        if (d > 1 && !market.correlation.empty())
            L = CorrelationEngine::cholesky(market.correlation);
        else if (d > 1) {  // default to independence
            L.assign(d * d, 0.0);
            for (std::size_t i = 0; i < d; ++i) L[i * d + i] = 1.0;
        }

        double h = rcfg.horizon;
        double sqrt_h = std::sqrt(h);
        std::vector<double> losses(rcfg.scenarios, 0.0);

        // Scenarios are independent draws keyed by (seed, scenario id), so the
        // loop parallelizes cleanly and results do not depend on thread count.
        auto evaluate_scenario = [&](std::int64_t s64) {
            const std::size_t s = static_cast<std::size_t>(s64);
            auto rng = make_path_rng(seed, s);
            double z_indep[256] = {}, z_corr[256] = {}, shocked[256] = {};
            for (std::size_t a = 0; a < d; ++a) z_indep[a] = rng.next_normal();
            if (d > 1) {
                CorrelationEngine::apply(
                    L, d, std::span<const double>(z_indep, d),
                    std::span<double>(z_corr, d));
            } else {
                z_corr[0] = z_indep[0];
            }
            for (std::size_t a = 0; a < d; ++a) {
                const auto& ad = market.assets[a];
                double drift = (market.risk_free_rate - ad.dividend_yield -
                                0.5 * ad.volatility * ad.volatility) * h;
                shocked[a] = ad.spot * std::exp(drift + ad.volatility * sqrt_h *
                                                        z_corr[a]);
            }
            double v = 0.0;
            for (const auto& ir : instr) {
                const OptionSpec& o = *ir.opt;
                const auto& ad = market.assets[static_cast<std::size_t>(ir.asset_index)];
                double s_new = shocked[static_cast<std::size_t>(ir.asset_index)];
                double price;
                if (ir.analytic_repricing) {
                    double remaining = o.maturity - h;
                    price = remaining <= 0.0
                                ? european_payoff(o.type, s_new, o.strike)
                                : BlackScholesPricer::price(o.type, s_new, o.strike,
                                                            market.risk_free_rate,
                                                            ad.dividend_yield,
                                                            ad.volatility, remaining);
                } else {
                    double dS = s_new - ad.spot;
                    price = ir.base_price + ir.delta * dS +
                            0.5 * ir.gamma * dS * dS;
                    if (price < 0.0) price = 0.0;  // option value floor
                }
                v += o.quantity * price;
            }
            losses[s] = -(v - v0);
        };

#ifdef QUANTCUDA_HAS_OPENMP
        if (backend == Backend::OpenMP) {
#pragma omp parallel for schedule(static)
            for (std::int64_t s64 = 0;
                 s64 < static_cast<std::int64_t>(rcfg.scenarios); ++s64)
                evaluate_scenario(s64);
        } else
#endif
        {
            for (std::int64_t s64 = 0;
                 s64 < static_cast<std::int64_t>(rcfg.scenarios); ++s64)
                evaluate_scenario(s64);
        }
        return losses;
    }

    static RiskResult run(const Portfolio& pf, const MarketData& market,
                          const SimulationConfig& price_cfg, const RiskConfig& rcfg,
                          Backend backend = Backend::OpenMP) {
        market.validate();
        pf.validate(market);
        if (backend == Backend::Cuda)
            throw std::invalid_argument("risk CPU engine requires scalar or openmp backend");
        if (price_cfg.paths == 0)
            throw std::invalid_argument("risk: pricing paths must be >= 1");
        if (price_cfg.paths > max_paths)
            throw std::invalid_argument("risk: pricing paths exceeds limit 100000000");
        if (rcfg.scenarios == 0)
            throw std::invalid_argument("risk: scenarios must be >= 1");
        if (!std::isfinite(rcfg.horizon) || rcfg.horizon <= 0.0)
            throw std::invalid_argument("risk: horizon must be finite and > 0");
        if (rcfg.confidence_levels.empty())
            throw std::invalid_argument("risk: at least one confidence level is required");
        for (double level : rcfg.confidence_levels)
            if (!std::isfinite(level) || level <= 0.0 || level >= 1.0)
                throw std::invalid_argument(
                    "risk: confidence levels must be finite and in (0,1)");
        for (const auto& option : pf.options)
            if (option.maturity < rcfg.horizon)
                throw std::invalid_argument(
                    "risk: horizon must not exceed an instrument maturity");
        std::size_t path_units = 0;
        for (const auto& option : pf.options) {
            const std::size_t steps = is_european(option.type)
                                          ? 1u
                                          : static_cast<std::size_t>(
                                                option.time_steps > 0
                                                    ? option.time_steps
                                                    : price_cfg.time_steps);
            if (steps == 0 || steps > 100'000 ||
                path_units > max_pricing_work - steps)
                throw std::invalid_argument("risk: invalid aggregate pricing workload");
            path_units += steps;
        }
        if (path_units != 0 && price_cfg.paths > max_pricing_work / path_units)
            throw std::invalid_argument(
                "risk: paths x steps x instruments exceeds limit 5000000000");
#ifndef QUANTCUDA_HAS_OPENMP
        if (backend == Backend::OpenMP)
            throw std::invalid_argument(
                "risk: OpenMP backend requested but unavailable in this build");
#endif

        RiskResult rr;
        SimulationConfig effective_price_cfg = price_cfg;
        effective_price_cfg.backend = backend;
        auto instr = prepare(pf, market, effective_price_cfg);
        rr.base_value = base_value(pf, instr);
        auto losses = scenario_losses(market, instr, rr.base_value, rcfg,
                                      price_cfg.seed, backend);

        // Distribution summary.
        double mean_loss = 0.0, m2_loss = 0.0;
        std::size_t count = 0;
        double worst = losses.empty() ? 0.0 : losses[0];
        double best = losses.empty() ? 0.0 : losses[0];
        for (double l : losses) {
            ++count;
            const double delta = l - mean_loss;
            mean_loss += delta / static_cast<double>(count);
            m2_loss += delta * (l - mean_loss);
            worst = std::max(worst, l);
            best  = std::min(best, l);
        }
        const double n = static_cast<double>(std::max<std::size_t>(count, 1));
        rr.pnl_mean = -mean_loss;  // report as P&L, not loss
        rr.pnl_stddev = std::sqrt(std::max(m2_loss / n, 0.0));
        rr.worst_loss = worst;
        rr.best_gain  = -best;

        rr.estimates = var_cvar_from_losses(std::move(losses), rcfg.confidence_levels);
        rr.scenarios = rcfg.scenarios;
        rr.horizon   = rcfg.horizon;
        rr.backend   = backend;
        return rr;
    }
};

inline std::vector<VarEstimate> var_cvar_from_losses(std::vector<double> losses,
                                                     const std::vector<double>& levels) {
    std::vector<VarEstimate> out;
    for (double alpha : levels)
        if (!std::isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0)
            throw std::invalid_argument(
                "VaR confidence level must be finite and in (0, 1)");
    if (losses.empty()) return out;
    for (double loss : losses)
        if (!std::isfinite(loss))
            throw std::invalid_argument("VaR losses must all be finite");
    std::sort(losses.begin(), losses.end());
    std::size_t n = losses.size();
    for (double alpha : levels) {
        const long double scaled_alpha =
            static_cast<long double>(alpha) * static_cast<long double>(n);
        // Quantile: smallest loss L such that P(loss <= L) >= alpha.
        std::size_t idx = static_cast<std::size_t>(
            std::ceil(scaled_alpha));
        if (idx == 0) idx = 1;
        if (idx > n) idx = n;
        VarEstimate e;
        e.confidence = alpha;
        e.var = losses[idx - 1];

        // Expected shortfall is the quantile integral
        //   ES_alpha = (1/(1-alpha)) * integral_alpha^1 VaR_u du.
        // For an empirical distribution, alpha can cut through one sample's
        // probability interval. Include only that fractional boundary mass,
        // then every strictly higher observation. This keeps the tail mass
        // exactly n * (1-alpha), including when alpha*n is not an integer.
        const std::size_t first_tail = static_cast<std::size_t>(
            std::floor(scaled_alpha));
        const long double boundary_mass =
            static_cast<long double>(first_tail + 1) - scaled_alpha;
        long double tail_sum = boundary_mass * losses[first_tail];
        for (std::size_t i = first_tail + 1; i < n; ++i)
            tail_sum += losses[i];
        const long double tail_mass = static_cast<long double>(n) - scaled_alpha;
        e.cvar = static_cast<double>(tail_sum / tail_mass);
        out.push_back(e);
    }
    return out;
}

}  // namespace quantcuda
