// QuantCUDA Risk Engine - command-line interface implementation.
//
//   quantcuda price     --portfolio F [--backend scalar|openmp|cuda]
//                       [--paths N] [--precision float|double|mixed]
//                       [--seed S] [--steps N] [--gpu-version v0..v7] [--greeks]
//   quantcuda risk      --portfolio F [--scenarios N] [--var-levels 0.95,0.99]
//                       [--horizon-days N] [--backend ...]
//   quantcuda validate  --portfolio F [--paths N]      (Black-Scholes comparison)
//   quantcuda greeks    --portfolio F [--paths N]      (analytic/FD/pathwise)
//   quantcuda benchmark --suite NAME [options]         (see benchmark.hpp)
//   quantcuda versions                                    (GPU generation table)
//   quantcuda info                                        (environment + device)
//
// All numeric arguments are strictly validated; the process never calls a
// shell and never writes outside explicitly requested --output paths.
#include "quantcuda/cli.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcuda/benchmark.hpp"
#include "quantcuda/black_scholes.hpp"
#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/greeks_cpu.hpp"
#include "quantcuda/json_config.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
#include "quantcuda/risk_cpu.hpp"
#include "quantcuda/timer.hpp"

namespace quantcuda {
namespace {

constexpr std::uint64_t kMaxPaths = 100'000'000;
constexpr std::uint64_t kMaxScenarios = 10'000'000;
constexpr std::uint64_t kMaxPricingWork = 5'000'000'000;
constexpr std::uint64_t kMaxScenarioValues = 50'000'000;

// ---------------------------------------------------------------------------
// Argument parsing helpers (strict: every numeric value fully parsed + ranged)
// ---------------------------------------------------------------------------
struct Args {
    std::string command;
    std::map<std::string, std::string> opt;
    std::set<std::string> implicit_values;
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) return a;
    a.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            std::string key, value;
            bool implicit_value = false;
            auto eq = s.find('=');
            if (eq != std::string::npos) {
                key = s.substr(2, eq - 2);
                value = s.substr(eq + 1);
            } else {
                key = s.substr(2);
                if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0)
                    value = argv[++i];
                else {
                    value = "true";
                    implicit_value = true;
                }
            }
            if (key.empty()) throw std::runtime_error("empty option name");
            if (!a.opt.emplace(key, value).second)
                throw std::runtime_error("duplicate option: --" + key);
            if (implicit_value) a.implicit_values.insert(key);
        } else {
            throw std::runtime_error("unexpected positional argument: " + s);
        }
    }
    return a;
}

void validate_options(const Args& a,
                      std::initializer_list<const char*> allowed_options,
                      std::initializer_list<const char*> flags = {}) {
    std::set<std::string> allowed;
    std::set<std::string> flag_set;
    for (const char* key : allowed_options) allowed.emplace(key);
    for (const char* key : flags) flag_set.emplace(key);

    for (const auto& entry : a.opt) {
        const std::string& key = entry.first;
        if (!allowed.count(key))
            throw std::runtime_error(a.command + ": unknown or inapplicable option --" +
                                     key);
        if (flag_set.count(key)) {
            if (!a.implicit_values.count(key))
                throw std::runtime_error("--" + key + " is a flag and accepts no value");
        } else if (a.implicit_values.count(key)) {
            throw std::runtime_error("--" + key + " requires a value");
        }
    }
}

std::string get_str(const Args& a, const char* key, const std::string& fallback = "") {
    auto it = a.opt.find(key);
    return it == a.opt.end() ? fallback : it->second;
}

std::uint64_t get_u64(const Args& a, const char* key, std::uint64_t fallback) {
    auto it = a.opt.find(key);
    if (it == a.opt.end()) return fallback;
    const std::string& s = it->second;
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("--") + key + ": must be a non-negative integer");
    std::uint64_t v = 0;
    try { v = std::stoull(s); } catch (...) {
        throw std::runtime_error(std::string("--") + key + ": integer overflow");
    }
    if (v > kMaxPaths)
        throw std::runtime_error(std::string("--") + key +
                                 ": exceeds safety limit 100000000");
    return v;
}

std::uint64_t get_seed(const Args& a, const char* key, std::uint64_t fallback) {
    auto it = a.opt.find(key);
    if (it == a.opt.end()) return fallback;
    const std::string& s = it->second;
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("--") + key +
                                 ": must be an unsigned 64-bit integer");
    try {
        std::size_t pos = 0;
        std::uint64_t value = std::stoull(s, &pos);
        if (pos != s.size()) throw std::runtime_error("trailing");
        return value;
    } catch (...) {
        throw std::runtime_error(std::string("--") + key +
                                 ": unsigned 64-bit integer overflow");
    }
}

std::vector<double> get_double_list(const Args& a, const char* key,
                                    std::vector<double> fallback) {
    auto it = a.opt.find(key);
    if (it == a.opt.end()) return fallback;
    std::vector<double> out;
    std::string s = it->second;
    std::size_t start = 0;
    while (start <= s.size()) {
        auto comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos
                                              ? std::string::npos : comma - start);
        if (tok.empty()) throw std::runtime_error(std::string("--") + key + ": empty element");
        try {
            std::size_t pos = 0;
            double v = std::stod(tok, &pos);
            if (pos != tok.size() || !std::isfinite(v))
                throw std::runtime_error("trailing or non-finite");
            out.push_back(v);
        } catch (...) {
            throw std::runtime_error(std::string("--") + key + ": bad number '" + tok + "'");
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

Backend get_backend(const Args& a, Backend fallback) {
    auto it = a.opt.find("backend");
    if (it == a.opt.end()) return fallback;
    Backend b;
    if (!parse_backend(it->second, b))
        throw std::runtime_error("--backend must be scalar|openmp|cuda");
    return b;
}

Precision get_precision(const Args& a, Precision fallback) {
    auto it = a.opt.find("precision");
    if (it == a.opt.end()) return fallback;
    Precision p;
    if (!parse_precision(it->second, p))
        throw std::runtime_error("--precision must be float|double|mixed");
    return p;
}

// ---------------------------------------------------------------------------
// Engine dispatch
// ---------------------------------------------------------------------------
McResult price_one(const OptionSpec& o, const MarketData& mkt,
                   const SimulationConfig& cfg) {
    int ai = mkt.asset_index(o.underlying);
    const AssetData& ad = mkt.assets[static_cast<std::size_t>(ai)];
    switch (cfg.backend) {
        case Backend::Scalar:
            return CpuMonteCarloPricer::price(o, ad.spot, mkt.risk_free_rate,
                                              ad.dividend_yield, ad.volatility, cfg);
        case Backend::OpenMP:
            return OpenMpMonteCarloPricer::price(o, ad.spot, mkt.risk_free_rate,
                                                 ad.dividend_yield, ad.volatility, cfg);
        case Backend::Cuda:
            if (!gpu::available())
                throw std::runtime_error(
                    "CUDA backend selected but no CUDA-capable GPU is available "
                    "on this machine (use --backend scalar|openmp instead)");
            return gpu::price_option(o, ad.spot, mkt.risk_free_rate,
                                     ad.dividend_yield, ad.volatility, cfg);
    }
    throw std::runtime_error("unknown backend");
}

SimulationConfig merged_sim_cfg(const EngineInput& in, const Args& a) {
    SimulationConfig cfg = in.simulation;
    cfg.paths = get_u64(a, "paths", cfg.paths);
    cfg.seed = get_seed(a, "seed", cfg.seed);
    std::uint64_t steps = get_u64(a, "steps", static_cast<std::uint64_t>(cfg.time_steps));
    if (steps == 0 || steps > 100'000)
        throw std::runtime_error("--steps must be in [1, 100000]");
    cfg.time_steps = static_cast<int>(steps);
    cfg.precision = get_precision(a, cfg.precision);
    cfg.backend = get_backend(a, cfg.backend);
#ifndef QUANTCUDA_HAS_OPENMP
    if (cfg.backend == Backend::OpenMP)
        throw std::runtime_error(
            "OpenMP backend requested, but this build has no OpenMP support");
#endif
    if (cfg.backend != Backend::Cuda && cfg.precision != Precision::Double)
        throw std::runtime_error(
            "float and mixed precision are CUDA-only; CPU backends use double");
    cfg.block_size = static_cast<int>(get_u64(a, "block-size", cfg.block_size));
    if (auto it = a.opt.find("gpu-version"); it != a.opt.end()) {
        if (!parse_gpu_version(it->second, cfg.gpu_version))
            throw std::runtime_error("--gpu-version must be v0..v7");
    }
    if (cfg.paths == 0) throw std::runtime_error("--paths must be >= 1");
    if (cfg.block_size < 32 || cfg.block_size > 1024)
        throw std::runtime_error("--block-size must be in [32, 1024]");
    if ((cfg.block_size & (cfg.block_size - 1)) != 0)
        throw std::runtime_error("--block-size must be a power of two");
    return cfg;
}

void validate_pricing_workload(const EngineInput& in, const SimulationConfig& cfg,
                               bool european_only = false) {
    std::uint64_t work = 0;
    for (const auto& opt : in.portfolio.options) {
        if (european_only && !is_european(opt.type)) continue;
        const std::uint64_t steps = is_european(opt.type)
                                        ? 1u
                                        : static_cast<std::uint64_t>(
                                              opt.time_steps > 0 ? opt.time_steps
                                                                 : cfg.time_steps);
        if (steps == 0 || cfg.paths > kMaxPricingWork / steps)
            throw std::runtime_error("requested pricing workload exceeds safety limit");
        const std::uint64_t instrument_work =
            static_cast<std::uint64_t>(cfg.paths) * steps;
        if (instrument_work > kMaxPricingWork - work)
            throw std::runtime_error(
                "requested paths x steps x instruments exceeds safety limit 5000000000");
        work += instrument_work;
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------
int cmd_price(const Args& a) {
    validate_options(a,
                     {"portfolio", "backend", "paths", "precision", "seed", "steps",
                      "block-size", "gpu-version", "greeks"},
                     {"greeks"});
    std::string file = get_str(a, "portfolio");
    if (file.empty()) throw std::runtime_error("price: --portfolio FILE is required");
    EngineInput in = load_engine_input_file(file);
    SimulationConfig cfg = merged_sim_cfg(in, a);
    validate_pricing_workload(in, cfg);
    if (a.opt.count("gpu-version") &&
        std::any_of(in.portfolio.options.begin(), in.portfolio.options.end(),
                    [](const OptionSpec& opt) { return !is_european(opt.type); })) {
        std::fprintf(stderr,
                     "quantcuda: note: --gpu-version selects the European V0-V7 "
                     "kernel; Asian/barrier instruments use their dedicated kernels\n");
    }
    bool with_greeks = a.opt.count("greeks") > 0;

    WallTimer wall;
    PricingResult res;
    res.backend = cfg.backend;
    res.precision = cfg.precision;
    res.paths = cfg.paths;
    res.seed = cfg.seed;

    std::printf("QuantCUDA Pricing Report\n");
    std::printf("------------------------------------------------\n");
    std::printf("Backend:                  %s\n", to_string(cfg.backend));
    if (cfg.backend == Backend::Cuda)
        std::printf("GPU:                      %s\n", gpu::device_name().c_str());
    std::printf("Simulation paths:         %zu\n", cfg.paths);
    std::printf("Precision:                %s\n", to_string(cfg.precision));
    std::printf("Seed:                     %llu\n",
                static_cast<unsigned long long>(cfg.seed));
    std::printf("------------------------------------------------\n");
    std::printf("%-16s %-18s %12s %12s", "instrument", "type", "price", "std_error");
    if (with_greeks) std::printf(" %10s %10s %10s", "delta", "gamma", "vega");
    std::printf("\n");

    for (const auto& o : in.portfolio.options) {
        OptionPricingResult opr;
        opr.id = o.id;
        opr.type = o.type;
        int ai = in.market.asset_index(o.underlying);
        const AssetData& ad = in.market.assets[static_cast<std::size_t>(ai)];

        McResult r = price_one(o, in.market, cfg);
        opr.price = r.price;
        opr.std_error = r.std_error;
        res.portfolio_value += o.quantity * r.price;

        if (is_european(o.type)) {
            opr.analytic_ref = BlackScholesPricer::price(
                o.type, ad.spot, o.strike, in.market.risk_free_rate,
                ad.dividend_yield, ad.volatility, o.maturity);
            opr.has_analytic = true;
        }
        if (with_greeks) {
            if (cfg.backend == Backend::Cuda && is_european(o.type)) {
                McGreeksResult g = gpu::price_greeks_european(
                    o, ad.spot, in.market.risk_free_rate, ad.dividend_yield,
                    ad.volatility, cfg);
                opr.greeks = g.greeks;
            } else {
                opr.greeks = CpuGreeksEngine::analytic_or_fd(
                    o, ad.spot, in.market.risk_free_rate, ad.dividend_yield,
                    ad.volatility, cfg);
            }
            opr.has_greeks = true;
        }
        res.options.push_back(opr);

        std::printf("%-16s %-18s %12.6f %12.6f", o.id.c_str(), to_string(o.type),
                    opr.price, opr.std_error);
        if (with_greeks)
            std::printf(" %10.4f %10.4f %10.4f", opr.greeks.delta, opr.greeks.gamma,
                        opr.greeks.vega);
        if (opr.has_analytic) {
            double dev = opr.price - opr.analytic_ref;
            std::printf("   (BS %.6f, dev %+.6f)", opr.analytic_ref, dev);
        }
        std::printf("\n");
    }

    res.end_to_end_ms = wall.elapsed_ms();
    res.kernel_time_ms = res.end_to_end_ms;
    double total_paths = static_cast<double>(cfg.paths) *
                         static_cast<double>(in.portfolio.options.size());
    res.paths_per_second = res.end_to_end_ms > 0.0
                               ? total_paths / (res.end_to_end_ms / 1000.0)
                               : 0.0;
    std::printf("------------------------------------------------\n");
    std::printf("Portfolio market value:   %.4f\n", res.portfolio_value);
    std::printf("End-to-end time:          %.1f ms\n", res.end_to_end_ms);
    std::printf("Throughput:               %.2f M paths/sec\n",
                res.paths_per_second / 1e6);
    return 0;
}

int cmd_risk(const Args& a) {
    validate_options(a,
                     {"portfolio", "backend", "paths", "precision", "seed", "steps",
                      "block-size", "gpu-version", "scenarios", "var-levels",
                      "horizon-days"});
    std::string file = get_str(a, "portfolio");
    if (file.empty()) throw std::runtime_error("risk: --portfolio FILE is required");
    EngineInput in = load_engine_input_file(file);
    SimulationConfig cfg = merged_sim_cfg(in, a);
    validate_pricing_workload(in, cfg);
    RiskConfig rcfg = in.risk;
    rcfg.scenarios = get_u64(a, "scenarios", rcfg.scenarios);
    if (rcfg.scenarios == 0) throw std::runtime_error("--scenarios must be >= 1");
    if (rcfg.scenarios > kMaxScenarios)
        throw std::runtime_error("--scenarios exceeds safety limit 10000000");
    if (!in.market.assets.empty() &&
        rcfg.scenarios > kMaxScenarioValues / in.market.assets.size())
        throw std::runtime_error(
            "requested scenarios x assets exceeds safety limit 50000000");
    rcfg.confidence_levels = get_double_list(a, "var-levels", rcfg.confidence_levels);
    if (rcfg.confidence_levels.empty())
        throw std::runtime_error("--var-levels must not be empty");
    for (double level : rcfg.confidence_levels) {
        if (!std::isfinite(level) || level <= 0.0 || level >= 1.0)
            throw std::runtime_error("--var-levels entries must be finite and in (0,1)");
    }
    if (a.opt.count("horizon-days")) {
        std::uint64_t days = get_u64(a, "horizon-days", 1);
        if (days == 0 || days > 252) throw std::runtime_error("--horizon-days in [1,252]");
        rcfg.horizon = static_cast<double>(days) / 252.0;
    }

    WallTimer wall;
    RiskResult rr;
    if (cfg.backend == Backend::Cuda) {
        rr = gpu::portfolio_risk(in.portfolio, in.market, cfg, rcfg);
    } else {
        rr = PortfolioRiskEngineCpu::run(in.portfolio, in.market, cfg, rcfg,
                                         cfg.backend);
        rr.end_to_end_ms = wall.elapsed_ms();
    }

    std::printf("QuantCUDA Portfolio Risk Report\n");
    std::printf("------------------------------------------------\n");
    std::printf("Backend:                  %s\n", to_string(cfg.backend));
    std::printf("Scenarios:                %zu\n", rr.scenarios);
    std::printf("Horizon:                  %.4f years (%.1f trading days)\n",
                rr.horizon, rr.horizon * 252.0);
    std::printf("Base portfolio value V0:  %.4f\n", rr.base_value);
    std::printf("Scenario P&L mean:        %+.4f\n", rr.pnl_mean);
    std::printf("Scenario P&L std dev:     %.4f\n", rr.pnl_stddev);
    std::printf("Worst loss:               %.4f\n", rr.worst_loss);
    std::printf("Best gain:                %.4f\n", rr.best_gain);
    std::printf("------------------------------------------------\n");
    for (const auto& e : rr.estimates) {
        std::printf("VaR  %4.1f%%:             %.4f\n", e.confidence * 100.0, e.var);
        std::printf("CVaR %4.1f%%:             %.4f\n", e.confidence * 100.0, e.cvar);
    }
    std::printf("------------------------------------------------\n");
    std::printf("End-to-end time:          %.1f ms\n", rr.end_to_end_ms);
    return 0;
}

int cmd_validate(const Args& a) {
    validate_options(a,
                     {"portfolio", "backend", "paths", "precision", "seed", "steps",
                      "block-size", "gpu-version"});
    std::string file = get_str(a, "portfolio");
    if (file.empty()) throw std::runtime_error("validate: --portfolio FILE is required");
    EngineInput in = load_engine_input_file(file);
    SimulationConfig cfg = merged_sim_cfg(in, a);
    validate_pricing_workload(in, cfg, true);

    std::printf("Black-Scholes Validation (European instruments)\n");
    std::printf("Backend: %s   paths: %zu   seed: %llu\n", to_string(cfg.backend),
                cfg.paths, static_cast<unsigned long long>(cfg.seed));
    std::printf("%-16s %14s %14s %14s %12s\n", "instrument", "BS_price", "MC_price",
                "MC_std_err", "dev/SE");
    bool all_ok = true;
    std::size_t checked = 0;
    for (const auto& o : in.portfolio.options) {
        if (!is_european(o.type)) continue;
        ++checked;
        int ai = in.market.asset_index(o.underlying);
        const AssetData& ad = in.market.assets[static_cast<std::size_t>(ai)];
        double bs = BlackScholesPricer::price(o.type, ad.spot, o.strike,
                                              in.market.risk_free_rate,
                                              ad.dividend_yield, ad.volatility,
                                              o.maturity);
        McResult mc = price_one(o, in.market, cfg);
        double deviation = std::fabs(mc.price - bs);
        double exact_tol = 1e-12 * std::max({1.0, std::fabs(mc.price), std::fabs(bs)});
        bool ok = false;
        double z = std::numeric_limits<double>::infinity();
        if (std::isfinite(mc.std_error) && mc.std_error > 0.0) {
            z = deviation / mc.std_error;
            ok = std::isfinite(z) && z < 4.0;
        } else if (std::isfinite(mc.price) && std::isfinite(bs) && deviation <= exact_tol) {
            z = 0.0;
            ok = true;
        }
        all_ok = all_ok && ok;
        std::printf("%-16s %14.6f %14.6f %14.6f %12.2f %s\n", o.id.c_str(), bs,
                    mc.price, mc.std_error, z, ok ? "OK" : "INVESTIGATE");
    }
    if (checked == 0)
        throw std::runtime_error("validate: portfolio contains no European instruments");
    return all_ok ? 0 : 3;
}

int cmd_greeks(const Args& a) {
    validate_options(a,
                     {"portfolio", "backend", "paths", "precision", "seed",
                      "block-size", "gpu-version"});
    std::string file = get_str(a, "portfolio");
    if (file.empty()) throw std::runtime_error("greeks: --portfolio FILE is required");
    EngineInput in = load_engine_input_file(file);
    SimulationConfig cfg = merged_sim_cfg(in, a);

    const std::size_t supported = static_cast<std::size_t>(std::count_if(
        in.portfolio.options.begin(), in.portfolio.options.end(),
        [](const OptionSpec& opt) { return is_european(opt.type); }));
    if (supported == 0)
        throw std::runtime_error("greeks: portfolio contains no European instruments");
    validate_pricing_workload(in, cfg, true);

    std::printf("Greeks Comparison (analytic vs finite-difference-CRN vs pathwise)\n");
    std::printf("%-16s %10s %12s %12s %12s %12s\n", "instrument", "greek", "analytic",
                "fd_crn", "pathwise", "paths");
    for (const auto& o : in.portfolio.options) {
        if (!is_european(o.type)) continue;
        int ai = in.market.asset_index(o.underlying);
        const AssetData& ad = in.market.assets[static_cast<std::size_t>(ai)];
        Greeks ga = BlackScholesPricer::greeks(o.type, ad.spot, o.strike,
                                               in.market.risk_free_rate,
                                               ad.dividend_yield, ad.volatility,
                                               o.maturity);
        McGreeksResult fd;
        McResult pw;
        if (cfg.backend == Backend::Cuda) {
            fd = gpu::price_greeks_european(o, ad.spot, in.market.risk_free_rate,
                                            ad.dividend_yield, ad.volatility, cfg);
            pw = gpu::delta_pathwise_european(o, ad.spot, in.market.risk_free_rate,
                                              ad.dividend_yield, ad.volatility, cfg);
        } else {
            fd = CpuGreeksEngine::european_fd(o.type, ad.spot, o.strike,
                                              in.market.risk_free_rate,
                                              ad.dividend_yield, ad.volatility,
                                              o.maturity, cfg.paths, cfg.seed);
            pw = CpuGreeksEngine::european_delta_pathwise(
                o.type, ad.spot, o.strike, in.market.risk_free_rate,
                ad.dividend_yield, ad.volatility, o.maturity, cfg.paths, cfg.seed);
        }
        std::printf("%-16s %10s %12.6f %12.6f %12.6f %12zu\n", o.id.c_str(), "delta",
                    ga.delta, fd.greeks.delta, pw.price, cfg.paths);
        std::printf("%-16s %10s %12.6f %12.6f %12s\n", "", "gamma", ga.gamma,
                    fd.greeks.gamma, "-");
        std::printf("%-16s %10s %12.6f %12.6f %12s\n", "", "vega", ga.vega,
                    fd.greeks.vega, "-");
    }
    return 0;
}

int cmd_versions() {
    std::printf("QuantCUDA GPU implementation generations\n");
    std::printf("------------------------------------------------\n");
    std::printf("v0-naive        AoS input, separate sim/payoff kernels, global arrays,\n");
    std::printf("                per-thread atomicAdd, synchronous copies\n");
    std::printf("v1-soa          structure-of-arrays batch layout (coalesced reads)\n");
    std::printf("v2-fused        fused RNG + simulation + payoff + block reduction\n");
    std::printf("v3-reduction    warp-shuffle reductions (4 flavors benchmarked)\n");
    std::printf("v4-specialized  European-specialized path; dedicated barrier studies\n");
    std::printf("v5-memory       __constant__ option params, shared-mem Cholesky staging\n");
    std::printf("v6-async        pinned host memory + multi-stream chunked execution\n");
    std::printf("v7-precision    float / double / mixed (float sim, double accumulate)\n");
    std::printf("------------------------------------------------\n");
    std::printf("Select with: --gpu-version vN   Benchmark: benchmark --suite version-study\n");
    std::printf("V0-V7 dispatch applies to European pricing; Asian/barrier use dedicated kernels.\n");
    return 0;
}

int cmd_info() {
    std::printf("%s", environment_report().c_str());
    return 0;
}

int cmd_benchmark(const Args& a) {
    validate_options(a,
                     {"suite", "output", "portfolio", "repeats",
                      "warmup", "seed", "backends", "paths", "steps", "instruments",
                      "scenarios"});
    BenchmarkOptions o;
    o.suite = get_str(a, "suite", "european-throughput");
    o.output_csv = get_str(a, "output");
    o.config_file = get_str(a, "portfolio");
    const std::uint64_t repeats = get_u64(a, "repeats", 5);
    const std::uint64_t warmup = get_u64(a, "warmup", 1);
    if (repeats < 1 || repeats > 101)
        throw std::runtime_error("--repeats must be in [1, 101]");
    if (warmup > 101)
        throw std::runtime_error("--warmup must be in [0, 101]");
    o.repeats = static_cast<int>(repeats);
    o.warmup = static_cast<int>(warmup);
    o.seed = get_seed(a, "seed", 42);
    o.backends.clear();
    std::string bs = get_str(a, "backends", "scalar,openmp,cuda");
    std::size_t start = 0;
    while (start <= bs.size()) {
        auto comma = bs.find(',', start);
        std::string tok = bs.substr(start, comma == std::string::npos
                                               ? std::string::npos : comma - start);
        Backend b;
        if (!parse_backend(tok, b))
            throw std::runtime_error("--backends: unknown backend '" + tok + "'");
        o.backends.push_back(b);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    // workload overrides
    BenchmarkOptions::Overrides ov;
    ov.paths = get_u64(a, "paths", 0);
    std::uint64_t benchmark_steps = get_u64(a, "steps", 0);
    if (benchmark_steps > 100'000)
        throw std::runtime_error("--steps must be <= 100000");
    ov.steps = static_cast<int>(benchmark_steps);
    std::uint64_t instruments = get_u64(a, "instruments", 0);
    if (instruments > 100'000)
        throw std::runtime_error("--instruments must be <= 100000");
    ov.instruments = static_cast<int>(instruments);
    ov.scenarios = get_u64(a, "scenarios", 0);
    if (ov.scenarios > kMaxScenarios)
        throw std::runtime_error("--scenarios exceeds safety limit 10000000");
    o.overrides = ov;
    return run_benchmark(o);
}

void print_usage() {
    std::printf(
        "QuantCUDA Risk Engine - C++20/CUDA derivatives pricing and portfolio risk\n"
        "\n"
        "usage: quantcuda <command> [options]\n"
        "\n"
        "commands:\n"
        "  price       price every instrument in a portfolio JSON\n"
        "  risk        portfolio VaR/CVaR from correlated scenarios\n"
        "  validate    Monte Carlo vs Black-Scholes comparison table\n"
        "  greeks      analytic vs FD-CRN vs pathwise Greeks\n"
        "  benchmark   run a benchmark suite (A-E + GPU studies)\n"
        "  versions    list GPU implementation generations V0-V7\n"
        "  info        print build/environment/device information\n"
        "\n"
        "common options:\n"
        "  --portfolio FILE      input JSON (see configs/)\n"
        "  --backend NAME        scalar | openmp | cuda\n"
        "  --paths N             Monte Carlo paths (<= 100000000)\n"
        "  --precision NAME      float | double | mixed\n"
        "  --seed S              RNG seed (default 42)\n"
        "  --steps N             path time steps (Asian/barrier)\n"
        "  --gpu-version vN      GPU generation v0..v7 (CUDA builds)\n"
        "  --greeks              add Greeks to the price report\n"
        "  --scenarios N         risk scenarios\n"
        "  --var-levels LIST     e.g. 0.95,0.99\n"
        "  --horizon-days N      risk horizon in trading days (default 1)\n"
        "  --output FILE         benchmark CSV output path\n"
        "\n"
        "examples:\n"
        "  quantcuda price --portfolio configs/european_call.json --backend openmp --paths 1000000\n"
        "  quantcuda risk --portfolio configs/multi_asset_portfolio.json --scenarios 100000\n"
        "  quantcuda benchmark --suite european-throughput --backends scalar,openmp --output results.csv\n");
}

}  // namespace

int run_cli(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.command.empty() || a.command == "help" || a.command == "--help" ||
        a.command == "-h") {
        validate_options(a, {});
        print_usage();
        return a.command.empty() ? 1 : 0;
    }
    if (a.command == "price") return cmd_price(a);
    if (a.command == "risk") return cmd_risk(a);
    if (a.command == "validate") return cmd_validate(a);
    if (a.command == "greeks") return cmd_greeks(a);
    if (a.command == "benchmark") return cmd_benchmark(a);
    if (a.command == "versions") {
        validate_options(a, {});
        return cmd_versions();
    }
    if (a.command == "info") {
        validate_options(a, {});
        return cmd_info();
    }
    std::fprintf(stderr, "quantcuda: unknown command '%s'\n", a.command.c_str());
    print_usage();
    return 1;
}

}  // namespace quantcuda
