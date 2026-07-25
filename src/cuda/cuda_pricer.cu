// QuantCUDA Risk Engine - public CUDA interface implementation.
//
// Host-side dispatch from (backend=cuda, gpu_version, precision) to the
// kernels kept in the other .cu files. Every public function mirrors the
// CPU engine's contract so the CLI/tests can swap backends transparently.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "cuda_internal.cuh"
#include "cuda_utils.cuh"
#include "quantcuda/cholesky.hpp"
#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/risk_cpu.hpp"

namespace quantcuda::gpu {
namespace {

constexpr int kMaxTimeSteps = 100'000;
constexpr std::size_t kMaxAssets = 256;
constexpr std::size_t kMaxPaths = 100'000'000;
constexpr std::size_t kMaxScenarios = 10'000'000;
constexpr std::size_t kMaxScenarioValues = 50'000'000;
constexpr std::size_t kMaxPricingWork = 5'000'000'000ull;

enum class OptionFamily { European, Asian, Barrier };

std::size_t checked_product(std::size_t a, std::size_t b,
                            const std::string& context) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::invalid_argument(context + ": size overflow");
    return a * b;
}

void validate_precision(Precision precision, const char* context) {
    switch (precision) {
        case Precision::Float:
        case Precision::Double:
        case Precision::Mixed: return;
    }
    throw std::invalid_argument(std::string(context) + ": invalid precision");
}

void validate_gpu_version(GpuVersion version, const char* context) {
    switch (version) {
        case GpuVersion::V0_Naive:
        case GpuVersion::V1_SoA:
        case GpuVersion::V2_Fused:
        case GpuVersion::V3_Reduction:
        case GpuVersion::V4_Specialized:
        case GpuVersion::V5_Memory:
        case GpuVersion::V6_Async:
        case GpuVersion::V7_Precision: return;
    }
    throw std::invalid_argument(std::string(context) + ": invalid GPU version");
}

void validate_simulation(const SimulationConfig& cfg, const char* context) {
    const std::string prefix(context);
    if (cfg.paths == 0)
        throw std::invalid_argument(prefix + ": paths must be >= 1");
    if (cfg.paths > kMaxPaths)
        throw std::invalid_argument(prefix + ": paths exceeds limit 100000000");
    if (cfg.time_steps <= 0 || cfg.time_steps > kMaxTimeSteps)
        throw std::invalid_argument(prefix + ": time steps must be in [1, 100000]");
    if (cfg.block_size < 32 || cfg.block_size > 1024 ||
        (cfg.block_size & (cfg.block_size - 1)) != 0)
        throw std::invalid_argument(
            prefix + ": CUDA block size must be a power of two in [32, 1024]");
    validate_precision(cfg.precision, context);
    validate_gpu_version(cfg.gpu_version, context);
    (void)checked_product(cfg.paths, sizeof(curandStatePhilox4_32_10_t),
                          prefix + ": RNG state allocation");
    (void)checked_product(cfg.paths, sizeof(double), prefix + ": path allocation");
}

void validate_option_inputs(const OptionSpec& opt, double spot, double r,
                            double q, double sigma, const SimulationConfig& cfg,
                            OptionFamily family, const char* context,
                            bool require_barrier_value = true) {
    const bool family_matches =
        (family == OptionFamily::European && is_european(opt.type)) ||
        (family == OptionFamily::Asian && is_asian(opt.type)) ||
        (family == OptionFamily::Barrier && is_barrier(opt.type));
    if (!family_matches)
        throw std::invalid_argument(std::string(context) + ": wrong option family");
    if (!std::isfinite(spot) || !std::isfinite(opt.strike) || !std::isfinite(r) ||
        !std::isfinite(q) || !std::isfinite(sigma) ||
        !std::isfinite(opt.maturity) || !std::isfinite(opt.barrier))
        throw std::invalid_argument(std::string(context) + ": inputs must be finite");
    if (spot < 0.0)
        throw std::invalid_argument(std::string(context) + ": spot must be >= 0");
    if (opt.strike < 0.0)
        throw std::invalid_argument(std::string(context) + ": strike must be >= 0");
    if (sigma < 0.0)
        throw std::invalid_argument(std::string(context) +
                                    ": volatility must be >= 0");
    if (opt.maturity < 0.0)
        throw std::invalid_argument(std::string(context) +
                                    ": maturity must be >= 0");
    if (opt.time_steps < 0 || opt.time_steps > kMaxTimeSteps)
        throw std::invalid_argument(std::string(context) +
                                    ": option time steps must be in [0, 100000]");
    if (family == OptionFamily::Barrier && require_barrier_value &&
        opt.barrier <= 0.0)
        throw std::invalid_argument(std::string(context) +
                                    ": barrier must be finite and > 0");
    validate_simulation(cfg, context);
    const std::size_t steps = family == OptionFamily::European
                                  ? 1u
                                  : static_cast<std::size_t>(
                                        opt.time_steps > 0 ? opt.time_steps
                                                           : cfg.time_steps);
    const std::size_t work = checked_product(
        cfg.paths, steps, std::string(context) + ": pricing workload");
    if (work > kMaxPricingWork)
        throw std::invalid_argument(
            std::string(context) +
            ": paths x steps exceeds limit 5000000000");
}

void validate_market(const MarketData& market, const char* context) {
    if (market.assets.size() > kMaxAssets)
        throw std::invalid_argument(std::string(context) +
                                    ": at most 256 assets supported");
    (void)checked_product(market.assets.size(), market.assets.size(),
                          std::string(context) + ": correlation matrix");
    market.validate();
}

dev::OptionParams make_params(const OptionSpec& opt, double spot, double r,
                              double q, double sigma, int default_steps) {
    dev::OptionParams p{};
    p.spot = spot;
    p.strike = opt.strike;
    p.rate = r;
    p.dividend = q;
    p.volatility = sigma;
    p.maturity = opt.maturity;
    p.barrier = opt.barrier;
    p.is_call = is_call(opt.type) ? 1 : 0;
    p.barrier_dir = (opt.type == OptionType::BarrierUpOutCall ||
                     opt.type == OptionType::BarrierUpOutPut)
                        ? 1
                        : ((opt.type == OptionType::BarrierDownOutCall ||
                            opt.type == OptionType::BarrierDownOutPut)
                               ? -1
                               : 0);
    p.time_steps = opt.time_steps > 0 ? opt.time_steps : default_steps;
    return p;
}

void ensure_device() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();  // clear the sticky error flag
        throw std::runtime_error(
            "CUDA backend selected but no usable GPU/driver was found "
            "(cudaGetDeviceCount failed). Run on a machine with an NVIDIA GPU "
            "and driver, or use --backend scalar|openmp.");
    }
    if (count < 1)
        throw std::runtime_error(
            "CUDA backend selected but no CUDA-capable GPU was detected. "
            "Run on a machine with an NVIDIA GPU, or use --backend scalar|openmp.");
    QC_CUDA_CHECK(cudaFree(nullptr));  // establish context
}

void validate_device_block_size(int block_size, const char* context) {
    int device = 0;
    QC_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    QC_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (block_size > prop.maxThreadsPerBlock)
        throw std::invalid_argument(std::string(context) +
                                    ": block size exceeds this device's limit");
}

}  // namespace

// ---------------------------------------------------------------------------
// Device information
// ---------------------------------------------------------------------------
bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

std::string device_name() {
    if (!available()) return "none (no CUDA-capable GPU)";
    cudaDeviceProp prop{};
    QC_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::string s(prop.name);
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  " (sm_%d%d, %.1f GB, %d SMs)",
                  prop.major, prop.minor,
                  static_cast<double>(prop.totalGlobalMem) / (1 << 30),
                  prop.multiProcessorCount);
    return s + buf;
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------
McResult price_european(const OptionSpec& opt, double spot, double r, double q,
                        double sigma, const SimulationConfig& cfg) {
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::European,
                           "european");
    ensure_device();
    validate_device_block_size(cfg.block_size, "european");
    auto p = make_params(opt, spot, r, q, sigma, cfg.time_steps);
    GpuMcRun run;
    if (cfg.gpu_version == GpuVersion::V6_Async)
        run = european_async_gpu(p, cfg.paths, cfg.seed, cfg.precision,
                                 cfg.block_size, 4);
    else
        run = european_gpu(p, cfg.paths, cfg.seed, cfg.precision,
                           cfg.gpu_version, cfg.block_size);
    return run.stats;
}

McResult price_asian(const OptionSpec& opt, double spot, double r, double q,
                     double sigma, const SimulationConfig& cfg) {
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::Asian,
                           "asian");
    ensure_device();
    validate_device_block_size(cfg.block_size, "asian");
    auto p = make_params(opt, spot, r, q, sigma, cfg.time_steps);
    return asian_gpu(p, cfg.paths, cfg.seed, cfg.precision, cfg.block_size).stats;
}

McResult price_barrier(const OptionSpec& opt, double spot, double r, double q,
                       double sigma, const SimulationConfig& cfg, int mode) {
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::Barrier,
                           "barrier");
    if (mode < 0 || mode > 2)
        throw std::invalid_argument("barrier: mode must be 0, 1, or 2");
    ensure_device();
    validate_device_block_size(cfg.block_size, "barrier");
    const bool up = opt.type == OptionType::BarrierUpOutCall ||
                    opt.type == OptionType::BarrierUpOutPut;
    if ((up && spot >= opt.barrier) || (!up && spot <= opt.barrier))
        return quantcuda::detail::deterministic_result(
            0.0, cfg.paths, std::exp(-r * opt.maturity));
    auto p = make_params(opt, spot, r, q, sigma, cfg.time_steps);
    return barrier_gpu(p, cfg.paths, cfg.seed, cfg.precision, cfg.block_size,
                       mode)
        .stats;
}

McResult price_option(const OptionSpec& opt, double spot, double r, double q,
                      double sigma, const SimulationConfig& cfg) {
    if (is_european(opt.type)) return price_european(opt, spot, r, q, sigma, cfg);
    if (is_asian(opt.type)) return price_asian(opt, spot, r, q, sigma, cfg);
    if (is_barrier(opt.type))
        return price_barrier(opt, spot, r, q, sigma, cfg, /*mode=*/0);
    throw std::invalid_argument("price_option: unsupported option type");
}

// ---------------------------------------------------------------------------
// Greeks
// ---------------------------------------------------------------------------
McGreeksResult price_greeks_european(const OptionSpec& opt, double spot, double r,
                                     double q, double sigma,
                                     const SimulationConfig& cfg) {
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::European,
                           "price_greeks_european");
    if (spot <= 0.0)
        throw std::invalid_argument("price_greeks_european: spot must be > 0");
    if (opt.maturity <= 0.0)
        throw std::invalid_argument("price_greeks_european: maturity must be > 0");
    ensure_device();
    validate_device_block_size(cfg.block_size, "price_greeks_european");
    auto p = make_params(opt, spot, r, q, sigma, cfg.time_steps);
    double h = 0.01 * spot;
    double hs = 0.01;
    GpuGreeksRun g = greeks_crn_gpu(p, cfg.paths, cfg.seed, h, hs, cfg.block_size);
    McGreeksResult out;
    out.price = g.price;
    out.price_std_error = g.price_se;
    out.greeks.delta = g.delta;
    out.greeks.gamma = g.gamma;
    out.greeks.vega = g.vega;
    out.paths = cfg.paths;
    return out;
}

McResult delta_pathwise_european(const OptionSpec& opt, double spot, double r,
                                 double q, double sigma,
                                 const SimulationConfig& cfg) {
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::European,
                           "delta_pathwise_european");
    if (spot <= 0.0)
        throw std::invalid_argument("delta_pathwise_european: spot must be > 0");
    if (opt.maturity <= 0.0)
        throw std::invalid_argument("delta_pathwise_european: maturity must be > 0");
    ensure_device();
    validate_device_block_size(cfg.block_size, "delta_pathwise_european");
    auto p = make_params(opt, spot, r, q, sigma, cfg.time_steps);
    return delta_pathwise_gpu(p, cfg.paths, cfg.seed, cfg.block_size).stats;
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------
std::vector<double> correlated_shocks(const MarketData& market,
                                      std::size_t scenarios, std::uint64_t seed,
                                      bool use_shared_tiles) {
    validate_market(market, "correlated_shocks");
    if (scenarios == 0)
        throw std::invalid_argument("correlated_shocks: scenarios must be >= 1");
    if (scenarios > kMaxScenarios)
        throw std::invalid_argument(
            "correlated_shocks: scenarios exceeds limit 10000000");
    const std::size_t d = market.assets.size();
    const std::size_t values =
        checked_product(scenarios, d, "correlated_shocks: output");
    if (values > kMaxScenarioValues)
        throw std::invalid_argument(
            "correlated_shocks: scenarios x assets exceeds limit 50000000");
    (void)checked_product(values, sizeof(double),
                          "correlated_shocks: output allocation");
    ensure_device();
    std::vector<double> chol;
    if (d == 1) {
        chol = {1.0};
    } else if (!market.correlation.empty()) {
        chol = CorrelationEngine::cholesky(market.correlation);
    } else {
        chol.assign(d * d, 0.0);
        for (std::size_t i = 0; i < d; ++i) chol[i * d + i] = 1.0;
    }
    double kernel_ms = 0.0;
    return correlated_shocks_gpu(chol, d, scenarios, seed, use_shared_tiles,
                                 &kernel_ms);
}

// ---------------------------------------------------------------------------
// Portfolio risk
// ---------------------------------------------------------------------------
RiskResult portfolio_risk(const Portfolio& pf, const MarketData& market,
                          const SimulationConfig& price_cfg,
                          const RiskConfig& rcfg) {
    validate_market(market, "risk");
    pf.validate(market);
    validate_simulation(price_cfg, "risk");
    for (const auto& option : pf.options)
        if (option.time_steps > kMaxTimeSteps)
            throw std::invalid_argument(
                "risk: option time steps must be in [0, 100000]");
    if (pf.options.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("risk: too many portfolio instruments");
    if (rcfg.scenarios == 0)
        throw std::invalid_argument("risk: scenarios must be >= 1");
    if (rcfg.scenarios > kMaxScenarios)
        throw std::invalid_argument("risk: scenarios exceeds limit 10000000");
    if (market.assets.size() > std::numeric_limits<std::size_t>::max() -
                                   pf.options.size())
        throw std::invalid_argument("risk: scenario workload size overflow");
    const std::size_t scenario_width =
        market.assets.size() + pf.options.size();
    const std::size_t scenario_work = checked_product(
        rcfg.scenarios, scenario_width, "risk: scenario workload");
    if (scenario_work > kMaxScenarioValues)
        throw std::invalid_argument(
            "risk: scenarios x (assets + instruments) exceeds limit 50000000");
    if (!std::isfinite(rcfg.horizon) || rcfg.horizon <= 0.0)
        throw std::invalid_argument("risk: horizon must be finite and > 0");
    if (rcfg.confidence_levels.empty())
        throw std::invalid_argument(
            "risk: at least one confidence level is required");
    for (double level : rcfg.confidence_levels)
        if (!std::isfinite(level) || level <= 0.0 || level >= 1.0)
            throw std::invalid_argument(
                "risk: confidence levels must be finite and in (0,1)");
    for (const auto& option : pf.options)
        if (option.maturity < rcfg.horizon)
            throw std::invalid_argument(
                "risk: horizon must not exceed an instrument maturity");
    const std::size_t shock_values = checked_product(
        rcfg.scenarios, market.assets.size(), "risk: scenario shock matrix");
    if (shock_values > kMaxScenarioValues)
        throw std::invalid_argument(
            "risk: scenarios x assets exceeds limit 50000000");
    (void)checked_product(shock_values, sizeof(double),
                          "risk: scenario shock allocation");
    (void)checked_product(rcfg.scenarios, sizeof(double),
                          "risk: scenario value allocation");
    ensure_device();
    validate_device_block_size(price_cfg.block_size, "risk");
    RiskResult rr;
    WallTimer wall;
    // Base valuation + Greeks on the host (once, reused by all scenarios).
    auto instr = PortfolioRiskEngineCpu::prepare(pf, market, price_cfg);
    rr.base_value = PortfolioRiskEngineCpu::base_value(pf, instr);

    // Correlated shocks on device.
    std::vector<double> shocks =
        correlated_shocks(market, rcfg.scenarios, price_cfg.seed, false);

    const std::size_t d = market.assets.size();
    std::vector<double> spots(d), vols(d), divs(d);
    for (std::size_t a = 0; a < d; ++a) {
        spots[a] = market.assets[a].spot;
        vols[a] = market.assets[a].volatility;
        divs[a] = market.assets[a].dividend_yield;
    }
    const std::size_t n = instr.size();
    std::vector<int> i_asset(n), i_call(n), i_analytic(n);
    std::vector<double> i_strike(n), i_mat(n), i_base(n), i_delta(n), i_gamma(n),
        i_qty(n);
    for (std::size_t i = 0; i < n; ++i) {
        i_asset[i] = instr[i].asset_index;
        i_call[i] = is_call(instr[i].opt->type) ? 1 : 0;
        i_analytic[i] = instr[i].analytic_repricing ? 1 : 0;
        i_strike[i] = instr[i].opt->strike;
        i_mat[i] = instr[i].opt->maturity;
        i_base[i] = instr[i].base_price;
        i_delta[i] = instr[i].delta;
        i_gamma[i] = instr[i].gamma;
        i_qty[i] = instr[i].opt->quantity;
    }
    double kernel_ms = 0.0;
    std::vector<double> values = portfolio_losses_gpu(
        shocks, rcfg.scenarios, d, spots, vols, divs, market.risk_free_rate,
        rcfg.horizon, i_asset, i_call, i_analytic, i_strike, i_mat, i_base,
        i_delta, i_gamma, i_qty, &kernel_ms);

    // losses = V0 - V_i (positive = loss)
    std::vector<double> losses(values.size());
    double pnl_mean = 0.0, pnl_m2 = 0.0;
    std::size_t pnl_count = 0;
    double worst = -std::numeric_limits<double>::infinity();
    double best_gain = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        double l = rr.base_value - values[i];
        losses[i] = l;
        double pnl = -l;
        ++pnl_count;
        const double delta = pnl - pnl_mean;
        pnl_mean += delta / static_cast<double>(pnl_count);
        pnl_m2 += delta * (pnl - pnl_mean);
        worst = std::max(worst, l);
        best_gain = std::max(best_gain, pnl);
    }
    rr.pnl_mean = pnl_count ? pnl_mean : 0.0;
    const double pnl_variance = pnl_count
                                    ? pnl_m2 / static_cast<double>(pnl_count)
                                    : 0.0;
    rr.pnl_stddev = std::sqrt(std::max(pnl_variance, 0.0));
    rr.worst_loss = losses.empty() ? 0.0 : worst;
    rr.best_gain = losses.empty() ? 0.0 : best_gain;

    rr.estimates = var_cvar_gpu(std::move(losses), rcfg.confidence_levels);
    rr.scenarios = rcfg.scenarios;
    rr.horizon = rcfg.horizon;
    rr.backend = Backend::Cuda;
    rr.end_to_end_ms = wall.elapsed_ms();
    return rr;
}

// ---------------------------------------------------------------------------
// Version + divergence studies (the optimization storyline)
// ---------------------------------------------------------------------------
std::vector<VersionRow> european_version_study(const OptionSpec& opt, double spot,
                                               double r, double q, double sigma,
                                               const SimulationConfig& cfg) {
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::European,
                           "version study");
    ensure_device();
    validate_device_block_size(cfg.block_size, "version study");
    auto p = make_params(opt, spot, r, q, sigma, cfg.time_steps);
    std::vector<VersionRow> rows;
    auto add = [&](GpuVersion v, const char* note, const GpuMcRun& run) {
        VersionRow row;
        row.version = v;
        row.kernel_ms = run.kernel_ms;
        row.e2e_ms = run.e2e_ms;
        row.price = run.stats.price;
        row.std_error = run.stats.std_error;
        row.note = note;
        rows.push_back(row);
    };
    add(GpuVersion::V0_Naive, "AoS, split kernels, atomicAdd, sync copies",
        european_gpu(p, cfg.paths, cfg.seed, cfg.precision, GpuVersion::V0_Naive,
                     cfg.block_size));
    add(GpuVersion::V1_SoA, "SoA parameter fields with path-major batch reads",
        european_gpu(p, cfg.paths, cfg.seed, cfg.precision, GpuVersion::V1_SoA,
                     cfg.block_size));
    add(GpuVersion::V2_Fused, "fused sim+payoff, shared-tree block reduction",
        european_gpu(p, cfg.paths, cfg.seed, cfg.precision, GpuVersion::V2_Fused,
                     cfg.block_size));
    add(GpuVersion::V3_Reduction, "fused + warp-shuffle reduction",
        european_gpu(p, cfg.paths, cfg.seed, cfg.precision,
                     GpuVersion::V3_Reduction, cfg.block_size));
    add(GpuVersion::V4_Specialized, "European-specialized fused path kernel",
        european_gpu(p, cfg.paths, cfg.seed, cfg.precision,
                     GpuVersion::V4_Specialized, cfg.block_size));
    add(GpuVersion::V5_Memory, "parameters via __constant__ memory",
        european_gpu(p, cfg.paths, cfg.seed, cfg.precision, GpuVersion::V5_Memory,
                     cfg.block_size));
    add(GpuVersion::V6_Async, "4 streams + pinned memory chunked execution",
        european_async_gpu(p, cfg.paths, cfg.seed, cfg.precision, cfg.block_size,
                           4));
    // V7: same fused kernel, three precision configurations.
    add(GpuVersion::V7_Precision, "precision=float (float accumulate)",
        european_gpu(p, cfg.paths, cfg.seed, Precision::Float,
                     GpuVersion::V3_Reduction, cfg.block_size));
    add(GpuVersion::V7_Precision, "precision=mixed (float sim, double accum)",
        european_gpu(p, cfg.paths, cfg.seed, Precision::Mixed,
                     GpuVersion::V3_Reduction, cfg.block_size));
    add(GpuVersion::V7_Precision, "precision=double",
        european_gpu(p, cfg.paths, cfg.seed, Precision::Double,
                     GpuVersion::V3_Reduction, cfg.block_size));
    return rows;
}

std::vector<LayoutRow> layout_study(int instruments, std::size_t paths_each,
                                    std::uint64_t seed, int warmup, int repeats) {
    if (instruments <= 0 || instruments > 100'000)
        throw std::invalid_argument(
            "layout study: instruments must be in [1, 100000]");
    if (paths_each == 0)
        throw std::invalid_argument("layout study: paths must be >= 1");
    if (warmup < 0 || warmup > 101)
        throw std::invalid_argument("layout study: warmup must be in [0, 101]");
    if (repeats <= 0 || repeats > 101)
        throw std::invalid_argument("layout study: repeats must be in [1, 101]");
    const std::size_t total_paths = checked_product(
        static_cast<std::size_t>(instruments), paths_each,
        "layout study: instrument/path product");
    if (paths_each > kMaxPaths || total_paths > kMaxPaths)
        throw std::invalid_argument(
            "layout study: aggregate paths exceeds limit 100000000");
    (void)checked_product(total_paths, sizeof(curandStatePhilox4_32_10_t),
                          "layout study: RNG state allocation");
    ensure_device();
    std::vector<dev::OptionParams> batch;
    batch.reserve(static_cast<std::size_t>(instruments));
    for (int i = 0; i < instruments; ++i) {
        dev::OptionParams p{};
        p.spot = 100.0;
        p.strike = 95.0 + (i % 21);
        p.rate = 0.0425;
        p.dividend = 0.01;
        p.volatility = 0.20;
        p.maturity = 1.0;
        p.barrier = 0.0;
        p.is_call = 1;
        p.barrier_dir = 0;
        p.time_steps = 1;
        batch.push_back(p);
    }
    std::vector<LayoutRow> rows;
    for (bool aos : {true, false}) {
        LayoutRow row;
        row.aos = aos;
        for (int rep = 0; rep < warmup; ++rep)
            (void)european_soa_batch_gpu(batch, paths_each, seed, 256, aos);
        const int measured_repeats = std::max(repeats, 1);
        std::vector<double> kernel_samples, e2e_samples;
        kernel_samples.reserve(static_cast<std::size_t>(measured_repeats));
        e2e_samples.reserve(static_cast<std::size_t>(measured_repeats));
        for (int rep = 0; rep < measured_repeats; ++rep) {
            auto run = european_soa_batch_gpu(batch, paths_each, seed, 256, aos);
            kernel_samples.push_back(run.kernel_ms);
            e2e_samples.push_back(run.e2e_ms);
        }
        std::sort(kernel_samples.begin(), kernel_samples.end());
        std::sort(e2e_samples.begin(), e2e_samples.end());
        row.kernel_ms = kernel_samples[kernel_samples.size() / 2];
        row.e2e_ms = e2e_samples[e2e_samples.size() / 2];
        rows.push_back(row);
    }
    return rows;
}

std::vector<DivergenceRow> barrier_divergence_study(const OptionSpec& opt,
                                                    double spot, double r,
                                                    double q, double sigma,
                                                    const SimulationConfig& cfg) {
    // The study supplies its own near/medium/far barriers, so only the base
    // instrument family and the other numeric inputs are required here.
    validate_option_inputs(opt, spot, r, q, sigma, cfg, OptionFamily::Barrier,
                           "barrier divergence study",
                           /*require_barrier_value=*/false);
    ensure_device();
    validate_device_block_size(cfg.block_size, "barrier divergence study");
    std::vector<DivergenceRow> rows;
    struct Case {
        const char* name;
        double barrier_mult;
    };
    const Case cases[] = {{"near", 1.10}, {"medium", 1.30}, {"far", 1.80}};
    for (const auto& c : cases) {
        for (int mode = 0; mode <= 2; ++mode) {
            OptionSpec o = opt;
            bool up = opt.type == OptionType::BarrierUpOutCall ||
                      opt.type == OptionType::BarrierUpOutPut;
            o.barrier = up ? spot * c.barrier_mult : spot / c.barrier_mult;
            auto p = make_params(o, spot, r, q, sigma, cfg.time_steps);
            GpuMcRun run = barrier_gpu(p, cfg.paths, cfg.seed, cfg.precision,
                                       cfg.block_size, mode);
            DivergenceRow row;
            row.barrier_position = c.name;
            row.barrier = o.barrier;
            row.mode = mode;
            row.kernel_ms = run.kernel_ms;
            row.price = run.stats.price;
            row.knockout_fraction = run.extra;
            rows.push_back(row);
        }
    }
    return rows;
}

}  // namespace quantcuda::gpu
