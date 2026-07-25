// QuantCUDA Risk Engine - core type definitions
// Shared vocabulary across CPU engines, CUDA kernels, benchmarks and tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace quantcuda {

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum class OptionType {
    EuropeanCall,
    EuropeanPut,
    AsianCall,       // arithmetic average
    AsianPut,
    BarrierUpOutCall,
    BarrierUpOutPut,
    BarrierDownOutCall,
    BarrierDownOutPut,
};

enum class Backend { Scalar, OpenMP, Cuda };

enum class Precision { Float, Double, Mixed };  // Mixed: float paths, double accumulation

// GPU implementation generations kept alive for the optimization storyline.
enum class GpuVersion {
    V0_Naive,          // AoS, split kernels, global arrays, atomic add, sync copies
    V1_SoA,            // coalesced structure-of-arrays layout
    V2_Fused,          // fused simulation + payoff + block reduction
    V3_Reduction,      // warp-shuffle reductions
    V4_Specialized,    // per-type kernels, barrier branch/mask study
    V5_Memory,         // constant memory params, shared-memory Cholesky tiles
    V6_Async,          // streams + pinned memory chunked execution
    V7_Precision,      // float / double / mixed / compensated study
};

// ---------------------------------------------------------------------------
// Market and portfolio
// ---------------------------------------------------------------------------

struct AssetData {
    std::string symbol;
    double spot           = 0.0;
    double volatility     = 0.0;   // annualized, decimal (0.20 = 20%)
    double dividend_yield = 0.0;   // continuous
};

struct MarketData {
    double risk_free_rate = 0.0;
    std::vector<AssetData> assets;
    // Row-major d x d correlation matrix. Empty => single-asset / independent.
    std::vector<std::vector<double>> correlation;

    int asset_index(const std::string& symbol) const;
    void validate() const;  // throws std::invalid_argument on bad input
};

struct OptionSpec {
    std::string id;
    OptionType  type = OptionType::EuropeanCall;
    std::string underlying;      // asset symbol
    double strike    = 0.0;
    double maturity  = 0.0;      // years
    double quantity  = 1.0;      // signed: long positive, short negative
    double barrier   = 0.0;      // only meaningful for barrier types
    int    time_steps = 0;       // 0 => use SimulationConfig default
};

struct Portfolio {
    std::vector<OptionSpec> options;
    void validate(const MarketData& market) const;
    bool empty() const { return options.empty(); }
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SimulationConfig {
    std::size_t paths      = 1'000'000;
    int         time_steps = 252;     // path steps for Asian / barrier
    std::uint64_t seed     = 42;
    Precision   precision  = Precision::Double;
    Backend     backend    = Backend::Scalar;
    int         block_size = 256;     // CUDA
    GpuVersion  gpu_version = GpuVersion::V7_Precision;
    int         openmp_threads = 0;   // 0 => library default
};

struct RiskConfig {
    std::size_t scenarios = 100'000;
    double horizon = 1.0 / 252.0;     // risk horizon in years (1 trading day)
    std::vector<double> confidence_levels {0.95, 0.99};
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

struct Greeks {
    double delta = 0.0;
    double gamma = 0.0;
    double vega  = 0.0;   // per 1.00 (100%) vol move; reports scale as needed
};

struct OptionPricingResult {
    std::string id;
    OptionType  type = OptionType::EuropeanCall;
    double price         = 0.0;   // per 1 unit of underlying
    double std_error     = 0.0;   // Monte Carlo standard error (0 for analytic)
    Greeks greeks;
    bool   has_greeks    = false;
    double analytic_ref  = 0.0;   // Black-Scholes where applicable
    bool   has_analytic  = false;
};

struct PricingResult {
    std::vector<OptionPricingResult> options;
    double portfolio_value = 0.0;  // sum(quantity * price)
    // timing
    double kernel_time_ms   = 0.0;  // device-only computation (== e2e on CPU)
    double end_to_end_ms    = 0.0;
    double paths_per_second = 0.0;
    Backend    backend   = Backend::Scalar;
    Precision  precision = Precision::Double;
    std::size_t paths    = 0;
    std::uint64_t seed   = 0;
};

struct VarEstimate {
    double confidence = 0.0;
    double var  = 0.0;   // loss quantile, positive = loss
    double cvar = 0.0;   // expected loss beyond VaR
};

struct RiskResult {
    double base_value = 0.0;                        // V0
    std::size_t scenarios = 0;
    double horizon = 0.0;
    std::vector<VarEstimate> estimates;
    // scenario distribution summary
    double pnl_mean   = 0.0;
    double pnl_stddev = 0.0;
    double worst_loss = 0.0;
    double best_gain  = 0.0;
    double end_to_end_ms = 0.0;
    Backend backend = Backend::Scalar;
};

// ---------------------------------------------------------------------------
// String conversion helpers (used by CLI, JSON loader, reports)
// ---------------------------------------------------------------------------

const char* to_string(OptionType t);
const char* to_string(Backend b);
const char* to_string(Precision p);
const char* to_string(GpuVersion v);

bool parse_option_type(const std::string& s, OptionType& out);
bool parse_backend(const std::string& s, Backend& out);
bool parse_precision(const std::string& s, Precision& out);
bool parse_gpu_version(const std::string& s, GpuVersion& out);

bool is_barrier(OptionType t);
bool is_asian(OptionType t);
bool is_european(OptionType t);
bool is_call(OptionType t);

// Payoff of a European-style option at terminal price ST (T already reached).
inline double european_payoff(OptionType t, double spot_T, double strike) {
    return is_call(t) ? (spot_T > strike ? spot_T - strike : 0.0)
                      : (strike > spot_T ? strike - spot_T : 0.0);
}

}  // namespace quantcuda
