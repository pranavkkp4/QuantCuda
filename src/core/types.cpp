// QuantCUDA Risk Engine - type helpers and portfolio/market validation.
#include "quantcuda/types.hpp"

#include "quantcuda/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace quantcuda {

// ---------------------------------------------------------------------------
// MarketData / Portfolio validation
// ---------------------------------------------------------------------------

int MarketData::asset_index(const std::string& symbol) const {
    for (std::size_t i = 0; i < assets.size(); ++i)
        if (assets[i].symbol == symbol) return static_cast<int>(i);
    return -1;
}

void MarketData::validate() const {
    if (assets.empty()) throw std::invalid_argument("market: at least one asset is required");
    if (!std::isfinite(risk_free_rate))
        throw std::invalid_argument("market: risk-free rate must be finite");
    std::unordered_set<std::string> symbols;
    for (const auto& a : assets) {
        if (a.symbol.empty()) throw std::invalid_argument("market: asset symbol is empty");
        if (!symbols.insert(a.symbol).second)
            throw std::invalid_argument("market: duplicate asset symbol '" + a.symbol + "'");
        if (!std::isfinite(a.spot) || !std::isfinite(a.volatility) ||
            !std::isfinite(a.dividend_yield))
            throw std::invalid_argument("market: asset values must be finite (" +
                                        a.symbol + ")");
        if (a.spot < 0.0)     throw std::invalid_argument("market: spot must be >= 0 (" + a.symbol + ")");
        if (a.volatility < 0.0)
            throw std::invalid_argument("market: volatility must be >= 0 (" + a.symbol + ")");
    }
    const std::size_t d = assets.size();
    if (!correlation.empty()) {
        if (correlation.size() != d)
            throw std::invalid_argument("market: correlation matrix row count != number of assets");
        for (const auto& row : correlation) {
            if (row.size() != d)
                throw std::invalid_argument("market: correlation matrix is not square");
            for (double v : row)
                if (!std::isfinite(v) || v < -1.0 || v > 1.0)
                    throw std::invalid_argument("market: correlation entry outside [-1, 1]");
        }
        for (std::size_t i = 0; i < d; ++i) {
            if (std::fabs(correlation[i][i] - 1.0) > 1e-9)
                throw std::invalid_argument("market: correlation diagonal must be 1");
            for (std::size_t j = i + 1; j < d; ++j)
                if (std::fabs(correlation[i][j] - correlation[j][i]) > 1e-9)
                    throw std::invalid_argument("market: correlation matrix must be symmetric");
        }
        // Shape/symmetry/range checks above produce domain-specific messages;
        // Cholesky then enforces the positive-definiteness promised by the JSON schema.
        (void)CorrelationEngine::cholesky(correlation);
    }
}

void Portfolio::validate(const MarketData& market) const {
    std::unordered_set<std::string> ids;
    for (const auto& o : options) {
        if (o.id.empty()) throw std::invalid_argument("option id must not be empty");
        if (!ids.insert(o.id).second)
            throw std::invalid_argument("portfolio: duplicate option id '" + o.id + "'");
        if (!is_european(o.type) && !is_asian(o.type) && !is_barrier(o.type))
            throw std::invalid_argument("option " + o.id + ": invalid option type");
        if (!std::isfinite(o.strike) || !std::isfinite(o.maturity) ||
            !std::isfinite(o.quantity) || !std::isfinite(o.barrier))
            throw std::invalid_argument("option " + o.id + ": numeric fields must be finite");
        if (o.strike < 0.0)
            throw std::invalid_argument("option " + o.id + ": strike must be >= 0");
        if (o.maturity < 0.0)
            throw std::invalid_argument("option " + o.id + ": maturity must be >= 0");
        if (o.quantity == 0.0)
            throw std::invalid_argument("option " + o.id + ": quantity must be non-zero");
        if (market.asset_index(o.underlying) < 0)
            throw std::invalid_argument("option " + o.id + ": unknown underlying '" +
                                        o.underlying + "'");
        if (is_barrier(o.type) && o.barrier <= 0.0)
            throw std::invalid_argument("option " + o.id + ": barrier option needs barrier > 0");
        if (o.time_steps < 0)
            throw std::invalid_argument("option " + o.id + ": time_steps must be >= 0");
    }
}

// ---------------------------------------------------------------------------
// String conversions
// ---------------------------------------------------------------------------

const char* to_string(OptionType t) {
    switch (t) {
        case OptionType::EuropeanCall: return "european_call";
        case OptionType::EuropeanPut: return "european_put";
        case OptionType::AsianCall: return "asian_call";
        case OptionType::AsianPut: return "asian_put";
        case OptionType::BarrierUpOutCall: return "barrier_up_out_call";
        case OptionType::BarrierUpOutPut: return "barrier_up_out_put";
        case OptionType::BarrierDownOutCall: return "barrier_down_out_call";
        case OptionType::BarrierDownOutPut: return "barrier_down_out_put";
    }
    return "unknown";
}

const char* to_string(Backend b) {
    switch (b) {
        case Backend::Scalar: return "scalar";
        case Backend::OpenMP: return "openmp";
        case Backend::Cuda: return "cuda";
    }
    return "unknown";
}

const char* to_string(Precision p) {
    switch (p) {
        case Precision::Float: return "float";
        case Precision::Double: return "double";
        case Precision::Mixed: return "mixed";
    }
    return "unknown";
}

const char* to_string(GpuVersion v) {
    switch (v) {
        case GpuVersion::V0_Naive: return "v0-naive";
        case GpuVersion::V1_SoA: return "v1-soa";
        case GpuVersion::V2_Fused: return "v2-fused";
        case GpuVersion::V3_Reduction: return "v3-reduction";
        case GpuVersion::V4_Specialized: return "v4-specialized";
        case GpuVersion::V5_Memory: return "v5-memory";
        case GpuVersion::V6_Async: return "v6-async";
        case GpuVersion::V7_Precision: return "v7-precision";
    }
    return "unknown";
}

bool parse_option_type(const std::string& s, OptionType& out) {
    static const std::pair<const char*, OptionType> table[] = {
        {"european_call", OptionType::EuropeanCall},
        {"european_put", OptionType::EuropeanPut},
        {"asian_call", OptionType::AsianCall},
        {"asian_put", OptionType::AsianPut},
        {"barrier_up_out_call", OptionType::BarrierUpOutCall},
        {"barrier_up_out_put", OptionType::BarrierUpOutPut},
        {"barrier_down_out_call", OptionType::BarrierDownOutCall},
        {"barrier_down_out_put", OptionType::BarrierDownOutPut},
        // convenient aliases
        {"call", OptionType::EuropeanCall},
        {"put", OptionType::EuropeanPut},
    };
    for (const auto& [k, v] : table)
        if (s == k) { out = v; return true; }
    return false;
}

bool parse_backend(const std::string& s, Backend& out) {
    if (s == "scalar") { out = Backend::Scalar; return true; }
    if (s == "openmp") { out = Backend::OpenMP; return true; }
    if (s == "cuda")   { out = Backend::Cuda;   return true; }
    return false;
}

bool parse_precision(const std::string& s, Precision& out) {
    if (s == "float")  { out = Precision::Float;  return true; }
    if (s == "double") { out = Precision::Double; return true; }
    if (s == "mixed")  { out = Precision::Mixed;  return true; }
    return false;
}

bool parse_gpu_version(const std::string& s, GpuVersion& out) {
    static const std::pair<const char*, GpuVersion> table[] = {
        {"v0", GpuVersion::V0_Naive},   {"v0-naive", GpuVersion::V0_Naive},
        {"v1", GpuVersion::V1_SoA},     {"v1-soa", GpuVersion::V1_SoA},
        {"v2", GpuVersion::V2_Fused},   {"v2-fused", GpuVersion::V2_Fused},
        {"v3", GpuVersion::V3_Reduction},{"v3-reduction", GpuVersion::V3_Reduction},
        {"v4", GpuVersion::V4_Specialized},{"v4-specialized", GpuVersion::V4_Specialized},
        {"v5", GpuVersion::V5_Memory},  {"v5-memory", GpuVersion::V5_Memory},
        {"v6", GpuVersion::V6_Async},   {"v6-async", GpuVersion::V6_Async},
        {"v7", GpuVersion::V7_Precision},{"v7-precision", GpuVersion::V7_Precision},
    };
    for (const auto& [k, v] : table)
        if (s == k) { out = v; return true; }
    return false;
}

bool is_barrier(OptionType t) {
    return t == OptionType::BarrierUpOutCall || t == OptionType::BarrierUpOutPut ||
           t == OptionType::BarrierDownOutCall || t == OptionType::BarrierDownOutPut;
}
bool is_asian(OptionType t) {
    return t == OptionType::AsianCall || t == OptionType::AsianPut;
}
bool is_european(OptionType t) {
    return t == OptionType::EuropeanCall || t == OptionType::EuropeanPut;
}
bool is_call(OptionType t) {
    return t == OptionType::EuropeanCall || t == OptionType::AsianCall ||
           t == OptionType::BarrierUpOutCall || t == OptionType::BarrierDownOutCall;
}

}  // namespace quantcuda
