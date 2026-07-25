// QuantCUDA Risk Engine - JSON configuration parsing and validation.
#include "quantcuda/json_config.hpp"

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "nlohmann/json.hpp"

using nlohmann::json;

namespace quantcuda {
namespace {

constexpr std::size_t kMaxConfigBytes = 8u * 1024u * 1024u;
constexpr std::uint64_t kMaxPaths = 100'000'000;
constexpr std::uint64_t kMaxScenarios = 10'000'000;
constexpr std::uint64_t kMaxTimeSteps = 100'000;
constexpr std::size_t kMaxAssets = 256;  // CUDA and risk kernels use this bound.
constexpr std::size_t kMaxOptions = 100'000;
constexpr std::size_t kMaxConfidenceLevels = 64;
constexpr std::uint64_t kMaxPricingWork = 5'000'000'000;
constexpr std::uint64_t kMaxRiskWork = 50'000'000;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("config error: " + msg);
}

void require_object(const json& value, const std::string& ctx) {
    if (!value.is_object()) fail(ctx + " must be an object");
}

void reject_unknown_keys(const json& value,
                         std::initializer_list<std::string_view> allowed,
                         const std::string& ctx) {
    require_object(value, ctx);
    for (auto it = value.begin(); it != value.end(); ++it) {
        bool known = false;
        for (std::string_view key : allowed) {
            if (it.key() == key) {
                known = true;
                break;
            }
        }
        if (!known) fail(ctx + ": unknown field '" + it.key() + "'");
    }
}

double get_double(const json& j, const char* key, const std::string& ctx) {
    if (!j.contains(key) || !j.at(key).is_number())
        fail(ctx + ": missing or non-numeric '" + key + "'");
    return j.at(key).get<double>();
}

std::uint64_t get_count(const json& j, const char* key, std::uint64_t fallback,
                        std::uint64_t maximum, const std::string& ctx) {
    if (!j.contains(key)) return fallback;
    if (!j.at(key).is_number_unsigned() && !j.at(key).is_number_integer())
        fail(ctx + ": '" + key + "' must be an integer");
    std::uint64_t value = 0;
    if (j.at(key).is_number_unsigned()) {
        value = j.at(key).get<std::uint64_t>();
    } else {
        const std::int64_t signed_value = j.at(key).get<std::int64_t>();
        if (signed_value <= 0) fail(ctx + ": '" + key + "' must be > 0");
        value = static_cast<std::uint64_t>(signed_value);
    }
    if (value == 0) fail(ctx + ": '" + key + "' must be > 0");
    if (value > maximum)
        fail(ctx + ": '" + key + "' exceeds safety limit " +
             std::to_string(maximum));
    return value;
}

std::uint64_t get_seed(const json& j, const char* key, std::uint64_t fallback,
                       const std::string& ctx) {
    if (!j.contains(key)) return fallback;
    if (!j.at(key).is_number_unsigned() && !j.at(key).is_number_integer())
        fail(ctx + ": '" + key + "' must be an unsigned integer");
    if (j.at(key).is_number_unsigned()) return j.at(key).get<std::uint64_t>();
    const std::int64_t value = j.at(key).get<std::int64_t>();
    if (value < 0) fail(ctx + ": '" + key + "' must be >= 0");
    return static_cast<std::uint64_t>(value);
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b,
                          const std::string& ctx) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        fail(ctx + " workload size overflows");
    return a + b;
}

std::uint64_t checked_product(std::uint64_t a, std::uint64_t b,
                              const std::string& ctx) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
        fail(ctx + " workload size overflows");
    return a * b;
}

EngineInput parse_engine_input(const json& j) {
    reject_unknown_keys(j, {"simulation", "market", "portfolio", "risk"},
                        "root");

    EngineInput in;

    // ---- simulation -------------------------------------------------------
    if (j.contains("simulation")) {
        const json& s = j.at("simulation");
        reject_unknown_keys(s, {"paths", "time_steps", "seed", "precision", "backend"},
                            "simulation");
        in.simulation.paths = static_cast<std::size_t>(
            get_count(s, "paths", 1'000'000, kMaxPaths, "simulation"));
        in.simulation.seed = get_seed(s, "seed", 42, "simulation");
        in.simulation.time_steps = static_cast<int>(
            get_count(s, "time_steps", 252, kMaxTimeSteps, "simulation"));
        if (s.contains("precision")) {
            if (!s.at("precision").is_string() ||
                !parse_precision(s.at("precision").get<std::string>(),
                                 in.simulation.precision))
                fail("simulation: precision must be float|double|mixed");
        }
        if (s.contains("backend")) {
            if (!s.at("backend").is_string() ||
                !parse_backend(s.at("backend").get<std::string>(), in.simulation.backend))
                fail("simulation: backend must be scalar|openmp|cuda");
        }
    }

    // ---- market -----------------------------------------------------------
    if (!j.contains("market")) fail("missing 'market' section");
    const json& m = j.at("market");
    reject_unknown_keys(m, {"risk_free_rate", "assets", "correlation"}, "market");
    in.market.risk_free_rate = get_double(m, "risk_free_rate", "market");
    if (in.market.risk_free_rate < -0.5 || in.market.risk_free_rate > 1.0)
        fail("market: risk_free_rate out of plausible range [-0.5, 1.0]");
    if (!m.contains("assets") || !m.at("assets").is_array() || m.at("assets").empty())
        fail("market: 'assets' must be a non-empty array");
    if (m.at("assets").size() > kMaxAssets)
        fail("market: at most " + std::to_string(kMaxAssets) + " assets supported");
    in.market.assets.reserve(m.at("assets").size());
    for (const json& a : m.at("assets")) {
        reject_unknown_keys(a, {"symbol", "spot", "volatility", "dividend_yield"},
                            "market.assets[]");
        AssetData ad;
        if (!a.contains("symbol") || !a.at("symbol").is_string())
            fail("market.assets[]: missing 'symbol'");
        ad.symbol = a.at("symbol").get<std::string>();
        if (ad.symbol.size() > 64) fail("market.assets[]: symbol too long");
        ad.spot = get_double(a, "spot", "asset " + ad.symbol);
        ad.volatility = get_double(a, "volatility", "asset " + ad.symbol);
        ad.dividend_yield = a.contains("dividend_yield")
                                ? get_double(a, "dividend_yield", "asset " + ad.symbol)
                                : 0.0;
        in.market.assets.push_back(std::move(ad));
    }
    if (m.contains("correlation")) {
        const json& c = m.at("correlation");
        if (!c.is_array() || c.size() != in.market.assets.size())
            fail("market: 'correlation' must have one row per asset");
        const std::uint64_t d = static_cast<std::uint64_t>(in.market.assets.size());
        (void)checked_product(d, d, "market correlation");
        in.market.correlation.reserve(c.size());
        for (const json& row : c) {
            if (!row.is_array() || row.size() != in.market.assets.size())
                fail("market: 'correlation' must be a square 2-D array");
            std::vector<double> parsed_row;
            parsed_row.reserve(row.size());
            for (const json& value : row) {
                if (!value.is_number()) fail("market: correlation entry not numeric");
                parsed_row.push_back(value.get<double>());
            }
            in.market.correlation.push_back(std::move(parsed_row));
        }
    }

    // ---- portfolio --------------------------------------------------------
    if (!j.contains("portfolio") || !j.at("portfolio").is_array())
        fail("missing 'portfolio' array");
    if (j.at("portfolio").size() > kMaxOptions)
        fail("portfolio: at most " + std::to_string(kMaxOptions) +
             " instruments supported");
    in.portfolio.options.reserve(j.at("portfolio").size());
    for (const json& o : j.at("portfolio")) {
        reject_unknown_keys(o, {"id", "type", "underlying", "strike", "maturity",
                                "quantity", "barrier", "time_steps"},
                            "portfolio[]");
        OptionSpec spec;
        if (!o.contains("id") || !o.at("id").is_string())
            fail("portfolio[]: missing or non-string 'id'");
        spec.id = o.at("id").get<std::string>();
        if (spec.id.size() > 128) fail("portfolio[]: id too long");
        if (!o.contains("type") || !o.at("type").is_string())
            fail("option " + spec.id + ": missing 'type'");
        const std::string type_str = o.at("type").get<std::string>();
        if (!parse_option_type(type_str, spec.type))
            fail("option " + spec.id + ": unknown option type '" + type_str + "'");
        if (!o.contains("underlying") || !o.at("underlying").is_string())
            fail("option " + spec.id + ": missing 'underlying'");
        spec.underlying = o.at("underlying").get<std::string>();
        spec.strike = get_double(o, "strike", "option " + spec.id);
        spec.maturity = get_double(o, "maturity", "option " + spec.id);
        spec.quantity = o.contains("quantity")
                            ? get_double(o, "quantity", "option " + spec.id)
                            : 1.0;
        spec.barrier = o.contains("barrier")
                           ? get_double(o, "barrier", "option " + spec.id)
                           : 0.0;
        if (o.contains("time_steps")) {
            spec.time_steps = static_cast<int>(
                get_count(o, "time_steps", 0, kMaxTimeSteps, "option " + spec.id));
        }
        in.portfolio.options.push_back(std::move(spec));
    }

    // ---- risk -------------------------------------------------------------
    if (j.contains("risk")) {
        const json& rk = j.at("risk");
        reject_unknown_keys(rk, {"scenarios", "horizon_years", "confidence_levels"},
                            "risk");
        in.risk.scenarios = static_cast<std::size_t>(
            get_count(rk, "scenarios", 100'000, kMaxScenarios, "risk"));
        if (rk.contains("horizon_years")) {
            const double h = get_double(rk, "horizon_years", "risk");
            if (h <= 0.0 || h > 1.0) fail("risk: horizon_years must be in (0, 1]");
            in.risk.horizon = h;
        }
        if (rk.contains("confidence_levels")) {
            const json& levels = rk.at("confidence_levels");
            if (!levels.is_array()) fail("risk: confidence_levels must be an array");
            if (levels.empty()) fail("risk: empty confidence_levels");
            if (levels.size() > kMaxConfidenceLevels)
                fail("risk: too many confidence_levels");
            in.risk.confidence_levels.clear();
            in.risk.confidence_levels.reserve(levels.size());
            for (const json& value : levels) {
                if (!value.is_number()) fail("risk: confidence level not numeric");
                const double alpha = value.get<double>();
                if (alpha <= 0.0 || alpha >= 1.0)
                    fail("risk: confidence level must be in (0,1)");
                in.risk.confidence_levels.push_back(alpha);
            }
        }
    }

    // Compute workload dimensions with checked arithmetic before any engine
    // uses them for indexing or allocation.
    std::uint64_t path_units = 0;
    for (const OptionSpec& option : in.portfolio.options) {
        const std::uint64_t steps = is_european(option.type)
                                        ? 1
                                        : static_cast<std::uint64_t>(
                                              option.time_steps > 0
                                                  ? option.time_steps
                                                  : in.simulation.time_steps);
        path_units = checked_add(path_units, steps, "simulation");
    }
    const std::uint64_t pricing_work = checked_product(
        static_cast<std::uint64_t>(in.simulation.paths), path_units, "simulation");
    if (pricing_work > kMaxPricingWork)
        fail("simulation aggregate path work exceeds safety limit " +
             std::to_string(kMaxPricingWork));
    const std::uint64_t scenario_units = checked_add(
        static_cast<std::uint64_t>(in.market.assets.size()),
        static_cast<std::uint64_t>(in.portfolio.options.size()), "risk");
    const std::uint64_t risk_work = checked_product(
        static_cast<std::uint64_t>(in.risk.scenarios), scenario_units, "risk");
    if (risk_work > kMaxRiskWork)
        fail("risk aggregate scenario work exceeds safety limit " +
             std::to_string(kMaxRiskWork));

    // Cross-field validation (also exercised by unit tests).
    try {
        in.market.validate();
        in.portfolio.validate(in.market);
    } catch (const std::invalid_argument& e) {
        fail(e.what());
    }
    return in;
}

}  // namespace

EngineInput load_engine_input(const std::string& json_text) {
    if (json_text.size() > kMaxConfigBytes)
        fail("JSON document exceeds 8 MiB size limit");
    try {
        return parse_engine_input(json::parse(json_text));
    } catch (const json::exception& e) {
        // Keep the public contract stable: callers only see std::runtime_error,
        // never nlohmann-specific parse/type/range exceptions.
        fail(std::string("invalid JSON: ") + e.what());
    }
}

EngineInput load_engine_input_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) fail("cannot open config file: " + path);
    const std::streampos end = file.tellg();
    if (end < 0) fail("cannot determine config file size: " + path);
    if (static_cast<std::uint64_t>(end) > kMaxConfigBytes)
        fail("config file exceeds 8 MiB size limit: " + path);
    std::string text(static_cast<std::size_t>(end), '\0');
    file.seekg(0, std::ios::beg);
    if (!text.empty() && !file.read(text.data(), static_cast<std::streamsize>(text.size())))
        fail("cannot read config file: " + path);
    return load_engine_input(text);
}

}  // namespace quantcuda
