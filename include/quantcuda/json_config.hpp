// QuantCUDA Risk Engine - JSON portfolio / simulation configuration loading.
#pragma once

#include <string>

#include "types.hpp"

namespace quantcuda {

struct EngineInput {
    SimulationConfig simulation;
    MarketData market;
    Portfolio portfolio;
    RiskConfig risk;
};

// Load and validate a portfolio JSON document (see configs/*.json for the
// schema). Throws std::runtime_error with a descriptive message on any
// malformed or missing field. Input size, workload dimensions, aggregate work,
// unknown keys, and negative values are rejected before engine dispatch.
EngineInput load_engine_input(const std::string& json_text);
EngineInput load_engine_input_file(const std::string& path);

}  // namespace quantcuda
