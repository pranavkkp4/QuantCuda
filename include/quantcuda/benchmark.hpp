// QuantCUDA Risk Engine - benchmark runner.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

namespace quantcuda {

struct BenchmarkOptions {
    std::string suite = "european-throughput";  // see --suite help
    std::vector<Backend> backends {Backend::Scalar, Backend::OpenMP, Backend::Cuda};
    std::string output_csv;                     // empty => stdout only
    std::string config_file;                    // optional portfolio for suite D
    int repeats = 5;                            // measured runs per data point
    int warmup = 1;
    std::uint64_t seed = 42;

    // Workload overrides (0 => suite default).
    struct Overrides {
        std::uint64_t paths = 0;
        int steps = 0;
        int instruments = 0;
        std::uint64_t scenarios = 0;
    } overrides;
};

// Executes a benchmark suite and optionally writes a CSV report. Returns process
// exit code. GPU rows are skipped (with a note) when no GPU is available.
int run_benchmark(const BenchmarkOptions& opts);

// Environment description recorded in every benchmark report.
std::string environment_report();

}  // namespace quantcuda
